import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import os
from PIL import Image
import torch.nn.functional as F

print("Data preprocessing started.")

# Path to dataset on Ranpur
# oasis_path = "/home/groups/comp3710/OASIS"

# load into 2D tensors needed that are grey (so colour cant be considered feature)
transform_image = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)), # Resize
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128), interpolation=Image.NEAREST), # Resize
    transforms.ToTensor()
])

def load_image(image_folder, mask_folder):
    images = []
    masks = []

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    
    for img in image_files:

        parts = img.split('_')
        slice_number = parts[-1].split('.')[0]  # last part before '.nii.png'

        case_number = int(parts[1])
        mask_case_number = f"{case_number:03d}"
        mask_name = f"seg_{mask_case_number}_slice_{slice_number}.nii.png"
        mask_path = os.path.join(mask_folder, mask_name)
        img_path = os.path.join(image_folder, img)

        if os.path.isfile(mask_path):
            img = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")

            img = transform_image(img)
            mask = transform_mask(mask)
            mask = (mask > 0.5).float()  # convert mask to binary
            
            images.append(img)
            masks.append(mask)
        else:
            continue

    return torch.stack(images), torch.stack(masks)

# Images and Masks are now loaded into tensors

def get_data_loaders(batch_size):

    # Process datasets together to ensure images and masks aligned
    train_set, train_mask = load_image("/home/groups/comp3710/OASIS/keras_png_slices_train",
                                        "/home/groups/comp3710/OASIS/keras_png_slices_seg_train")
    train_mask = train_mask.float()

    validation_set, validation_mask = load_image("/home/groups/comp3710/OASIS/keras_png_slices_validate",
                                                  "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate")
    validation_mask = validation_mask.float()

    test_set, test_mask = load_image("/home/groups/comp3710/OASIS/keras_png_slices_test",
                                      "/home/groups/comp3710/OASIS/keras_png_slices_seg_test")
    test_mask = test_mask.float()

    # Put tensors into data loaders, matching images with masks 

    train_tensor_set = TensorDataset(train_set, train_mask)
    validation_tensor_set = TensorDataset(validation_set, validation_mask)
    test_tensor_set = TensorDataset(test_set, test_mask)

    # batch_size previously 128
    train_loader = DataLoader(train_tensor_set, batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_tensor_set, batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_tensor_set, batch_size, shuffle=False, num_workers=2)

    print("Data preprocessing complete.")

    return train_tensor_set, validation_tensor_set, test_tensor_set, train_loader, validation_loader, test_loader


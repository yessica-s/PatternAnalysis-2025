import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import os
from PIL import Image

# Path to dataset on Ranpur
# oasis_path = "/home/groups/comp3710/OASIS"

# load into 2D tensors needed that are grey (so colour cant be considered feature)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)), # Resize
    transforms.ToTensor()
])

# Return two tensors of image and masks
def load_images(folder):
    image_list = []
    for filename in sorted(os.listdir(folder)): # Make sure images and masks matched correctly by sorting
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = transform(img) # greyscale, tensor, resize from above
            image_list.append(img)

    images_tensor = torch.stack(image_list) # put them all into one giant tensor
    return images_tensor

# Images and Masks are now loaded into tensors

train_set = load_images("/home/groups/comp3710/OASIS/keras_png_slices_train")
validation_set = load_images("/home/groups/comp3710/OASIS/keras_png_slices_validate")
test_set = load_images("/home/groups/comp3710/OASIS/keras_png_slices_test")

train_mask = load_images("/home/groups/comp3710/OASIS/keras_png_slices_seg_train").float()
validation_mask = load_images("/home/groups/comp3710/OASIS/keras_png_slices_seg_validate").float()
test_mask = load_images("/home/groups/comp3710/OASIS/keras_png_slices_seg_test").float()

# Put tensors into data loaders, matching images with masks 

train_tensor_set = TensorDataset(train_set, train_mask)
validation_tensor_set = TensorDataset(validation_set, validation_mask)
test_tensor_set = TensorDataset(test_set, test_mask)

train_loader = DataLoader(train_tensor_set, batch_size=128, shuffle=True, num_workers=2)
validation_loader = DataLoader(validation_tensor_set, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(test_tensor_set, batch_size=128, shuffle=False, num_workers=2)
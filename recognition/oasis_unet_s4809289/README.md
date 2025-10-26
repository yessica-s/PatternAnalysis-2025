# Segmentation of 2D OASIS Brain Dataset with Improved UNet

##### COMP3710 Pattern Recognition & Analysis | Semester 2 2025
##### Yessica Samtani | 48092892

## Problem

This task aims to solve the problem of segmenting the 2D brain dataset. 
This involves identifying distinct features and areas of brain tissue for 
the purposes of medical imaging and diagnoses. We aim to develop a Machine 
Learning model which can accurately extract and identify patterns in brain tissue from MRI Scans. 

## Implementation

This model implements an Improved UNet model. This model aims to take the images of brain scans
and extract features from them via an Encoder. From there, these layers and features are attempted
to be recreated by the model in order to predict an accurate brain tissue mask. Finally, the model 
produces a 2D segmentation map to represent the predicted different features of the brain.

The Improved UNet is different from the Simple UNet model in the following main ways:
* LeakyReLU activation function used over ReLU
* Batch Normalization after all convolutional layers

#### Data Preprocessing

The following steps are involved in pre-processing the dataset.

1. Images and Masks are resized to 128, 128
2. Images and Masks converted to grayscale
3. Interpolation implemented for image masks
4. Masks binarized
5. Images and Masks gathered in Tensors and Data Loaders
   (these were separated by Training, Validation, and Test Images)

#### Training

* During this stage, the model undergoes 50 epochs of training, and typically reaches the desired dice coefficient of > 0.9 after 2-5 epochs.
* The model calculated both training and validaton loss.
* Dice Loss loss function was implemented

#### Testing

After testing, the average dice coefficient similarity score was found to be 0.9816, alongside an average Dice Loss of 0.0185.

## Visualization

An example of some model predictions in comparison to ground truth masks is shown below in Figures 1 and 2, representing predictions during Training and Testing respectively.

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/61d1153a-5d23-4c13-88e9-4305a1b19626" />

_Figure 1. Ground Truth vs Predictions during Training_

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/2b8027b8-2337-4b02-be67-b2fa9f4942c5" />

_Figure 2. Ground Truth vs Predictions on Test Set_

## Dependencies

The following commands were used to setup the environment and install required packages/libraries.

```
conda create --name torch python=3.10
conda activate torch
pip install torch torchvision
pip install numpy matplotlib
```

## Usage

The following options can be used to run the model on the Ranpur cluster. It is assumed
that the model is run on the Ranpur cluster as the model directly references file paths
relative to the location of the 2D OASIS dataset on Ranpur.

#### Option 1

```
vim runner
```
Paste the following contents in the editor and save the file.

```
#/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4       
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00           # 2 hour time limit
#SBATCH --partition=a100
#SBATCH --job-name=improved_unet
#SBATCH -o unet.out

conda activate torch
python predict.py
```
Run the following command in the location of the runner file.
```
sbatch runner
```

#### Option 2
Alternatively, run the following commands directly in the terminal
```
conda activate torch
srun -p a100-test --gres=shard:1 python predict.py
```



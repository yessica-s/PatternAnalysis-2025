import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# REFERENCE: https://github.com/fwrhine/ImprovedUNet/blob/master/improved_unet.py
class uNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base_channels=16, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p
        self.base_channels = base_channels

        # Encode
        self.enc1 = self._conv_block(in_channels, 32, dropout_p)
        self.enc2 = self._conv_block(32, 64, dropout_p)
        self.enc3 = self._conv_block(64, 128, dropout_p)
        self.enc4 = self._conv_block(128, 256, dropout_p)

        self.bottleneck = self._conv_block(256, 512, dropout_p)

        # Deocde
        self.dec4 = self._conv_block(512 + 256, 256, dropout_p)
        self.dec3 = self._conv_block(256 + 128, 128, dropout_p)
        self.dec2 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec1 = self._conv_block(64 + 32, 32, dropout_p)

        self.final = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _conv_block(self, in_channels, out_channels, dropout_p=0.3):
        # Convulution block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )
    
    # Encoder function
    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        return e1, e2, e3, e4, b
    
    # Decoder function
    def decode(self, e1, e2, e3, e4, b):
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

        return d1

    def forward(self, x):
        e1, e2, e3, e4, bottleneck = self.encode(x)
        d1 = self.decode(e1, e2, e3, e4, bottleneck)

        return torch.sigmoid(self.final(d1)) # Return sigmoid output

# Multi-class dice loss: REFERENCE: https://www.kaggle.com/code/doyeonkimmm/multiclass-segmentation-unet-modified-dice
    #   - calculate DICE on each class
    #   - average is the final loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predicted, true):
        # REFERENCE: https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
        predicted = predicted.contiguous()
        true = true.contiguous()
        
        intersection = (predicted * true).sum(dim=(2,3))
        union = (predicted.sum(dim=(2,3)) + true.sum(dim=(2,3)))

        # REFERENCE: https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
        # addition of smooth avoids div 0 error
        dice_coefficient = (2 * intersection + self.smooth) / (union + self.smooth)
        avg_dice_coefficient = dice_coefficient.mean() # mean dice coeff across the 4 classes
        dice_loss = 1 - avg_dice_coefficient

        return dice_loss
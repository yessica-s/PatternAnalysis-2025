import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# leaky_relu_activation = nn.LeakyReLU(negative_slope=0.01) # slope of 10^-2

class uNet (nn.Module):

    def __init__(self, in_channels=1, out_channels=4, base_channels=16, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p
        self.base_channels = base_channels

        # Encode
        self.enc1 = self._conv_block(in_channels, 16, dropout_p)
        self.enc2 = self._conv_block(16, 32, dropout_p)
        self.enc3 = self._conv_block(32, 64, dropout_p)
        self.enc4 = self._conv_block(64, 128, dropout_p)

        self.bottleneck = self._conv_block(128, 256, dropout_p)

        # Deocde
        self.dec4 = self._conv_block(256 + 128, 128, dropout_p)
        self.dec3 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec2 = self._conv_block(64 + 32, 32, dropout_p)
        self.dec1 = self._conv_block(32 + 16, 16, dropout_p)

        self.final = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels, dropout_p=0.3):
        # Convulution block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(dropout_p)
        )
    
    def forward(self, x):
        e1, e2, e3, e4, bottleneck = self.encode(x)
        d1 = self.decode(e1, e2, e3, e4, bottleneck)
        final = self.final(d1)

        return final
    

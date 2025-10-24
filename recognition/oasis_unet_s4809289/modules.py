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

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
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
        final = self.final(d1)
        final = torch.softmax(final, dim=1) # activation function - 

        return final
    


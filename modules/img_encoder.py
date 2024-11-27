import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.gelu = nn.GELU()
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.norm2(x)
        x = self.conv2(x)
        return x + residual 


class ImageEncoder(nn.Module):
    def __init__(self, input_channels = 3, base_channels = 64, num_k = 4):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size = 7, stride = 2, padding = 3),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size = 3, padding = 1)
        )

        self.num_k = num_k
        self.num_blocks = [3, 2, 2, 1] # assume there are four levels of features

        self.residual_blocks = nn.ModuleList([
            nn.Sequential(*[ResidualConvBlock(base_channels) for j in self.num_blocks])
            for i in range(num_k) 
        ])

        self.mean_pooling = nn.ModuleList([
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1) for i in range(num_k - 1)
        ])

        self.to_feature = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(base_channels),
                nn.Conv2d(base_channels, base_channels, kernel_size = 1) # out_channels might be different
            )
            for i in range(num_k) # matches the length of residual_blocks
        ])

    def forward(self, x):
        """
        Extract image features at multiple levels.

        Args:
            x: Input image tensor [B, 3, H, W] (* H, W: resized)

        Returns:
            List of feature tensors [I_0, I_2, I_4, I_8]
        """
        features = []
        x = self.initial_conv(x)  
        
        for i, res_block in enumerate(self.residual_blocks):
            x = res_block(x)
            features.append(self.to_feature[i](x)) 
            if i != (self.num_k - 1):
                x = self.mean_pooling[i](x)

        return features
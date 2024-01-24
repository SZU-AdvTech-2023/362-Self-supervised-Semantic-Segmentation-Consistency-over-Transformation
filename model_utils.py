import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np


class LearnableAffineTransform(nn.Module):
    def __init__(self):
        super(LearnableAffineTransform, self).__init__()
        # 假设输入特征图的形状为 N x 64 x 256 x 256
        # 定义局部化网络，用来预测仿射变换的6个参数
        self.localization = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),##64*64*256
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        # 全连接层，输出6个仿射变换参数
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # 使用身份变换初始化权重/bias
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 256)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        input_feature_map = x

        grid = F.affine_grid(theta, input_feature_map.size(), align_corners=False)
        x = F.grid_sample(input_feature_map, grid, align_corners=False)

        return x


class InceptionLKAModule(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the Inception LKA (I-LKA) module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        """
        super(InceptionLKAModule, self).__init__()
        
        kernels = [3, 5, 7]
        paddings = [1, 4, 9]
        dilations = [1, 2, 3]
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.spatial_convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=kernels[i], stride=1,
                                                    padding=paddings[i], groups=in_channels,
                                                    dilation=dilations[i]) for i in range(len(kernels))])
        self.conv1 = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        attn = self.conv0(x)
        spatial_attns = [conv(attn) for conv in self.spatial_convs]
        attn = torch.cat(spatial_attns, dim=1)
        attn = self.conv1(attn)
        return original_input * attn
    
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the Attention module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        
        :return: Output tensor after applying attention module
        """
        super(AttentionModule, self).__init__()
        self.proj_1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating = InceptionLKAModule(in_channels)
        self.proj_1x1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        x = self.proj_1x1_1(x)
        x = self.activation(x)
        x = self.spatial_gating(x)
        x = self.proj_1x1_2(x)
        
        return original_input + x
    
    
class DeformableConv(nn.Module):
    def __init__(self, nChannel, batch=1, height=256, width=256):
        """
        Initialize the Deformable Convolution module.

        :param batch: Batch size
        :type batch: int
        :param height: Height of input feature maps
        :type height: int
        :param width: Width of input feature maps
        :type width: int
        
        :return: Output tensor after applying deformable convolution
        """
        super().__init__()
        kh, kw = 3, 3
        self.weight = nn.Parameter(torch.rand(nChannel, nChannel, kh, kw))
        self.offset = nn.Parameter(torch.rand(batch, 2 * kh * kw, height, width))
        self.mask = nn.Parameter(torch.rand(batch, kh * kw, height, width))

    def forward(self, x):
        return deform_conv2d(x, self.offset, self.weight, mask=self.mask, padding=1)
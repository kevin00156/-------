import torch.nn as nn
import torch
from torch.nn import functional as F

# 定義 CNN 模型結構
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = nn.functional.relu(out)
        return out + x



class ResNetModel(nn.Module):
    def __init__(self,
                 input_channel: int = 32, 
                 output_channel : int =32,
                 input_size: int = 3,
                 output_size: int = 2,
                 n_blocks: int = 10):
        super(ResNetModel, self).__init__()

        self.n_channels = input_channel 

        self.conv1 = nn.Conv2d(input_size, self.n_channels, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(self.n_channels) for _ in range(n_blocks)]
        )
        self.fc1 = nn.Linear(input_channel * 8 * 8, output_channel)
        self.fc2 = nn.Linear(output_channel, output_size)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

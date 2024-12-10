import torch.nn as nn
from typing import Sequence
import torch

# 定義 CNN 模型結構
class CNNModel(nn.Module):
    def __init__(self,
                 input_size: int = 32, 
                 num_classes: int = 120,
                 hidden_layers: Sequence[int] = [256, 128],
                 activation_function: nn.Module = nn.ReLU(),
                 dropout_rate: float = 0.3):
        super(CNNModel, self).__init__()
        
        # 捲積層
        num_pool_layers = 1
        start_size = 16
        conv_sizes = [start_size * (2 ** i) for i in range(num_pool_layers)]
        conv_layers = []
        prev_size = 3
        for i, conv_size in enumerate(conv_sizes):
            conv_layers.append(nn.Conv2d(prev_size, conv_size, kernel_size=5, stride=1, padding=1))
            prev_size = conv_size
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            conv_layers.append(nn.Dropout(p=dropout_rate))
            conv_layers.append(nn.BatchNorm2d(conv_size))
            conv_layers.append(activation_function)
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 使用虛擬張量來計算卷積層的輸出大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.conv_layers(dummy_input)
            conv_output_size = dummy_output.view(1, -1).size(1)

        # 全連接層
        layers = []
        prev_size = conv_output_size

        # 添加隱藏層
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_function)
            if i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_size = hidden_size
        
        # 添加輸出層
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
if __name__ == "__main__":
    model = CNNModel()
    print(model)

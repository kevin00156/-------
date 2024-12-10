import torch.nn as nn
from typing import Sequence
import torch

# 定義 CNN 模型結構
class CNNModelForCIFAR100_Simple(nn.Module):
    def __init__(self,
                 input_size: int = 64, 
                 num_classes: int = 100,
                 hidden_layers: Sequence[int] = [256, 128, 64],
                 activation_function: str = 'LeakyReLU',
                 dropout_rate: float = 0.3,
                 num_pool_layers: int = 2,
                 conv_start_size: int = 16):
        super(CNNModelForCIFAR100_Simple, self).__init__()
        
        # 根據字符串選擇激活函數
        if activation_function == 'LeakyReLU':
            self.activation_function = nn.LeakyReLU()
        elif activation_function == 'ReLU':
            self.activation_function = nn.ReLU()
        elif activation_function == 'Sigmoid':
            self.activation_function = nn.Sigmoid()
        elif activation_function == 'Tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'ELU':
            self.activation_function = nn.ELU()
        elif activation_function == 'SELU':
            self.activation_function = nn.SELU()
        elif activation_function == 'GELU':
            self.activation_function = nn.GELU()
        elif activation_function == 'RReLU':
            self.activation_function = nn.RReLU()
        elif activation_function == 'PReLU':
            self.activation_function = nn.PReLU()
        elif activation_function == 'Softplus':
            self.activation_function = nn.Softplus()
        elif activation_function == 'Mish':
            self.activation_function = nn.Mish()
        else:
            raise ValueError(f"不支援的激活函數: {activation_function}")

        # 捲積層
        conv_sizes = [conv_start_size * (2 ** i) for i in range(num_pool_layers)]
        conv_layers = []
        prev_size = 3
        for i, conv_size in enumerate(conv_sizes):
            conv_layers.append(nn.Conv2d(prev_size, conv_size, kernel_size=3, stride=1, padding=1))
            conv_layers.append(self.activation_function)
            conv_layers.append(nn.Conv2d(conv_size, conv_size, kernel_size=3, stride=1, padding=1))
            conv_layers.append(self.activation_function)
            prev_size = conv_size
            #conv_layers.append(nn.Dropout(p=dropout_rate))
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
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
            #layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation_function)
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
    model = CNNModelForCIFAR100_Simple()
    print(model)

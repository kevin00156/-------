import os
import sys

"""
此腳本用於計算資料集的平均值和標準差，用於正規化資料
"""

# 將專案根目錄添加到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import CoffeeBeanDataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def calculate_mean_std(dataset_path: str, batch_size: int = 32, num_workers: int = 1):
    """
    計算資料集的平均值和標準差
    
    Args:
        dataset_path: 資料集json檔案的路徑
        batch_size: 批次大小
        num_workers: 資料載入的工作程序數量
        
    Returns:
        tuple: (mean, std) 分別為各通道的平均值和標準差
    """
    # 建立基本轉換
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 載入資料集
    dataset = CoffeeBeanDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    # 初始化變數
    mean = 0.
    std = 0.
    total_images = 0
    
    # 計算平均值
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_samples
    
    mean = mean / total_images
    
    # 計算標準差
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0,2])
    
    std = torch.sqrt(std / (total_images * 256 * 256))
    
    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    dataset_path = "Coffee bean dataset/dataset.json"
    mean, std = calculate_mean_std(dataset_path)
    print(f"資料集的平均值: {mean}")
    print(f"資料集的標準差: {std}")

import yaml
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim  # 引入優化器模組
import os
import importlib
import inspect
import torch
from .Models import *


def repeat_channels(x):
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x

def load_transforms(transforms_config):
    """從配置中加載數據轉換"""
    transform_list = []
    for transform in transforms_config:
        if transform['type'] == "Resize":
            transform_list.append(transforms.Resize(tuple(transform['size'])))
        elif transform['type'] == "RandomCrop":
            transform_list.append(transforms.RandomCrop(size=tuple(transform['size']), padding=transform['padding']))
        elif transform['type'] == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif transform['type'] == "Lambda":
            transform_list.append(transforms.Lambda(repeat_channels))
        elif transform['type'] == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip())
        elif transform['type'] == "RandomRotation":
            transform_list.append(transforms.RandomRotation(degrees=transform['degrees']))
        elif transform['type'] == "ColorJitter":
            transform_list.append(transforms.ColorJitter(
                brightness=transform['brightness'],
                contrast=transform['contrast'],
                saturation=transform['saturation'],
                hue=transform['hue']
            ))
        elif transform['type'] == "Normalize":
            transform_list.append(transforms.Normalize(mean=transform['mean'], std=transform['std']))

    return transforms.Compose(transform_list)

def get_model_classes():
    """顯示可用的模型類型"""
    models_module = importlib.import_module('.Models', __package__)
    model_classes = {name: cls for name, cls in inspect.getmembers(models_module, inspect.isclass) if issubclass(cls, nn.Module)}
    return model_classes

def get_optimizer_classes():
    """顯示可用的優化器類型"""
    optimizer_classes = {name: cls for name, cls in inspect.getmembers(optim, inspect.isclass) if name != 'Optimizer'}
    return optimizer_classes

def get_scheduler_classes():
    """顯示可用的學習率調度器類型"""
    scheduler_classes = {name: cls for name, cls in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass)}
    return scheduler_classes


def load_model(model_name: str, model_parameters: dict):
    """根據模型名稱和參數動態構建模型"""
    # 使用 get_model_classes 獲取所有模型類
    model_classes = get_model_classes()

    if model_name in model_classes:
        #print(model_classes[model_name])
        #print(model_parameters)
        return model_classes[model_name](**model_parameters)  # 使用已定義的模型類
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_optimizer(optimizer_config, model):
    """根據配置加載優化器"""
    optimizer_type = optimizer_config['type']
    params = optimizer_config['params']
    
    # 使用 get_optimizer_classes 獲取所有優化器類
    optimizer_classes = get_optimizer_classes()

    if optimizer_type in optimizer_classes:
        return optimizer_classes[optimizer_type](model.parameters(), **params)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def load_scheduler(scheduler_config, optimizer):
    """根據配置加載學習率調度器"""
    scheduler_type = scheduler_config['type']
    params = scheduler_config['params']
    
    # 使用 get_scheduler_classes 獲取所有調度器類
    scheduler_classes = get_scheduler_classes()

    if scheduler_type in scheduler_classes:
        return scheduler_classes[scheduler_type](optimizer, **params)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def load_config(config_path: str):
    """從 YAML 配置檔案中加載模型、轉換和訓練參數"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 確保 config 是一個字典
    if not isinstance(config, dict):
        raise ValueError("配置文件應該是一個字典")

    # 獲取模型配置
    model_config = config['model']  # 獲取模型配置
    model_name = model_config['name']  # 獲取模型名稱
    model_parameters = model_config['parameters']  # 獲取模型參數
    model = load_model(model_name, model_parameters)  # 使用名稱和參數構建模型

    # 加載數據轉換
    transforms_config = config['transforms']  # 獲取轉換配置
    preprocess = load_transforms(transforms_config)

    # 加載訓練參數
    training_config = config['training']  # 獲取訓練配置
    training_params = {
        'batch_size': training_config['batch_size'],
        'num_workers': training_config['num_workers'],
        'learning_rate': training_config['learning_rate'],
        'max_epochs': training_config['max_epochs'],
        'early_stopping_patience': training_config['early_stopping_patience']
    }

    # 加載優化器
    optimizer_config = config['optimizer']  # 獲��優化器配置
    optimizer = load_optimizer(optimizer_config, model)

    # 加載學習率調度器（如果有的話）
    scheduler = None
    if 'scheduler' in config:
        scheduler_config = config['scheduler']  # 獲取調度器配置
        scheduler = load_scheduler(scheduler_config, optimizer)

    return model, preprocess, training_params, optimizer, scheduler


"""
YAML 配置範例：

configs:
  model:
    name: "CNNModelForCIFAR100"  # 可選擇: CNNModelForCIFAR100, LeNet, VGG, ResNet18
    parameters:
      num_classes: 100
      input_size: 32
  training:
    batch_size: 120
    num_workers: 4
    learning_rate: 0.001
    max_epochs: 1000
    early_stopping_patience: 15
  transforms:
    - type: "Resize"
      size: [32, 32]
    - type: "RandomCrop"
      size: [32, 32]
      padding: 4
    - type: "ToTensor"
    - type: "Lambda"
      function: "repeat_channels"
    - type: "RandomHorizontalFlip"
    - type: "RandomRotation"
      degrees: 30
    - type: "ColorJitter"
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    - type: "Normalize"
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2470, 0.2435, 0.2616]
  optimizer:
    type: "Adam"  # 可選擇: SGD, Adam, RMSprop, Adagrad, AdamW
    params:
      weight_decay: 0.0001
  scheduler:
    type: "StepLR"  # 可選擇: StepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
    params:
      step_size: 30
      gamma: 0.1
"""


if __name__ == "__main__":
    print("模型類型：", get_model_classes())
    print("優化器類型：", get_optimizer_classes())
    print("調度器類型：", get_scheduler_classes())

    # 讀取 YAML 檔案
    import yaml
    import os
    
    # 取得當前工作目錄下的 train_configs 資料夾中的所有 yaml 檔案
    config_dir = "train_configs"
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    
    for yaml_file in yaml_files:
        file_path = os.path.join(config_dir, yaml_file)
        print(f"\n正在讀取設定檔: {yaml_file}")
        print("-" * 50)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 確保 config 是一個字典
        if not isinstance(config, dict):
            raise ValueError("配置文件應該是一個字典")

        # 直接使用 config 中的鍵
        model_config = config['model']  # 獲取模型配置
        print(f"模型名稱: {model_config['name']}")  # 使用 model_config 而不是 config_item
        print("模型參數:", model_config['parameters'])

        training_config = config['training']  # 獲取訓練配置
        print("訓練參數:", training_config)

        transforms_config = config['transforms']  # 獲取轉換配置
        print("轉換配置:", transforms_config)
        
        optimizer_config = config['optimizer']  # 獲取優化器配置
        print("優化器配置:", optimizer_config)
        
        scheduler_config = config['scheduler']  # 獲取調度器配置
        print("調度器配置:", scheduler_config)
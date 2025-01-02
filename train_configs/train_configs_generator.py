import yaml
import os

"""
此腳本用於生成訓練配置檔案，可以生成多個訓練配置檔案，並將其儲存到 train_configs 資料夾中。  
請搭配utils/load_parameters.py使用

使用方法：
在main函式中，修改訓練參數，比如model, optimizer, train_transforms, val_transforms, test_transforms, optimizer, scheduler等，
也可以根據自己的需要，新增新的訓練參數，以list的形式在每一個感興趣的參數後面增加
更改後執行此檔案就會生成一堆yaml檔案
"""


def generate_train_config(config_dict, base_filename='train_config',file_path='train_configs'):
    """
    將訓練配置字典寫入YAML檔案
    
    Args:
        config_dict (dict): 包含模型配置、訓練參數等的字典
        base_filename (str): 基礎檔案名稱，預設為'train_config'
        file_path (str): 檔案路徑，預設為'train_configs'
        
    Returns:
        str: 生成的檔案完整路徑
    """
    # 確保train_configs目錄存在
    os.makedirs(file_path, exist_ok=True)
    
    # 尋找可用的檔案編號
    counter = 1
    while True:
        filename = f"{base_filename}_{counter}.yaml"
        filepath = os.path.join(file_path, filename)
        if not os.path.exists(filepath):
            break
        counter += 1
    
    # 寫入YAML檔案
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"配置檔案已生成: {filepath}")
    return filepath

def create_default_config(
    model: dict,
    training_batch_size: int,
    training_num_workers: int,
    training_learning_rate: float,
    training_max_epochs: int,
    training_early_stopping_patience: int,
    train_transforms: list[dict],
    val_transforms: list[dict],
    test_transforms: list[dict],
    optimizer: dict,
    scheduler: dict
    ):
    """
    創建預設的訓練配置字典
    
    Returns:
        dict: 預設配置字典
    """
    return {
        'model': model,
        'training': {
            'batch_size': training_batch_size,
            'num_workers': training_num_workers,
            'learning_rate': training_learning_rate,
            'max_epochs': training_max_epochs,
            'early_stopping_patience': training_early_stopping_patience
        },
        'train_transforms': train_transforms,
        'val_transforms': val_transforms,
        'test_transforms': test_transforms,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
if __name__ == '__main__':
    
    with open('settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)  # 讀取配置文件

    num_classes = settings['dataset_info']['num_classes']
    input_size = settings['dataset_info']['input_size']
    mean = settings['dataset_info_merged']['mean']
    std = settings['dataset_info_merged']['std']
    mean_val = settings['dataset_info_merged']['mean']
    std_val = settings['dataset_info_merged']['std']
    mean_test = settings['dataset_info_merged']['mean']
    std_test = settings['dataset_info_merged']['std']
    models = [
        {'name': 'ResNet50Model',#你可以改成自定義的model，或是pytorch官方的model
         'parameters': {
             'model_name': 'resnext101_32x8d',
             'num_classes': num_classes,
             'weights': 'DEFAULT'
             }},
        {'name': 'ResNet50Model',#你可以改成自定義的model，或是pytorch官方的model
         'parameters': {
             'model_name': 'resnext50_32x4d',
             'num_classes': num_classes,
             'weights': 'DEFAULT'
             }},
        {'name': 'ResNet50Model',#你可以改成自定義的model，或是pytorch官方的model
         'parameters': {
             'model_name': 'resnet101',
             'num_classes': num_classes,
             'weights': 'DEFAULT'
             }},
        {'name': 'ResNet50Model',#你可以改成自定義的model，或是pytorch官方的model
         'parameters': {
             'model_name': 'wide_resnet50_2',
             'num_classes': num_classes,
             'weights': 'DEFAULT'
             }},
    ]
    training_batch_size = 10
    training_num_workers = 6
    training_learning_rates = 0.001
    training_max_epochs = 1000
    training_early_stopping_patience = 100
    train_transforms = [
        [
            {'type': 'Resize', 'size': [input_size, input_size]},
            {'type': 'ToTensor'},
            {'type': 'Lambda', 'function': 'repeat_channels'},
            {'type': 'RandomHorizontalFlip', 'p': 0.5},
            {'type': 'RandomVerticalFlip', 'p': 0.5},
            {'type': 'RandomRotation', 'degrees': 360, 'p': 0.5},  # 允許任意角度旋轉
            {'type': 'RandomPerspective', 'distortion_scale': 0.5, 'p': 0.5},
            {'type': 'RandomAffine', 'degrees': 360, 'translate': 0.2, 'scale': 0.2, 'shear': 10, 'p': 0.5},
            {'type': 'ColorJitter', 'brightness': 0.3, 'contrast': 0.3,
             'saturation': 0.3, 'hue': 0.2, 'p': 0.5},
            {'type': 'GaussianBlur', 'kernel_size': 5, 'sigma': 1.0, 'p': 0.5},  # 加入噪聲的設置
            {'type': 'Normalize', 
             'mean': mean,
             'std': std}
        ],
    ]
    val_transforms = [
        [
            {'type': 'Resize', 'size': [input_size, input_size]},
            {'type': 'ToTensor'},
            {'type': 'Lambda', 'function': 'repeat_channels'},
            {'type': 'Normalize', 
                'mean': mean_val,
                'std': std_val},
            ],
    ]
    test_transforms = [
        [
            {'type': 'Resize', 'size': [input_size, input_size]},
            {'type': 'ToTensor'},
            {'type': 'Lambda', 'function': 'repeat_channels'},
            {'type': 'Normalize', 
                'mean': mean_test,
                'std': std_test},
        ],
    ]
    optimizer = [
        {
            'type': 'AdamW',
            'params': {
                'lr': 0.00005,
                'weight_decay': 0.0001,
            }
        },
    ]
    
    scheduler = [
        {
            'type': 'ReduceLROnPlateau',
            'params': {
                'mode': 'min',
                'factor': 0.1,
                'patience': 10,
                'min_lr': 0.00000001
            }
        },
    ]
    
    index = 0
    for model in models:
        for opt in optimizer:
            for train_transform in train_transforms:
                for val_transform in val_transforms:
                    for test_transform in test_transforms:
                        for sched in scheduler:
                            config_dict = create_default_config(
                                model, 
                                training_batch_size, 
                                training_num_workers, 
                                training_learning_rates, 
                                training_max_epochs, 
                                training_early_stopping_patience, 
                                train_transform, 
                                val_transform,
                                test_transform,
                                opt,
                                sched,
                            )
                            print(yaml.dump(config_dict, allow_unicode=True, sort_keys=False))
                            generate_train_config(config_dict)
                            index += 1
                    
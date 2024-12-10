import yaml
import os

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
    transforms: list[dict],
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
        'transforms': transforms,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
if __name__ == '__main__':
    
    num_classes = 2
    input_size = 256

    models = [
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.1,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.1,
             'num_pool_layers': 4,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.1,
             'num_pool_layers': 5,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.1,
             'num_pool_layers': 3,
             'conv_start_size': 128
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [512, 256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.1,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.3,
             'num_pool_layers': 3,
             'conv_start_size': 128
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'ReLU',
             'dropout_rate': 0.3,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'ReLU',
             'dropout_rate': 0.1,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'RReLU',
             'dropout_rate': 0.3,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'ELU',
             'dropout_rate': 0.3,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'Mish',
             'dropout_rate': 0.3,
             'num_pool_layers': 3,
             'conv_start_size': 64
             }},
        {'name': 'CNNModelForCIFAR10',
         'parameters': {
             'num_classes': num_classes,
             'input_size': input_size,
             'hidden_layers': [256, 128, 64],
             'activation_function': 'LeakyReLU',
             'dropout_rate': 0.3,
             'num_pool_layers': 4,
             'conv_start_size': 64
             }},
    ]
    training_batch_size = 100
    training_num_workers = 4
    training_learning_rates = 0.001
    training_max_epochs = 1000
    training_early_stopping_patience = 20
    transforms = [
            {'type': 'Resize', 'size': [32, 32]},
            {'type': 'RandomCrop', 'size': [32, 32], 'padding': 4},
            {'type': 'ToTensor'},
            {'type': 'Lambda', 'function': 'repeat_channels'},
            {'type': 'RandomHorizontalFlip'},
            {'type': 'RandomRotation', 'degrees': 30},
            {'type': 'ColorJitter', 'brightness': 0.2, 'contrast': 0.2,
             'saturation': 0.2, 'hue': 0.1},
            {'type': 'Normalize', 
             'mean': [0.4914, 0.4822, 0.4465],
             'std': [0.2470, 0.2435, 0.2616]}
        ]
    optimizer = [
        {
            'type': 'Adam',
            'params': {
                'lr': 0.001,
            }
        },
        {
            'type': 'SGD',
            'params': {
                'lr': 0.001,
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
            for sched in scheduler:
                config_dict = create_default_config(
                    model, 
                    training_batch_size, 
                    training_num_workers, 
                    training_learning_rates, 
                    training_max_epochs, 
                    training_early_stopping_patience, 
                    transforms, 
                    opt,
                    sched,
                )
                print(yaml.dump(config_dict, allow_unicode=True, sort_keys=False))
                generate_train_config(config_dict)
                index += 1
        
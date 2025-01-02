import torch.nn as nn
import torch.nn.modules.activation as activation

def get_available_activations():
    # 獲取 nn 模組中所有屬性
    all_classes = dir(activation)
    # 過濾出所有在 nn.activation 中的類
    activation_classes = [cls for cls in all_classes if cls in activation.__dict__ and isinstance(activation.__dict__[cls], type) and issubclass(activation.__dict__[cls], activation.Module) and cls not in ['Module', 'Container']]
    return activation_classes

def activation_function_parser(activation_function:str)->nn.Module:
    try:
        # 嘗試從 nn 模組中獲取對應的激活函數類
        activation_class = getattr(nn, activation_function)
        return activation_class()
    except AttributeError:
        available_activations = get_available_activations()
        raise ValueError(f"不支援的激活函數: {activation_function}, 請確認該函數存在於 torch.nn 中，可用列表：{available_activations}")

if __name__ == "__main__":
    module = activation_function_parser('ReLU')
    print(module)
    module = activation_function_parser('NotAnModule')
    print(module)
    

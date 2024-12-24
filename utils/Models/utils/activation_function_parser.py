import torch.nn as nn

def activation_function_parser(activation_function:str)->nn.Module:
    if activation_function == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation_function == 'ReLU':
        return nn.ReLU()
    elif activation_function == 'Sigmoid':
        return nn.Sigmoid()
    elif activation_function == 'Tanh':
        return nn.Tanh()
    elif activation_function == 'ELU':
        return nn.ELU()
    elif activation_function == 'SELU':
        return nn.SELU()
    elif activation_function == 'GELU':
        return nn.GELU()
    elif activation_function == 'RReLU':
        return nn.RReLU()
    elif activation_function == 'PReLU':
        return nn.PReLU()
    elif activation_function == 'Softplus':
        return nn.Softplus()
    elif activation_function == 'Mish':
        return nn.Mish()
    else:
        raise ValueError(f"不支援的激活函數: {activation_function}")
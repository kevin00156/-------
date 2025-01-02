import torch
from pytorch_lightning import LightningModule
from torchvision import models

class ResNetModel(LightningModule):
    def __init__(self,model_name='resnet50', num_classes=2, weights=None):
        super(ResNetModel, self).__init__()
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=weights)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights=weights)
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights=weights)
        elif model_name == 'resnet152':
            self.model = models.resnet152(weights=weights)
        elif model_name == 'resnext50_32x4d':
            self.model = models.resnext50_32x4d(weights=weights)
        elif model_name == 'resnext101_32x8d':
            self.model = models.resnext101_32x8d(weights=weights)
        elif model_name == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=weights)
        elif model_name == 'wide_resnet101_2':
            self.model = models.wide_resnet101_2(weights=weights)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def load_model_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        new_state_dict = {}
        for key in checkpoint['state_dict']:
            new_key = key.replace('model.model.', 'model.')  # 去掉多餘的前綴
            new_state_dict[new_key] = checkpoint['state_dict'][key]
        self.load_state_dict(new_state_dict)
        
# 加載模型
if __name__ == "__main__":
    model = ResNetModel.load_from_checkpoint('trained_models/final_model_ResNet.ckpt')
    model.eval()
import torch
from pytorch_lightning import LightningModule
from torchvision import models

class ResNet50Model(LightningModule):
    def __init__(self,model_name='resnet50', num_classes=2, weights=None):
        super(ResNet50Model, self).__init__()
        
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

# 加載模型
if __name__ == "__main__":
    model = ResNet50Model.load_from_checkpoint('trained_models/resnet50.ckpt')
    model.eval()
import torch
from pytorch_lightning import LightningModule
from torchvision import models

class ResNet50Model(LightningModule):
    def __init__(self, num_classes=2, weights=None):
        super(ResNet50Model, self).__init__()
        
        self.model = models.resnet50(weights=weights)
        
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 加載模型
if __name__ == "__main__":
    model = ResNet50Model.load_from_checkpoint('trained_models/resnet50.ckpt')
    model.eval()
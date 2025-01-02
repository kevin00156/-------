import torch
from pytorch_lightning import LightningModule
from torchvision import models

class ResNet101Model(LightningModule):
    def __init__(self, num_classes=2, weights=None):
        super(ResNet101Model, self).__init__()
        
        self.model = models.resnet50(weights=weights)
        
        self.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

# 加載模型
if __name__ == "__main__":
    model = ResNet101Model.load_from_checkpoint('trained_models/final_model_ResNet101.ckpt')
    model.eval()
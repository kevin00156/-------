import torch
import torch.nn as nn
import torchvision.models as models

class VGG_Pretrained(nn.Module):
    def __init__(self, num_classes=10, model_name='vgg16', weights='DEFAULT'):
        super(VGG_Pretrained, self).__init__()
        
        # 載入預訓練的VGG模型
        if model_name == 'vgg16':
            self.model = models.vgg16(weights=weights)
        elif model_name == 'vgg19':
            self.model = models.vgg19(weights=weights)
        else:
            raise ValueError("model_name must be 'vgg16' or 'vgg19'")
        
        # 凍結所有預訓練層的參數
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 修改最後的全連接層以適應新的分類數量
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def unfreeze_features(self, num_layers=0):
        """
        解凍最後num_layers層的參數進行微調
        如果num_layers=0，則保持所有特徵提取層凍結
        """
        if num_layers > 0:
            children = list(self.model.features.children())
            for child in children[-num_layers:]:
                for param in child.parameters():
                    param.requires_grad = True
                    
if __name__ == "__main__":
    model = VGG_Pretrained(
        num_classes=2,
        model_name='vgg16',
        weights='DEFAULT'
    )
    print(model)
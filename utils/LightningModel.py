import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class LightningModel(pl.LightningModule):
    def __init__(
            self,
            *,
            num_classes: int = 10,
            model: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler._LRScheduler,
            show_progress_bar: bool = True,
            show_result_every_epoch: bool = False
        ):
        super().__init__()  
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.show_progress_bar = show_progress_bar
        self.show_result_every_epoch = show_result_every_epoch
        self.save_hyperparameters(ignore=['model'])
        
        # 新增列表來儲存損失和準確度
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss, prog_bar=self.show_progress_bar)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=self.show_progress_bar)
        self.log('val_acc', acc, prog_bar=self.show_progress_bar)

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(avg_loss)

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics['val_loss'].item()
        avg_val_acc = self.trainer.callback_metrics['val_acc'].item()
        self.val_losses.append(avg_val_loss)
        self.val_accs.append(avg_val_acc)
        if self.show_result_every_epoch == True:
            print(".")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('test_loss', loss, prog_bar=self.show_progress_bar)

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': 'val_loss'}

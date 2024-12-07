from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import threading
import logging
import numpy as np

from utils import LightningModel, CoffeeBeanDataset, CNNModel

"""
參數定義
"""

image_size = 128 
batch_size = 32
num_workers = 4
train_size_ratio = 0.8
val_size_ratio = 0.15
# 初始化模型和訓練器

def repeat_channels(x):
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomCrop(size=(image_size, image_size), padding=4),
    transforms.ToTensor(),
    transforms.Lambda(repeat_channels),  # 使用普通函數替代 lambda
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
])

train_dataset = CoffeeBeanDataset(
    json_file="coffee_bean_dataset/dataset.json",
    transform=preprocess
)
model_label_count = train_dataset.get_label_count()
train_model = CNNModel(num_classes=model_label_count, input_size=image_size)
optimizer = optim.Adam(train_model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


# 設定計算設備(GPU或CPU)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')



"""
初始化必要參數
"""



"""
    Dash網頁設定
    可以顯示訓練過程的損失和準確率，每次epoch更新一次
"""
# Dash app setup
app = dash.Dash(__name__)

# 禁用 Dash 的開發工具日誌
app.config.suppress_callback_exceptions = True
app.logger.setLevel(logging.ERROR)  # 設置 Dash 日誌級別

# 禁用 Flask 的日誌
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app.layout = html.Div([
    dcc.Graph(id='train-loss-graph'),
    dcc.Graph(id='val-loss-graph'),
    dcc.Graph(id='val-acc-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # 每秒更新一次
        n_intervals=0
    )
])

@app.callback(
    [Output('train-loss-graph', 'figure'),
     Output('val-loss-graph', 'figure'),
     Output('val-acc-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    # Train Loss Plot
    train_loss_fig = go.Figure()
    train_loss_fig.add_trace(go.Scatter(x=list(range(len(model.train_losses))), y=model.train_losses, mode='lines+markers', name='Train Loss'))
    train_loss_fig.update_layout(title='Train Loss', xaxis_title='Epoch', yaxis_title='Loss')

    # Validation Loss Plot
    val_loss_fig = go.Figure()
    val_loss_fig.add_trace(go.Scatter(x=list(range(len(model.val_losses))), y=model.val_losses, mode='lines+markers', name='Val Loss', line=dict(color='orange')))
    val_loss_fig.update_layout(title='Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')

    # Validation Accuracy Plot
    val_acc_fig = go.Figure()
    val_acc_fig.add_trace(go.Scatter(x=list(range(len(model.val_accs))), y=model.val_accs, mode='lines+markers', name='Val Accuracy', line=dict(color='green')))
    val_acc_fig.update_layout(title='Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')

    return train_loss_fig, val_loss_fig, val_acc_fig

def run_dash():
    app.run_server(debug=False, use_reloader=False)


"""
主程式
"""
if __name__ == "__main__":
    print("開始執行程式")
    
    data_path = os.getcwd()
    
    print("初始化完成")

    # 拆分訓練集和驗證集
    dataset_size = len(train_dataset)
    train_size = int(train_size_ratio * dataset_size)
    val_size = int(val_size_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True)
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True)
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True)
    print("資料集拆分完成")

    logger = TensorBoardLogger(save_dir='lightning_logs')


    model = LightningModel(
        num_classes=model_label_count,
        model=train_model,
        optimizer=optimizer,
        scheduler=scheduler,
        show_progress_bar=True,
        show_result_every_epoch=False
    )
    # 設定訓練器
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir='lightning_logs',  # 儲存檢查點和日誌的目錄
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=20, mode="min")]
    )
    
    # 開始訓練
    dash_thread = threading.Thread(target=run_dash,daemon=True) # 啟動 Dash 線程
    dash_thread.start()
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # 儲存最終模型
    trainer.save_checkpoint("final_model.ckpt")
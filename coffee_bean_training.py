from torch.utils.data import DataLoader, Dataset
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
import torchvision
import yaml
from typing import Dict, Any

from utils import LightningModel, load_config, CoffeeBeanDataset

"""
參數定義
"""

model = None

config_path = "train_configs"
config_filenames = [path for path in os.listdir(config_path) if path.endswith(".yaml")]
# 根據配置文件名中的數字進行排序
config_filenames.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))


# 確保配置文件存在
if not config_filenames:
    raise FileNotFoundError("未找到任何 YAML 配置文件")

#print(f"找到的配置文件: {config_filenames}")
# 加載所有配置
training_configs = [load_config(config_path + "/" + config_filename) for config_filename in config_filenames]





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
def main(training_config, config_index):
    global model
    train_model, train_transforms, val_transforms, test_transforms, training_params, optimizer, scheduler = training_config
    
    # 初始化模型和訓練器
    train_dataset_original = CoffeeBeanDataset(
        json_file="Coffee bean dataset/dataset.json"
    )
    model_label_count = train_dataset_original.get_label_count()
    # 計算訓練集和驗證集的大小
    total_size = len(train_dataset_original)
    train_size = int(0.8 * total_size)  # 80% 用於訓練
    val_size = int(0.15 * total_size)  # 15% 用於驗證
    test_size = total_size - train_size - val_size  # 剩下的 5% 用於測試

    # 使用 random_split 將數據集分割成訓練集和驗證集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset_original, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 為每個數據集設置不同的 transform
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = test_transforms
    
    print("初始化完成")

    # 使用配置中的批次大小和工作數
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        num_workers=training_params['num_workers'],
        pin_memory=True,
        persistent_workers=True)
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=training_params['num_workers'],
        pin_memory=True,
        persistent_workers=True)
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=training_params['num_workers'],
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
        #max_epochs=training_params['max_epochs'],
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir='lightning_logs',
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=training_params['early_stopping_patience'], mode="min")]
    )
    
    # 開始訓練
    dash_thread = threading.Thread(target=run_dash, daemon=True)  # 啟動 Dash 線程
    dash_thread.start()
    try:
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
    except KeyboardInterrupt:
        print("訓練過程手動中斷，繼續下一個模型的訓練")
        return

    # 儲存最終模型
    trainer.save_checkpoint(f"final_model_{config_index}.ckpt")


if __name__ == "__main__":
    print("開始執行程式")
    for training_config in training_configs:
        print(f"開始訓練第{training_configs.index(training_config)+1}個模型")
        main(training_config, training_configs.index(training_config)+1)


from ultralytics import YOLO
import os
import torch

# 設定基礎路徑
base_path = "Coffee_bean_dataset_YOLOv11"

def main(train = True):
    # 初始化 YOLOv11 模型
    model = YOLO('yolo11x.pt')  

    if train == True:
        # 訓練參數設定
        training_args = {
            'data': 'coffee_beans.yaml',  # 您需要創建這個配置文件
            'epochs': 500,
            'imgsz': 256,
            'batch': 32,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'patience': 20,
            'project': 'coffee_bean_detection_yolo11x',
            'name': 'yolov11_training'
        }

        # 開始訓練
        try:
            results = model.train(**training_args)
        except KeyboardInterrupt:
            print("訓練過程手動中斷")
            return

    # 驗證模型
    model.val()

if __name__ == "__main__":
    print("開始執行程式")
    main()
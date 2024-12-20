import cv2  # 導入OpenCV庫，用於視頻處理
import numpy as np
import threading
import time
import yaml
import torch
import torchvision.transforms as transforms
from utils import process_coffee_beans, repeat_channels, ResNet50Model
from PIL import Image

"""
這是可以開啟影片檔案，並以多線程方式，一邊播放影片，一邊進行咖啡豆辨識的程式
"""


frame = None
result = None
with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)  # 讀取配置文件
pixel_threshold_lower = settings['coffee_bean_pixel_threshold']['lower']  # 獲取像素下限
pixel_threshold_upper = settings['coffee_bean_pixel_threshold']['upper']  # 獲取像素上限
# 加載 ResNet50 模型
model = ResNet50Model.load_from_checkpoint('trained_models/resnet50_merged.ckpt')  # 從檢查點加載訓練後的模型
model.eval()  # 設置模型為評估模式
input_size = settings['dataset_info']['input_size']
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),  # 將圖像調整為256x256大小
    transforms.ToTensor(),  # 將PIL圖像轉換為張量
    transforms.Lambda(repeat_channels),  # 應用重複通道的函數
    transforms.Normalize(  # 對圖像進行標準化
        mean=settings['dataset_info_merged']['mean'],  # 計算的均值
        std=settings['dataset_info_merged']['std']  # 計算的標準差
    ),
])


def get_video_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def play_video(video_path):
    global frame, result

    # 讀取視頻文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("無法打開視頻文件")
        return
    
    while True:
        ret, frame = cap.read()  # 逐幀讀取視頻
        #if frame.shape[0] > frame.shape[1]:  # 如果高度大於寬度，則旋轉圖像
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 旋轉90度
        if not ret:
            print("視頻讀取結束")
            break
            
        # 檢查 result 是否為有效的 numpy 陣列
        display_frame = frame.copy()
        if result is not None and isinstance(result, np.ndarray) and result.shape == frame.shape:
            display_frame = cv2.add(frame, result)
            
        height, width = display_frame.shape[:2]
        new_width = int((width / height) * 768)  # 根據比例計算新的寬度
        display_frame = cv2.resize(display_frame, (new_width, 768))  # 調整顯示幀的大小
        cv2.imshow('Video Playback', display_frame)  # 顯示當前幀
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 鍵退出播放
            break
    
    cap.release()  # 釋放視頻捕獲對象
    cv2.destroyAllWindows()  # 關閉所有OpenCV窗口

def predict_coffee_bean(image):
    # 對咖啡豆圖像進行預測
    #cv2.imwrite("coffee_bean_rgb.jpg", image)  # 可選：保存圖像
    #image = Image.open("coffee_bean_rgb.jpg")  # 可選：從文件中加載圖像
    if image.shape[2] == 3:  # 確保圖像是三通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將BGR轉換為RGB格式
    image = Image.fromarray(image)  # 將NumPy數組轉換為PIL圖像
    # 確保預處理與訓練時一致
    coffee_bean_transformed = transform(image)  # 對圖像進行轉換
    coffee_bean_input = coffee_bean_transformed.unsqueeze(0)  # 增加一個維度以符合模型輸入要求

    with torch.no_grad():  # 在不計算梯度的情況下進行預測
        prediction = model(coffee_bean_input)  # 獲取模型預測
        predicted_class = prediction.argmax(dim=1).item()  # 獲取預測的類別
        return predicted_class  # 返回預測的類別
def predict_coffee_bean_from_frame():
    global result
    while True:
        try:
            expanded_beans = process_coffee_beans(
                frame,
                show_image=False, 
                pixel_threshold_lower=pixel_threshold_lower, 
                pixel_threshold_upper=pixel_threshold_upper
            )
        
            empty_image_with_predict_result = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)  # 建立一個與圖像相同大小的透明mask
            color = [(0, 0, 255), (0, 255, 0)]  # 定義顏色列表
            for index, (bean, (x, y, w, h)) in enumerate(expanded_beans):
                # 記錄預測開始時間
                start_predict_time = time.time()  
                predicted_class = predict_coffee_bean(bean)  # 對咖啡豆進行預測
                end_predict_time = time.time()  # 記錄預測結束時間
                cv2.rectangle(empty_image_with_predict_result, (x, y), (x+w, y+h), color[predicted_class], 2)  # 繪製矩形框
                print(f"""predicted_class: {predicted_class}, index: {index}, h: {h}, w: {w}, take time ={end_predict_time - start_predict_time:.4f}sec""")  # 輸出預測結果
        

                cv2.putText(
                    img=empty_image_with_predict_result, 
                    text=str(index+1),  # 標記編號
                    org=(x, y-10),  # 標記位置
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.9, 
                    color=color[predicted_class], 
                    thickness=2
                )
            result = empty_image_with_predict_result
        except ValueError as e:
            print(f"Error: {e}")

def main():
    #path = 'http://192.168.1.2:8080/video'  # 替換為你的IP Webcam的URL
    path = 'coffee_bean_test_video/PXL_20241216_163729711.mp4'
    frame = get_video_first_frame(path)
    #if frame.shape[0] > frame.shape[1]:  # 如果高度大於寬度，則旋轉圖像
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 旋轉90度
    
    result_thread = threading.Thread(target=predict_coffee_bean_from_frame,daemon=True)
    result_thread.start()

    video_thread = threading.Thread(
        target=play_video, 
        args=(path,),
        daemon=True
    )
    video_thread.start()  # 啟動播放視頻的線程


    video_thread.join()

if __name__ == "__main__":
    main()
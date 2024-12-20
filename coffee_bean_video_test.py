import cv2  # 導入OpenCV庫，用於視頻處理
import numpy as np
import threading
import time
import yaml
from ultralytics import YOLO  # 新增 YOLO 引入
from utils import process_coffee_beans

"""
這是可以開啟影片檔案，並以多線程方式，一邊播放影片，一邊進行咖啡豆辨識的程式
"""


frame = None
result = None
with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)  # 讀取配置文件
pixel_threshold_lower = settings['coffee_bean_pixel_threshold']['lower']  # 獲取像素下限
pixel_threshold_upper = settings['coffee_bean_pixel_threshold']['upper']  # 獲取像素上限
# 載入 YOLO 模型
model = YOLO('coffee_bean_detection/yolov11_training4/weights/best.pt')  # 替換成您的 YOLO 權重檔路徑


def get_video_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def play_video(video_path, store_video=True):
    global frame, result

    # 讀取視頻文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("無法打開視頻文件")
        return
    
    if store_video:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4編碼
        output_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
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
        
        if store_video:
            output_video.write(display_frame)

        height, width = display_frame.shape[:2]
        new_width = int((width / height) * 768)  # 根據比例計算新的寬度
        display_frame = cv2.resize(display_frame, (new_width, 768))  # 調整顯示幀的大小
        cv2.imshow('Video Playback', display_frame)  # 顯示當前幀

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 鍵退出播放
            break
    
    cap.release()  # 釋放視頻捕獲對象
    cv2.destroyAllWindows()  # 關閉所有OpenCV窗口

def predict_coffee_bean(image):
    # 對單顆咖啡豆圖像進行預測
    image = cv2.resize(image, (512, 512))
    results = model(image)  # YOLO 預測
    
    # 獲取預測結果
    if len(results) > 0 and len(results[0].boxes) > 0:
        # 取得最高信心度的預測結果
        confidence = float(results[0].boxes.conf[0])
        predicted_class = int(results[0].boxes.cls[0])
        return predicted_class, confidence
    return 0, 0.0  # 如果沒有檢測到，返回預設值

def predict_coffee_bean_from_frame():
    global result
    while True:
        try:
            if frame is None:
                continue
                
            expanded_beans = process_coffee_beans(
                frame,
                show_image=False, 
                pixel_threshold_lower=pixel_threshold_lower, 
                pixel_threshold_upper=pixel_threshold_upper
            )
        
            empty_image_with_predict_result = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            color = [(0, 0, 255), (0, 255, 0)]  # 定義顏色列表
            
            for index, (bean, (x, y, w, h)) in enumerate(expanded_beans):
                start_predict_time = time.time()
                predicted_class, confidence = predict_coffee_bean(bean)
                end_predict_time = time.time()
                
                # 繪製邊界框
                cv2.rectangle(empty_image_with_predict_result, (x, y), (x+w, y+h), 
                            color[predicted_class], 2)
                
                print(f"""predicted_class: {predicted_class}, confidence: {confidence:.2f}, 
                      index: {index}, h: {h}, w: {w}, 
                      take time={end_predict_time - start_predict_time:.4f}sec""")
                
                # 添加標籤文字
                cv2.putText(
                    img=empty_image_with_predict_result,
                    text=f"{index+1} ({confidence:.2f})",
                    org=(x, y-10),
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
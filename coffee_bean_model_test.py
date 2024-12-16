import cv2  # 導入OpenCV庫，用於圖像處理
import torch  # 導入PyTorch庫，用於深度學習
import numpy as np  # 導入NumPy庫，用於數值計算
from torchvision import models, transforms  # 導入torchvision中的模型和轉換工具
from utils import ResNet50Model  # 導入自定義的ResNet50模型
import os  # 導入os庫，用於文件和目錄操作
from PIL import Image  # 導入PIL庫，用於圖像處理
import time  # 確保導入time模組，用於計時
import yaml  # 導入yaml庫，用於讀取配置文件

from utils import process_coffee_beans

with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)  # 讀取配置文件


# 加載 ResNet50 模型
model = ResNet50Model.load_from_checkpoint('trained_models/resnet50.ckpt')  # 從檢查點加載訓練後的模型
model.eval()  # 設置模型為評估模式

def repeat_channels(x):
    # 如果輸入的通道數為1，則重複三次以形成三通道圖像
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x

# 定義圖像轉換
input_size = settings['dataset_info']['input_size']
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),  # 將圖像調整為256x256大小
    transforms.ToTensor(),  # 將PIL圖像轉換為張量
    transforms.Lambda(repeat_channels),  # 應用重複通道的函數
    transforms.Normalize(  # 對圖像進行標準化
        mean=settings['dataset_info']['mean'],  # 計算的均值
        std=settings['dataset_info']['std']  # 計算的標準差
    ),
])

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



def process_and_predict(image, resize_size=None, show_image=True, pixel_threshold_lower=None, pixel_threshold_upper=None):
    # 處理圖像並進行預測
    expanded_beans = process_coffee_beans(
        image=image,  # 輸入圖像
        show_image=False,  # 是否顯示圖像
        pixel_threshold_lower=pixel_threshold_lower,  # 像素下限
        pixel_threshold_upper=pixel_threshold_upper  # 像素上限
    )
    
    result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # 建立一個與圖像相同大小的透明mask
    color = [(0, 0, 255), (0, 255, 0)]  # 定義顏色列表
    for index, (bean, (x, y, w, h)) in enumerate(expanded_beans):
        # 記錄預測開始時間
        start_predict_time = time.time()  
        predicted_class = predict_coffee_bean(bean)  # 對咖啡豆進行預測
        end_predict_time = time.time()  # 記錄預測結束時間

        cv2.rectangle(result, (x, y), (x+w, y+h), color[predicted_class], 2)  # 繪製矩形框
        print(f"""predicted_class: {predicted_class}, index: {index}, h: {h}, w: {w}, take time ={end_predict_time - start_predict_time:.4f}sec""")  # 輸出預測結果
        if show_image:
            cv2.imshow('Coffee Bean', bean)  # 顯示咖啡豆圖像
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.putText(
            img=result, 
            text=str(index+1),  # 標記編號
            org=(x, y-10),  # 標記位置
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.9, 
            color=color[predicted_class], 
            thickness=2
        )
        end_draw_time = time.time()  # 記錄繪製結束時間

    # 調整大小
    if resize_size is not None:
        height, width = result.shape[:2]  # 獲取結果圖像的高度和寬度
        new_width = resize_size  # 設定新的寬度
        new_height = int((new_width / width) * height)  # 根據比例計算新的高度
        result = cv2.resize(result, (new_width, new_height))  # 調整圖像大小
    
    if show_image:
        cv2.imshow('Coffee Beans Contours', result)  # 顯示結果圖像
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result  # 返回處理後的結果


def test_model_from_dataset(path, show_image=False):
    # 測試模型的預測效果
    for file in os.listdir(path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 檢查文件類型
            continue
        coffee_bean = cv2.imread(f"{path}/{file}")  # 讀取咖啡豆圖像
        h, w = coffee_bean.shape[:2]  # 獲取圖像的高度和寬度
        predicted_class = predict_coffee_bean(coffee_bean)  # 進行預測
        print(f"{file} predicted_class: {predicted_class}, h: {h}, w: {w}")  # 輸出預測結果
        if show_image:
            cv2.imshow('Coffee Beans Contours', coffee_bean)  # 顯示咖啡豆圖像
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def test_model_from_original_images(base_source_path, base_target_path, show_image=False):
    
    pixel_threshold_lower = settings['coffee_bean_pixel_threshold']['lower']  # 獲取像素下限
    pixel_threshold_upper = settings['coffee_bean_pixel_threshold']['upper']  # 獲取像素上限

    for file in os.listdir(base_source_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 檢查文件類型
            continue
        img = cv2.imread(f'{base_source_path}/{file}')  # 讀取圖像
        print(f"file: {file}")  # 輸出文件名
        height, width = img.shape[:2]  # 獲取圖像的高度和寬度
        print(f"height: {height}, width: {width}")  # 輸出圖像尺寸
        if height > width:  # 如果高度大於寬度，則旋轉圖像
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(f"height: {height}, width: {width}")  # 再次輸出圖像尺寸
        # 使用示例  
        if show_image:
            cv2.imshow('Coffee Beans', img)  # 顯示咖啡豆圖像
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        execution_times = []  # 用於儲存執行時間的列表
        start_process_time = time.time()  # 記錄開始時間
        result = process_and_predict(
            image=img,  # 輸入圖像
            show_image=show_image,  # 是否顯示圖像
            pixel_threshold_lower=pixel_threshold_lower,  # 像素下限
            pixel_threshold_upper=pixel_threshold_upper  # 像素上限
        )
        img = cv2.add(img, result)  #合成僅有框的圖片與原始圖片
        end_process_time = time.time()  # 記錄結束時間
        execution_times.append(end_process_time - start_process_time)  # 將執行時間放入列表
        print(f"process_and_predict 執行時間: {execution_times[-1]} 秒")  # 印出執行時間
        cv2.imwrite(f'{base_target_path}/{file}', img)  # 保存處理後的圖像

if __name__ == "__main__":
    import time  # 確保導入time模組
    start_time = time.time()  # 記錄開始時間
    test_model_from_original_images("Coffee bean dataset/NG", "coffee_bean_predict/NG", show_image=False)  # 處理NG類別的咖啡豆圖像
    test_model_from_original_images("Coffee bean dataset/OK", "coffee_bean_predict/OK", show_image=False)  # 處理OK類別的咖啡豆圖像
    end_time = time.time()  # 記錄結束時間
    print(f"執行時間: {end_time - start_time} 秒")  # 印出執行時間
    #test_model_from_dataset("Coffee bean dataset/OK/coffee_beans", show_image=False)  # 可選：測試模型
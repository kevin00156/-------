import cv2
import torch
import numpy as np
from torchvision import models, transforms
from utils import ResNet50Model
import os
from PIL import Image
# 加載 ResNet50 模型



# 確保加載的模型是訓練後的最終模型
model = ResNet50Model.load_from_checkpoint('trained_models/resnet50.ckpt')
model.eval()

def repeat_channels(x):
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x
# 定義圖像轉換
transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(repeat_channels),
    transforms.Normalize(
        mean=[0.5551365613937378, 0.5310814380645752, 0.4438391327857971],
        std=[0.30236372351646423, 0.2883330285549164, 0.22104455530643463]
    ),
])

def predict_coffee_bean(image):
    #cv2.imwrite("coffee_bean_rgb.jpg", image)
    #image = Image.open("coffee_bean_rgb.jpg")
    if image.shape[2] == 3:  # 確保圖像是三通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將BGR轉換為RGB格式
    image = Image.fromarray(image)  # 將NumPy數組轉換為PIL圖像
    # 確保預處理與訓練時一致
    coffee_bean_transformed = transform(image)
    coffee_bean_input = coffee_bean_transformed.unsqueeze(0)

    with torch.no_grad():
        prediction = model(coffee_bean_input)
        predicted_class = prediction.argmax(dim=1).item()
        return predicted_class

def process_coffee_beans(
        image, 
        show_image=False, 
        pixel_threshold_lower=5000, 
        pixel_threshold_upper=50000
    ):
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_image:
        cv2.imshow('灰度圖', gray)
        cv2.waitKey(0)
    
    # 使用高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if show_image:
        cv2.imshow('高斯模糊', blurred)
        cv2.waitKey(0)
    
    # 使用Otsu's二值化方法
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show_image:
        cv2.imshow('二值化', binary)
        cv2.waitKey(0)
    
    # 形態學操作：開運算去除雜訊
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if show_image:
        cv2.imshow('開運算', opening)
        cv2.waitKey(0)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 在原圖上個別框出每顆咖啡豆
    result = image.copy()
    
    filtered_contours = [
        contour for contour in contours 
        if pixel_threshold_lower < cv2.contourArea(contour) < pixel_threshold_upper]
        
    for i, contour in enumerate(filtered_contours):
        # 計算輪廓面積
        area = cv2.contourArea(contour)
        # 取得輪廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 繪製矩形框
        #cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 在框上標記編號
        #cv2.putText(result, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 顯示每個輪廓的面積
        print(f'咖啡豆 #{i+1} 面積: {area:.2f} 像素')
        
        # 在圖片上顯示面積
        #cv2.putText(result, f'Area: {area:.0f}', (x, y+h+20), 
                   # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 建立一個列表來儲存擴展後的咖啡豆區域
    expanded_beans = []
    
    # 遍歷每個咖啡豆輪廓
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area > pixel_threshold_lower and area < pixel_threshold_upper:
            x, y, w, h = cv2.boundingRect(contour)
            # 擴展邊界3像素
            x_expanded = max(0, x - 3)
            y_expanded = max(0, y - 3) 
            w_expanded = min(image.shape[1] - x_expanded, w + 6)
            h_expanded = min(image.shape[0] - y_expanded, h + 6)
            
            # 將擴展後的區域加入列表
            expanded_beans.append([x_expanded, y_expanded, w_expanded, h_expanded])
    return result, expanded_beans

def process_and_predict(image, resize_size=None, show_image=True):
    # 使用 process_coffee_beans 函數
    result, expanded_beans = process_coffee_beans(image, show_image=False)
    
    for index, (x, y, w, h) in enumerate(expanded_beans):
        coffee_bean = image[y:y+h, x:x+w]
        #cv2.imwrite(f"coffee_bean_{index}.png", coffee_bean)
        #coffee_bean_rgb = cv2.cvtColor(coffee_bean, cv2.COLOR_BGR2RGB)
        predicted_class = predict_coffee_bean(coffee_bean)
        # 在原圖上繪製預測類別
        color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        print(f"predicted_class: {predicted_class}, index: {index}, h: {h}, w: {w}")
        if show_image:
            cv2.imshow('Coffee Bean', coffee_bean)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.putText(
            img=result, 
            text=str(index+1), 
            org=(x, y-10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.9, 
            color=color, 
            thickness=2
            )
    

    # 調整大小
    if resize_size is not None:
        height, width = result.shape[:2]
        new_width = resize_size
        new_height = int((new_width / width) * height)
        result = cv2.resize(result, (new_width, new_height))
    
    if show_image:
        cv2.imshow('Coffee Beans Contours', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def test_model(path, show_image=False):
    for file in os.listdir(path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        coffee_bean = cv2.imread(f"{path}/{file}")
        #coffee_bean = cv2.cvtColor(coffee_bean, cv2.COLOR_BGR2RGB)
        h,w = coffee_bean.shape[:2]
        predicted_class = predict_coffee_bean(coffee_bean)
        print(f"{file} predicted_class: {predicted_class}, h: {h}, w: {w}")
        if show_image:
            cv2.imshow('Coffee Beans Contours', coffee_bean)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main(base_source_path, base_target_path, show_image=False):

    for file in os.listdir(base_source_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(f'{base_source_path}/{file}')
        print (f"file: {file}")
        height, width = img.shape[:2]
        print(f"height: {height}, width: {width}")
        if height > width:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #pass
        print (f"height: {height}, width: {width}")
        # 使用示例  
        if show_image:
            cv2.imshow('Coffee Beans', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        img = process_and_predict(img, show_image=show_image)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
        cv2.imwrite(f'{base_target_path}/{file}', img)

if __name__ == "__main__":
    main("Coffee bean dataset/NG", "coffee_bean_predict/NG", show_image=False)
    main("Coffee bean dataset/OK", "coffee_bean_predict/OK", show_image=False)
    #test_model("Coffee bean dataset/OK/coffee_beans", show_image=False)
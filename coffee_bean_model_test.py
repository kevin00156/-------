import cv2
import numpy as np
import os
from ultralytics import YOLO

# 載入 YOLOv11 模型
model = YOLO("coffee_bean_detection_yolo11x/yolov11_training3/weights/best.pt")  # 使用適當的模型權重文件


def predict_coffee_bean_yolo(image):
    # 使用 YOLO 模型進行預測
    results = model(image)
    # 使用 results.boxes 來獲取偵測結果
    predictions = results[0].boxes  # 取得預測結果

    return predictions

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
    # 使用 process_coffee_beans 提取咖啡豆
    _, expanded_beans = process_coffee_beans(image, show_image=show_image)
    
    # 創建一個空白圖像來顯示所有預測結果
    result_image = image.copy()

    for (x, y, w, h) in expanded_beans:
        # 提取每顆咖啡豆
        coffee_bean = image[y:y+h, x:x+w]
        
        # 使用 YOLO 模型進行驗證
        predictions = predict_coffee_bean_yolo(coffee_bean)
        
        # 找到信心度最高的預測
        if predictions:
            best_box = max(predictions, key=lambda box: box.conf[0])
            conf = best_box.conf[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            class_id = int(best_box.cls[0])

            # 在原圖上繪製預測類別
            color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(result_image, (x1 + x, y1 + y), (x2 + x, y2 + y), color, 2)
            cv2.putText(
                img=result_image, 
                text=f'ID: {class_id} Conf: {conf:.2f}', 
                org=(x1 + x, y1 + y - 10), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.9, 
                color=color, 
                thickness=2
            )
    
    # 調整大小
    if resize_size is not None:
        height, width = result_image.shape[:2]
        new_width = resize_size
        new_height = int((new_width / width) * height)
        result_image = cv2.resize(result_image, (new_width, new_height))
    
    if show_image:
        cv2.imshow('Coffee Beans Contours', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image


def test_model(path, show_image=False):
    for file in os.listdir(path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        coffee_bean = cv2.imread(f"{path}/{file}")
        #coffee_bean = cv2.cvtColor(coffee_bean, cv2.COLOR_BGR2RGB)
        h,w = coffee_bean.shape[:2]
        predicted_class = predict_coffee_bean_yolo(coffee_bean)
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
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            pass
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
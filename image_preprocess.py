import cv2
import numpy as np
import glob
import os

def save_image(image_folder, image, namespace):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    # 找到images資料夾中最大的號碼
    existing_images = glob.glob(os.path.join(image_folder, f'{namespace}_*.jpg'))
    if existing_images:
        latest_image = max(existing_images, key=os.path.getctime)
        latest_image_number = int(os.path.basename(latest_image).split('_')[-1].split('.')[0])
    else:
        latest_image_number = 0
    
    # 將圖像寫入到images資料夾中，命名是namespace_{i}.jpg
    image_path = os.path.join(image_folder, f'{namespace}_{latest_image_number + 1}.jpg')
    cv2.imwrite(image_path, image)

def process_coffee_beans(image, show_image=False, pixel_threshold_lower=10000, pixel_threshold_upper=50000):
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
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 在框上標記編號
        cv2.putText(result, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 顯示每個輪廓的面積
        print(f'咖啡豆 #{i+1} 面積: {area:.2f} 像素')
        
        # 在圖片上顯示面積
        cv2.putText(result, f'Area: {area:.0f}', (x, y+h+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
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

def main():
    show_image = True
    original_image_folder = "Coffee bean dataset/OK"
    processed_image_folder = "images/result"
    if not os.path.exists(original_image_folder):
        print(f"資料夾 {original_image_folder} 不存在")
        return
    if not os.path.exists(processed_image_folder):
        os.makedirs(processed_image_folder)
    
    for image_path in glob.glob(f"{original_image_folder}/*.jpg"):
        # 使用示例
        image = cv2.imread(image_path)
        processed_image, expanded_beans = process_coffee_beans(image, show_image=False)
        
        # 儲存結果
        cv2.imwrite(f"{processed_image_folder}/{os.path.basename(image_path)}", processed_image)
        for i, bean_range in enumerate(expanded_beans):
            x, y, w, h = bean_range
            crop_image = image[y:y+h, x:x+w]
            save_image(processed_image_folder, crop_image, f"{os.path.basename(image_path).split('.')[0]}_coffee_bean")

        # 顯示結果
        if show_image:
            processed_image = cv2.resize(processed_image, (1024,768))
            cv2.imshow('Coffee Beans Contours', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
if __name__ == '__main__':  
    main()

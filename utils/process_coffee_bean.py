import cv2
import numpy as np

"""
這是用findcontour來尋找整張照片中可能存在的咖啡豆的程式

args:
    image: 圖像(cv2格式)
    show_image: 若為true 則會顯示處理過程中的圖像
    pixel_threshold_lower: 最低contour area的閥值(單位pixel)，小於則濾除
    pixel_threshold_lower: 最高contour area的閥值(單位pixel)，大於則濾除
return:
    list(image, (x ,y ,w ,h))

    image: 咖啡豆的圖像
    (x ,y ,w ,h): 咖啡豆在原始圖片中的位置資訊
"""

def process_coffee_beans(
        image, 
        show_image=False, 
        pixel_threshold_lower=5000, 
        pixel_threshold_upper=25000
    ):
    # 處理咖啡豆圖像，並返回處理結果和擴展的咖啡豆區域
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉換為灰度圖
    if show_image:
        cv2.imshow('灰度圖', gray)  # 顯示灰度圖
        cv2.waitKey(0)
    
    # 使用高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 應用高斯模糊
    if show_image:
        cv2.imshow('高斯模糊', blurred)  # 顯示模糊圖
        cv2.waitKey(0)
    
    # 使用Otsu's二值化方法
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 進行二值化
    if show_image:
        cv2.imshow('二值化', binary)  # 顯示二值化圖
        cv2.waitKey(0)
    
    # 形態學操作：開運算去除雜訊
    kernel = np.ones((3,3), np.uint8)  # 定義形態學操作的核
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # 開運算去除雜訊
    if show_image:
        cv2.imshow('開運算', opening)  # 顯示開運算結果
        cv2.waitKey(0)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 獲取輪廓
    #result = image.copy()  # 複製原圖以便繪製結果
    
    # 過濾輪廓，根據面積篩選
    filtered_contours = [
        contour for contour in contours 
        if pixel_threshold_lower < cv2.contourArea(contour) < pixel_threshold_upper]
        
    for i, contour in enumerate(filtered_contours):
        # 計算輪廓面積
        area = cv2.contourArea(contour)  # 獲取輪廓面積
        # 取得輪廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)  # 獲取外接矩形的坐標和大小
        
        # 顯示每個輪廓的面積
        print(f'咖啡豆 #{i+1} 面積: {area:.2f} 像素')  # 輸出咖啡豆面積
        
    # 建立一個列表來儲存擴展後的咖啡豆區域
    expanded_beans = [
        (image[y:y+h, x:x+w], (x, y, w, h))  # 儲存擴展的咖啡豆區域及其坐標
        for contour in filtered_contours 
        for (x, y, w, h) in [cv2.boundingRect(contour)]
    ]
    return expanded_beans  # 返回擴展的咖啡豆圖像
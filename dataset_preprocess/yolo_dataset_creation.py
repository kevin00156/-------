import os

"""
    這是YOLOv11的資料集建立程式
    這個程式會將指定資料夾的圖片統一做標註，並將標註存成yolo的標籤格式
"""


# 定義資料集路徑
base_path = "Coffee bean dataset_YOLOv11"
ng_path = os.path.join(base_path, "NG/coffee_beans")
ok_path = os.path.join(base_path, "OK/coffee_beans")

# 定義類別ID
class_ids = {"NG": 0, "OK": 1}

# 創建標籤文件
def create_labels(image_dir, class_id):
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # 創建對應的標籤文件
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(image_dir, label_file)
            
            # 寫入標籤
            with open(label_path, 'w') as f:
                # 假設物件佔據整個圖片
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# 為NG和OK類別創建標籤
create_labels(ng_path, class_ids["NG"])
create_labels(ok_path, class_ids["OK"])

print("標籤文件已生成。")
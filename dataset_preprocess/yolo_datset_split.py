import os
import random
import shutil

# 設定隨機種子以確保結果可重現
random.seed(42)

# 資料集路徑
base_path = "coffee_bean_dataset_pixel7"
ng_path = os.path.join(base_path, "NG/coffee_beans")
ok_path = os.path.join(base_path, "OK/coffee_beans")

# 創建訓練、驗證和測試資料夾
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_path, "labels", split), exist_ok=True)

# 定義類別ID
class_ids = {"NG": 0, "OK": 1}

# 創建標籤文件
def create_labels(image_dir, class_id):
    label_dir = image_dir.replace("images", "labels")
    os.makedirs(label_dir, exist_ok=True)
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # 創建對應的標籤文件
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)
            
            # 寫入標籤
            with open(label_path, 'w') as f:
                # 假設物件佔據整個圖片
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# 為NG和OK類別創建標籤
create_labels(ng_path, class_ids["NG"])
create_labels(ok_path, class_ids["OK"])

# 獲取所有圖片文件
image_files = [(os.path.join(ng_path, f), class_ids["NG"]) for f in os.listdir(ng_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files += [(os.path.join(ok_path, f), class_ids["OK"]) for f in os.listdir(ok_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 隨機打亂圖片文件
random.shuffle(image_files)

# 設定切分比例
train_ratio = 0.8
val_ratio = 0.15
test_ratio = 0.05

# 計算每個集合的大小
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

# 分配圖片到不同的集合
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# 移動文件到對應的資料夾
def move_files(file_list, split):
    for file_path, class_id in file_list:
        file_name = os.path.basename(file_path)
        # 移動圖片
        shutil.move(file_path, os.path.join(base_path, "images", split, file_name))
        # 移動對應的標籤文件
        label_file = os.path.splitext(file_name)[0] + ".txt"
        shutil.move(file_path.replace("images", "labels").replace(".jpg", ".txt"), os.path.join(base_path, "labels", split, label_file))

move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

print("資料集已自動切分完成。")
import os
import json

"""
這支程式是用來建立資料集的
1. 需要先執行image_processing.py來處理影像，將所有結果都放入coffee_bean_dataset資料夾中
2. 再來需要先去result資料夾中，手動刪除錯誤或不必要的影像
3. 最後執行這支程式，來建立dataset.json
"""


def create_dataset_dict(source_folders, output_path):
    dataset_dict = {}

    for folder, label in source_folders.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 支援的圖片格式，處理大寫情況
                file_path = os.path.join(folder, filename)
                dataset_dict[file_path] = label

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, ensure_ascii=False, indent=4)

# 使用範例
base_path = "Coffee bean dataset"
source_folders = {
    f"{base_path}/OK/coffee_beans": 'OK',
    f"{base_path}/NG/coffee_beans": 'NG',
}

output_path = f"{base_path}/dataset.json"
create_dataset_dict(source_folders, output_path)

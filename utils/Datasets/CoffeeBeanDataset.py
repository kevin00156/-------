import json
from torch.utils.data import Dataset
from PIL import Image

class CoffeeBeanDataset(Dataset):
    def __init__(self, json_file, transform=None):
        # 讀取 JSON 檔案
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.transform = transform
        self.image_paths = list(self.data.keys())
        self.labels = list(self.data.values())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 取得影像路徑和標籤
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 開啟影像
        image = Image.open(img_path)

        # 如果有提供轉換，則應用轉換
        if self.transform:
            image = self.transform(image)

        # 將標籤轉換為數字
        label = 1 if label == "OK" else 0

        return image, label
    def get_label_count(self):
        """回傳目前資料集所擁有的標籤數量"""
        return len(set(self.labels))

# 使用範例
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])
# dataset = CoffeeBeanDataset('dataset.json', transform=transform)

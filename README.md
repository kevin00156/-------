# Coffee Bean Detection

這是一個使用 PyTorch 和 PyTorch Lightning 進行咖啡豆檢測的專案。專案包含資料預處理、模型訓練和結果可視化等功能。
# 注意

使用此專案的python版本為3.11.10，CUDA版本為12.6
若要使用其他版本，請不要直接安裝requirements.txt，請先安裝pytorch，然後根據缺少的package，自行使用pip安裝  
本專案中沒有放入訓練資料集和訓練過的權重檔，有需要可以在作者自己的雲端中下載
資料集連結：https://u.pcloud.link/publink/show?code=XZWSM55ZnW9fOTYhwLmekgJDJqv3vSTsL16V

## 專案結構

- `utils/`: 包含資料集和模型的工具程式。
  - `coffee_bean_datasets.py`: 定義了 PyTorch 資料集類別。
  - `Models/`: 包含 CNN 模型的定義。
- `coffee_bean_training.py`: 主訓練腳本，負責模型訓練和驗證。
- `image_preprocess.py`: 處理影像資料的腳本。
- `coffee_bean_dataset/`: 包含訓練和測試資料的資料夾。
- `dataset.json`: 包含影像路徑和標籤的 JSON 檔案。

## 安裝

1. 克隆此專案到本地端：

   ```bash
   git clone https://github.com/kevin00156/coffee_bean_detection
   cd coffee_bean_detection
   ```  

2. 先安裝pytorch，請參考[pytorch官網](https://pytorch.org/get-started/locally/)

3. 安裝所需的 Python 套件 (記得切換到虛擬環境)：

   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. **資料預處理**：

   使用 `dataset_preprocess內的功能` 來處理影像資料，確保所有影像都已正確儲存並標記。  
   詳細參考該資料夾內的 `README.md`
   

4. **模型訓練**：

   使用 `coffee_bean_training.py` 來訓練模型。此腳本會自動拆分資料集並開始訓練。
   請自行參考檔案前段參數定義，調整模型訓練參數
   你可以在utils/Models中新建專屬自己的Model，並在coffee_bean_training.py中引用

   ```bash
   python coffee_bean_training.py
   ```

3. **結果可視化**：

   你應該可以使用coffee_bean_model_test.py來測試模型，
   測試結果會被保存在指定的資料夾中，

4. **影片測試**

   你應該可以使用coffee_bean_video_test.py來測試影片或攝影機，
   方法與main branch相同，請自行參考


## 貢獻

歡迎對此專案進行貢獻！請提交問題或拉取請求。

## 授權

此專案採用 MIT 授權條款。詳情請參閱 [LICENSE](LICENSE) 文件。

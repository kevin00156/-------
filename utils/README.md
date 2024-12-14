# utils 資料夾

`utils` 資料夾包含了多個模組和工具，用於支援整個專案的運行和開發。以下是各個模組的簡要介紹：

## 模組列表

### 1. Datasets
- **CoffeeBeanDataset.py**: 定義了 `CoffeeBeanDataset` 類別，用於處理咖啡豆數據集。該類別從 JSON 文件中讀取數據，並提供數據增強的功能。

### 2. Models
- **CNNModel.py**: 定義了一個基本的 CNN 模型。
- **CNNModelForCIFAR10.py**: 定義了一個專門用於 CIFAR-10 數據集的 CNN 模型。
- **CNNModelForCIFAR100_Complex.py**: 定義了一個複雜的 CNN 模型，適用於 CIFAR-100 數據集。
- **CNNModelForCIFAR100_Simple.py**: 定義了一個簡單的 CNN 模型，適用於 CIFAR-100 數據集。
- **LeNet.py**: 定義了經典的 LeNet 模型。
- **ResNet.py**: 定義了 ResNet 模型的基本結構。
- **ResNet50Model.py**: 定義了一個基於 ResNet-50 的模型，並使用 PyTorch Lightning 進行封裝。
- **VGG.py**: 定義了 VGG 模型。

### 3. LightningModel
- **LightningModel.py**: 使用 PyTorch Lightning 封裝的模型類別，提供了訓練、驗證和測試的功能，並支���混淆矩陣的可視化。

### 4. load_parameters
- **load_parameters.py**: 提供了從 YAML 配置文件中加載模型、優化器、學習率調度器和數據增強的功能。

## 使用說明

- **CoffeeBeanDataset**: 用於加載和處理咖啡豆數據集，支持數據增強。
- **Models**: 提供多種神經網絡模型的實現，適用於不同的數據集和任務。
- **LightningModel**: 提供了基於 PyTorch Lightning 的訓練框架，支持自動化的訓練流程。
- **load_parameters**: 支持從配置文件中動態加載模型和訓練參數，方便進行實驗配置。

## 注意事項

- 確保在使用這些模組時，已經安裝了所需的 Python 庫，如 `torch`, `torchvision`, `pytorch_lightning` 等。
- 在使用 `load_parameters.py` 時，請確保配置文件的格式正確，並包含所需的所有參數。

</rewritten_file>

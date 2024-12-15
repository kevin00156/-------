#注意：
請自行提取資料集之後，使用底下兩個程式進行資料自動建立與標註

yolo資料集的格式應該要像是這樣:

```
datasets/
├── images/
│   ├── train/
│   │   ├── coffee_bean_1.jpg
│   │   ├── coffee_bean_2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── coffee_bean_1.jpg
│   │   ├── coffee_bean_2.jpg
│   │   └── ...
```

而原本的資料集應該要像是這樣:

```
Coffee bean dataset
├── NG/
│   ├── coffee_beans/
│   │   ├── coffee_bean_1.jpg
│   │   ├── coffee_bean_2.jpg
│   │   └── ...
└── OK/
    ├── coffee_beans/
    │   ├── coffee_bean_1.jpg
    │   ├── coffee_bean_2.jpg
    │   └── ...

```

這兩個程式的目的就是為了將原本的資料集轉換成yolo資料集的格式
**注意**：
1. 需要先從master分支中建立自己的資料集，然後再進行底下兩個程式的操作

1. yolo_dataset_creation.py  
> 這是用來將已經分類好的NG與OK資料夾中的咖啡豆，逐個標記為yolo資料集的格式，最終應該要生成一大堆跟咖啡豆名稱相同的txt檔案，
> 這些txt檔案的內容應該要像是這樣:
> ```
> 0 0.5 0.5 1 1
> ```
> ```
> 1 0.5 0.5 1 1
> ```
> 第一個0或1表示咖啡豆的類別，0表示NG，1表示OK
> 第二個0.5表示咖啡豆的中心點x座標，第三個0.5表示咖啡豆的中心點y座標，第四個1表示咖啡豆的寬度，第五個1表示咖啡豆的高度

2. yolo_dataset_split.py
> 這是用來將yolo資料集的圖片與標籤檔案進行分割，最終應該要在目標資料夾底下images/與labels/底下各生成train,val,test三個資料夾，分別存放訓練、驗證、測試的圖片與標籤檔案
> 檔案結構看起來會像這樣
> ```
> Coffee bean dataset_YOLOv11/
> ├── images/
> │   ├── train/
> │   │   ├── coffee_bean_1.jpg
> │   │   ├── coffee_bean_1.txt
> │   │   ├── coffee_bean_2.jpg
> │   │   ├── coffee_bean_2.txt
> │   │   └── ...
> │   ├── val/
> │   │   ├── coffee_bean_3.jpg
> │   │   ├── coffee_bean_3.txt
> │   │   ├── coffee_bean_4.jpg
> │   │   ├── coffee_bean_4.txt
> │   │   └── ...
> ```

最後需要建立一個空的`datasets/`資料夾，然後把整組資料夾丟進去，以滿足YOLOv11的資料集格式
最後在專案根目錄下應該會有以下的資料
> ```
> datasets/
> ├── Coffee bean dataset_YOLOv11/
> │   ├── images/
> │   │   ├── train/
> │   │   │   ├── coffee_bean_1.jpg
> │   │   │   ├── coffee_bean_1.txt
> │   │   │   │   ├── coffee_bean_2.jpg
> │   │   │   │   ├── coffee_bean_2.txt
> │   │   │   └── ...
> │   │   ├── val/
> │   │   │   ├── coffee_bean_3.jpg
> │   │   │   ├── coffee_bean_3.txt
> │   │   │   ├── coffee_bean_4.jpg
> │   │   │   ├── coffee_bean_4.txt
> │   │   │   └── ...
> ```
接著就可以開始訓練了




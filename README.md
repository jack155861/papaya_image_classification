## 完整的推論流程
- 詳見 [09_final.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/09_final.ipynb)
- 讀兩張圖一正一反，且 padding & resize，320 * 320 * 3
- 圖片左右合併，320 * 640 * 3
- segmentation model 進行預測
- 預測結果以信賴度 0.5 為界，轉為 0 與 255
- 得到去背結果
- 找出在 320 * 640 * 3 的圖片中，屬於木瓜的邊界 (理論上有兩個木瓜，若不足兩顆木瓜，表示元影像為『其他』類別)
- 若有兩個以上的木瓜區域，僅切割前兩個面積最大的
- 分割成兩張圖片，且高 >= 寬，否則轉 90 度
- resize & padding to 224 * 224 * 3
- rotate image 180 degree (產生 4 張影像，正+反，0度+180度)
- 將四張影像堆疊起來 (224 * 224 * 3) 進行推論

![image](https://user-images.githubusercontent.com/45505414/202710570-48f2dd3a-e391-4a60-9d4f-59688d8c8e88.png)
- 影像數值正規化
- classification model 進行預測

## 原始影像的分類明細
- data_label.xlsx：紀錄哪些木瓜影像為同一顆木瓜 (為正反兩面)
<img width="600" alt="data_label_inxlsx" src="https://user-images.githubusercontent.com/45505414/202681081-dead0c69-b0cb-4a15-a45f-641069a362e4.png">

## 資料夾結構
- 01_dataset：所有的 image dataset
- 02_padding_image：透過 [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb) 產生
- 02_mask_image：透過 [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb) 產生
- 03_unet_image：透過 [03_segmentation_image.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/03_segmentation_image.ipynb) 產生
- 04_unet_training：透過 04_training_unet.py 來訓練
- 06_classification_image：透過 [06_classification_image_aigo.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image_aigo.ipynb) 產生
- 07_classification_training_XXX：透過 07_classification_training.py 來訓練
- 09_final_weight：結合兩個模型，整合所有 inference 的流程


## 各步驟
### step1 安裝套件與下載原始影像
- requirements.txt：需要安裝的套件版本
- 安裝 [graphviz](https://graphviz.org/download/)
- https://1drv.ms/u/s!AjNFywK-OwvPm415xaeHb1eNht-_bg?e=mIGbb8

### step2 產生「02」資料夾
- [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb)：將原圖 padding 為正方形後，等比例縮放至 320*320，並透過「[rembg](https://pypi.org/project/rembg/)」進行去背產生遮罩圖  

### step3 產生「03」資料夾
- [03_segmentation_image.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/03_segmentation_image.ipynb)：一顆木瓜有正反兩張影像，每張影像可以旋轉4個90度，兩張影像可以產生16張 (4*4) 圖，又可以變成「左邊放正面影像右邊放反面影像」以及「左邊放反面影像右邊放正面影像」，因此一張圖片可以擴展成為 32 張影像。原圖為 320*320，擴展成為的 32 張影像皆為 320*640。
- 「03」資料夾內有兩個資料夾，「image」與「mask」，後續 unet 的訓練，需要這兩個資料夾內的檔案。

### step4 segmentation 的訓練
- [04_segmentation_training.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/04_segmentation_training.ipynb)：使用[開源程式](https://github.com/qubvel/segmentation_models)碼進行訓練，資料源為「03」資料夾
- 使用 resnet18 為 backbone
- 最後的分類只有一種，是否為 papaya
- 最後產生的模型放在「04_unet_training」裏面
- training segmentstion in terminal
```
!CUDA_VISIBLE_DEVICES=0,1 python3 04_training_unet.py
```
- 下載 [segmentation model](https://1drv.ms/u/s!AjNFywK-OwvPmv9yiqXRPhMPH2h1OA?e=6ouRyv)

### step5 衡量 segmentation 的成效
- [05_segmentation_evaluation.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/05_segmentation_evaluation.ipynb)：衡量 segmentation 的成效
- 原圖 - 實際遮罩圖 - 我們訓練出來的 model predict 的結果
<img width="600" alt="image" src="https://user-images.githubusercontent.com/45505414/202700969-972f0373-c5d5-4c51-ac1f-18ebe5fea76e.png">
<img width="600" alt="image" src="https://user-images.githubusercontent.com/45505414/202700993-a8fc95ed-f9da-423a-8193-f1ca08d3d58e.png">

- 訓練過程
<img width="600" alt="image" src="https://user-images.githubusercontent.com/45505414/202701237-a8008a2f-94c3-43c9-af99-160834871d5c.png">

### step6 產生訓練分類模型時所需的 dataset
- [06_classification_image_aigo.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image_aigo.ipynb)
- 有五種型態的dataset

### step7 訓練分類模型
* 使用 mobilenetv2 為 backbon
* model training
  * parser.add_argument('--l2', type=str) #使用 l2 regularization
  * parser.add_argument('--optimizer', type=str)  #使用優化器 sgd, adam, nadam
  * parser.add_argument('--pooling_type', type=str) #max-pooling, average-pooling, 1*1 convolution, combine all
  * parser.add_argument('--image_type', type=str) #五種dataset
  * parser.add_argument('--loss', type=str, default='sparse') #使用 label-encoding or onehot-encoding
```
!CUDA_VISIBLE_DEVICES=0 python3 07_classification_training.py --l2 0 --optimizer sgd --pooling_type max --image_type multi
```

### step8 衡量 classification 的成效
- [08_classification_evaluation.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/08_classification_evaluation.ipynb)
- 07_classification_training_chanel3/sgd/ave/relu_batch_0.1/weights_acc.h5 有最佳準確度

### step9 combine two model
- [09_save_weight.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/09_save_weight.ipynb)：須包含權重與模型架構
- [09_final.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/09_final.ipynb)：所有 inference 的流程
- [下載 unet 模型](https://1drv.ms/u/s!AjNFywK-OwvPmv9yiqXRPhMPH2h1OA?e=6ouRyv)：重新命名為unet_resnet18.h5

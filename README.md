


### 原始影像的分類明細
- data_label.xlsx：紀錄哪些木瓜影像為同一顆木瓜 (為正反兩面)
<img width="206" alt="data_label_inxlsx" src="https://user-images.githubusercontent.com/45505414/202681081-dead0c69-b0cb-4a15-a45f-641069a362e4.png">

### 資料夾結構
- 01_dataset：所有的 image dataset
- 02_padding_image：透過 [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb) 產生
- 02_mask_image：透過 [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb) 產生

### step1 安裝套件與下載原始影像
- requirements.txt：需要安裝的套件版本
- 安裝 [graphviz](https://graphviz.org/download/)
- https://1drv.ms/u/s!AjNFywK-OwvPm415xaeHb1eNht-_bg?e=mIGbb8
### step2 產生「02」資料夾
- [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb)：將原圖 padding 為正方形後，等比例縮放至 320*320，並透過「[rembg](https://pypi.org/project/rembg/)」進行去背產生遮罩圖  
### step3 產生「03」資料夾
- [03_segmentation_image.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/03_segmentation_image.ipynb)：一顆木瓜有正反兩張影像，每張影像可以旋轉4個90度，兩張影像可以產生16張 (4*4) 圖，又可以變成「左邊放正面影像右邊放反面影像」以及「左邊放反面影像右邊放正面影像」，因此一張圖片可以擴展成為 32 張影像。原圖為 320*320，擴展成為的 32 張影像皆為 320*640。
- 「03」資料夾內有兩個資料夾，「image」與「mask」，後續 unet 的訓練，需要這兩個資料夾內的檔案。


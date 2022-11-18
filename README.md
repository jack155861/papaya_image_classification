


### 原始影像的分類明細
- data_label.xlsx：紀錄哪些木瓜影像為同一顆木瓜 (為正反兩面)
<img width="206" alt="data_label_inxlsx" src="https://user-images.githubusercontent.com/45505414/202681081-dead0c69-b0cb-4a15-a45f-641069a362e4.png">

### 資料夾結構
- 01_dataset：所有的 image dataset
- 02_padding_image：透過 [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb) 產生
- 02_mask_image：透過 [02_mask_prepare.ipynb](https://github.com/jack155861/papaya_image_classification/blob/main/02_mask_prepare.ipynb) 產生

### step1 安裝套件
- requirements.txt：需要安裝的套件版本
### step2 下載原始影像
- https://1drv.ms/u/s!AjNFywK-OwvPm415xaeHb1eNht-_bg?e=mIGbb8
### step3 產生「02」資料夾
- 02_mask_prepare.ipynb：將原圖 padding 為正方形後，等比例縮放至 320*320，並透過「[rembg](https://pypi.org/project/rembg/)」進行去背產生遮罩圖  





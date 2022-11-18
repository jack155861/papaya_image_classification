### different dataset input

classification_dataset_chanel3.pkl
- one input and input size is 224 * 224 * 3 (正面與背面分別轉90度後相疊在一起，但仍為 3 個 channel)
- sample input

![image](https://user-images.githubusercontent.com/45505414/202710570-48f2dd3a-e391-4a60-9d4f-59688d8c8e88.png)
- [model structure](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image/classification_dataset_chanel3.png)
- [download dataset](https://1drv.ms/u/s!AjNFywK-OwvPm4VCH1DB9HZB_TIDXA?e=xxP7G7)

classification_dataset_chanel6.pkl
- one input and input size is 224 * 224 * 6 (正面與背面相疊在一起變成 6 個 channel)
- [model structure](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image/classification_dataset_chanel6.png)
- [download dataset](https://1drv.ms/u/s!AjNFywK-OwvPmuURdtKLVDqNwnYq7A?e=QtACPW)

classification_dataset_chanel12.pkl
- four multiple input (each input size is 224 * 224 * 3，此為一條 mobilenetv2 branch)
- sample input
<img width="896" alt="image" src="https://user-images.githubusercontent.com/45505414/202706921-2c8828e2-98aa-4e5d-b18f-64a37b28cb8f.png">

- [model structure](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image/classification_dataset_chanel12.png)
- [download dataset](https://1drv.ms/u/s!AjNFywK-OwvPmukkxKHfBoRfbNrusQ?e=9D0M00)

classification_dataset_merge.pkl
- one input and input size is 224 * 224 * 3 (正面與背面合併在一張圖片)
- sample input

![image](https://user-images.githubusercontent.com/45505414/202711684-729e4a61-9ab8-4264-9cd6-c3b9f4853761.png)
- [model structure](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image/classification_dataset_merge.png)
- [download dataset](https://1drv.ms/u/s!AjNFywK-OwvPmuNi5CBmDg2O_x_Owg?e=vXQXL5)

classification_dataset_multi.pkl
- two multiple input (each input size is 224 * 224 * 3，此為兩條 mobilenetv2 branch，一條正面，一條反面)
- [model structure](https://github.com/jack155861/papaya_image_classification/blob/main/06_classification_image/classification_dataset_multi.png)
- [download dataset](https://1drv.ms/u/s!AjNFywK-OwvPmuMdSCiLuDGMxaEKHg?e=U56YeZ)

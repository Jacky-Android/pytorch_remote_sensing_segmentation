# pytorch_remote_sensing_segmentation
此软件仓库包含针对不同遥感数据集的不同语义分割模型的 PyTorch 实现
👨‍💻👨‍💻👨‍💻

This repository contains PyTorch implementations of different semantic segmentation models for different remote sensing datasets

# 所提供模型（models）
Loading..........

# 支持数据集
[Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)

[Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

[LoveDA](https://github.com/Junjue-Wang/LoveDA)

[OpenEarthMap](https://open-earth-map.org/)

# 数据集准备

## Potsdam
数据集为城市区域的遥感影像语义分割数据集，特征影像数据分辨率为8米，包括38幅遥感影像，波段为红色、绿色，近红外波段，DSM波段，其中24幅完全标注，标注的土地覆盖类别包括：耕地：[0, 255, 255]，林地：[255, 0, 0]，草地：[0, 255, 0]，建筑用地：[255, 0, 255]，水域：[0, 0, 255]，未利用地及其他：[255, 255, 0])。数据详情：[http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)

文件夹例子
```python
Potsdam
│   ├── train_images (original)
│   ├── train_masks (original)
│   ├── test_images (original)
│   ├── test_masks (original)
│   ├── test_masks_eroded (original)
│   ├── train (processed)
│   ├── test (processed)
```
因为Potsdam数据集有38块，可以选择块来做为训练集，剩下的做为测试集
也可以使用我切好的
https://www.123pan.com/s/YnBgjv-D8j1H.html提取码:pots

https://drive.google.com/file/d/1NWLLVlUVaIZYwePbPx7Ca60AdyW-cdMT/view?usp=sharing
```python 
python pytorch_remote_sensing_senmentation/tools/potsdam_cut.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images_1024" --output-mask-dir "data/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 512 --rgb-image 
```
## Vaihingen
数据集包含33幅不同大小的遥感图像，每幅图像都是从一个更大的顶层正射影像图片提取的，图像选择的过程避免了出现没有数据的情况。顶层影像和DSM的空间分辨率为9 cm。遥感图像格式为8位TIFF文件，由近红外、红色和绿色3个波段组成。DSM是单波段的TIFF文件，灰度等级（对应于DSM高度）为32位浮点值编码。

切割好的数据集：
123盘：https://www.123pan.com/s/YnBgjv-rST4H.html提取码:Vsb8

Google Dirver:https://drive.google.com/file/d/1L2rxAzMm-pEV2dY111285t04hOm2L1hv/view?usp=drive_link

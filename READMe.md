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


## LoveDa数据集
类别标签：背景 - 1、建筑 - 2、道路 - 3、水 - 4、贫瘠 - 5、森林 - 6、农业 - 7。无数据区域被指定为 0，应忽略。
南京、常州、武汉5987张高空间分辨率（0.3 m）遥感影像
关注城乡不同地理环境
推进语义分割和领域适应任务
三个巨大的挑战：
多尺度物体
复杂背景样本
班级分布不一致
数据集：LoveDA数据集在[Zenodo](https://zenodo.org/records/5706578)
![image](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation/assets/55181594/c3f49559-d00f-4847-ae70-05a84cb6f1bb)

## OpenEarthMap
![image](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation/assets/55181594/4f50a81b-2189-4aed-a357-f184aaff9870)

OpenEarthMap 是全球高分辨率土地覆被制图的基准数据集。OpenEarthMap 包含 5000 幅航空和卫星图像，其中有人工标注的 8 类土地覆被标签和 220 万个片段，地面采样距离为 0.25-0.5 米，覆盖 6 大洲 44 个国家的 97 个地区。在 OpenEarthMap 上训练的土地覆被测绘模型可在全球范围内通用，并可在各种应用中作为现成模型使用。
数据集：[OpenEarthMap](https://zenodo.org/records/7223446)

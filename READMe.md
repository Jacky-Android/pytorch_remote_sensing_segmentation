# pytorch_remote_sensing_segmentation
此软件仓库包含针对不同遥感数据集的不同语义分割模型的 PyTorch 实现
👨‍💻👨‍💻👨‍💻

This repository contains PyTorch implementations of different semantic segmentation models for different remote sensing datasets

# 所提供模型（models）
Loading..........

# 支持数据集
[Potsdam](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation?tab=readme-ov-file#potsdam)

[Vaihingen](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation?tab=readme-ov-file#vaihingen)

[LoveDA](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation?tab=readme-ov-file#loveda%E6%95%B0%E6%8D%AE%E9%9B%86)

[OpenEarthMap](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation?tab=readme-ov-file#openearthmap)
[EarthVQA 又叫LoveDa2](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation?tab=readme-ov-file#earthvqa)

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

## EarthVQA
多模态多任务 VQA 数据集（EarthVQA）旨在连接地球视觉和地球科学语言，其中包括共同注册的遥感图像、土地覆盖语义掩码和任务驱动的语言文本。
EarthVQA 数据集包含 6000 幅 0.3 米的图像和 208593 个嵌入了城市和农村治理要求的 QA 对。这些 QA 对设计用于交通状况、教育设施、绿色生态、耕地状况等多种推理任务中的判断、计数、对象状况分析和综合分析类型。这种多模式、多任务的数据集提出了新的挑战，需要对遥感图像进行地理空间关系推理和归纳。
![image](https://github.com/Jacky-Android/pytorch_remote_sensing_segmentation/assets/55181594/0041f6ff-d578-4b9b-9c87-27a01a2c3c4e)
[下载](http://rsidea.whu.edu.cn/EarthVQA.htm)

# 鸣谢
[GeoSeg](https://github.com/WangLibo1995/GeoSeg)

[Timm](https://github.com/huggingface/pytorch-image-models)

[pytorch](https://github.com/pytorch/pytorch)

[Albumentations](https://github.com/albumentations-team/albumentations)


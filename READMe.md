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

# 数据集准备

## Potsdam
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
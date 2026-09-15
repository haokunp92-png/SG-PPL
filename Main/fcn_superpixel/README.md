# FCN 超像素 Pipeline（医学影像适配）

从 SpixelFCN 抽取的核心功能，**固定 3 通道 RGB 输入**（灰度自动转 RGB），针对医学影像优化。

## 目录结构

```
fcn_superpixel/
├── model/           # SpixelNet 模型
├── dataset.py       # 自定义数据集（支持图像+标签 或 仅图像+SLIC伪标签）
├── train_util.py    # 训练工具（poolfeat, upfeat, 网格初始化等）
├── loss.py          # 语义+位置损失
├── train.py         # 训练脚本
├── infer.py         # 推理脚本
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

若需**连通性后处理**（enforce_connectivity），需编译 superpixel_fcn-master 的 Cython 模块：

```bash
cd ../superpixel_fcn-master/third_party/cython
python setup.py install --user
```

不编译也可运行，推理时加 `--enforce_connect` 会跳过该步骤。

## 数据格式

### 方式一：图像 + 标签（有标注）

- `image_dir/`: 图像 (.jpg, .png 等)
- `label_dir/`: 标签图，与图像同名，像素值为类别 ID (0~49)

### 方式二：仅图像 + SLIC 伪标签（无标注）

- 只提供 `image_dir/`
- 使用 `--use_slic`，自动用 SLIC 生成伪标签进行训练

### 方式三：患者文件夹嵌套（如鼻咽癌病理切片）

```
数据集根目录/
├── Patient_001/
│   ├── slice1.png
│   ├── slice2.png
│   └── ...
├── Patient_002/
│   └── ...
```
- 加 `--recursive` 递归搜索所有子目录

## 医学影像默认配置

- **输入**：固定 3 通道 RGB（灰度图自动 `cv2.COLOR_GRAY2RGB`）
- **归一化**：mean=[0.5,0.5,0.5], std=[1,1,1]
- **SLIC**：compactness=15, sigma=1（针对组织纹理）
- **染色抖动**：医学模式默认开启（亮度/对比度）
- 加 `--no_medical` 可恢复自然图像配置 [0.411,0.432,0.45]

## 训练

```bash
# 医学影像（默认，灰度图自动转 RGB）
python train.py --image_dir ./pathology --save_dir ./ckpt --recursive --use_slic

# 有标签
python train.py --image_dir ./my_images --label_dir ./my_labels --save_dir ./ckpt

# 自然图像（禁用医学模式）
python train.py --image_dir ./my_images --save_dir ./ckpt --use_slic --no_medical

# 自定义归一化、SLIC
python train.py --image_dir ./pathology --save_dir ./ckpt --recursive --norm_mean 0.45,0.42,0.38 --slic_compactness 20
```

**注意**：`img_size` 需为 16 的倍数，如 208、320。

## 推理

```bash
python infer.py --image_dir ./test_images --output ./results --pretrained ./ckpt/checkpoint_epoch10.pth

# 患者文件夹嵌套 + 保留输出结构
python infer.py --image_dir /root/autodl-tmp/sPixel_CXGNN/filtered_img --output /root/autodl-tmp/sPixel_CXGNN/dataset --pretrained ./ckpt/checkpoint_epoch100.pth --recursive --preserve_structure --suffix png

# 启用连通性后处理（需先编译 cython）
python infer.py --image_dir ./test_images --output ./results --pretrained ./ckpt/best.pth --enforce_connect
```

输出：
- `results/spixel_viz/`: 超像素边界可视化
- `results/map_csv/`: 超像素 ID 图（CSV）
- 加 `--preserve_structure` 时：`output/患者编号/spixel_viz/` 和 `map_csv/`

## 使用预训练权重

若已有 SpixelFCN 在 BSDS500 上的权重（如 `SpixelNet_bsd_ckpt.tar`），可直接用于推理：

```bash
python infer.py --image_dir ./my_images --output ./out --pretrained /path/to/SpixelNet_bsd_ckpt.tar
```

也可作为训练初始化：

```bash
python train.py --image_dir ./my_images --save_dir ./ckpt --pretrained /path/to/SpixelNet_bsd_ckpt.tar
```

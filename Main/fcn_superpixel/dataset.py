"""
Custom dataset for FCN superpixel training.
支持两种数据格式:
1. 图像+标签: image_dir 和 label_dir，文件名一一对应
2. 仅图像+SLIC伪标签: 用 SLIC 生成初始超像素作为监督（无标注时使用）
"""
import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

try:
    from skimage.segmentation import slic
    _HAS_SLIC = True
except ImportError:
    _HAS_SLIC = False


class SuperpixelDataset(Dataset):
    """
    Args:
        image_dir: 图像根目录
        label_dir: 标签目录（可选）
        recursive: 若 True，递归搜索子目录（适用于 患者编号/切片.png 结构）
        img_size: 训练时裁剪尺寸 (H, W)，需为 16 的倍数
        use_slic: 无标签时用 SLIC 生成伪标签
    """
    def __init__(self, image_dir, label_dir=None, recursive=False, img_size=(208, 208), max_classes=50,
                 use_slic=True, slic_n_segments=200, slic_compactness=10, slic_sigma=1, stain_aug=False,
                 extensions=('jpg', 'jpeg', 'png', 'bmp')):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.recursive = recursive
        self.img_size = img_size
        self.max_classes = max_classes
        self.use_slic = use_slic and _HAS_SLIC
        self.slic_n_segments = slic_n_segments
        self.slic_compactness = slic_compactness
        self.slic_sigma = slic_sigma
        self.stain_aug = stain_aug

        self.samples = []
        for ext in extensions:
            if recursive:
                self.samples.extend(glob.glob(os.path.join(image_dir, '**', f'*.{ext}'), recursive=True))
            else:
                self.samples.extend(glob.glob(os.path.join(image_dir, f'*.{ext}')))
        self.samples = sorted(set(self.samples))

        if not self.samples:
            raise ValueError(f"No images found in {image_dir}" + (" (recursive)" if recursive else ""))

        if label_dir and not os.path.isdir(label_dir):
            raise ValueError(f"Label dir not found: {label_dir}")

    def __len__(self):
        return len(self.samples)

    def _get_label_path(self, img_path):
        if self.label_dir is None:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        if self.recursive:
            rel_dir = os.path.dirname(os.path.relpath(img_path, self.image_dir))
            for ext in ('png', 'jpg', 'bmp'):
                p = os.path.join(self.label_dir, rel_dir, f'{base}.{ext}')
                if os.path.isfile(p):
                    return p
        for ext in ('png', 'jpg', 'bmp'):
            p = os.path.join(self.label_dir, f'{base}.{ext}')
            if os.path.isfile(p):
                return p
        return None

    def _load_label(self, label_path):
        lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            return None
        return lbl

    def _slic_label(self, img):
        if not _HAS_SLIC:
            raise RuntimeError("skimage not installed. pip install scikit-image")
        segments = slic(img, n_segments=self.slic_n_segments, compactness=self.slic_compactness, sigma=self.slic_sigma)
        # 将 segment id 压缩到 0..max_classes-1
        unique = np.unique(segments)
        mapping = {u: i % self.max_classes for i, u in enumerate(unique)}
        out = np.zeros_like(segments, dtype=np.int64)
        for u in unique:
            out[segments == u] = mapping[u]
        return out

    def _remap_label(self, label):
        """将标签 remap 到 0..max_classes-1"""
        unique = np.unique(label)
        if len(unique) > self.max_classes:
            # 保留出现频率最高的 max_classes 个
            counts = [(np.sum(label == u), u) for u in unique]
            counts.sort(reverse=True)
            keep = set(u for _, u in counts[:self.max_classes])
            mapping = {u: i for i, u in enumerate(sorted(keep))}
            out = np.zeros_like(label, dtype=np.int64)
            for u in unique:
                if u in mapping:
                    out[label == u] = mapping[u]
                else:
                    out[label == u] = 0
            return out
        mapping = {u: i for i, u in enumerate(unique)}
        out = np.zeros_like(label, dtype=np.int64)
        for u in unique:
            out[label == u] = mapping[u]
        return out

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load {img_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 灰度转 RGB
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self._get_label_path(img_path)
        if label_path:
            label = self._load_label(label_path)
            if label is None:
                if self.use_slic:
                    label = self._slic_label(img)
                else:
                    raise ValueError(f"Failed to load label {label_path}")
            else:
                label = self._remap_label(label)
        else:
            if self.use_slic:
                label = self._slic_label(img)
            else:
                raise ValueError("No label_dir and use_slic=False. Provide labels or set use_slic=True.")

        # 随机裁剪
        h, w = img.shape[:2]
        th, tw = self.img_size
        if h >= th and w >= tw:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            img = img[i:i+th, j:j+tw]
            label = label[i:i+th, j:j+tw]
        else:
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 随机翻转
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            label = np.fliplr(label).copy()
        if random.random() < 0.5:
            img = np.flipud(img).copy()
            label = np.flipud(label).copy()

        # 医学影像：H&E 染色抖动（亮度/对比度）
        if self.stain_aug:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return img, label


def collate_fn(batch):
    imgs = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.from_numpy(np.stack([b[1] for b in batch])).unsqueeze(1).long()
    return imgs, labels

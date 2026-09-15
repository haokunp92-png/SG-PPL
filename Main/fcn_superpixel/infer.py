"""
FCN 超像素推理脚本 - 对任意图像生成超像素
用法:
  python infer.py --image_dir ./my_images --output ./results --pretrained ./ckpt/checkpoint_epoch10.pth
  python infer.py --image_dir ./test --output ./out --pretrained ./ckpt/best.pth --enforce_connect
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from model import SpixelNet1l_bn
from train_util import shift9pos, update_spixl_map, get_spixel_image

parser = argparse.ArgumentParser(description='FCN Superpixel Inference')
parser.add_argument('--image_dir', required=True, help='path to images folder')
parser.add_argument('--output', default='./output', help='path to save results')
parser.add_argument('--pretrained', required=True, help='path to checkpoint')
parser.add_argument('--suffix', default='png', help='image suffix (jpg, png, etc.)')
parser.add_argument('--recursive', action='store_true', help='递归搜索子目录（患者编号/切片.png 结构）')
parser.add_argument('--preserve_structure', action='store_true', help='输出保留患者文件夹结构 output/患者编号/')
parser.add_argument('--downsize', type=int, default=16, help='superpixel grid size (same as training)')
parser.add_argument('--enforce_connect', action='store_true', help='enforce connectivity (requires cython)')
parser.add_argument('--gpu', default='0')

# 医学影像（与训练一致，默认从 checkpoint 读取）
parser.add_argument('--norm_mean', type=str, default=None, help='归一化均值，默认从 checkpoint 或 0.5,0.5,0.5')
parser.add_argument('--norm_std', type=str, default=None, help='归一化标准差')


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 灰度转 RGB
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_paths = []
    root = Path(args.image_dir)
    for ext in (args.suffix, 'jpg', 'jpeg', 'png', 'bmp'):
        if args.recursive:
            image_paths.extend(root.rglob(f'*.{ext}'))
        else:
            image_paths.extend(root.glob(f'*.{ext}'))
    image_paths = sorted(set(str(p) for p in image_paths))

    if not image_paths:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Found {len(image_paths)} images')

    # Load model
    ckpt = torch.load(args.pretrained, map_location='cpu')
    model = SpixelNet1l_bn(ckpt)
    model = model.to(device)
    model.eval()

    os.makedirs(os.path.join(args.output, 'spixel_viz'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'map_csv'), exist_ok=True)

    # 归一化：优先 checkpoint，其次医学默认 [0.5,0.5,0.5]
    norm_mean = ckpt.get('norm_mean')
    norm_std = ckpt.get('norm_std')
    if args.norm_mean is not None:
        norm_mean = [float(x) for x in args.norm_mean.split(',')]
    elif norm_mean is None:
        norm_mean = [0.5, 0.5, 0.5]  # 医学影像默认
    if args.norm_std is not None:
        norm_std = [float(x) for x in args.norm_std.split(',')]
    elif norm_std is None:
        norm_std = [1.0, 1.0, 1.0]
    mean_tensor = torch.tensor(norm_mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(norm_std, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    total_time = 0
    for idx, img_path in enumerate(image_paths):
        img = load_image(img_path)
        if img is None:
            print(f'Skip {img_path}')
            continue

        H, W = img.shape[:2]
        H_ = int(np.ceil(H / 16) * 16)
        W_ = int(np.ceil(W / 16) * 16)

        n_spixl_h = int(np.floor(H_ / args.downsize))
        n_spixl_w = int(np.floor(W_ / args.downsize))
        spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
        spix_idx_tensor_ = shift9pos(spix_values)
        spix_idx_tensor = np.repeat(
            np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2
        )
        spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).float().to(device)

        img_resized = cv2.resize(img, (W_, H_), interpolation=cv2.INTER_CUBIC)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = (img_tensor.to(device) - mean_tensor) / std_tensor

        t0 = time.time()
        with torch.no_grad():
            output = model(img_tensor)
        curr_spixl_map = update_spixl_map(spixeIds, output)
        ori_sz_map = F.interpolate(curr_spixl_map.float(), size=(H_, W_), mode='nearest').long()
        t1 = time.time()
        total_time += t1 - t0

        n_spixel = n_spixl_h * n_spixl_w
        ori_img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        spixel_viz, spixel_label_map = get_spixel_image(
            ori_img_tensor[0],
            ori_sz_map.squeeze(),
            n_spixels=n_spixel,
            b_enforce_connect=args.enforce_connect,
        )

        rel_path = os.path.relpath(img_path, args.image_dir)
        rel_dir = os.path.dirname(rel_path)
        base = os.path.splitext(os.path.basename(img_path))[0]

        if args.preserve_structure and rel_dir != '.':
            viz_dir = os.path.join(args.output, rel_dir, 'spixel_viz')
            csv_dir = os.path.join(args.output, rel_dir, 'map_csv')
            out_base = base
        else:
            viz_dir = os.path.join(args.output, 'spixel_viz')
            csv_dir = os.path.join(args.output, 'map_csv')
            out_base = os.path.join(rel_dir, base).replace(os.sep, '_') if rel_dir != '.' else base

        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        cv2.imwrite(os.path.join(viz_dir, f'{out_base}_sPixel.png'),
                    (spixel_viz.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1])
        np.savetxt(os.path.join(csv_dir, f'{out_base}.csv'),
                   (spixel_label_map + 1).astype(int), fmt='%i', delimiter=',')

        if (idx + 1) % 10 == 0:
            print(f'Processed {idx + 1}/{len(image_paths)}')

    print(f'Done. Avg time per image: {total_time / len(image_paths):.3f}s')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()

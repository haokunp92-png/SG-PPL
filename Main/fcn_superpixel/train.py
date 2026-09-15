"""
FCN 超像素训练脚本 - 支持自定义数据集
用法:
  python train.py --image_dir ./my_images --label_dir ./my_labels --save_dir ./ckpt
  python train.py --image_dir ./my_images --save_dir ./ckpt  # 无标签时用 SLIC 伪标签
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SpixelNet1l_bn
from dataset import SuperpixelDataset, collate_fn
from loss import compute_semantic_pos_loss
from train_util import *

parser = argparse.ArgumentParser(description='FCN Superpixel Training on Custom Dataset')
parser.add_argument('--image_dir', required=True, help='path to images (根目录，患者编号/切片.png 结构时加 --recursive)')
parser.add_argument('--label_dir', default=None, help='path to labels (optional, use SLIC if None)')
parser.add_argument('--recursive', action='store_true', help='递归搜索子目录，适用于 患者编号/切片.png 结构')
parser.add_argument('--save_dir', default='./ckpt', help='path to save checkpoints')
parser.add_argument('--pretrained', default=None, help='path to pretrained checkpoint')

parser.add_argument('--img_size', type=int, nargs=2, default=[208, 208], help='train crop size (H W), must be 16*n')
parser.add_argument('--downsize', type=int, default=16, help='superpixel grid size')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--workers', type=int, default=8, help='DataLoader 多进程 worker 数量，0=单线程')
parser.add_argument('--pin_memory', action='store_true', help='pin memory 加速 CPU->GPU 传输 (CUDA 时)')
parser.add_argument('--persistent_workers', action='store_true', help='worker 常驻，避免每 epoch 重启')
parser.add_argument('--prefetch_factor', type=int, default=2, help='每个 worker 预取的 batch 数')
parser.add_argument('--num_threads', type=int, default=None, help='PyTorch CPU 线程数 (默认自动)')
parser.add_argument('--gpu', default='0', help='GPU id')

parser.add_argument('--pos_weight', type=float, default=0.003, help='position loss weight')
parser.add_argument('--max_classes', type=int, default=50, help='max label classes')
parser.add_argument('--use_slic', action='store_true', help='use SLIC pseudo-labels when no labels')
parser.add_argument('--print_freq', type=int, default=20)

# 医学影像适配（固定 3 通道 RGB，灰度自动转 RGB）
parser.add_argument('--no_medical', action='store_true', help='禁用医学模式，使用自然图像归一化 [0.411,0.432,0.45]')
parser.add_argument('--norm_mean', type=str, default=None, help='归一化均值，逗号分隔，如 0.5,0.5,0.5')
parser.add_argument('--norm_std', type=str, default=None, help='归一化标准差，逗号分隔')
parser.add_argument('--slic_compactness', type=float, default=None, help='SLIC compactness（医学默认 15）')
parser.add_argument('--slic_sigma', type=float, default=None, help='SLIC sigma（医学默认 1.0）')
parser.add_argument('--stain_aug', action='store_true', help='H&E 染色抖动（医学模式默认开启）')
parser.add_argument('--no_stain_aug', action='store_true', help='禁用染色抖动')


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Warning: CUDA not available, using CPU (may be slow)')

    # 多线程/多进程：DataLoader workers + pin_memory + persistent_workers
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    use_cuda = device.type == 'cuda'
    pin_memory = args.pin_memory or use_cuda  # CUDA 时默认开启 pin_memory
    persistent_workers = (args.persistent_workers or args.workers > 0) and args.workers > 0  # workers>0 时默认常驻
    print(f'[多线程] workers={args.workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}, prefetch_factor={args.prefetch_factor}')

    args.train_img_height, args.train_img_width = args.img_size
    args.input_img_height, args.input_img_width = args.img_size
    args.batch_size = args.batch_size

    # 归一化：医学模式 [0.5,0.5,0.5]，自然图像 [0.411,0.432,0.45]
    use_medical = not args.no_medical  # 默认医学影像模式
    if args.norm_mean is not None:
        norm_mean = [float(x) for x in args.norm_mean.split(',')]
    elif use_medical:
        norm_mean = [0.5, 0.5, 0.5]
    else:
        norm_mean = [0.411, 0.432, 0.45]
    if args.norm_std is not None:
        norm_std = [float(x) for x in args.norm_std.split(',')]
    else:
        norm_std = [1.0, 1.0, 1.0]
    norm_mean_t = torch.tensor(norm_mean, dtype=torch.float32)
    norm_std_t = torch.tensor(norm_std, dtype=torch.float32)
    print(f'[归一化] mean={norm_mean}, std={norm_std}')

    # SLIC 参数：医学模式针对组织纹理
    slic_compactness = args.slic_compactness if args.slic_compactness is not None else (15.0 if use_medical else 10.0)
    slic_sigma = args.slic_sigma if args.slic_sigma is not None else (1.0 if use_medical else 1.0)

    # Dataset（固定 3 通道，灰度自动转 RGB）
    train_set = SuperpixelDataset(
        args.image_dir,
        label_dir=args.label_dir,
        recursive=args.recursive,
        img_size=tuple(args.img_size),
        max_classes=args.max_classes,
        use_slic=args.use_slic or (args.label_dir is None),
        slic_compactness=slic_compactness,
        slic_sigma=slic_sigma,
        stain_aug=(args.stain_aug or use_medical) and not args.no_stain_aug,
    )
    print(f'Loaded {len(train_set)} images from {args.image_dir}')
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
    )
    if args.workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    train_loader = DataLoader(train_set, **loader_kwargs)

    # Model
    model = SpixelNet1l_bn()
    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f'Loaded pretrained from {args.pretrained}')
    model = model.to(device)

    param_groups = [
        {'params': model.bias_parameters(), 'weight_decay': 0},
        {'params': model.weight_parameters(), 'weight_decay': 4e-4},
    ]
    optimizer = optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.999))

    os.makedirs(args.save_dir, exist_ok=True)

    # Init grid
    spixelID, XY_feat = init_spixel_grid(
        args.train_img_height, args.train_img_width,
        downsize=args.downsize, batch_size=args.batch_size, device=device
    )

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            # Normalize
            m = norm_mean_t.to(device).view(1, 3, 1, 1)
            s = norm_std_t.to(device).view(1, 3, 1, 1)
            images = (images - m) / s

            label_1hot = label2one_hot_torch(labels, C=args.max_classes, device=device)
            XY_batch = XY_feat[:images.size(0)]
            LABXY_feat = build_LABXY_feat(label_1hot, XY_batch)

            output = model(images)
            loss, _, _ = compute_semantic_pos_loss(
                output, LABXY_feat,
                pos_weight=args.pos_weight,
                kernel_size=args.downsize,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

            if (i + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}] [{i+1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == 0:
            path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'arch': 'SpixelNet1l_bn',
                'norm_mean': norm_mean,
                'norm_std': norm_std,
            }, path)
            print(f'Saved {path}')

    print('Training done.')


if __name__ == '__main__':
    main()

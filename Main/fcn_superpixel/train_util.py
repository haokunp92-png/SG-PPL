"""
Training utilities for FCN superpixel: grid init, poolfeat, upfeat, spixel map, etc.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries

# Optional: Cython connectivity (for enforce_connectivity)
try:
    import sys
    import os
    _cython_path = os.path.join(os.path.dirname(__file__), '..', 'superpixel_fcn-master', 'third_party', 'cython')
    if os.path.exists(_cython_path):
        sys.path.insert(0, _cython_path)
    from connectivity import enforce_connectivity
    _HAS_CONNECTIVITY = True
except ImportError:
    _HAS_CONNECTIVITY = False


def init_spixel_grid(img_height, img_width, downsize=16, batch_size=1, device='cuda'):
    """Initialize superpixel grid indices and XY coordinates."""
    n_spixl_h = int(np.floor(img_height / downsize))
    n_spixl_w = int(np.floor(img_width / downsize))
    spixel_height = int(img_height / n_spixl_h)
    spixel_width = int(img_width / n_spixl_w)

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)
    spix_idx_tensor = np.repeat(np.repeat(spix_idx_tensor_, spixel_height, axis=1), spixel_width, axis=2)
    torch_spix_idx = torch.from_numpy(np.tile(spix_idx_tensor, (batch_size, 1, 1, 1))).float().to(device)

    all_h = np.arange(0, img_height, 1)
    all_w = np.arange(0, img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h, all_w, indexing='ij'))
    coord_tensor = np.concatenate([curr_pxl_coord[1:2], curr_pxl_coord[:1]])
    all_XY_feat = torch.from_numpy(np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)).to(device)
    return torch_spix_idx, all_XY_feat


def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)
    top = input_pd[:, :-2*h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom = input_pd[:, 2*h_shift_unit:, w_shift_unit:-w_shift_unit]
    left = input_pd[:, h_shift_unit:-h_shift_unit, :-2*w_shift_unit]
    right = input_pd[:, h_shift_unit:-h_shift_unit, 2*w_shift_unit:]
    center = input_pd[:, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom_right = input_pd[:, 2*h_shift_unit:, 2*w_shift_unit:]
    bottom_left = input_pd[:, 2*h_shift_unit:, :-2*w_shift_unit]
    top_right = input_pd[:, :-2*h_shift_unit, 2*w_shift_unit:]
    top_left = input_pd[:, :-2*h_shift_unit, :-2*w_shift_unit]
    return np.concatenate([top_left, top, top_right, left, center, right, bottom_left, bottom, bottom_right], axis=0)


def poolfeat(input, prob, sp_h=2, sp_w=2):
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum = feat_sum + shift_feat[:, :-1, :, :]
        prob_sum = prob_sum + shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape
    device = input.device
    h_shift_unit, w_shift_unit = 1, 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w], device=device)], dim=1)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    send_to_tl = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2*h_shift_unit:, 2*w_shift_unit:]
    feat_sum, prob_sum = send_to_tl[:, :-1, :, :].clone(), send_to_tl[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2*h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2*h_shift_unit:, :-2*w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2*w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, :-2*w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2*h_shift_unit, 2*w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2*h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2*h_shift_unit, :-2*w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    return feat_sum / (prob_sum + 1e-8)


def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape
    h_shift, w_shift = 1, 1
    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    feat_sum = F.interpolate(feat_pd[:, :, :-2*h_shift, :-2*w_shift], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 0, 1)
    feat_sum += F.interpolate(feat_pd[:, :, :-2*h_shift, w_shift:-w_shift], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 1, 1)
    feat_sum += F.interpolate(feat_pd[:, :, :-2*h_shift, 2*w_shift:], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 2, 1)
    feat_sum += F.interpolate(feat_pd[:, :, h_shift:-h_shift, :-2*w_shift], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 3, 1)
    feat_sum += F.interpolate(input, (h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 4, 1)
    feat_sum += F.interpolate(feat_pd[:, :, h_shift:-h_shift, 2*w_shift:], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 5, 1)
    feat_sum += F.interpolate(feat_pd[:, :, 2*h_shift:, :-2*w_shift], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 6, 1)
    feat_sum += F.interpolate(feat_pd[:, :, 2*h_shift:, w_shift:-w_shift], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 7, 1)
    feat_sum += F.interpolate(feat_pd[:, :, 2*h_shift:, 2*w_shift:], size=(h*up_h, w*up_w), mode='nearest') * prob.narrow(1, 8, 1)
    return feat_sum


def update_spixl_map(spixl_map_idx_in, assig_map_in):
    assig_map = assig_map_in.clone()
    b, _, h, w = assig_map.shape
    _, _, id_h, id_w = spixl_map_idx_in.shape
    spixl_map_idx = spixl_map_idx_in if (id_h == h and id_w == w) else F.interpolate(spixl_map_idx_in, size=(h, w), mode='nearest')
    assig_max, _ = torch.max(assig_map, dim=1, keepdim=True)
    assignment_ = (assig_map == assig_max).float().to(assig_map.device)
    new_spixl_map = torch.sum(spixl_map_idx * assignment_, dim=1, keepdim=True).long()
    return new_spixl_map


def get_spixel_image(given_img, spix_index, n_spixels=600, b_enforce_connect=False):
    if not isinstance(given_img, np.ndarray):
        given_img_np_ = given_img.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        given_img_np_ = given_img
    if not isinstance(spix_index, np.ndarray):
        spix_index_np = spix_index.detach().cpu().numpy()
        if spix_index_np.ndim == 3:
            spix_index_np = spix_index_np[0]
    else:
        spix_index_np = spix_index.copy()

    h, w = spix_index_np.shape
    given_img_np = cv2.resize(given_img_np_, (w, h), interpolation=cv2.INTER_CUBIC)

    if b_enforce_connect and _HAS_CONNECTIVITY:
        spix_index_np = spix_index_np.astype(np.int64)
        segment_size = (h * w) / (n_spixels * 1.0)
        min_size = int(0.06 * segment_size)
        max_size = int(3 * segment_size)
        spix_index_np = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]

    cur_max = np.max(given_img_np) or 1.0
    spixel_bd_image = mark_boundaries(given_img_np / cur_max, spix_index_np.astype(int), color=(0, 1, 1))
    return (cur_max * spixel_bd_image).astype(np.float32).transpose(2, 0, 1), spix_index_np


def build_LABXY_feat(label_in, XY_feat):
    img_lab = label_in.clone().float()
    b, _, ch, cw = XY_feat.shape
    scale_img = F.interpolate(img_lab, size=(ch, cw), mode='nearest')
    return torch.cat([scale_img, XY_feat], dim=1)


def label2one_hot_torch(labels, C=50, device='cuda'):
    b, _, h, w = labels.shape
    one_hot = torch.zeros(b, C, h, w, dtype=torch.long, device=device)
    target = one_hot.scatter_(1, labels.long().clamp(0, C-1), 1)
    return target.float()


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

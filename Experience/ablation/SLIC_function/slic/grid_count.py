"""
与 `fcn_superpixel/infer.py` 一致的超像素「目标节点数」计算。

infer 中：
  H_ = ceil(H/16)*16, W_ = ceil(W/16)*16  （与网络输入对齐的 padding）
  n_spixl_h = floor(H_ / downsize)
  n_spixl_w = floor(W_ / downsize)
  n_spixel = n_spixl_h * n_spixl_w
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def expected_grid_shape(
    height: int,
    width: int,
    *,
    pad_multiple: int = 16,
    downsize: int = 16,
) -> Tuple[int, int, int, int]:
    """
    返回 (H_, W_, n_spixl_h, n_spixl_w)，与 FCN infer 中用于生成初始网格的量一致。
    """
    H_ = int(np.ceil(int(height) / pad_multiple) * pad_multiple)
    W_ = int(np.ceil(int(width) / pad_multiple) * pad_multiple)
    n_h = int(np.floor(H_ / downsize))
    n_w = int(np.floor(W_ / downsize))
    return H_, W_, n_h, n_w


def target_superpixel_count(
    height: int,
    width: int,
    *,
    pad_multiple: int = 16,
    downsize: int = 16,
) -> int:
    """FCN 风格目标超像素数：n_spixl_h * n_spixl_w。"""
    _, _, n_h, n_w = expected_grid_shape(
        height, width, pad_multiple=pad_multiple, downsize=downsize
    )
    return max(1, n_h * n_w)

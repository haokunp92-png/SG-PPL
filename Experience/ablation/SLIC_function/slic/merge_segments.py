"""
当 SLIC 产生的连通域标签数多于目标时，贪合并至目标数量（仅「压少」不「拆多」）。
每轮将两块 4-邻接区域中「面积较小者」并入「面积较大者」的标签。
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np

try:
    from skimage.segmentation import relabel_sequential
except ImportError:  # pragma: no cover
    relabel_sequential = None


def _build_adjacent_label_pairs(lab: np.ndarray) -> Set[Tuple[int, int]]:
    H, W = lab.shape
    pairs: Set[Tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            a = int(lab[r, c])
            if c + 1 < W:
                b = int(lab[r, c + 1])
                if a != b:
                    pairs.add((a, b) if a < b else (b, a))
            if r + 1 < H:
                b = int(lab[r + 1, c])
                if a != b:
                    pairs.add((a, b) if a < b else (b, a))
    return pairs


def merge_superpixels_to_count(lab: np.ndarray, target_n: int) -> np.ndarray:
    """
    若当前唯一标签数 > target_n，反复合并邻接超像素，直到数量 <= target_n。
    若已 <= target_n，原样返回（不做拆分）。
    """
    if relabel_sequential is None:
        raise RuntimeError("需要 scikit-image：pip install scikit-image")

    lab = np.asarray(lab, dtype=np.int64).copy()
    if target_n < 1:
        raise ValueError("target_n 须 >= 1")

    lab, _, _ = relabel_sequential(lab)

    while True:
        u = np.unique(lab)
        if len(u) <= target_n:
            return lab.astype(np.int64)

        counts = np.bincount(lab.ravel().astype(np.int64))
        pairs = _build_adjacent_label_pairs(lab)
        if not pairs:
            return lab.astype(np.int64)

        best_key: Optional[Tuple[int, int, int, int]] = None
        best_pair: Optional[Tuple[int, int]] = None
        for a, b in pairs:
            ca, cb = int(counts[a]), int(counts[b])
            key = (min(ca, cb), max(ca, cb), a, b)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (a, b)

        assert best_pair is not None
        a, b = best_pair
        ca, cb = int(counts[a]), int(counts[b])
        if ca <= cb:
            lab[lab == a] = b
        else:
            lab[lab == b] = a

        lab, _, _ = relabel_sequential(lab)

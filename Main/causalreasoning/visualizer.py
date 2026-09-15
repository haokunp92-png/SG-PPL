"""
ITE 归因可视化器
严格按照每个超像素节点的贡献度（per_superpixel_importance）在原始超像素图上涂色。

设计原则：
1. 每个超像素节点的颜色 = 其归因分数（不使用任何默认占位 0.5）。
2. 每张切片内做 min-max 归一化，拉满 colormap 的颜色对比度。
3. 缺失分数的节点用该切片的均值填补（中性，不影响相对差异）。
4. 仅在与超像素重合且原图非“纯黑背景”的像素上叠加热力图；大图黑底及节点内黑色空洞保持纯黑输出。
5. PNG 上不含任何文字，所有数值信息在 JSON 中。
6. 原始超像素图路径可配置（--dataset_root）。
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def _parse_slice_number(slice_id) -> Optional[int]:
    """从复杂 slice_id 中提取 4 位数字编号。
    支持 '10083224_t1c_image_z_0022' / '0022' / 22 / '022' 等。
    返回 None 表示无法解析。
    """
    if isinstance(slice_id, (int, float)):
        return int(slice_id)
    s = str(slice_id).strip()
    m = re.search(r"z[_]?(\d+)", s)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", s)
    if nums:
        return int(nums[-1])
    return None


def _resolve_image_path(
    dataset_root: Path,
    patient_id: str,
    slice_id,
    pattern: str,
) -> Optional[Path]:
    """根据 pattern 拼接原图路径；找不到则返回 None。"""
    slice_num = _parse_slice_number(slice_id)
    if slice_num is None:
        return None
    candidate = pattern.format(
        dataset_root=str(dataset_root),
        patient_id=patient_id,
        slice_num=slice_num,
        slice_id=slice_id,
    )
    p = Path(candidate)
    if p.exists():
        return p
    return None


def _build_roi_mask(
    h: int,
    w: int,
    pixel_coords: Dict[int, np.ndarray],
) -> np.ndarray:
    """所有超像素覆盖像素的并集（与 step1 CSV 一致）。"""
    mask = np.zeros((h, w), dtype=bool)
    for coords in pixel_coords.values():
        if coords is None or len(coords) == 0:
            continue
        rows = coords[:, 0].astype(np.int64)
        cols = coords[:, 1].astype(np.int64)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        mask[rows[valid], cols[valid]] = True
    return mask


def _dark_background_mask(img_rgb: np.ndarray, bg_threshold: int) -> np.ndarray:
    """近黑背景：max(R,G,B) <= bg_threshold 视为背景/空洞，不涂热力图。"""
    if img_rgb.ndim != 3:
        raise ValueError("img_rgb 必须为 HWC")
    return np.max(img_rgb, axis=2) <= int(bg_threshold)


def _normalize_per_slice_scores(
    pixel_coords: Dict[int, np.ndarray],
    per_sp_scores: Dict[int, float],
) -> Dict[int, float]:
    """对当前切片内所有节点做 min-max 归一化，缺失节点用均值填补。
    返回 {node_id: normalized_score in [0, 1]}。
    """
    node_ids = list(pixel_coords.keys())
    raw = np.array(
        [float(per_sp_scores.get(int(nid), np.nan)) for nid in node_ids],
        dtype=np.float64,
    )
    valid_mask = ~np.isnan(raw)
    if not valid_mask.any():
        return {int(nid): 0.5 for nid in node_ids}

    valid_mean = float(raw[valid_mask].mean())
    raw[~valid_mask] = valid_mean

    vmin, vmax = float(raw.min()), float(raw.max())
    if vmax - vmin < 1e-12:
        # 所有分数相同：统一给 0.5（中性）
        normalized = np.full_like(raw, 0.5)
    else:
        normalized = (raw - vmin) / (vmax - vmin)

    return {int(nid): float(score) for nid, score in zip(node_ids, normalized)}


def _overlay_heatmap_on_image(
    img_rgb: np.ndarray,
    pixel_coords: Dict[int, np.ndarray],
    normalized_scores: Dict[int, float],
    alpha: float = 0.6,
    colormap: int = None,
    bg_threshold: int = 15,
) -> np.ndarray:
    """在「超像素 ROI ∩ 非黑背景」像素上叠加热力图，其余输出纯黑。

    - 大图黑边、以及 ROI 节点内的黑色空洞（原图近黑）均不涂色，保持黑。
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) 未安装，请先 pip install opencv-python")
    if colormap is None:
        colormap = cv2.COLORMAP_JET

    h, w = img_rgb.shape[:2]
    roi_mask = _build_roi_mask(h, w, pixel_coords)
    dark_mask = _dark_background_mask(img_rgb, bg_threshold)
    fill_mask = roi_mask & (~dark_mask)

    score_map = np.zeros((h, w), dtype=np.float32)

    for node_id, coords in pixel_coords.items():
        if coords is None or len(coords) == 0:
            continue
        score = float(normalized_scores.get(int(node_id), 0.5))
        rows = coords[:, 0].astype(np.int64)
        cols = coords[:, 1].astype(np.int64)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rows, cols = rows[valid], cols[valid]
        if len(rows) == 0:
            continue
        # 仅对非黑背景像素写入分数，避免黑色区域被 colormap 低值着色
        m = fill_mask[rows, cols]
        if m.any():
            score_map[rows[m], cols[m]] = score

    score_u8 = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(score_u8, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # 非绘制区域：热力图层也为 0（纯黑），避免 JET 的暗蓝色泄漏到背景
    heat_rgb[~fill_mask] = 0

    out = np.zeros((h, w, 3), dtype=np.uint8)
    if fill_mask.any():
        blend = (
            (1.0 - alpha) * img_rgb.astype(np.float32)
            + alpha * heat_rgb.astype(np.float32)
        )
        out[fill_mask] = np.clip(blend[fill_mask], 0, 255).astype(np.uint8)
    return out


class AttributionVisualizer:
    """单患者归因可视化器。"""

    def __init__(
        self,
        step1_pkl: Path,
        dataset_root: Path,
        image_pattern: str = "{dataset_root}/{patient_id}/spixel_viz/{patient_id}_t1c_image_z_{slice_num:04d}_sPixel.png",
        alpha: float = 0.6,
        bg_threshold: int = 15,
    ):
        self.step1_pkl = Path(step1_pkl)
        self.dataset_root = Path(dataset_root)
        self.image_pattern = image_pattern
        self.alpha = alpha
        self.bg_threshold = int(bg_threshold)

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) 未安装，请先 pip install opencv-python")

        self.step1_data = self._load_step1()

    def _load_step1(self) -> Dict:
        if not self.step1_pkl.exists():
            raise FileNotFoundError(
                f"未找到 step1 输出: {self.step1_pkl}\n"
                f"请先运行 run_step1.py 生成 step1_slice_graphs.pkl"
            )
        with open(self.step1_pkl, "rb") as f:
            return pickle.load(f)

    def visualize_patient(
        self,
        patient_id: str,
        per_superpixel_importance: Dict[str, Dict[int, float]],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """为一个患者的所有切片生成热力图。

        Args:
            patient_id: 患者 ID
            per_superpixel_importance: {slice_id: {node_id: score}}
            output_dir: 输出目录

        Returns:
            {slice_id: 保存的 PNG 路径}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if patient_id not in self.step1_data:
            print(f"[visualizer] 警告：患者 {patient_id} 不在 step1 数据中，跳过")
            return {}

        patient_slices = self.step1_data[patient_id]
        saved: Dict[str, Path] = {}
        skipped: list = []

        for slice_id, per_sp_scores in per_superpixel_importance.items():
            slice_data = patient_slices.get(slice_id)
            if slice_data is None:
                # 兼容数字 key
                slice_num = _parse_slice_number(slice_id)
                if slice_num is not None:
                    for k, v in patient_slices.items():
                        if _parse_slice_number(k) == slice_num:
                            slice_data = v
                            break
            if slice_data is None:
                skipped.append((slice_id, "step1 无对应切片"))
                continue

            pixel_coords = slice_data.get("pixel_coords")
            if not pixel_coords:
                skipped.append((slice_id, "step1 缺少 pixel_coords"))
                continue

            img_path = _resolve_image_path(
                self.dataset_root, patient_id, slice_id, self.image_pattern
            )
            if img_path is None:
                skipped.append((slice_id, "未找到原图"))
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append((slice_id, f"无法读取图像 {img_path}"))
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            normalized = _normalize_per_slice_scores(pixel_coords, per_sp_scores)
            overlay = _overlay_heatmap_on_image(
                img_rgb,
                pixel_coords,
                normalized,
                alpha=self.alpha,
                bg_threshold=self.bg_threshold,
            )

            out_path = output_dir / f"slice_{slice_id}_attribution.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved[slice_id] = out_path

        print(
            f"[visualizer] 患者 {patient_id} 共生成 {len(saved)} 张归因热力图"
            + (f"（跳过 {len(skipped)} 张）" if skipped else "")
        )
        for sid, reason in skipped:
            print(f"  - 跳过 slice={sid}: {reason}")

        return saved


__all__ = ["AttributionVisualizer"]

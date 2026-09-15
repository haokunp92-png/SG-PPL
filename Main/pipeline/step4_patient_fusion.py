"""
第四步：多切片融合为患者级图像特征

将同一患者的多张切片表征进行平均池化，得到患者级图像特征 img_feat_patient。
融合公式：patient_img_feat = (1/K) * sum(slice_feat_i for i=1 to K)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

from .step3_gnn_encoder import D_SLICE


# 患者级图像特征维度，等于 d_slice
D_IMG_PATIENT = D_SLICE


def load_step3_feats(pkl_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    加载第三步输出的切片特征。
    Returns: {patient_id: {slice_id: slice_feat}}
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def fuse_slices_to_patient(
    slice_feats: Dict[str, np.ndarray],
    d_slice: int = D_SLICE,
) -> np.ndarray:
    """
    对单患者的多张切片特征做平均池化。

    Args:
        slice_feats: {slice_id: slice_feat}，每个 slice_feat 为 (d_slice,)
        d_slice: 切片特征维度，用于空切片时的占位

    Returns:
        img_feat_patient: (d_img_patient,) = (d_slice,)
    """
    feats = list(slice_feats.values())
    if not feats:
        return np.zeros(d_slice, dtype=np.float32)
    return np.mean(feats, axis=0).astype(np.float32)


def run_step4_fusion(
    step3_pkl: str,
    output_dir: str,
    patient_ids: Optional[List[str]] = None,
    d_slice: int = D_SLICE,
    step3_pkl_causal: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    执行第四步：多切片融合为患者级图像特征。

    Args:
        step3_pkl: 第三步输出 step3_slice_feats.pkl 路径
        output_dir: 输出目录
        patient_ids: 可选，指定患者 ID 列表
        d_slice: 切片特征维度，需与第三步一致

    Returns:
        {patient_id: img_feat_patient}
    """
    slice_feats = load_step3_feats(step3_pkl)
    if patient_ids is not None:
        slice_feats = {k: v for k, v in slice_feats.items() if k in patient_ids}

    # 从数据推断 d_slice
    for slices in slice_feats.values():
        for feat in slices.values():
            if feat.size > 0:
                d_slice = int(feat.shape[-1])
                break
        else:
            continue
        break

    result = {}
    for pid, slices in slice_feats.items():
        result[pid] = fuse_slices_to_patient(slices, d_slice=d_slice)

    result_causal = None
    if step3_pkl_causal and Path(step3_pkl_causal).exists():
        slice_feats_causal = load_step3_feats(step3_pkl_causal)
        if patient_ids is not None:
            slice_feats_causal = {k: v for k, v in slice_feats_causal.items() if k in patient_ids}
        common_pids = sorted(set(result.keys()) & set(slice_feats_causal.keys()))
        result_causal = {}
        for pid in common_pids:
            result_causal[pid] = fuse_slices_to_patient(slice_feats_causal[pid], d_slice=d_slice)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d_img_patient = d_slice
    out_path = output_dir / "step4_patient_img_feats.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"已保存到 {out_path}，d_img_patient={d_img_patient}")

    if result_causal is not None:
        out_path_causal = output_dir / "step4_patient_img_feats_causal.pkl"
        with open(out_path_causal, "wb") as f:
            pickle.dump(result_causal, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"已保存因果版患者特征到 {out_path_causal}")

    meta = {
        pid: {
            "n_slices": len(slices),
            "d_img_patient": d_img_patient,
        }
        for pid, slices in slice_feats.items()
    }
    meta["global"] = {
        "d_img_patient": d_img_patient,
        "has_causal_img_feat": bool(result_causal is not None),
    }
    meta_path = output_dir / "step4_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已保存元信息到 {meta_path}")

    return result

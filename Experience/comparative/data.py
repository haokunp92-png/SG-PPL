"""
对比实验的数据装载：把任意前端（我的超像素 / SLIC）的 step4 患者级图像特征
与 clinical.csv 组装成基线模型可用的多模态输入。

设计约束：
1. 基线不含治疗变量 youdao —— 关联模型只能估计 as-treated 结局，
   保持治疗无关输入才与 DH-CaS 的 joint_feat_causal 用同一套协变量。
2. 划分优先从 step5 产物读取，保证与主流水线逐例一致，指标可直接并列。
3. 标准化统计量只在训练集上拟合，避免信息泄漏到验证/测试集。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.step5_clinical_fusion import load_clinical_csv  # noqa: E402

# 与主流水线 step5 一致的默认划分比例与种子
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 42

INPUT_MODES = ("concat", "joint_causal")


class ComparativeDataset:
    """一个前端 × 一种输入模式下的对比实验数据集。"""

    def __init__(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        youdao: np.ndarray,
        patient_ids: List[str],
        split: Dict[str, np.ndarray],
        feature_names: List[str],
        meta: Dict[str, Any],
    ):
        self.X = X
        self.time = time
        self.event = event
        self.youdao = youdao
        self.patient_ids = patient_ids
        self.split = split
        self.feature_names = feature_names
        self.meta = meta

    @property
    def d_in(self) -> int:
        return int(self.X.shape[1])

    def subset(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """取某个划分的 (X, time, event)。"""
        idx = self.split[name]
        return self.X[idx], self.time[idx], self.event[idx]

    def describe(self) -> str:
        parts = [
            f"样本 {len(self.patient_ids)} 例, 输入维度 {self.d_in} ({self.meta['input_mode']})",
            "划分 " + ", ".join(f"{k}={len(v)}" for k, v in self.split.items()),
            f"事件率 {float(self.event.mean()):.3f}",
        ]
        return " | ".join(parts)


def load_step4_img_feats(step4_output: str) -> Dict[str, np.ndarray]:
    """读取 step4 患者级图像特征；step4_output 决定用哪个前端的超像素。"""
    path = Path(step4_output)
    if path.is_dir():
        path = path / "step4_patient_img_feats.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到患者级图像特征: {path}\n"
            "请先对该前端跑完 Step1-4（SLIC 分支见 SLIC_function/run_slic_pipeline.py）"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_step5(step5_output: Optional[str]) -> Optional[Dict[str, Any]]:
    if not step5_output:
        return None
    path = Path(step5_output)
    if path.is_dir():
        path = path / "step5_aligned_data.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _split_from_step5(
    step5: Dict[str, Any], patient_ids: List[str]
) -> Optional[Dict[str, np.ndarray]]:
    """复用 step5 的患者划分，保证与主流水线逐例一致。"""
    keys = ("train_ids", "val_ids", "test_ids")
    if not all(k in step5 for k in keys):
        return None
    pos = {pid: i for i, pid in enumerate(patient_ids)}
    split = {}
    for name, key in zip(("train", "val", "test"), keys):
        split[name] = np.array(
            [pos[p] for p in step5[key] if p in pos], dtype=np.int64
        )
    if len(split["train"]) == 0:
        return None
    return split


def _split_fresh(
    n: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """与 step5 相同的划分逻辑（seed → permutation → 70/15/15）。"""
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def _build_concat_features(
    img_feats: Dict[str, np.ndarray],
    clinical: Dict[str, Dict[str, Any]],
    patient_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    多模态拼接输入：[图像特征(d_img), ajcc, age]，不含 youdao。

    这是 DeepSurv 等基线的标准喂法（原始协变量直接进网络），
    不借用主流水线的门控交互融合，避免把本文的融合贡献让给基线。
    """
    d_img = int(next(iter(img_feats.values())).shape[0])
    rows = []
    for pid in patient_ids:
        c = clinical[pid]
        rows.append(
            np.concatenate(
                [
                    np.asarray(img_feats[pid], dtype=np.float32).ravel(),
                    np.array([float(c["ajcc"]), float(c["age"])], dtype=np.float32),
                ]
            )
        )
    names = [f"img_{i}" for i in range(d_img)] + ["ajcc", "age"]
    return np.stack(rows).astype(np.float32), names


def _build_joint_causal_features(
    step5: Optional[Dict[str, Any]], patient_ids: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """直接使用 step5 的 joint_feat_causal（含主流水线的门控交互融合）。"""
    if step5 is None:
        raise ValueError("input_mode=joint_causal 需要提供 --step5_output")
    by_pid = {r["patient_id"]: r for r in step5.get("aligned", [])}
    missing = [p for p in patient_ids if p not in by_pid]
    if missing:
        raise ValueError(f"step5 中缺少 {len(missing)} 例患者，例如 {missing[:3]}")
    if "joint_feat_causal" not in next(iter(by_pid.values())):
        raise ValueError(
            "step5 产物为旧版 schema（无 joint_feat_causal，含治疗泄漏）。\n"
            "请重跑 step5，或改用 --input_mode concat"
        )
    X = np.stack([by_pid[p]["joint_feat_causal"] for p in patient_ids]).astype(np.float32)
    return X, [f"joint_{i}" for i in range(X.shape[1])]


def standardize(
    X: np.ndarray, train_idx: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """按训练集统计量做 z-score；图像特征方差极小，不标准化会严重拖慢收敛。"""
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32), {"mean": mu, "std": sd}


def build_dataset(
    step4_output: str,
    clinical_csv: str,
    step5_output: Optional[str] = None,
    input_mode: str = "concat",
    seed: int = DEFAULT_SEED,
    do_standardize: bool = True,
) -> ComparativeDataset:
    """
    组装对比实验数据集。

    Args:
        step4_output: 某个前端的 step4 输出目录（决定"我的超像素"还是 SLIC）
        clinical_csv: clinical.csv 路径
        step5_output: 可选；用于复用主流水线划分，input_mode=joint_causal 时必需
        input_mode: concat（图像⊕ajcc⊕age）或 joint_causal（step5 门控融合输出）
        seed: 无 step5 划分可复用时，自建划分的随机种子
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode 须为 {INPUT_MODES} 之一，收到 {input_mode}")

    img_feats = load_step4_img_feats(step4_output)
    clinical = load_clinical_csv(clinical_csv)
    step5 = _load_step5(step5_output)

    patient_ids = sorted(set(img_feats) & set(clinical))
    if not patient_ids:
        raise ValueError("图像特征与临床数据无交集患者，请检查 patient_id 是否一致")

    if input_mode == "concat":
        X, feature_names = _build_concat_features(img_feats, clinical, patient_ids)
    else:
        X, feature_names = _build_joint_causal_features(step5, patient_ids)

    time = np.array([clinical[p]["deadtime"] for p in patient_ids], dtype=np.float64)
    event = np.array([clinical[p]["event"] for p in patient_ids], dtype=np.int32)
    youdao = np.array([clinical[p]["youdao"] for p in patient_ids], dtype=np.int32)

    split = _split_from_step5(step5, patient_ids) if step5 else None
    split_source = "step5"
    if split is None:
        split = _split_fresh(len(patient_ids), seed=seed)
        split_source = f"fresh(seed={seed})"

    scaler = None
    if do_standardize:
        X, scaler = standardize(X, split["train"])

    meta = {
        "input_mode": input_mode,
        "step4_output": str(step4_output),
        "step5_output": str(step5_output) if step5_output else None,
        "split_source": split_source,
        "n_patients": len(patient_ids),
        "d_in": int(X.shape[1]),
        "includes_treatment": False,
        "event_rate": float(event.mean()),
        "standardized": bool(do_standardize),
    }
    if scaler is not None:
        meta["scaler_std_min"] = float(scaler["std"].min())

    return ComparativeDataset(
        X=X,
        time=time,
        event=event,
        youdao=youdao,
        patient_ids=patient_ids,
        split=split,
        feature_names=feature_names,
        meta=meta,
    )

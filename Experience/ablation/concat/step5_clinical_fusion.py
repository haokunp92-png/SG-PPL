"""
第五步（concat 消融）：临床数据读取与模态对齐

读取 clinical.csv，将图像特征与临床特征直接拼接后投影到 d_joint，得到 joint_feat。
划分训练集、验证集、测试集（患者不跨集）。
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import numpy as np

# 临床特征维度
D_AJCC_EMB = 4
D_CLINIC = D_AJCC_EMB + 1 + 1  # ajcc_emb + age + youdao
D_JOINT = 128


def load_clinical_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """加载 clinical.csv。"""
    import csv

    result = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_event = "event" in fieldnames
        for row in reader:
            pid = str(row["patient_id"]).strip()
            result[pid] = {
                "deadtime": float(row["deadtime"]),
                "youdao": int(row["youdao"]),
                "ajcc": int(row["ajcc"]),
                "age": float(row["age"]),
                "event": int(row["event"]) if has_event else 1,
            }
    return result


def build_ajcc_embedding(
    ajcc_values: List[int],
    d_emb: int = D_AJCC_EMB,
    seed: int = 42,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """构建 ajcc 分期 embedding 查找表。"""
    unique = sorted(set(ajcc_values))
    rng = np.random.default_rng(seed)
    emb = (rng.standard_normal((len(unique), d_emb)).astype(np.float32) * 0.1)
    lookup = {v: emb[i] for i, v in enumerate(unique)}
    return lookup, emb


def build_clinic_feat(
    ajcc: int,
    age: float,
    youdao: int,
    ajcc_lookup: Dict[int, np.ndarray],
) -> np.ndarray:
    """构建临床特征向量：concat([ajcc_emb, age, youdao])"""
    ajcc_emb = ajcc_lookup.get(ajcc, ajcc_lookup.get(3, np.zeros(D_AJCC_EMB)))
    return np.concatenate([
        np.asarray(ajcc_emb, dtype=np.float32),
        np.array([float(age)], dtype=np.float32),
        np.array([float(youdao)], dtype=np.float32),
    ])


def build_clinic_feat_causal(
    ajcc: int,
    age: float,
    ajcc_lookup: Dict[int, np.ndarray],
) -> np.ndarray:
    """构建因果模型用临床特征（不含治疗 youdao）。"""
    ajcc_emb = ajcc_lookup.get(ajcc, ajcc_lookup.get(3, np.zeros(D_AJCC_EMB)))
    return np.concatenate([
        np.asarray(ajcc_emb, dtype=np.float32),
        np.array([float(age)], dtype=np.float32),
    ])


def load_step4_feats(pkl_path: str) -> Dict[str, np.ndarray]:
    """加载第四步输出的患者级图像特征。"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + eps)


def project_img_feat_to_joint_components(
    img_feat: np.ndarray,
    d_joint: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    上游联合图模式：仅做患者级图像投影（不拼接临床）。
    """
    x = img_feat.astype(np.float32)
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_joint, x.shape[0])).astype(np.float32) * 0.02)
    b = np.zeros(d_joint, dtype=np.float32)
    z_raw = w @ x + b
    z = _layer_norm(z_raw)
    return {
        "img_input": x.astype(np.float32),
        "img_proj_raw": z_raw.astype(np.float32),
        "joint_feat": z.astype(np.float32),
    }


def init_concat_projection_params(
    d_img: int,
    d_clinic: int,
    d_joint: int = D_JOINT,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    初始化 concat 投影参数：
      concat([img_feat, clinic_feat]) -> Linear -> LayerNorm
    """
    rng = np.random.default_rng(seed)
    d_in = int(d_img + d_clinic)
    return {
        "W_concat": (rng.standard_normal((d_joint, d_in)).astype(np.float32) * 0.02),
        "b_concat": np.zeros(d_joint, dtype=np.float32),
    }


def fuse_modalities_concat_components(
    img_feat: np.ndarray,
    clinic_feat: np.ndarray,
    fusion_params: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    直接拼接融合：
      1) x_cat = concat([img_feat, clinic_feat])
      2) z_raw = W_concat @ x_cat + b_concat
      3) joint_feat = LayerNorm(z_raw)
    """
    x_img = img_feat.astype(np.float32)
    x_cli = clinic_feat.astype(np.float32)
    x_cat = np.concatenate([x_img, x_cli], axis=0).astype(np.float32)
    z_raw = fusion_params["W_concat"] @ x_cat + fusion_params["b_concat"]
    z = _layer_norm(z_raw).astype(np.float32)
    return {
        "img_input": x_img,
        "clinic_input": x_cli,
        "concat_input": x_cat,
        "joint_pre_norm": z_raw.astype(np.float32),
        "joint_feat": z,
    }


def run_step5_fusion(
    clinical_csv: str,
    step4_pkl: str,
    output_dir: str,
    patient_ids: Optional[List[str]] = None,
    d_joint: int = D_JOINT,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    step4_pkl_causal: Optional[str] = None,
    use_joint_graph_upstream: bool = False,
) -> Dict[str, Any]:
    """
    执行第五步（concat 消融）：
    读取临床 + 图像特征，直接拼接融合后划分 train/val/test。
    """
    clinical = load_clinical_csv(clinical_csv)
    img_feats = load_step4_feats(step4_pkl)
    img_feats_causal = None
    if step4_pkl_causal:
        img_feats_causal = load_step4_feats(step4_pkl_causal)

    if patient_ids is not None:
        clinical = {k: v for k, v in clinical.items() if k in patient_ids}
        img_feats = {k: v for k, v in img_feats.items() if k in patient_ids}
        if img_feats_causal is not None:
            img_feats_causal = {k: v for k, v in img_feats_causal.items() if k in patient_ids}

    common = sorted(set(clinical.keys()) & set(img_feats.keys()))
    if not common:
        raise ValueError("临床数据与图像特征无交集患者，请检查 patient_id 是否一致")

    ajcc_values = [clinical[pid]["ajcc"] for pid in common]
    ajcc_lookup, _ = build_ajcc_embedding(ajcc_values, seed=seed)
    d_img = int(img_feats[common[0]].shape[0])

    fusion_params = None
    fusion_params_causal = None
    if not use_joint_graph_upstream:
        fusion_params = init_concat_projection_params(
            d_img=d_img,
            d_clinic=D_CLINIC,
            d_joint=d_joint,
            seed=seed,
        )
        fusion_params_causal = init_concat_projection_params(
            d_img=d_img,
            d_clinic=D_AJCC_EMB + 1,
            d_joint=d_joint,
            seed=seed + 7,
        )

    aligned = []
    for pid in common:
        img_feat = img_feats[pid]
        has_dedicated_causal_img = bool(img_feats_causal is not None and pid in img_feats_causal)
        img_feat_causal = img_feats_causal[pid] if has_dedicated_causal_img else img_feat
        c = clinical[pid]
        clinic_feat = build_clinic_feat(c["ajcc"], c["age"], c["youdao"], ajcc_lookup)
        clinic_feat_causal = build_clinic_feat_causal(c["ajcc"], c["age"], ajcc_lookup)
        attribution_branch = None

        if use_joint_graph_upstream:
            proj_full = project_img_feat_to_joint_components(
                img_feat=img_feat,
                d_joint=d_joint,
                seed=seed,
            )
            causal_proj_seed = (seed + 13) if has_dedicated_causal_img else seed
            proj_causal = project_img_feat_to_joint_components(
                img_feat=img_feat_causal,
                d_joint=d_joint,
                seed=causal_proj_seed,
            )
            joint_feat = proj_full["joint_feat"]
            joint_feat_causal = proj_causal["joint_feat"]
            joint_graph = None
            joint_graph_causal = None
            attribution_branch = {
                "mode": "upstream_projection",
                "full": proj_full,
                "causal": proj_causal,
            }
        else:
            fusion_full = fuse_modalities_concat_components(
                img_feat=img_feat,
                clinic_feat=clinic_feat,
                fusion_params=fusion_params,
            )
            fusion_causal = fuse_modalities_concat_components(
                img_feat=img_feat_causal,
                clinic_feat=clinic_feat_causal,
                fusion_params=fusion_params_causal,
            )
            joint_feat = fusion_full["joint_feat"]
            joint_feat_causal = fusion_causal["joint_feat"]
            joint_graph = None
            joint_graph_causal = None
            attribution_branch = {
                "mode": "concat_linear",
                "full": fusion_full,
                "causal": fusion_causal,
            }

        aligned.append({
            "patient_id": pid,
            "joint_feat": joint_feat,
            "joint_feat_causal": joint_feat_causal,
            "joint_graph": joint_graph,
            "joint_graph_causal": joint_graph_causal,
            "clinic_feat_raw": clinic_feat,
            "clinic_feat_causal_raw": clinic_feat_causal,
            "img_feat_patient_raw": img_feat.astype(np.float32),
            "img_feat_patient_causal_raw": img_feat_causal.astype(np.float32),
            "attribution_branch": attribution_branch,
            "youdao": c["youdao"],
            "deadtime": c["deadtime"],
            "event": c["event"],
        })

    np.random.seed(seed)
    perm = np.random.permutation(len(common))
    n = len(perm)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_ids = [common[i] for i in train_idx]
    val_ids = [common[i] for i in val_idx]
    test_ids = [common[i] for i in test_idx]

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    data = {
        "aligned": aligned,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "ajcc_lookup": {k: v.tolist() for k, v in ajcc_lookup.items()},
        "d_joint": d_joint,
        "d_clinic": D_CLINIC,
        "d_img_patient": img_feats[common[0]].shape[0] if common else 0,
        "fusion_type": (
            "upstream_joint_graph_patient_projection"
            if use_joint_graph_upstream else "concat_linear"
        ),
        "use_joint_graph_upstream": bool(use_joint_graph_upstream),
        "joint_graph_schema": None,
        "attribution_branch": {
            "enabled": True,
            "version": "concat_v1",
            "note": "concat 消融：保存拼接输入与投影输出",
        },
    }

    out_path = output_dir_p / "step5_aligned_data.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"已保存到 {out_path}，共 {len(aligned)} 例，d_joint={d_joint}")

    meta = {
        "n_total": len(aligned),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "d_joint": d_joint,
        "d_clinic": D_CLINIC,
        "fusion_type": (
            "upstream_joint_graph_patient_projection"
            if use_joint_graph_upstream else "concat_linear"
        ),
        "use_joint_graph_upstream": bool(use_joint_graph_upstream),
        "enable_attribution_branch": True,
    }
    meta_path = output_dir_p / "step5_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    return data

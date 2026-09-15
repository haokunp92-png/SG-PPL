"""
第五步：临床数据读取与模态对齐

读取 clinical.csv，将图像特征与临床特征对齐融合，得到 joint_feat。
划分训练集、验证集、测试集（患者不跨集）。
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import numpy as np

# 避免导入 step4（step4 依赖 step3/torch），维度从数据推断


# 临床特征维度
D_AJCC_EMB = 4  # ajcc embedding 维度
D_CLINIC = D_AJCC_EMB + 1 + 1  # ajcc_emb + age + youdao
D_JOINT = 128  # 联合表征维度


def load_clinical_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 clinical.csv。
    Returns: {patient_id_str: {deadtime, youdao, ajcc, age, event?}}
    若 CSV 无 event 列，默认 event=1。
    """
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
    """
    构建 ajcc 分期的 embedding 查找表。
    ajcc 取值为 3 或 4，映射为 d_emb 维向量。
    使用可学习的随机初始化（实际训练时可由模型学习）。
    """
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
    """
    构建临床特征向量：clinic_feat = concat([ajcc_emb, age, youdao])
    """
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
    """
    构建因果模型用临床特征（不含治疗 youdao）。
    因果推断需用治疗前协变量预测潜在结局 Y(0)/Y(1)，治疗不应出现在输入中。
    clinic_feat_causal = concat([ajcc_emb, age])
    """
    ajcc_emb = ajcc_lookup.get(ajcc, ajcc_lookup.get(3, np.zeros(D_AJCC_EMB)))
    return np.concatenate([
        np.asarray(ajcc_emb, dtype=np.float32),
        np.array([float(age)], dtype=np.float32),
    ])


def load_step4_feats(pkl_path: str) -> Dict[str, np.ndarray]:
    """加载第四步输出的患者级图像特征"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def project_img_feat_to_joint(
    img_feat: np.ndarray,
    d_joint: int,
    seed: int,
) -> np.ndarray:
    """
    将患者级图像特征线性映射到 d_joint（不再重复注入临床）。
    用于“临床已在节点层注入”的联合图上游模式。
    """
    x = img_feat.astype(np.float32)
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_joint, x.shape[0])).astype(np.float32) * 0.02)
    b = np.zeros(d_joint, dtype=np.float32)
    z = w @ x + b
    z = (z - z.mean()) / (z.std() + 1e-6)
    return z.astype(np.float32)


def project_img_feat_to_joint_components(
    img_feat: np.ndarray,
    d_joint: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    返回患者级投影分解，供后续归因分支复用。
    """
    x = img_feat.astype(np.float32)
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_joint, x.shape[0])).astype(np.float32) * 0.02)
    b = np.zeros(d_joint, dtype=np.float32)
    z_raw = w @ x + b
    z = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)
    return {
        "img_input": x.astype(np.float32),
        "img_proj_raw": z_raw.astype(np.float32),
        "joint_feat": z.astype(np.float32),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + eps)


def init_interaction_fusion_params(
    d_img: int,
    d_clinic: int,
    d_joint: int = D_JOINT,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    初始化跨模态交互融合参数（全局一次，随后对所有患者复用）。
    这里不再用 concat，而是双路投影 + 条件门控 + 交叉项融合。
    """
    rng = np.random.default_rng(seed)
    return {
        "W_img": (rng.standard_normal((d_joint, d_img)).astype(np.float32) * 0.02),
        "b_img": np.zeros(d_joint, dtype=np.float32),
        "W_cli": (rng.standard_normal((d_joint, d_clinic)).astype(np.float32) * 0.02),
        "b_cli": np.zeros(d_joint, dtype=np.float32),
        "W_gate": (rng.standard_normal((d_joint, d_clinic)).astype(np.float32) * 0.02),
        "b_gate": np.zeros(d_joint, dtype=np.float32),
    }


def fuse_modalities_interaction(
    img_feat: np.ndarray,
    clinic_feat: np.ndarray,
    fusion_params: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    跨模态特征交互融合（非 concat）：
      1) 图像与临床分别投影到同一潜空间；
      2) 使用临床条件门控调制图像表示；
      3) 使用 Hadamard 交叉项显式建模交互；
      4) 残差相加并做 layer norm，得到 joint_feat。
    """
    comps = fuse_modalities_interaction_components(
        img_feat=img_feat,
        clinic_feat=clinic_feat,
        fusion_params=fusion_params,
    )
    return comps["joint_feat"]


def fuse_modalities_interaction_components(
    img_feat: np.ndarray,
    clinic_feat: np.ndarray,
    fusion_params: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    交互融合分解结果（用于后续归因），包含 joint_feat 及关键中间量。
    """
    x_img = img_feat.astype(np.float32)
    x_cli = clinic_feat.astype(np.float32)
    img_proj = fusion_params["W_img"] @ x_img + fusion_params["b_img"]
    cli_proj = fusion_params["W_cli"] @ x_cli + fusion_params["b_cli"]
    gate = _sigmoid(fusion_params["W_gate"] @ x_cli + fusion_params["b_gate"])
    img_cond = img_proj * gate
    cross = img_proj * cli_proj
    joint_pre_norm = img_cond + cli_proj + 0.5 * cross
    joint = _layer_norm(joint_pre_norm)
    return {
        "img_input": x_img.astype(np.float32),
        "clinic_input": x_cli.astype(np.float32),
        "img_proj": img_proj.astype(np.float32),
        "clinic_proj": cli_proj.astype(np.float32),
        "gate": gate.astype(np.float32),
        "img_cond": img_cond.astype(np.float32),
        "cross": cross.astype(np.float32),
        "joint_pre_norm": joint_pre_norm.astype(np.float32),
        "joint_feat": joint.astype(np.float32),
    }


def build_joint_multimodal_graph(
    img_feat: np.ndarray,
    clinic_feat: np.ndarray,
    fusion_params: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    构建患者级联合图（影像节点 + 临床节点 + 交互节点）。
    - 节点特征统一到 d_joint 维，便于后续图交互建模或解释；
    - 边采用星型连接：各临床节点与影像节点相连，并连接交互节点。
    """
    d_joint = fusion_params["W_img"].shape[0]
    x_img = img_feat.astype(np.float32)
    x_cli = clinic_feat.astype(np.float32)
    n_cli = int(x_cli.shape[0])

    img_node = fusion_params["W_img"] @ x_img + fusion_params["b_img"]
    gate = _sigmoid(fusion_params["W_gate"] @ x_cli + fusion_params["b_gate"])
    cli_proj = fusion_params["W_cli"] @ x_cli + fusion_params["b_cli"]
    cross_node = img_node * gate

    # 每个临床维度单独成节点：保留“哪个临床因子参与交互”的可解释性
    clinic_nodes = []
    for i in range(n_cli):
        one_hot_scalar = np.zeros(n_cli, dtype=np.float32)
        one_hot_scalar[i] = x_cli[i]
        node_i = fusion_params["W_cli"] @ one_hot_scalar + fusion_params["b_cli"]
        clinic_nodes.append(node_i.astype(np.float32))

    node_feats = [img_node.astype(np.float32)] + clinic_nodes + [cross_node.astype(np.float32), cli_proj.astype(np.float32)]
    node_types = ["image_global"] + [f"clinical_dim_{i}" for i in range(n_cli)] + ["cross_interaction", "clinical_global"]
    n_nodes = len(node_feats)

    # 无向图的双向边：影像节点(0)与各临床节点连接；交互节点与影像/临床全局连接
    edges: List[List[int]] = []
    cross_idx = n_nodes - 2
    cli_global_idx = n_nodes - 1
    for i in range(1, 1 + n_cli):
        edges.append([0, i])
        edges.append([i, 0])
    edges.extend([
        [0, cross_idx], [cross_idx, 0],
        [cross_idx, cli_global_idx], [cli_global_idx, cross_idx],
        [0, cli_global_idx], [cli_global_idx, 0],
    ])

    return {
        "node_features": np.stack(node_feats).astype(np.float32),  # (N, d_joint)
        "edge_index": np.asarray(edges, dtype=np.int64).T,         # (2, E)
        "node_types": node_types,
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
    执行第五步：临床数据读取与模态对齐，划分数据集。

    Returns:
        包含 aligned_data, train_ids, val_ids, test_ids, ajcc_lookup 等
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

    # 对齐：取临床与图像的交集
    common = sorted(set(clinical.keys()) & set(img_feats.keys()))
    if not common:
        raise ValueError("临床数据与图像特征无交集患者，请检查 patient_id 是否一致")

    ajcc_values = [clinical[pid]["ajcc"] for pid in common]
    ajcc_lookup, _ = build_ajcc_embedding(ajcc_values, seed=seed)
    d_img = int(img_feats[common[0]].shape[0])
    fusion_params = None
    fusion_params_causal = None
    if not use_joint_graph_upstream:
        fusion_params = init_interaction_fusion_params(
            d_img=d_img,
            d_clinic=D_CLINIC,
            d_joint=d_joint,
            seed=seed,
        )
        fusion_params_causal = init_interaction_fusion_params(
            d_img=d_img,
            d_clinic=D_AJCC_EMB + 1,
            d_joint=d_joint,
            seed=seed + 7,
        )

    aligned = []
    for pid in common:
        img_feat = img_feats[pid]
        has_dedicated_causal_img = bool(img_feats_causal is not None and pid in img_feats_causal)
        img_feat_causal = (
            img_feats_causal[pid] if has_dedicated_causal_img else img_feat
        )
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
            # 若因果图像特征回退为事实图像特征，复用同一投影参数，避免引入人为随机差异。
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
            fusion_full = fuse_modalities_interaction_components(
                img_feat=img_feat,
                clinic_feat=clinic_feat,
                fusion_params=fusion_params,
            )
            fusion_causal = fuse_modalities_interaction_components(
                img_feat=img_feat_causal,
                clinic_feat=clinic_feat_causal,
                fusion_params=fusion_params_causal,
            )
            joint_feat = fusion_full["joint_feat"]
            joint_feat_causal = fusion_causal["joint_feat"]
            joint_graph = build_joint_multimodal_graph(
                img_feat=img_feat, clinic_feat=clinic_feat, fusion_params=fusion_params
            )
            joint_graph_causal = build_joint_multimodal_graph(
                img_feat=img_feat_causal,
                clinic_feat=clinic_feat_causal,
                fusion_params=fusion_params_causal,
            )
            attribution_branch = {
                "mode": "interaction_gated_cross",
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

    # 划分 train/val/test
    np.random.seed(seed)
    perm = np.random.permutation(len(common))
    n = len(perm)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_ids = [common[i] for i in train_idx]
    val_ids = [common[i] for i in val_idx]
    test_ids = [common[i] for i in test_idx]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存
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
            if use_joint_graph_upstream else "interaction_gated_cross"
        ),
        "use_joint_graph_upstream": bool(use_joint_graph_upstream),
        "joint_graph_schema": {
            "nodes": ["image_global", "clinical_dim_*", "cross_interaction", "clinical_global"],
            "edges": "bidirectional star + cross links",
        },
        "attribution_branch": {
            "enabled": True,
            "version": "v1",
            "note": "预留归因分支，保存融合中间量与原始模态输入",
        },
    }
    out_path = output_dir / "step5_aligned_data.pkl"
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
            if use_joint_graph_upstream else "interaction_gated_cross"
        ),
        "use_joint_graph_upstream": bool(use_joint_graph_upstream),
        "enable_attribution_branch": True,
    }
    meta_path = output_dir / "step5_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    return data

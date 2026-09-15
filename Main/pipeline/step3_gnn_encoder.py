"""
第三步：GNN 编码器

对 step2 输出的切片图进行图神经网络编码，得到固定维度的切片级特征。
使用纯 PyTorch 实现（无 PyG）。
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .step2_feature_extractor import D_NODE_INIT

D_SLICE = 64


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """对 index 指定的目标位置做 mean 聚合。src: (E, F), index: (E,) -> (dim_size, F)"""
    E, F = src.shape
    out = torch.zeros(dim_size, F, dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, F, dtype=src.dtype, device=src.device)
    idx = index.unsqueeze(-1).expand(-1, F)
    out.scatter_add_(0, idx, src)
    count.scatter_add_(0, idx, torch.ones_like(src))
    count = count.clamp(min=1.0)
    return out / count


class SimpleGCNLayer(nn.Module):
    """单层 GCN：邻居特征均值聚合后与自身特征拼接，再线性变换。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim), edge_index: (2, E) [src, dst]
        对每个节点：聚合邻居特征均值，与自身拼接后线性变换。
        """
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)
        if edge_index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            agg = _scatter_mean(x[src], dst, N)
        combined = torch.cat([x, agg], dim=-1)
        return self.linear(combined)


class SliceGNNEncoder(nn.Module):
    """3 层 GCN + 全局 mean pool + 线性到 d_slice。"""

    def __init__(self, d_node_init: int = D_NODE_INIT, d_hidden: int = 64, d_slice: int = D_SLICE):
        super().__init__()
        self.d_slice = d_slice
        self.gcn1 = SimpleGCNLayer(d_node_init, d_hidden)
        self.gcn2 = SimpleGCNLayer(d_hidden, d_hidden)
        self.gcn3 = SimpleGCNLayer(d_hidden, d_hidden)
        self.pool_to_slice = nn.Linear(d_hidden, d_slice)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d_node_init), edge_index: (2, E)
        返回: (d_slice,) 切片级特征向量
        """
        h = torch.relu(self.gcn1(x, edge_index))
        h = torch.relu(self.gcn2(h, edge_index))
        h = torch.relu(self.gcn3(h, edge_index))
        h_pool = h.mean(dim=0)
        return self.pool_to_slice(h_pool)


def encode_slice(
    graph_data: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
    node_feat_key: str = "node_feat_init_slice",
    edge_key: str = "edge_index",
) -> np.ndarray:
    """
    对单张切片图进行编码。

    Args:
        graph_data: step2 输出的单切片图
        model: SliceGNNEncoder
        device: 计算设备
        node_feat_key: 节点特征字段名
        edge_key: 边字段名

    Returns:
        (d_slice,) 的 numpy 数组
    """
    x = torch.from_numpy(graph_data[node_feat_key]).float().to(device)
    edge_index = torch.from_numpy(graph_data[edge_key]).long().to(device)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)

    return out.cpu().numpy()


def load_step2_graphs(pkl_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    加载 step2 输出的图数据。

    Returns:
        {patient_id: {slice_id: graph_data}}
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_step3_encode(
    step2_pkl: str,
    output_dir: str,
    d_slice: int = 64,
    device: Optional[torch.device] = None,
    patient_ids: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    use_joint_graph: bool = False,
) -> None:
    """
    执行第三步：对所有切片进行 GNN 编码，保存切片特征。

    Args:
        step2_pkl: step2 输出 pkl 路径（含 step2_slice_graphs.pkl）
        output_dir: 输出目录
        d_slice: 切片特征维度
        device: 计算设备，None 则自动选择
        patient_ids: 可选，指定患者 ID 列表
        ckpt_path: 可选，GNN 预训练权重路径
        use_joint_graph: 若为 True，优先编码 step2 中的联合图字段
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_step2_graphs(step2_pkl)
    if patient_ids is not None:
        graphs = {k: v for k, v in graphs.items() if k in patient_ids}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SliceGNNEncoder(d_node_init=D_NODE_INIT, d_hidden=64, d_slice=d_slice).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"已加载 GNN 权重: {ckpt_path}")

    node_key = "node_feat_joint_slice" if use_joint_graph else "node_feat_init_slice"
    edge_key = "edge_index_joint_slice" if use_joint_graph else "edge_index"
    has_causal_joint = False
    if use_joint_graph:
        for slices in graphs.values():
            for g in slices.values():
                has_causal_joint = (
                    "node_feat_joint_slice_causal" in g and "edge_index_joint_slice_causal" in g
                )
                break
            if has_causal_joint:
                break

    result = {}
    result_causal = {} if (use_joint_graph and has_causal_joint) else None
    for pid, slices in graphs.items():
        result[pid] = {}
        if result_causal is not None:
            result_causal[pid] = {}
        for sid, g in slices.items():
            if node_key not in g or edge_key not in g:
                raise KeyError(
                    f"缺少联合图字段: {node_key}/{edge_key}。"
                    "请先使用 run_step2.py --joint_graph 重新生成 step2 输出。"
                )
            feat = encode_slice(g, model, device, node_feat_key=node_key, edge_key=edge_key)
            result[pid][sid] = feat
            if result_causal is not None:
                feat_c = encode_slice(
                    g,
                    model,
                    device,
                    node_feat_key="node_feat_joint_slice_causal",
                    edge_key="edge_index_joint_slice_causal",
                )
                result_causal[pid][sid] = feat_c

    feats_path = output_dir / "step3_slice_feats.pkl"
    encoder_ckpt = output_dir / "step3_slice_encoder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d_slice": int(d_slice),
            "d_node_init": int(D_NODE_INIT),
            "use_joint_graph": bool(use_joint_graph),
            "note": "与 step3_slice_feats.pkl 同步保存，供遮蔽扰动实验等离线重编码使用",
        },
        encoder_ckpt,
    )

    with open(feats_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_msg = (
        f"已保存到 {feats_path}，d_slice={d_slice}"
        f"；编码器权重: {encoder_ckpt.name}"
    )
    if use_joint_graph:
        save_msg += "（联合图编码）"
    print(save_msg)

    if result_causal is not None:
        feats_path_causal = output_dir / "step3_slice_feats_causal.pkl"
        with open(feats_path_causal, "wb") as f:
            pickle.dump(result_causal, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"已保存因果版切片特征到 {feats_path_causal}")

    meta = {
        pid: {
            "n_slices": len(slices),
            "slice_ids": list(slices.keys()),
            "d_slice": d_slice,
            "encoded_with_joint_graph": bool(use_joint_graph),
        }
        for pid, slices in result.items()
    }
    meta["global"] = {
        "d_slice": d_slice,
        "use_joint_graph": bool(use_joint_graph),
        "has_causal_joint_output": bool(result_causal is not None),
    }
    meta_path = output_dir / "step3_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已保存元信息到 {meta_path}")

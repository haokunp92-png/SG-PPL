"""
第一步：单张切片的图构建

对每个患者的每张病理切片，构建超像素图结构：
- 节点：每个超像素区域
- 边：空间邻接关系（相邻超像素之间建立连接）
- 节点属性：空间位置（质心）、像素值信息
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class SliceGraph:
    """单张切片对应的图数据结构"""
    patient_id: str
    slice_id: str
    num_nodes: int
    node_ids: np.ndarray           # 节点ID（0-based，对应超像素）
    edge_index: np.ndarray         # 边 (2, E)，PyG 格式 [src; dst]
    node_pos: np.ndarray           # 每个节点的质心坐标 (N, 2)，(y, x) 或 (row, col)
    node_pixel_feat: np.ndarray    # 每个节点的原始像素特征 (N, C)，C=3 为 mean RGB
    pixel_coords: Optional[Dict[int, np.ndarray]] = None  # 可选：每个超像素的像素坐标 (n_pixels, 2)
    spixel_map: Optional[np.ndarray] = None  # 可选：原始超像素映射 H×W

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "slice_id": self.slice_id,
            "num_nodes": self.num_nodes,
            "node_ids": self.node_ids,
            "edge_index": self.edge_index,
            "node_pos": self.node_pos,
            "node_pixel_feat": self.node_pixel_feat,
            "pixel_coords": self.pixel_coords,
        }


def load_spixel_map(csv_path: str) -> np.ndarray:
    """
    加载超像素映射 CSV 文件。
    格式：每行对应图像一行，逗号分隔，值为超像素 ID（1-based）。
    返回：H×W 的整数数组。
    """
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.int32)
    if data.ndim == 1:
        # 单行情况，reshape 为 (1, W)
        data = data.reshape(1, -1)
    return data


def build_adjacency_from_map(spixel_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据超像素映射构建邻接关系（4-连通：上下左右）。
    返回：(edge_index (2, E), unique_spixel_ids)
    """
    H, W = spixel_map.shape
    edges = set()

    # 4-连通：右、下
    for r in range(H):
        for c in range(W):
            sp_id = spixel_map[r, c]
            # 右邻居
            if c + 1 < W:
                sp_right = spixel_map[r, c + 1]
                if sp_id != sp_right:
                    edges.add((min(sp_id, sp_right), max(sp_id, sp_right)))
            # 下邻居
            if r + 1 < H:
                sp_down = spixel_map[r + 1, c]
                if sp_id != sp_down:
                    edges.add((min(sp_id, sp_down), max(sp_id, sp_down)))

    # 转为 edge_index 格式 (2, E)，双向边
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.unique(spixel_map)

    edges = list(edges)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = np.array([src, dst], dtype=np.int64)

    unique_ids = np.unique(spixel_map)
    return edge_index, unique_ids


def compute_node_features(
    spixel_map: np.ndarray,
    unique_ids: np.ndarray,
    image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, np.ndarray]]]:
    """
    计算每个超像素节点的空间位置和像素特征。
    - node_pos: (N, 2) 质心坐标 (y, x)，归一化到 [0, 1]
    - node_pixel_feat: (N, C) 若 image 存在则为 mean RGB，否则为 0
    - pixel_coords: 每个超像素的像素坐标 (n_pixels, 2)，用于后续特征提取
    """
    H, W = spixel_map.shape
    n_nodes = len(unique_ids)
    id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}

    centroids = np.zeros((n_nodes, 2), dtype=np.float32)
    pixel_feats = np.zeros((n_nodes, 3), dtype=np.float32)  # RGB
    pixel_coords_dict = {}

    for sp_id in unique_ids:
        mask = spixel_map == sp_id
        rows, cols = np.where(mask)
        n_pix = len(rows)
        if n_pix == 0:
            continue
        idx = id_to_idx[sp_id]
        cy, cx = np.mean(rows), np.mean(cols)
        centroids[idx] = [cy, cx]
        pixel_coords_dict[sp_id] = np.column_stack([rows, cols])

        if image is not None and image.size > 0:
            ih, iw = int(image.shape[0]), int(image.shape[1])
            valid = (rows >= 0) & (rows < ih) & (cols >= 0) & (cols < iw)
            r_v, c_v = rows[valid], cols[valid]
            if r_v.size == 0:
                pixel_coords_dict[sp_id] = np.empty((0, 2), dtype=np.int64)
                continue
            if image.ndim == 2:
                vals = image[r_v, c_v]
                pixel_feats[idx] = [np.mean(vals), np.mean(vals), np.mean(vals)]
            else:
                pixel_feats[idx] = [
                    np.mean(image[r_v, c_v, 0]),
                    np.mean(image[r_v, c_v, 1]),
                    np.mean(image[r_v, c_v, 2]),
                ]
            pixel_coords_dict[sp_id] = np.column_stack([r_v, c_v])

    # 归一化质心到 [0, 1]
    centroids[:, 0] /= max(H - 1, 1)
    centroids[:, 1] /= max(W - 1, 1)

    return centroids, pixel_feats, pixel_coords_dict


def remap_node_indices(
    edge_index: np.ndarray,
    unique_ids: np.ndarray,
) -> np.ndarray:
    """
    将超像素 ID（可能非连续、1-based）重映射为 0-based 连续节点索引。
    """
    id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
    n = len(unique_ids)
    new_edge = np.zeros_like(edge_index)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u in id_to_idx and v in id_to_idx:
            new_edge[0, i] = id_to_idx[u]
            new_edge[1, i] = id_to_idx[v]
    return new_edge


def build_slice_graph(
    map_csv_path: str,
    patient_id: str,
    slice_id: str,
    image_path: Optional[str] = None,
    store_pixel_coords: bool = True,
) -> SliceGraph:
    """
    对单张切片构建图结构。

    Args:
        map_csv_path: 超像素映射 CSV 路径
        patient_id: 患者编号
        slice_id: 切片编号（如 "025"）
        image_path: 可选，原始病理图像路径，用于提取像素特征
        store_pixel_coords: 是否存储每个超像素的像素坐标（供第二步特征提取使用）

    Returns:
        SliceGraph 实例
    """
    spixel_map = load_spixel_map(map_csv_path)
    edge_index, unique_ids = build_adjacency_from_map(spixel_map)

    image = None
    if image_path and Path(image_path).exists():
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass  # 无 cv2 时跳过图像加载，node_pixel_feat 将为 0

    node_pos, node_pixel_feat, pixel_coords_dict = compute_node_features(
        spixel_map, unique_ids, image
    )

    # 重映射边索引为 0-based
    edge_index_remap = remap_node_indices(edge_index, unique_ids)

    # 过滤无效边（若存在未在 unique_ids 中的 ID）
    valid = (edge_index_remap[0] >= 0) & (edge_index_remap[0] < len(unique_ids)) & \
            (edge_index_remap[1] >= 0) & (edge_index_remap[1] < len(unique_ids))
    edge_index_remap = edge_index_remap[:, valid]

    pixel_coords = None
    if store_pixel_coords and pixel_coords_dict:
        # 用 0-based 索引作为 key
        pixel_coords = {i: pixel_coords_dict[sid] for i, sid in enumerate(unique_ids)}

    return SliceGraph(
        patient_id=patient_id,
        slice_id=slice_id,
        num_nodes=len(unique_ids),
        node_ids=np.arange(len(unique_ids)),
        edge_index=edge_index_remap,
        node_pos=node_pos,
        node_pixel_feat=node_pixel_feat,
        pixel_coords=pixel_coords,
        spixel_map=spixel_map,
    )


def collect_slice_graphs(
    dataset_root: str,
    image_dir: Optional[str] = None,
    patient_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, SliceGraph]]:
    """
    收集所有患者的切片图。

    Args:
        dataset_root: dataset 根目录，结构为 dataset/{patient_id}/map_csv/{patient_id}_slice_{NNN}.csv
        image_dir: 可选，原始图像根目录，结构为 image_dir/{patient_id}/slice_{NNN}.png
        patient_ids: 可选，指定患者 ID 列表；若为 None 则扫描 dataset_root 下所有患者

    Returns:
        {patient_id: {slice_id: SliceGraph}}
    """
    dataset_root = Path(dataset_root)
    result = {}

    if patient_ids is None:
        patient_ids = [d.name for d in dataset_root.iterdir() if d.is_dir()]

    for pid in patient_ids:
        map_dir = dataset_root / pid / "map_csv"
        if not map_dir.exists():
            continue
        result[pid] = {}
        for csv_path in sorted(map_dir.glob("*.csv")):
            # 解析切片 ID，如 10105856_slice_025.csv -> 025
            name = csv_path.stem
            if "_slice_" in name:
                slice_id = name.split("_slice_")[-1]
            else:
                slice_id = name
            img_path = None
            if image_dir:
                img_path = str(Path(image_dir) / pid / f"slice_{slice_id}.png")
            g = build_slice_graph(
                str(csv_path),
                patient_id=pid,
                slice_id=slice_id,
                image_path=img_path,
                store_pixel_coords=True,
            )
            result[pid][slice_id] = g

    return result

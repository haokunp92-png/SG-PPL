"""
第二步：单张切片的节点特征提取

对每张切片的超像素图进行初始节点特征提取。
特征包括：一阶统计量（均值、方差、偏度）、纹理特征、形状特征、质心坐标。
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from scipy import stats

try:
    from skimage.feature import graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# 特征维度，供后续步骤使用
D_NODE_INIT = 20  # 一阶9 + 纹理4 + 形状5 + 空间2
D_CLINICAL_RAW = 3  # ajcc, age, youdao


def _safe_skew(x: np.ndarray) -> float:
    """安全计算偏度，避免空数组或零方差"""
    x = np.asarray(x).flatten()
    if len(x) < 2 or np.std(x) < 1e-8:
        return 0.0
    return float(stats.skew(x))


def extract_first_order(pixels: np.ndarray) -> np.ndarray:
    """
    一阶统计量：均值、方差、偏度。
    pixels: (N, C) 或 (N,) 像素值
    返回: (9,) - 每通道 mean, std, skew，若灰度则复制3份
    """
    if pixels.ndim == 1:
        pixels = pixels.reshape(-1, 1)
    n_ch = min(3, pixels.shape[1])
    feats = []
    for c in range(n_ch):
        ch = pixels[:, c].astype(np.float64)
        feats.extend([
            np.mean(ch),
            np.std(ch) if np.std(ch) > 1e-8 else 0.0,
            _safe_skew(ch),
        ])
    # 补齐到 9 维
    while len(feats) < 9:
        feats.append(0.0)
    return np.array(feats[:9], dtype=np.float32)


def extract_texture(gray: np.ndarray) -> np.ndarray:
    """
    纹理特征：GLCM 的 contrast, correlation, energy, homogeneity。
    gray: 灰度像素值数组，需离散化到 0-255
    返回: (4,)
    """
    if not _HAS_SKIMAGE or len(gray) < 4:
        return np.zeros(4, dtype=np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    n = int(np.sqrt(len(gray_u8)))
    n = max(2, min(n, 64))
    patch = gray_u8[: n * n].reshape(n, n)
    try:
        glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]
        energy = graycoprops(glcm, "energy")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
        if np.isnan(correlation):
            correlation = 0.0
        return np.array([contrast, correlation, energy, homogeneity], dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


def extract_shape(coords: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    形状特征（完整版，依赖 scipy.ndimage）：面积、周长比、紧致度、长宽比、圆度。
    """
    n = len(coords)
    if n < 2:
        return np.zeros(5, dtype=np.float32)
    try:
        from scipy.ndimage import binary_erosion
        mask = np.zeros((img_h, img_w), dtype=bool)
        rows = np.clip(coords[:, 0].astype(int), 0, img_h - 1)
        cols = np.clip(coords[:, 1].astype(int), 0, img_w - 1)
        mask[rows, cols] = True
        eroded = binary_erosion(mask)
        perimeter = np.sum(mask & ~eroded)
    except Exception:
        return extract_shape_fast(coords, img_h, img_w)
    area = n / (img_h * img_w) if img_h * img_w > 0 else 0.0
    perimeter_ratio = perimeter / (2 * np.sqrt(np.pi * n) + 1e-8) if n > 0 else 0.0
    compactness = (4 * np.pi * n) / (perimeter ** 2 + 1e-8) if perimeter > 0 else 0.0
    h_bb, w_bb = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    aspect = max(h_bb, w_bb) / (min(h_bb, w_bb) + 1e-8)
    circularity = 4 * np.pi * n / (perimeter ** 2 + 1e-8) if perimeter > 0 else 0.0
    return np.array([area, perimeter_ratio, compactness, aspect, circularity], dtype=np.float32)


def extract_shape_fast(coords: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    形状特征（快速版，不依赖 scipy.ndimage）：面积、bbox 比例、紧致度近似。
    """
    n = len(coords)
    if n < 2:
        return np.zeros(5, dtype=np.float32)
    area = n / (img_h * img_w) if img_h * img_w > 0 else 0.0
    rows, cols = coords[:, 0], coords[:, 1]
    r_min, r_max, c_min, c_max = rows.min(), rows.max(), cols.min(), cols.max()
    h_bb = r_max - r_min + 1
    w_bb = c_max - c_min + 1
    bbox_area = h_bb * w_bb
    fill_ratio = n / (bbox_area + 1e-8)
    aspect = max(h_bb, w_bb) / (min(h_bb, w_bb) + 1e-8)
    # 周长近似：凸包或边界点数
    perimeter_approx = 2 * (h_bb + w_bb)
    compactness = (4 * np.pi * n) / (perimeter_approx ** 2 + 1e-8)
    circularity = compactness
    return np.array([area, fill_ratio, compactness, aspect, circularity], dtype=np.float32)


def extract_node_features(
    graph_data: Dict[str, Any],
    image: Optional[np.ndarray],
    image_dir: Optional[str],
    patient_id: str,
    slice_id: str,
    use_fast_shape: bool = True,
) -> np.ndarray:
    """
    对单张切片的所有节点提取 node_feat_init_slice。

    Args:
        graph_data: 第一步输出的单张切片图数据
        image: 原始图像 (H, W, 3) 或 (H, W)，None 则尝试从 image_dir 加载
        image_dir: 图像根目录
        patient_id: 患者编号
        slice_id: 切片编号
        use_fast_shape: 是否使用快速形状特征（不依赖 scipy.ndimage）

    Returns:
        node_feat_init_slice: (N, d_node_init)
    """
    node_pos = graph_data["node_pos"]
    pixel_coords = graph_data.get("pixel_coords")
    node_pixel_feat = graph_data.get("node_pixel_feat")

    n_nodes = len(node_pos)
    if pixel_coords is None or n_nodes == 0:
        # 无像素坐标时，仅用 node_pos 和 node_pixel_feat 拼接
        feat = np.zeros((n_nodes, D_NODE_INIT), dtype=np.float32)
        feat[:, -2:] = node_pos
        if node_pixel_feat is not None and node_pixel_feat.shape[1] >= 3:
            feat[:, :9] = np.tile(
                extract_first_order(node_pixel_feat.mean(axis=0)), (n_nodes, 1)
            )
        return feat

    if image is None and image_dir:
        try:
            import cv2

            from .path_utils import resolve_slide_image

            resolved = resolve_slide_image(image_dir, patient_id, slice_id)
            if resolved:
                img = cv2.imread(resolved)
                if img is not None:
                    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass

    img_h, img_w = 1, 1
    if image is not None:
        img_h, img_w = image.shape[0], image.shape[1]
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

    shape_fn = extract_shape_fast if use_fast_shape else extract_shape
    feats_list = []
    for i in range(n_nodes):
        coords = pixel_coords.get(i)
        if coords is None or len(coords) == 0:
            feats_list.append(np.zeros(D_NODE_INIT, dtype=np.float32))
            continue
        rows = coords[:, 0].astype(int)
        cols = coords[:, 1].astype(int)
        rows = np.clip(rows, 0, img_h - 1)
        cols = np.clip(cols, 0, img_w - 1)

        first_order = np.zeros(9, dtype=np.float32)
        texture = np.zeros(4, dtype=np.float32)
        if image is not None and image.size > 0:
            pixels = image[rows, cols]  # (n, 3)
            first_order = extract_first_order(pixels)
            gray = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
            texture = extract_texture(gray)
        elif node_pixel_feat is not None and i < node_pixel_feat.shape[0]:
            first_order = extract_first_order(node_pixel_feat[i : i + 1].flatten())

        shape_feat = shape_fn(coords, img_h, img_w)
        spatial = node_pos[i]
        feats_list.append(np.concatenate([first_order, texture, shape_feat, spatial]))

    return np.stack(feats_list, axis=0).astype(np.float32)


def load_clinical_csv(csv_path: str) -> Dict[str, Dict[str, float]]:
    """
    轻量读取 clinical.csv（仅 step2 需要字段）。
    Returns: {patient_id: {"ajcc": float, "age": float, "youdao": float}}
    """
    import csv

    clinical: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["patient_id"]).strip()
            clinical[pid] = {
                "ajcc": float(row["ajcc"]),
                "age": float(row["age"]),
                "youdao": float(row["youdao"]),
            }
    return clinical


def _build_clinical_vector(
    clinical_row: Dict[str, float],
    include_treatment: bool = True,
) -> np.ndarray:
    """
    构建临床原始向量并做基础归一化：
    - ajcc: 约束到 [0, 1]（按 4 归一）
    - age: 约束到 [0, 1]（按 100 归一）
    - youdao: 0/1
    """
    ajcc = float(clinical_row.get("ajcc", 0.0)) / 4.0
    age = float(clinical_row.get("age", 0.0)) / 100.0
    youdao = float(clinical_row.get("youdao", 0.0))
    if include_treatment:
        return np.array([ajcc, age, youdao], dtype=np.float32)
    return np.array([ajcc, age], dtype=np.float32)


def _init_clinical_injector(
    d_clinical: int,
    d_node: int = D_NODE_INIT,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    初始化临床->节点的调制参数（固定参数，所有患者共享，保证可复现）。
    """
    rng = np.random.default_rng(seed)
    return {
        "W_gate": (rng.standard_normal((d_node, d_clinical)).astype(np.float32) * 0.08),
        "b_gate": np.zeros(d_node, dtype=np.float32),
        "W_shift": (rng.standard_normal((d_node, d_clinical)).astype(np.float32) * 0.05),
        "b_shift": np.zeros(d_node, dtype=np.float32),
        "W_node": (rng.standard_normal((d_node, d_clinical)).astype(np.float32) * 0.06),
        "b_node": np.zeros(d_node, dtype=np.float32),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def inject_clinical_joint_graph(
    graph_data: Dict[str, Any],
    node_feat_init_slice: np.ndarray,
    clinical_row: Dict[str, float],
    injector: Dict[str, np.ndarray],
    include_treatment: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将临床信息直接作用到每个超像素节点，并构建联合图：
    1) 对每个节点进行 FiLM 式调制（临床条件门控 + 平移）
    2) 新增一个临床全局节点
    3) 临床节点与每个超像素节点双向连接

    Returns:
      node_feat_joint_slice: (N+1, d_node)
      edge_index_joint: (2, E_joint)
    """
    edge_index = np.asarray(graph_data["edge_index"], dtype=np.int64)
    x = node_feat_init_slice.astype(np.float32)
    n_nodes, d_node = x.shape

    c = _build_clinical_vector(clinical_row, include_treatment=include_treatment)
    gate = _sigmoid(injector["W_gate"] @ c + injector["b_gate"])            # (d_node,)
    shift = injector["W_shift"] @ c + injector["b_shift"]                    # (d_node,)
    x_mod = x * (1.0 + gate[None, :]) + shift[None, :]                       # (N, d_node)

    c_node = injector["W_node"] @ c + injector["b_node"]                     # (d_node,)
    c_node = c_node.astype(np.float32)[None, :]                              # (1, d_node)

    node_joint = np.vstack([x_mod, c_node]).astype(np.float32)               # (N+1, d_node)
    c_idx = n_nodes

    # 将临床节点连到每个超像素节点（双向）
    src_add = np.concatenate([np.full(n_nodes, c_idx, dtype=np.int64), np.arange(n_nodes, dtype=np.int64)])
    dst_add = np.concatenate([np.arange(n_nodes, dtype=np.int64), np.full(n_nodes, c_idx, dtype=np.int64)])
    edge_add = np.stack([src_add, dst_add], axis=0)

    edge_joint = np.concatenate([edge_index, edge_add], axis=1) if edge_index.size > 0 else edge_add
    return node_joint, edge_joint

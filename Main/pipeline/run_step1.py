"""
第一步：单张切片的图构建 - 运行脚本

扫描 dataset 目录下所有患者的超像素映射，为每张切片构建图结构，
并保存到 output 目录供后续步骤使用。
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np

from .graph_builder import collect_slice_graphs, SliceGraph


def save_graphs(
    graphs: dict,
    output_dir: str,
    format: str = "pickle",
) -> None:
    """
    保存图数据。
    format: "pickle" 或 "npz"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为 pickle，便于后续步骤直接加载
    save_path = output_dir / "step1_slice_graphs.pkl"
    # 转为可序列化格式（去掉 spixel_map 以节省空间）
    to_save = {}
    for pid, slices in graphs.items():
        to_save[pid] = {}
        for sid, g in slices.items():
            to_save[pid][sid] = {
                "patient_id": g.patient_id,
                "slice_id": g.slice_id,
                "num_nodes": g.num_nodes,
                "node_ids": g.node_ids,
                "edge_index": g.edge_index,
                "node_pos": g.node_pos,
                "node_pixel_feat": g.node_pixel_feat,
                "pixel_coords": g.pixel_coords,
            }
    with open(save_path, "wb") as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"已保存 {len(graphs)} 个患者的切片图到 {save_path}")

    # 保存元信息（每个患者有多少切片等）
    meta = {
        pid: {
            "n_slices": len(slices),
            "slice_ids": list(slices.keys()),
            "total_nodes": sum(g.num_nodes for g in slices.values()),
        }
        for pid, slices in graphs.items()
    }
    with open(output_dir / "step1_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_graphs(pkl_path: str) -> dict:
    """加载第一步保存的图数据"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="第一步：单张切片的图构建")
    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="超像素 map_csv 根目录",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="可选：原始病理图像根目录，用于提取像素特征",
    )
    parser.add_argument(
        "--output",
        default="./output/step1",
        help="输出目录",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="可选：指定患者 ID 列表，不指定则处理全部",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"错误：dataset 目录不存在 {dataset_root}")
        return

    print("正在构建切片图...")
    graphs = collect_slice_graphs(
        dataset_root=str(dataset_root),
        image_dir=args.image_dir,
        patient_ids=args.patients,
    )

    if not graphs:
        print("未找到任何切片数据")
        return

    total_slices = sum(len(s) for s in graphs.values())
    total_nodes = sum(g.num_nodes for slices in graphs.values() for g in slices.values())
    print(f"共处理 {len(graphs)} 个患者，{total_slices} 张切片，{total_nodes} 个超像素节点")

    save_graphs(graphs, args.output)
    print("第一步完成。")


if __name__ == "__main__":
    main()

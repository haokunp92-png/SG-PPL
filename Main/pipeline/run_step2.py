"""
第二步：单张切片的节点特征提取 - 运行脚本

加载第一步输出的切片图，对每个节点提取影像组学特征（一阶、纹理、形状、空间），
输出带 node_feat_init_slice 的图数据。
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, List

from .run_step1 import load_graphs
from .step2_feature_extractor import (
    extract_node_features,
    load_clinical_csv,
    inject_clinical_joint_graph,
    _init_clinical_injector,
    D_NODE_INIT,
)


def run_step2(
    step1_pkl: str,
    output_dir: str,
    image_dir: Optional[str] = None,
    patient_ids: Optional[List[str]] = None,
    clinical_csv: Optional[str] = None,
    use_joint_graph: bool = False,
) -> None:
    """
    执行第二步：为所有切片添加 node_feat_init_slice。
    """
    graphs = load_graphs(step1_pkl)
    if patient_ids:
        graphs = {k: v for k, v in graphs.items() if k in patient_ids}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical = None
    injector_full = None
    injector_causal = None
    if use_joint_graph:
        if not clinical_csv:
            raise ValueError("use_joint_graph=True 时必须提供 clinical_csv")
        clinical = load_clinical_csv(clinical_csv)
        injector_full = _init_clinical_injector(d_clinical=3, d_node=D_NODE_INIT, seed=42)
        injector_causal = _init_clinical_injector(d_clinical=2, d_node=D_NODE_INIT, seed=49)

    result = {}
    for pid, slices in graphs.items():
        result[pid] = {}
        c_row = clinical.get(pid) if clinical is not None else None
        for sid, g in slices.items():
            feat = extract_node_features(
                graph_data=g,
                image=None,
                image_dir=image_dir,
                patient_id=pid,
                slice_id=sid,
                use_fast_shape=True,
            )
            row = {
                **g,
                "node_feat_init_slice": feat,
            }
            if use_joint_graph and c_row is not None:
                node_joint, edge_joint = inject_clinical_joint_graph(
                    graph_data=g,
                    node_feat_init_slice=feat,
                    clinical_row=c_row,
                    injector=injector_full,
                    include_treatment=True,
                )
                node_joint_c, edge_joint_c = inject_clinical_joint_graph(
                    graph_data=g,
                    node_feat_init_slice=feat,
                    clinical_row=c_row,
                    injector=injector_causal,
                    include_treatment=False,
                )
                row["node_feat_joint_slice"] = node_joint
                row["edge_index_joint_slice"] = edge_joint
                row["node_feat_joint_slice_causal"] = node_joint_c
                row["edge_index_joint_slice_causal"] = edge_joint_c
            result[pid][sid] = row

    save_path = output_dir / "step2_slice_graphs.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    msg = f"已保存到 {save_path}，node_feat 维度 d_node_init={D_NODE_INIT}"
    if use_joint_graph:
        msg += "，已构建临床-超像素联合图"
    print(msg)

    meta = {
        pid: {
            "n_slices": len(slices),
            "slice_ids": list(slices.keys()),
            "d_node_init": D_NODE_INIT,
            "has_joint_graph": bool(use_joint_graph),
        }
        for pid, slices in result.items()
    }
    meta["global"] = {
        "d_node_init": D_NODE_INIT,
        "use_joint_graph": bool(use_joint_graph),
        "joint_node_feature_keys": (
            ["node_feat_joint_slice", "node_feat_joint_slice_causal"] if use_joint_graph else []
        ),
    }
    with open(output_dir / "step2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="第二步：单张切片的节点特征提取")
    parser.add_argument(
        "--step1_output",
        default="./output/step1",
        help="第一步输出目录（含 step1_slice_graphs.pkl）",
    ) 
    parser.add_argument(
        "--output",
        default="./output/step2",
        help="第二步输出目录",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="可选：原始病理图像根目录，用于提取像素/纹理特征",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="可选：指定患者 ID 列表",
    )
    parser.add_argument(
        "--clinical",
        default=None,
        help="可选：clinical.csv 路径（联合图模式必填）",
    )
    parser.add_argument(
        "--joint_graph",
        action="store_true",
        help="启用临床-超像素联合图：将临床信息直接注入每个超像素节点",
    )
    args = parser.parse_args()

    pkl_path = Path(args.step1_output) / "step1_slice_graphs.pkl"
    if not pkl_path.exists():
        print(f"错误：未找到 {pkl_path}，请先运行第一步")
        return

    print("正在提取节点特征...")
    run_step2(
        step1_pkl=str(pkl_path),
        output_dir=args.output,
        image_dir=args.image_dir,
        patient_ids=args.patients,
        clinical_csv=args.clinical,
        use_joint_graph=bool(args.joint_graph),
    )
    print("第二步完成。")


if __name__ == "__main__":
    main()

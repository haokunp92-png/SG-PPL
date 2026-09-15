"""
第三步：GNN 编码 - 运行脚本

加载 step2 输出的切片图，使用 GNN 编码得到固定维度的切片特征。
"""

import argparse
from pathlib import Path
from typing import Optional, List

from .step3_gnn_encoder import run_step3_encode


def main():
    parser = argparse.ArgumentParser(description="第三步：GNN 编码切片图")
    parser.add_argument(
        "--step2_output",
        default="./output/step2",
        help="第二步输出目录（含 step2_slice_graphs.pkl）",
    )
    parser.add_argument(
        "--output",
        default="./output/step3",
        help="第三步输出目录",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="可选：指定患者 ID 列表",
    )
    parser.add_argument(
        "--d_slice",
        type=int,
        default=64,
        help="slice_feat 维度",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="可选：计算设备，如 cuda 或 cpu",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="可选：GNN 预训练权重路径",
    )
    parser.add_argument(
        "--joint_graph",
        action="store_true",
        help="启用联合图编码：使用 step2 中临床注入后的节点与边字段",
    )
    args = parser.parse_args()

    pkl_path = Path(args.step2_output) / "step2_slice_graphs.pkl"
    if not pkl_path.exists():
        print(f"错误：未找到 {pkl_path}，请先运行第二步")
        return

    device = None
    if args.device is not None:
        import torch
        device = torch.device(args.device)

    print("正在 GNN 编码...")
    run_step3_encode(
        step2_pkl=str(pkl_path),
        output_dir=args.output,
        d_slice=args.d_slice,
        device=device,
        patient_ids=args.patients,
        ckpt_path=args.ckpt,
        use_joint_graph=bool(args.joint_graph),
    )
    print("第三步完成。")


if __name__ == "__main__":
    main()

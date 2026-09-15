"""
第四步：多切片融合为患者级图像特征 - 运行脚本

加载第三步输出的切片特征，对每个患者做平均池化得到 img_feat_patient。
"""

import argparse
from pathlib import Path
from typing import Optional, List

from .step4_patient_fusion import run_step4_fusion
from .step3_gnn_encoder import D_SLICE


def main():
    parser = argparse.ArgumentParser(description="第四步：多切片融合为患者级图像特征")
    parser.add_argument(
        "--step3_output",
        default="./output/step3",
        help="第三步输出目录（含 step3_slice_feats.pkl）",
    )
    parser.add_argument(
        "--output",
        default="./output/step4",
        help="第四步输出目录",
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
        default=D_SLICE,
        help=f"切片特征维度，需与第三步一致，默认 {D_SLICE}",
    )
    parser.add_argument(
        "--joint_graph",
        action="store_true",
        help="启用联合图模式：同时读取 step3_slice_feats_causal.pkl 并输出因果版患者特征",
    )
    args = parser.parse_args()

    pkl_path = Path(args.step3_output) / "step3_slice_feats.pkl"
    if not pkl_path.exists():
        print(f"错误：未找到 {pkl_path}，请先运行第三步")
        return

    print("正在融合多切片...")
    step3_causal_pkl = None
    if args.joint_graph:
        pkl_path_causal = Path(args.step3_output) / "step3_slice_feats_causal.pkl"
        if not pkl_path_causal.exists():
            print(f"错误：未找到 {pkl_path_causal}，请先用联合图模式运行第三步")
            return
        step3_causal_pkl = str(pkl_path_causal)
    run_step4_fusion(
        step3_pkl=str(pkl_path),
        output_dir=args.output,
        patient_ids=args.patients,
        d_slice=args.d_slice,
        step3_pkl_causal=step3_causal_pkl,
    )
    print("第四步完成。")


if __name__ == "__main__":
    main()

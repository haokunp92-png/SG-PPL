"""
第六步（单头消融）：因果生存模型构建与反事实推理 - 运行脚本

训练单头因果生存模型，强制通过 treatment 翻转得到反事实风险，
并计算个体治疗效应 ITE 及 bootstrap。
"""

import argparse
from pathlib import Path

from step6_causal_model import run_step6_single_head


def main():
    parser = argparse.ArgumentParser(description="第六步（单头消融）：因果生存模型构建与反事实推理")
    parser.add_argument(
        "--step5_output",
        default="./output/step5",
        help="第五步输出目录（含 step5_aligned_data.pkl）",
    )
    parser.add_argument(
        "--output",
        default="./output/step6_single_head",
        help="第六步输出目录（单头）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率",
    )
    parser.add_argument(
        "--head_hidden",
        type=int,
        default=32,
        help="head 中间层维度（>0 启用 2 层 head；0 为单层）",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="AdamW 权重衰减",
    )
    parser.add_argument(
        "--balance_reg",
        type=float,
        default=0.0,
        help="平衡正则化权重（youdao 组间编码分布相似）",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=15,
        help="学习率衰减 patience",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="学习率衰减因子",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=200,
        help="bootstrap 次数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--inference_all",
        action="store_true",
        help="对全部患者（train+val+test）进行反事实推理；默认仅对验证集（无验证集则测试集）",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        metavar="N",
        help="仅使用 step5 aligned 中按顺序前 N 个不同患者（与 train/val/test 求交）",
    )
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        default="c_index",
        choices=["c_index", "loss"],
        help="验证集选优依据：c_index 或 loss",
    )
    args = parser.parse_args()

    pkl_path = Path(args.step5_output) / "step5_aligned_data.pkl"
    if not pkl_path.exists():
        print(f"错误：未找到 {pkl_path}，请先运行第五步")
        return

    print("正在训练单头因果模型（S-learner 消融）...")
    run_step6_single_head(
        step5_pkl=str(pkl_path),
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        balance_reg=args.balance_reg,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        inference_all=args.inference_all,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        head_hidden=args.head_hidden,
        checkpoint_metric=args.checkpoint_metric,
        max_patients=args.max_patients,
    )
    print("第六步（单头消融）完成。")


if __name__ == "__main__":
    main()

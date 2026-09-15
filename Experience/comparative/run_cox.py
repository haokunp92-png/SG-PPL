"""
对比实验运行脚本：Cox 比例风险基线（线性风险函数 + ridge），报告 C-index。

前端由 --step4_output 决定：
  SLIC 分支      → SLIC_function 跑完 Step1-4 后的 step4 目录
  我的超像素分支  → 主项目 ./output/step4

用法::

    # SLIC + Cox（λ 在网格上按验证集 C-index 选优）
    py -m comparative.run_cox --step4_output ./SLIC_function/output/step4 \
        --step5_output ./output/step5 --tag slic_cox

    # 固定 λ
    py -m comparative.run_cox --step4_output ./SLIC_function/output/step4 --l2_penalty 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .cox import DEFAULT_L2_GRID, DEFAULT_LBFGS_MAX_ITER, train_cox
from .data import DEFAULT_SEED, INPUT_MODES, build_dataset
from .report import build_summary, print_c_index, print_dataset_banner, save_results

_NOTE = (
    "关联模型（线性 Cox），不含治疗变量，无法产出 ITE/PITE；仅报告 C-index。"
    "目标函数 -partial_likelihood + λ||w||^2 为凸，L-BFGS 解到收敛即 penalized MLE，"
    "零初始化，结果与随机种子无关。"
)


def _parse_grid(text: str) -> tuple:
    vals = tuple(float(t) for t in str(text).replace(" ", "").split(",") if t)
    if not vals:
        raise argparse.ArgumentTypeError("l2_grid 至少需要一个值，例如 0,1,10,100")
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("l2_grid 各值须非负")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比实验：Cox 比例风险基线（多模态治疗无关输入，报告 C-index）"
    )
    parser.add_argument(
        "--step4_output",
        default="./output/step4",
        help="患者级图像特征目录，决定用哪个前端的超像素（SLIC 或我的超像素）",
    )
    parser.add_argument(
        "--step5_output",
        default=None,
        help="可选；用于复用主流水线的 train/val/test 划分，保证与其它格子逐例可比",
    )
    parser.add_argument("--clinical", default="./clinical.csv", help="clinical.csv 路径")
    parser.add_argument("--output", default="./output/comparative", help="结果输出目录")
    parser.add_argument(
        "--tag",
        default=None,
        help="实验标识，用于结果文件名（如 slic_cox）；默认按前端目录推断",
    )
    parser.add_argument(
        "--input_mode",
        default="concat",
        choices=INPUT_MODES,
        help="concat=图像⊕ajcc⊕age（基线标准喂法）；joint_causal=复用 step5 门控融合输出",
    )
    parser.add_argument(
        "--solver",
        default="lbfgs",
        choices=("lbfgs", "adamw"),
        help="lbfgs=解到收敛的 penalized MLE（默认）；adamw=与神经基线严格同训练协议",
    )
    parser.add_argument(
        "--l2_penalty",
        type=float,
        default=None,
        help="ridge 惩罚 λ；不设则在 --l2_grid 上按验证集 C-index 选优",
    )
    parser.add_argument(
        "--l2_grid",
        type=_parse_grid,
        default=DEFAULT_L2_GRID,
        help="λ 候选网格，逗号分隔（默认 0,1,3,10,30,100,300）",
    )
    parser.add_argument(
        "--max_iter", type=int, default=DEFAULT_LBFGS_MAX_ITER, help="L-BFGS 最大迭代数"
    )
    parser.add_argument("--epochs", type=int, default=600, help="solver=adamw 时的训练轮数")
    parser.add_argument("--lr", type=float, default=1e-2, help="solver=adamw 时的学习率")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument(
        "--top_k", type=int, default=10, help="打印系数绝对值最大的前 K 个特征"
    )
    parser.add_argument("--quiet", action="store_true", help="减少训练日志")
    args = parser.parse_args()

    dataset = build_dataset(
        step4_output=args.step4_output,
        clinical_csv=args.clinical,
        step5_output=args.step5_output,
        input_mode=args.input_mode,
        seed=args.seed,
    )
    tag = args.tag or f"cox_{Path(args.step4_output).resolve().parent.name}"

    print_dataset_banner(tag, "Cox", dataset)

    model, history = train_cox(
        dataset,
        solver=args.solver,
        l2_penalty=args.l2_penalty,
        l2_grid=tuple(args.l2_grid),
        max_iter=args.max_iter,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        verbose=not args.quiet,
    )
    runs: List[dict] = [history]

    summary = build_summary(
        tag=tag,
        model_name="cox",
        dataset=dataset,
        runs=runs,
        supports_counterfactual=False,
        note=_NOTE,
        extra={
            "coefficients_summary": history["coefficients_summary"],
            "l2_selection": history["l2_selection"],
            "solver": history["solver"],
        },
    )
    metrics_path = save_results(args.output, tag, model, dataset, summary)

    print_c_index(summary)
    coef = history["coefficients_summary"]
    print(
        f"系数（已标准化输入，正=风险升高）  ‖w‖₂={coef['l2_norm']:.4f}"
        f"  [图像部分 {coef['l2_norm_image']:.4f} / 临床部分 {coef['l2_norm_clinical']:.4f}]"
    )
    for row in coef["top_by_abs"][: args.top_k]:
        print(
            f"  {row['feature']:>10s}  coef={row['coef']:+.4f}  HR={row['hazard_ratio']:.3f}"
        )
    print("=" * 60)
    print(f"已保存到 {metrics_path}")


if __name__ == "__main__":
    main()

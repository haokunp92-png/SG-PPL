"""
对比实验运行脚本：T-learner 基线（两治疗臂各自独立建模），报告 C-index。

前端由 --step4_output 决定：
  SLIC 分支      → SLIC_function 跑完 Step1-4 后的 step4 目录
  我的超像素分支  → 主项目 ./output/step4

用法::

    # SLIC + T-learner
    py -m comparative.run_tlearner --step4_output ./SLIC_function/output/step4 \
        --step5_output ./output/step5 --tag slic_tlearner

    # 每臂改用线性 Cox
    py -m comparative.run_tlearner --step4_output ./SLIC_function/output/step4 \
        --base_learner cox --tag slic_tlearner_cox
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .data import DEFAULT_SEED, INPUT_MODES, build_dataset
from .deepsurv import DEFAULT_DROPOUT, DEFAULT_HIDDEN, DEFAULT_WEIGHT_DECAY
from .report import build_summary, print_c_index, print_dataset_banner, save_results
from .tlearner import BASE_LEARNERS, train_tlearner

_NOTE = (
    "T-learner：两治疗臂完全独立建模，不共享参数也不联合训练。"
    "可做反事实推理（本脚本只报 C-index，未产出 ITE）。"
    "合并 C-index 依赖各臂风险按训练集均值中心化，跨臂尺度差异无法完全消除，"
    "组内 C-index（c_index_by_arm）才是无歧义的。"
)


def _parse_hidden(text: str) -> tuple:
    dims = tuple(int(t) for t in str(text).replace(" ", "").split(",") if t)
    if not dims:
        raise argparse.ArgumentTypeError("hidden 至少需要一层，例如 64,32")
    if any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("hidden 各层维度须为正整数")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比实验：T-learner 基线（多模态治疗无关协变量，分臂建模，报告 C-index）"
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
        help="实验标识，用于结果文件名（如 slic_tlearner）；默认按前端目录推断",
    )
    parser.add_argument(
        "--input_mode",
        default="concat",
        choices=INPUT_MODES,
        help="concat=图像⊕ajcc⊕age（基线标准喂法）；joint_causal=复用 step5 门控融合输出",
    )
    parser.add_argument(
        "--base_learner",
        default="deepsurv",
        choices=BASE_LEARNERS,
        help="每臂的基学习器；deepsurv 与 DH-CaS 容量可比，是主要对照",
    )
    parser.add_argument(
        "--hidden",
        type=_parse_hidden,
        default=DEFAULT_HIDDEN,
        help="每臂 MLP 隐藏层维度（base_learner=deepsurv 时生效）",
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout 比例")
    parser.add_argument("--epochs", type=int, default=600, help="每臂训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument(
        "--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW 权重衰减"
    )
    parser.add_argument(
        "--l2_penalty", type=float, default=10.0, help="base_learner=cox 时每臂的固定 ridge λ"
    )
    parser.add_argument(
        "--no_center_risk",
        action="store_true",
        help="不做各臂风险中心化（合并 C-index 将失去定义，仅用于诊断）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="重复训练次数（种子递增，划分不变）；评估集小、噪声大，>1 可报 mean±std",
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
    tag = args.tag or f"tlearner_{Path(args.step4_output).resolve().parent.name}"

    print_dataset_banner(tag, f"T-learner ({args.base_learner})", dataset)

    runs: List[dict] = []
    model = None
    for i in range(max(1, args.n_repeats)):
        seed = args.seed + i
        if args.n_repeats > 1:
            print(f"\n--- 第 {i + 1}/{args.n_repeats} 次 (seed={seed}) ---")
        model, history = train_tlearner(
            dataset,
            base_learner=args.base_learner,
            hidden=tuple(args.hidden),
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_penalty=args.l2_penalty,
            center_risk=not args.no_center_risk,
            seed=seed,
            verbose=not args.quiet,
        )
        runs.append(history)

    summary = build_summary(
        tag=tag,
        model_name="tlearner",
        dataset=dataset,
        runs=runs,
        supports_counterfactual=True,
        note=_NOTE,
        extra={
            "base_learner": args.base_learner,
            "n_by_arm": runs[0]["n_by_arm"],
            "center_risk": runs[0]["center_risk"],
        },
    )
    metrics_path = save_results(args.output, tag, model, dataset, summary)

    print_c_index(summary)
    print("组内 C-index（不受跨臂偏移影响，无歧义）")
    for arm_key in ("arm0", "arm1"):
        vals = [r["c_index_by_arm"][arm_key] for r in runs]
        cells = []
        for split_name in ("train", "val", "test"):
            xs = [v[split_name] for v in vals]
            n = runs[0]["n_by_arm"][arm_key][split_name]
            mean = sum(xs) / len(xs)
            cells.append(
                f"{split_name}={mean:.4f}(n={n})" if mean == mean else f"{split_name}=N/A"
            )
        print(f"  {arm_key}: " + "  ".join(cells))
    print("=" * 60)
    print(f"已保存到 {metrics_path}")


if __name__ == "__main__":
    main()

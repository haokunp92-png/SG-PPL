"""
第七步：绘制生存分析图像 - 运行脚本

图一：单患者事实 vs 反事实生存曲线（需 --patients）
图二：PITE 分类柱状图
图四：ITE 瀑布图
"""

import argparse
from pathlib import Path
from typing import Optional, List

from .step7_visualization import run_step7_plot


def parse_patient_args(patients_str: Optional[str], indices_str: Optional[str]) -> tuple:
    """解析 --patients 和 --patient_indices 参数"""
    patient_ids: Optional[List[str]] = None
    patient_indices: Optional[List[int]] = None
    if patients_str:
        patient_ids = [p.strip() for p in patients_str.split(",") if p.strip()]
    if indices_str:
        try:
            patient_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
        except ValueError:
            raise ValueError("--patient_indices 应为逗号分隔的整数，如 0,1,2,3")
    return patient_ids, patient_indices


def main():
    parser = argparse.ArgumentParser(description="第七步：绘制生存分析图像")
    parser.add_argument(
        "--step6_output",
        default="./output/step6",
        help="第六步输出目录（含 step6_counterfactual_results.pkl）",
    )
    parser.add_argument(
        "--output",
        default="./output/step7",
        help="第七步输出目录",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="患者 ID 列表，逗号分隔，如 10071200,10093300",
    )
    parser.add_argument(
        "--patient_indices",
        type=str,
        default=None,
        help="患者序号列表（0-based），逗号分隔，如 0,1,2,3,4",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=200,
        help="生存率差异指标的 bootstrap 次数（用于 P 值）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出 step6 结果中的患者 ID，不绘图",
    )
    args = parser.parse_args()

    patient_ids, patient_indices = parse_patient_args(args.patients, args.patient_indices)

    if args.list:
        from .step7_visualization import load_step6_results
        pkl_path = Path(args.step6_output) / "step6_counterfactual_results.pkl"
        if not pkl_path.exists():
            print(f"Error: {pkl_path} not found. Run step 6 first.")
            return
        results = load_step6_results(str(pkl_path))
        print(f"Step 6 results contain {len(results)} patient(s):")
        for i, r in enumerate(results):
            print(f"  [{i}] {r['patient_id']}")
        return

    run_step7_plot(
        step6_output=args.step6_output,
        output_dir=args.output,
        patient_ids=patient_ids,
        patient_indices=patient_indices,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

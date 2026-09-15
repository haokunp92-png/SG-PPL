"""
Causal Reasoning & Attribution for sPixel_CXGNN
单患者 ITE 值归因分析（基于 Captum Integrated Gradients）+ 超像素归因可视化。

用法:
    # 仅生成归因 JSON
    python -m causalreasoning.run_attribution --patient 10071200

    # 同时生成超像素归因热力图（默认开启）
    python -m causalreasoning.run_attribution --patient 10071200 \
        --dataset_root ./dataset

    # 关闭可视化
    python -m causalreasoning.run_attribution --patient 10071200 --no_visualize

注意：内部 step6 pickle 数据键仍为 "IC"（保持向后兼容），但 JSON 输出统一使用 ITE。
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

from .attribution import ICAttributor

try:
    from .visualizer import AttributionVisualizer
except Exception as _viz_err:
    AttributionVisualizer = None  # type: ignore
    _VIZ_IMPORT_ERROR = _viz_err
else:
    _VIZ_IMPORT_ERROR = None


def main():
    parser = argparse.ArgumentParser(description="ITE 值因果归因（Integrated Gradients）")
    parser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="要分析的患者编号 (例如 10071200)",
    )
    parser.add_argument(
        "--step5_output",
        type=str,
        default="./output/step5",
        help="Step5 输出目录 (含 step5_aligned_data.pkl)",
    )
    parser.add_argument(
        "--step6_output",
        type=str,
        default="./output/step6",
        help="Step6 输出目录 (含 step6_model.pt 和 step6_counterfactual_results.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/causalreasoning_output",
        help="归因结果输出根目录",
    )
    parser.add_argument(
        "--step2_output",
        type=str,
        default="./output/step2",
        help="Step2 输出目录 (含 step2_slice_graphs.pkl，用于 per-superpixel 精细归因)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="integrated_gradients",
        choices=["integrated_gradients", "saliency"],
        help="归因方法",
    )

    # 可视化相关参数
    parser.add_argument(
        "--step1_output",
        type=str,
        default="./output/step1",
        help="Step1 输出目录 (含 step1_slice_graphs.pkl，用于可视化时定位每个超像素的像素坐标)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./dataset",
        help="原始数据集根目录，用于读取超像素原图（按 image_pattern 拼接）",
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="{dataset_root}/{patient_id}/spixel_viz/{patient_id}_t1c_image_z_{slice_num:04d}_sPixel.png",
        help="原始超像素图路径模板，可用变量：{dataset_root}, {patient_id}, {slice_num:04d}, {slice_id}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="热力图与原图叠加时的透明度 (0~1，越大越突出热力图)",
    )
    parser.add_argument(
        "--bg_threshold",
        type=int,
        default=15,
        help="原图近黑背景判定阈值：max(R,G,B)<=该值视为黑底/空洞，不叠加热力图 (0-255，默认 15)",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="跳过超像素热力图生成（默认开启可视化）",
    )

    args = parser.parse_args()

    patient_id = args.patient.strip()
    output_dir = Path(args.output) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始对患者 {patient_id} 进行 ITE 值归因分析...")

    # 1. 初始化归因器（传入 step2 用于 per-superpixel 精细归因）
    attributor = ICAttributor(
        step5_path=Path(args.step5_output) / "step5_aligned_data.pkl",
        step6_path=Path(args.step6_output),
        step2_path=Path(args.step2_output) / "step2_slice_graphs.pkl",
        method=args.method,
        n_steps=80,
    )

    # 2. 执行归因
    attribution_result = attributor.attribute_patient(patient_id)

    # 3. 提取并清理 image_attribution_summary（去掉过长的原始 attr_vector，避免 LLM 上下文过大）
    img_summary_raw = attribution_result.get("image_attribution_summary") or {}
    image_attribution_summary = {
        k: v for k, v in img_summary_raw.items() if k != "attr_vector"
    }

    # 4. 提取 per-superpixel 归因（dict 序列化时 key 转 str，便于跨语言读取）
    per_sp_raw = attribution_result.get("per_superpixel_importance", {}) or {}
    per_sp_serializable: Dict[str, Dict[str, float]] = {}
    for sid, node_dict in per_sp_raw.items():
        per_sp_serializable[str(sid)] = {
            str(int(nid)): float(score) for nid, score in (node_dict or {}).items()
        }

    # 5. 保存 JSON 结果（统一使用 ITE 命名）
    result_json = {
        "patient_id": patient_id,
        "ite_value": float(attribution_result.get("ic_value", 0.0)),
        "image_attribution_summary": image_attribution_summary,
        "clinical_contribution": {
            k: float(v) for k, v in attribution_result.get("clinical_contribution", {}).items()
        },
        "slice_importance": {
            k: float(v)
            for k, v in attribution_result.get("slice_importance", {}).items()
        },
        "per_superpixel_importance": per_sp_serializable,
        "overall_notes": attribution_result.get("notes", ""),
        "method": args.method,
    }

    json_path = output_dir / f"patient_{patient_id}_attribution.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\n分析完成！结果已保存到: {output_dir}")
    print(f"JSON 报告: {json_path.name}")
    print(f"图像 vs 临床贡献: {image_attribution_summary}")
    print(f"临床变量贡献: {result_json['clinical_contribution']}")

    # 6. 可视化：生成超像素归因热力图（默认开启）
    if not args.no_visualize:
        if AttributionVisualizer is None:
            print(
                f"\n[可视化] 跳过：依赖未就绪 ({_VIZ_IMPORT_ERROR})。"
                f"\n  请运行: pip install opencv-python"
            )
        else:
            heatmaps_dir = output_dir / "heatmaps"
            try:
                visualizer = AttributionVisualizer(
                    step1_pkl=Path(args.step1_output) / "step1_slice_graphs.pkl",
                    dataset_root=Path(args.dataset_root),
                    image_pattern=args.image_pattern,
                    alpha=args.alpha,
                    bg_threshold=args.bg_threshold,
                )
                saved = visualizer.visualize_patient(
                    patient_id=patient_id,
                    per_superpixel_importance=per_sp_raw,
                    output_dir=heatmaps_dir,
                )
                if saved:
                    print(f"\n[可视化] 共生成 {len(saved)} 张超像素热力图: {heatmaps_dir}")
                else:
                    print(f"\n[可视化] 未生成任何热力图（请检查 --dataset_root / --image_pattern）")
            except FileNotFoundError as e:
                print(f"\n[可视化] 跳过：{e}")


if __name__ == "__main__":
    main()

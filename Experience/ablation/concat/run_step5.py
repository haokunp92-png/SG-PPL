"""
第五步（concat 消融）：临床数据读取与模态对齐 - 运行脚本

读取 clinical.csv，与第四步图像特征对齐后，使用直接拼接方式构建 joint_feat。
"""

import argparse
import json
from pathlib import Path

from step5_clinical_fusion import run_step5_fusion


def main():
    parser = argparse.ArgumentParser(description="第五步（concat 消融）：临床数据读取与模态对齐")
    parser.add_argument(
        "--clinical",
        default="./clinical.csv",
        help="临床数据 CSV 路径",
    )
    parser.add_argument(
        "--step4_output",
        default="./output/step4",
        help="第四步输出目录（含 step4_patient_img_feats.pkl）",
    )
    parser.add_argument(
        "--output",
        default="./output/step5_concat",
        help="第五步输出目录",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="可选：指定患者 ID 列表",
    )
    parser.add_argument(
        "--d_joint",
        type=int,
        default=128,
        help="联合表征维度 d_joint",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--joint_graph_upstream",
        action="store_true",
        help="上游已使用联合图模式时启用：step5 不再重复注入临床，仅做患者级投影",
    )
    parser.add_argument(
        "--step4_causal_output",
        default=None,
        help="可选：因果版 step4 特征 pkl 路径或目录（含 step4_patient_img_feats_causal.pkl）",
    )
    args = parser.parse_args()

    clinical_path = Path(args.clinical)
    if not clinical_path.exists():
        print(f"错误：未找到 {clinical_path}")
        return

    pkl_path = Path(args.step4_output) / "step4_patient_img_feats.pkl"
    if not pkl_path.exists():
        print(f"错误：未找到 {pkl_path}，请先运行第四步")
        return

    step4_pkl_causal = None
    if args.step4_causal_output:
        step4_causal_path = Path(args.step4_causal_output)
        if step4_causal_path.is_dir():
            step4_causal_path = step4_causal_path / "step4_patient_img_feats_causal.pkl"
        if not step4_causal_path.exists():
            print(f"错误：未找到因果版特征文件 {step4_causal_path}")
            return
        step4_pkl_causal = str(step4_causal_path)

    if args.joint_graph_upstream:
        step4_meta_path = Path(args.step4_output) / "step4_meta.json"
        has_causal_img_feat = False
        if step4_meta_path.exists():
            try:
                with open(step4_meta_path, "r", encoding="utf-8") as f:
                    step4_meta = json.load(f)
                global_meta = step4_meta.get("global", {}) if isinstance(step4_meta, dict) else {}
                has_causal_img_feat = bool(global_meta.get("has_causal_img_feat", False))
            except (OSError, json.JSONDecodeError):
                has_causal_img_feat = False

        if not has_causal_img_feat and step4_pkl_causal is None:
            print(
                "错误：启用 --joint_graph_upstream 时，未检测到因果版患者图像特征。"
                "请先按联合图链路运行 step3/step4（生成 step4_patient_img_feats_causal.pkl），"
                "或通过 --step4_causal_output 显式提供该文件。"
            )
            return

    print("正在读取临床数据并进行 concat 融合...")
    run_step5_fusion(
        clinical_csv=str(clinical_path),
        step4_pkl=str(pkl_path),
        output_dir=args.output,
        patient_ids=args.patients,
        d_joint=args.d_joint,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        step4_pkl_causal=step4_pkl_causal,
        use_joint_graph_upstream=bool(args.joint_graph_upstream),
    )
    print("第五步（concat 消融）完成。")


if __name__ == "__main__":
    main()

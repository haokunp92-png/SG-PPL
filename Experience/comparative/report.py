"""
对比实验结果的统一汇总与落盘，保证各基线产出同一套 JSON 字段，
便于之后直接拼成对比表格。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .trainer import SPLITS


def build_summary(
    tag: str,
    model_name: str,
    dataset,
    runs: List[Dict[str, Any]],
    supports_counterfactual: bool = False,
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把多次重复训练的 history 汇总成一份结果。"""
    summary: Dict[str, Any] = {
        "tag": tag,
        "model": model_name,
        "supports_counterfactual": supports_counterfactual,
        "note": note,
        "data_meta": dataset.meta,
        "n_repeats": len(runs),
        "runs": runs,
    }
    for split_name in SPLITS:
        vals = [
            r["c_index"][split_name]
            for r in runs
            if not np.isnan(r["c_index"][split_name])
        ]
        if vals:
            summary[f"c_index_{split_name}"] = float(np.mean(vals))
            summary[f"c_index_{split_name}_std"] = float(np.std(vals))
    if extra:
        summary.update(extra)
    return summary


def save_results(
    output_dir: str,
    tag: str,
    model: torch.nn.Module,
    dataset,
    summary: Dict[str, Any],
) -> Path:
    """保存权重与指标 JSON，返回指标文件路径。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "d_in": dataset.d_in, "tag": tag},
        out_dir / f"{tag}_model.pt",
    )
    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return metrics_path


def print_c_index(summary: Dict[str, Any]) -> None:
    """打印 C-index 结果块。"""
    print("\n" + "=" * 60)
    print(f"C-index 结果  [{summary['tag']}]")
    print("=" * 60)
    n_repeats = summary.get("n_repeats", 1)
    for split_name in SPLITS:
        key = f"c_index_{split_name}"
        if key not in summary:
            print(f"  {split_name:5s}: N/A (样本不足 2 例，无可比对)")
            continue
        line = f"  {split_name:5s}: {summary[key]:.4f}"
        if n_repeats > 1:
            line += f" ± {summary[key + '_std']:.4f}"
        print(line)
    print("=" * 60)


def print_dataset_banner(tag: str, model_name: str, dataset) -> None:
    """打印数据与设置概览。"""
    print("=" * 60)
    print(f"对比实验: {model_name}  [{tag}]")
    print("=" * 60)
    print(f"  {dataset.describe()}")
    print(
        f"  划分来源: {dataset.meta['split_source']}  | "
        f"含治疗变量: {dataset.meta['includes_treatment']}"
    )
    if dataset.meta["event_rate"] >= 1.0:
        print(
            "  [warn] 事件率为 1.000（clinical.csv 无 event 列，全部视为已发生），"
            "C-index 无删失处理"
        )

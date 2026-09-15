#!/usr/bin/env python3
"""
SLIC_function 内一键流程：由**原图**经 SLIC 生成 ``map_csv`` + ``spixel_viz``，再顺序跑 Step1–6。

必选 / 常用参数（参见 ``README.md``）：
  - ``-i/--input_dir``（或 ``--image_dir``）：原图根目录 ``{patient_id}/*.png``
  - ``-d/--dataset``（或 ``--slic-out``）：SLIC 写出根目录（map_csv / spixel_viz），兼 Step1 的 dataset
  - ``--out_root``（或 ``--run-output``）：Step1–6 的中间与模型输出根（其下 ``step1``…``step6``）
  - ``-c/--clinical``：clinical.csv

在 **SLIC_function** 下执行示例::

    python run_slic_pipeline.py -i D:/my/img --slic-out D:/exp/slic_dataset --run-output D:/exp/run --clinical ./clinical.csv

方法论与仓库 ``ours/pipeline`` 一致。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_cmd(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    r = subprocess.run(argv, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SLIC map_csv → Step1→2→3→4→5→6（在 SLIC_function 目录运行）",
    )
    ap.add_argument(
        "-i",
        "--input_dir",
        "--image_dir",
        dest="input_dir",
        default=None,
        help="【必填】原图根目录（如 ./img）；结构 {patient_id}/*.png，内含 z_序号 文件名",
    )
    ap.add_argument(
        "-d",
        "--dataset",
        "--slic-out",
        dest="dataset",
        default="./dataset",
        metavar="PATH",
        help="【SLIC 输出 / Step1 dataset】map_csv 与 spixel_viz 的根目录（可绝对路径自定义）",
    )
    ap.add_argument(
        "--out_root",
        "--run-output",
        dest="out_root",
        default="./output",
        metavar="PATH",
        help="【训练流水线输出】Step1–6 根目录，其下自动创建 step1…step6（可绝对路径自定义）",
    )
    ap.add_argument(
        "-c",
        "--clinical",
        default="./clinical.csv",
        help="clinical.csv（相对 SLIC_function 或绝对路径）",
    )
    ap.add_argument(
        "--skip_slic",
        action="store_true",
        help="跳过 SLIC（dataset 目录已含有 map_csv）",
    )
    ap.add_argument("--no_viz", action="store_true", help="SLIC 不生成 spixel_viz")
    py = sys.executable

    ap.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Step6 训练轮数",
    )
    ap.add_argument(
        "--no_inference_all",
        action="store_true",
        help="Step6 不传 --inference_all（默认会传便于全队列反事实）",
    )

    args = ap.parse_args()

    if not args.input_dir:
        ap.error("请指定原图目录：-i / --input_dir（或旧参数 --image_dir）")

    dataset = Path(args.dataset)
    dataset = dataset.resolve() if dataset.is_absolute() else (ROOT / dataset).resolve()

    clinical = Path(args.clinical)
    clinical = clinical.resolve() if clinical.is_absolute() else (ROOT / clinical).resolve()

    image_dir_arg = Path(args.input_dir)
    image_abs = (
        image_dir_arg
        if image_dir_arg.is_absolute()
        else (ROOT / image_dir_arg).resolve()
    )

    if not args.skip_slic:
        slic_cmd = [
            py,
            "-m",
            "slic.run_slic",
            "--image_dir",
            str(image_abs),
            "--output",
            str(dataset),
            "--recursive",
            "--canonical_step1_names",
            "--preserve_structure",
        ]
        if args.no_viz:
            slic_cmd.append("--no_viz")
        run_cmd(slic_cmd)

    if not dataset.is_dir():
        print(f"dataset 不存在: {dataset}")
        sys.exit(1)
    if not clinical.is_file():
        print(f"未找到 clinical: {clinical}（可复制项目根目录的 clinical.csv）")
        sys.exit(1)

    out_root = args.out_root

    run_cmd(
        [
            py,
            "run_step1.py",
            "--dataset",
            str(dataset),
            "--output",
            f"{out_root}/step1",
            "--image_dir",
            str(image_abs),
        ]
    )

    run_cmd(
        [
            py,
            "run_step2.py",
            "--step1_output",
            f"{out_root}/step1",
            "--output",
            f"{out_root}/step2",
            "--image_dir",
            str(image_abs),
        ]
    )

    run_cmd(
        [
            py,
            "run_step3.py",
            "--step2_output",
            f"{out_root}/step2",
            "--output",
            f"{out_root}/step3",
        ]
    )

    run_cmd(
        [
            py,
            "run_step4.py",
            "--step3_output",
            f"{out_root}/step3",
            "--output",
            f"{out_root}/step4",
        ]
    )

    run_cmd(
        [
            py,
            "run_step5.py",
            "--clinical",
            str(clinical.resolve()),
            "--step4_output",
            f"{out_root}/step4",
            "--output",
            f"{out_root}/step5",
        ]
    )

    step6 = [
        py,
        "run_step6.py",
        "--step5_output",
        f"{out_root}/step5",
        "--output",
        f"{out_root}/step6",
        "--epochs",
        str(args.epochs),
    ]
    if not args.no_inference_all:
        step6.append("--inference_all")
    run_cmd(step6)

    _bout = Path(args.out_root)
    _bout = _bout.resolve() if _bout.is_absolute() else (ROOT / _bout).resolve()
    print("\n流水线结束。模型与结果:", _bout / "step6")


if __name__ == "__main__":
    main()

"""
SLIC 批量处理**原图**，产出与 `fcn_superpixel/infer.py`（ours）一致的目录与命名。

输入示例：``img/10070385/10070385_t1c_image_z_0020.png``（患者子目录 + 原图文件名）

输出（与 infer 相同的两类产物）：
  - ``{dataset}/{patient}/map_csv/{patient}_slice_{序}.csv``：H×W，逗号分隔，超像素 ID **1-based**
  - ``{dataset}/{patient}/spixel_viz/{patient}_slice_{序}_sPixel.png``：**超像素边界可视化图**（默认始终写出）

节点数：默认使用 FCN infer 相同公式
  H_ = ceil(H/16)*16, W_ = ceil(W/16)*16, target = floor(H_/downsize)*floor(W_/downsize)
若 SLIC 连通域数 > target，默认贪合并至 <= target（见 merge_segments）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage import img_as_float
    from skimage.segmentation import mark_boundaries, relabel_sequential, slic
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "SLIC 需要 scikit-image。请执行: pip install scikit-image\n"
        f"原始错误: {e}"
    ) from e

try:
    from .grid_count import expected_grid_shape, target_superpixel_count
    from .merge_segments import merge_superpixels_to_count
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grid_count import expected_grid_shape, target_superpixel_count
    from merge_segments import merge_superpixels_to_count


def _canonical_csv_stem(rel_dir: Path, image_stem: str, slice_zero_pad: int) -> str:
    """
    生成与 Step1 匹配的 ``{patient_id}_slice_{序}`` basename（不含 .csv）。

    patient_id：优先文件名 ``数字_`` 前缀（``{患者ID}_t1c_...``），否则用直接父目录
    ``rel_dir.name``；输出目录为 ``{output}/{patient_id}/map_csv/``。
    切片序号：优先 ``z_`` 捕获；否则用 stem 最后一个整数并按 ``slice_zero_pad`` 格式化。
    """
    if slice_zero_pad < 1 or slice_zero_pad > 8:
        raise ValueError("slice_zero_pad 应在 1~8")
    # 优先从文件名取患者 ID（与 FCN 原图 ``{患者ID}_...`` 一致），再退回父文件夹名
    m_head = re.match(r"^(\d+)_", image_stem)
    if m_head:
        pid = m_head.group(1)
    elif rel_dir.name:
        pid = rel_dir.name
    else:
        raise ValueError(
            "无法确定患者 ID：图在根目录且文件名不以「患者ID_」开头；"
            "请改为「编号_其他.png」或放入「患者ID/图.png」子目录。"
        )
    # 与 FCN 原图命名一致时保留 z_0020 中的「0020」，避免与 t1c_image_z_0020 冲突
    mz = re.search(r"z[_\s]?(\d+)", image_stem, re.I)
    if mz:
        sid = mz.group(1)
    else:
        nums = [int(x) for x in re.findall(r"\d+", image_stem)]
        if not nums:
            raise ValueError("文件名中无数字，无法确定切片序号")
        sid = f"{nums[-1]:0{slice_zero_pad}d}"
    return f"{pid}_slice_{sid}"


def load_image(path: Path) -> np.ndarray | None:
    """与 fcn_superpixel/infer.load_image 一致：返回 RGB uint8。"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def run_slic_labels(
    img_rgb: np.ndarray,
    *,
    pad_multiple: int = 16,
    downsize: int = 16,
    compactness: float = 10.0,
    sigma: float = 1.0,
    max_num_iter: int = 10,
    merge_to_target: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    在 FCN 对齐尺寸 (H_, W_) 上运行 SLIC，返回 label 图 (H_, W_)，元素为 0..K-1。
    附 info dict：含 target_count, n_unique_before_merge, n_unique_final, H_, W_。
    """
    H, W = img_rgb.shape[:2]
    H_, W_, _, _ = expected_grid_shape(
        H, W, pad_multiple=pad_multiple, downsize=downsize
    )
    target = target_superpixel_count(
        H, W, pad_multiple=pad_multiple, downsize=downsize
    )

    img_rs = cv2.resize(img_rgb, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img_f = img_as_float(img_rs)

    slic_base = dict(
        n_segments=target,
        compactness=compactness,
        sigma=sigma,
        max_num_iter=max_num_iter,
        start_label=0,
    )

    def _slic_attempt(img: np.ndarray, kw: dict) -> np.ndarray:
        errors: list[str] = []
        for call in (
            lambda: slic(img, channel_axis=-1, **kw),
            lambda: slic(
                img,
                channel_axis=-1,
                **{k: v for k, v in kw.items() if k != "enforce_connectivity"},
            ),
            lambda: slic(
                img,
                multichannel=True,
                **{k: v for k, v in kw.items() if k != "enforce_connectivity"},
            ),
            lambda: slic(img, multichannel=True, **kw),
        ):
            try:
                return call()
            except TypeError as e:
                errors.append(str(e))
                continue
        raise RuntimeError(
            "skimage.segmentation.slic 调用失败（尝试了多种参数组合）。最后错误: "
            + errors[-1]
        )

    segments: np.ndarray | None = None
    for add_enforce in (True, False):
        kw = dict(slic_base)
        if add_enforce:
            kw["enforce_connectivity"] = True
        try:
            segments = _slic_attempt(img_f, kw)
            break
        except RuntimeError:
            continue

    if segments is None:
        raise RuntimeError("无法调用 skimage.segmentation.slic，请检查 scikit-image 版本")

    segments = np.asarray(segments, dtype=np.int64)
    n0 = len(np.unique(segments))

    if merge_to_target and n0 > target:
        segments = merge_superpixels_to_count(segments, target)

    segments, _, _ = relabel_sequential(segments)
    n1 = len(np.unique(segments))

    info = {
        "H": H,
        "W": W,
        "H_": H_,
        "W_": W_,
        "target_superpixel_count": target,
        "n_unique_before_merge": int(n0),
        "n_unique_final": int(n1),
        "below_target": n1 < target,
    }
    return segments, info


def _save_visualization(img_rgb: np.ndarray, labels: np.ndarray, out_png: Path) -> None:
    """与 infer 类似：在原尺寸图上画边界 — 先将 label 升为原图尺寸或把图缩至 label。"""
    H, W = img_rgb.shape[:2]
    lh, lw = labels.shape[:2]
    if (lh, lw) != (H, W):
        show = cv2.resize(img_rgb, (lw, lh), interpolation=cv2.INTER_CUBIC)
    else:
        show = img_rgb
    bd = mark_boundaries(
        img_as_float(show), labels.astype(np.int32), color=(0, 1, 1)
    )
    bgr = (np.clip(bd, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), bgr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SLIC 超像素 → map_csv（与 FCN infer 同一目标节点数约定）"
    )
    parser.add_argument("--image_dir", required=True, help="输入图像根目录")
    parser.add_argument("--output", default="./slic_out", help="输出根目录")
    parser.add_argument(
        "--suffix",
        default="png",
        help="搜索扩展名（亦会尝试 jpg/jpeg/png/bmp）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录（患者/切片 结构）",
    )
    parser.add_argument(
        "--preserve_structure",
        action="store_true",
        help="保留相对路径：output/患者/.../map_csv 与 infer 一致",
    )
    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=16,
        help="与 FCN infer 一致：将边长 pad 为该数的倍数后再算网格",
    )
    parser.add_argument(
        "--downsize",
        type=int,
        default=16,
        help="与 FCN infer --downsize 一致：网格步长（默认 16）",
    )
    parser.add_argument(
        "--compactness",
        type=float,
        default=10.0,
        help="SLIC compactness（越大越规则）",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="高斯预处理 sigma（像素），0 表示关闭",
    )
    parser.add_argument(
        "--max_num_iter",
        type=int,
        default=10,
        help="SLIC 迭代次数",
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="不在 SLIC 后合并；唯一域数可能大于 FCN target",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="不写出超像素可视化图（默认同 infer：始终生成 spixel_viz/*_sPixel.png）",
    )
    parser.add_argument(
        "--canonical_step1_names",
        action="store_true",
        help=(
            "将 CSV 命名为 {patient}_slice_{slice}.csv（与 downstream Step1/2 一致）；"
            "推荐与 --preserve_structure 联用（首级目录为患者 ID）"
        ),
    )
    parser.add_argument(
        "--slice_zero_pad",
        type=int,
        default=3,
        help=(
            "无 z_序号时 fall back：用最后一位整数并按此宽度补齐；"
            "有 z_XXXX 时沿用文件名中的位数（与本仓库 *_t1c_image_z_0020*.png 一致）"
        ),
    )

    args = parser.parse_args()
    merge_to_target = not args.no_merge

    rootsuf = Path(args.image_dir)
    exts = {args.suffix.lower(), "jpg", "jpeg", "png", "bmp"}
    image_paths: list[Path] = []
    for ext in exts:
        if args.recursive:
            image_paths.extend(rootsuf.rglob(f"*.{ext}"))
        else:
            image_paths.extend(rootsuf.glob(f"*.{ext}"))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"未在 {args.image_dir} 找到图像")
        return

    print(f"共 {len(image_paths)} 张图；输出目录 {args.output}")
    warned_below = False

    for idx, img_path in enumerate(image_paths):
        img = load_image(img_path)
        if img is None:
            print(f"跳过无法读取: {img_path}")
            continue

        labels, info = run_slic_labels(
            img,
            pad_multiple=args.pad_multiple,
            downsize=args.downsize,
            compactness=args.compactness,
            sigma=args.sigma,
            max_num_iter=args.max_num_iter,
            merge_to_target=merge_to_target,
        )

        if info["below_target"] and not warned_below:
            print(
                "[提示] 部分切片 SLIC 唯一域数少于 FCN target；"
                "可尝试减小 --compactness 或增大 --max_num_iter / --sigma。"
            )
            warned_below = True

        try:
            rel_path = img_path.relative_to(rootsuf.resolve())
        except ValueError:
            rel_path = Path(img_path.name)

        base = img_path.stem
        rel_dir = rel_path.parent

        out_root = Path(args.output)

        if args.canonical_step1_names:
            try:
                out_base = _canonical_csv_stem(rel_dir, base, args.slice_zero_pad)
                pid = out_base.split("_slice_", 1)[0]
                csv_dir = out_root / pid / "map_csv"
                viz_dir = out_root / pid / "spixel_viz"
            except ValueError as e:
                print(f"[canonical_step1_names] {e}，退回 infer 命名布局")
                if args.preserve_structure and str(rel_dir) != ".":
                    viz_dir = out_root / rel_dir / "spixel_viz"
                    csv_dir = out_root / rel_dir / "map_csv"
                    out_base = base
                else:
                    viz_dir = out_root / "spixel_viz"
                    csv_dir = out_root / "map_csv"
                    out_base = (
                        str(rel_dir).replace(os.sep, "_") + "_" + base
                        if str(rel_dir) != "."
                        else base
                    )
        else:
            if args.preserve_structure and str(rel_dir) != ".":
                viz_dir = out_root / rel_dir / "spixel_viz"
                csv_dir = out_root / rel_dir / "map_csv"
                out_base = base
            else:
                viz_dir = out_root / "spixel_viz"
                csv_dir = out_root / "map_csv"
                out_base = (
                    str(rel_dir).replace(os.sep, "_") + "_" + base
                    if str(rel_dir) != "."
                    else base
                )

        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{out_base}.csv"
        to_save = (labels + 1).astype(np.int32)
        np.savetxt(str(csv_path), to_save, fmt="%i", delimiter=",")

        if not args.no_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)
            _save_visualization(img, labels, viz_dir / f"{out_base}_sPixel.png")

        if (idx + 1) % 10 == 0:
            print(
                f"  已处理 {idx + 1}/{len(image_paths)} "
                f"(target={info['target_superpixel_count']}, "
                f"final={info['n_unique_final']})"
            )

    print("完成。")


if __name__ == "__main__":
    main()

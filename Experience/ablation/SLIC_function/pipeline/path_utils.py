"""原图路径解析：兼容 ``slice_NNN.png`` 与 FCN 风格 ``*z_*0025*.png``。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

_NUMERIC_ID_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*$")


def normalize_patient_id(pid) -> str:
    """
    将临床表 / 目录名 / pickle 键上的患者 ID 规范为同一字符串。

    常见情况：Excel 导出 ``10070385.0``、前后空格、数值型文本，与影像侧 ``10070385`` 对齐。
    非纯数字 ID 仅做 strip 与 BOM 去除，不强行改写。
    """
    if pid is None:
        return ""
    s = str(pid).strip().lstrip("\ufeff")
    if not s:
        return s
    if _NUMERIC_ID_RE.fullmatch(s):
        try:
            x = float(s)
            if x.is_integer():
                return str(int(x))
        except (ValueError, OverflowError):
            pass
    return s


def resolve_slide_image(
    image_dir: str | Path, patient_id: str, slice_id: str
) -> Optional[str]:
    """
    在 ``{image_dir}/{patient_id}/`` 下查找与 slice_id 对应的切片图。

    尝试顺序：
    1. ``slice_{slice_id}.{ext}``、``{slice_id}.{ext}``
    2. 文件名中 ``z_*连续数字`` 的整数与 slice_id 一致
    3. stem 中任一数字段与 slice_id 的整数值一致（匹配多文件时取字典序第一个）

    若直接子目录名与 ``patient_id`` 不一致（如 ``10070385.0`` 文件夹），会按
    :func:`normalize_patient_id` 匹配同一切患者目录。
    """
    base = Path(image_dir)
    root = base / str(patient_id)
    if not root.is_dir():
        want = normalize_patient_id(patient_id)
        found: Optional[Path] = None
        for d in base.iterdir():
            if d.is_dir() and normalize_patient_id(d.name) == want:
                found = d
                break
        if found is None:
            return None
        root = found
    sid = str(slice_id).strip()
    for ext in _EXT:
        for name in (f"slice_{sid}{ext}", f"{sid}{ext}"):
            p = root / name
            if p.is_file():
                return str(p.resolve())

    try:
        target = int(sid.lstrip("0") or "0")
    except ValueError:
        return None

    for ext in _EXT:
        for p in sorted(root.glob(f"*{ext}")):
            m = re.search(r"z[_\s]?(\d+)", p.stem, re.I)
            if m and int(m.group(1)) == target:
                return str(p.resolve())

    for ext in _EXT:
        for p in sorted(root.glob(f"*{ext}")):
            nums = [int(x) for x in re.findall(r"\d+", p.stem)]
            if nums and nums[-1] == target:
                return str(p.resolve())

    return None

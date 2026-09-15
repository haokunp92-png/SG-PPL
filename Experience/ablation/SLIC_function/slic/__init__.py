"""
SLIC 超像素导出：与 `fcn_superpixel/infer.py` 同一「网格目标节点数」约定，
产出与 Step1 `graph_builder.load_spixel_map` 一致的 map_csv。

- `expected_grid_shape` / `target_superpixel_count`：与 FCN 推理脚本一致；
- CLI：`python -m slic.run_slic` 或 `python slic/run_slic.py`。
"""

from .grid_count import expected_grid_shape, target_superpixel_count

__all__ = ["expected_grid_shape", "target_superpixel_count"]

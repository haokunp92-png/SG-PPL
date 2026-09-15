本说明仅针对 **SLIC_function** 子项目。入口与选项与根目录 `slic` 包相同，请在 **`SLIC_function`** 目录下执行：

```bash
cd SLIC_function
python -m slic.run_slic --help
```

数据结构见上级 **`README.md`**（与仓库根目录 `dataset/` 对齐：`{patient_id}/map_csv/{patient_id}_slice_{NNN}.csv`，canonical 默认三位 `NNN`）。

与 Step1–6 的一键衔接见 **`run_slic_pipeline.py`**。

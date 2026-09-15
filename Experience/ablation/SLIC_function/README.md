# SLIC_function：概述与用法

## 概述

本目录在 **不改变主项目生存分析方法论** 的前提下，把 **超像素分割** 换成 **SLIC**：从**原图**批量生成与 `ours` / FCN `infer.py` 对齐的 **`map_csv` + `spixel_viz`**，再沿用内部 **Step1→6**（图 → 组学 → GNN → 患者融合 → 临床融合 → 因果生存模型）。

| 阶段 | 做什么 |
|------|--------|
| SLIC | 读 `{患者}/原图.png`，写 `{患者}/map_csv/*.csv` 与 `{患者}/spixel_viz/*_sPixel.png` |
| Step1–6 | 与仓库 `pipeline`、`CXGNN` 一致；`--dataset` 指向上述 SLIC 根目录，`--image_dir` 指向同一批原图 |

**原图示例路径**：`./img/10070385/10070385_t1c_image_z_0020.png`（患者子目录 + 含 `z_0020` 的文件名 → 输出 `10070385_slice_0020.csv` 等）。

---

## 依赖与运行位置

```bash
cd SLIC_function
pip install torch
pip install -r slic/requirements.txt
```

以下命令均在 **`SLIC_function`** 目录下执行；路径支持 **相对本目录** 或 **绝对路径**。

---

## 一键流水线：参数说明

脚本：**`run_slic_pipeline.py`**

| 参数（简写） | 含义 | 默认 |
|--------------|------|------|
| **`-i` / `--input_dir`**（或 **`--image_dir`**） | **传入数据：原图根目录**（递归；`{patient_id}/*.png`） | （必填） |
| **`-d` / `--dataset`**（或 **`--slic-out`**） | **SLIC 输出根目录**：写入 `map_csv/`、`spixel_viz/`；也是 Step1 的 `--dataset` | `./dataset` |
| **`--out_root`**（或 **`--run-output`**） | **Step1–6 输出根目录**；其下自动生成 `step1`…`step6` | `./output` |
| **`-c` / `--clinical`** | `clinical.csv` | `./clinical.csv` |
| **`--skip_slic`** | 跳过 SLIC（`--dataset` 下已有 `map_csv`） | 关 |
| **`--no_viz`** | 不生成 `spixel_viz/*_sPixel.png` | 关 |
| **`--epochs`** | Step6 训练轮数 | `600` |
| **`--no_inference_all`** | Step6 不加 `--inference_all` | 关 |

### 示例：自定义输入 / 输出目录

将原图、SLIC 产物、训练中间结果**全部放在你指定的盘上**（可把路径换成你的真实目录）：

```bash
python run_slic_pipeline.py ^
  -i D:/data/npc_raw/img ^
  -d D:/exp/slic_ablation/dataset ^
  --run-output D:/exp/slic_ablation/run ^
  -c D:/data/clinical.csv
```

Linux / macOS：

```bash
python run_slic_pipeline.py \
  -i /data/npc_raw/img \
  -d /exp/slic_ablation/dataset \
  --run-output /exp/slic_ablation/run \
  -c /data/clinical.csv
```

等价写法（长参数名与旧名兼容）：

```bash
python run_slic_pipeline.py --input_dir ./img --dataset ./my_slic_data --out_root ./my_train_out --clinical ./clinical.csv
```

仅重跑 Step1–6、**不跑 SLIC** 时：

```bash
python run_slic_pipeline.py -i ./img -d D:/exp/slic_dataset --run-output D:/exp/run2 -c ./clinical.csv --skip_slic
```

（`--skip_slic` 时仍需要提供 **`-i`**，供 Step1/2 对齐原图路径。）

查看全部选项：

```bash
python run_slic_pipeline.py --help
```

---

## 仅运行 SLIC（不跑 Step1–6）

**`python -m slic.run_slic`**：原图目录由 **`--image_dir`** 指定，**输出根**由 **`--output`** 指定（与一键里的 **`-d` / `--dataset` / `--slic-out`** 含义相同）。

```bash
python -m slic.run_slic --image_dir ./img --output ./my_slic_dataset --recursive ^
  --canonical_step1_names --preserve_structure
```

可选 **`--no_viz`**；默认会写 `spixel_viz` 与 `map_csv`。更多选项：`python -m slic.run_slic --help`。

---

## 与原 FCN（ours）产物对齐

| 项目 | 说明 |
|------|------|
| 目录 | `{patient}/map_csv` + `{patient}/spixel_viz` |
| 文件 | `{patient}_slice_{序}.csv`、`{patient}_slice_{序}_sPixel.png` |
| CSV | 与 `graph_builder.load_spixel_map` 一致：**(H_,W_)**、`1-based` |

---

## 分步等价命令

若不想用一键脚本，可手动串联（请将 `./img`、`./dataset`、`./output` 换成你的 **`--input_dir` / `-d` / `--out_root`**）：

```bash
python -m slic.run_slic --image_dir ./img --output ./dataset --recursive --canonical_step1_names --preserve_structure
python run_step1.py --dataset ./dataset --output ./output/step1 --image_dir ./img
python run_step2.py --step1_output ./output/step1 --output ./output/step2 --image_dir ./img
python run_step3.py --step2_output ./output/step2 --output ./output/step3
python run_step4.py --step3_output ./output/step3 --output ./output/step4
python run_step5.py --clinical ./clinical.csv --step4_output ./output/step4 --output ./output/step5
python run_step6.py --step5_output ./output/step5 --output ./output/step6 --inference_all
```

---

## Step1 / Step2 原图解析

**`pipeline/path_utils.resolve_slide_image`**：按 `slice_{序}.png`、文件名中的 **`z_`** 等与切片序号匹配。

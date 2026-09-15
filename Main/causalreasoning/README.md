# Causal Reasoning & ITE Attribution（因果归因模块）

基于 Step6 **双头因果生存模型**，对个体 **ITE**（个体治疗效应；内部 step6 pickle 仍用键 `IC` 以兼容旧数据）做解释：Captum **Integrated Gradients**、切片/超像素重要性、临床变量贡献，并**默认生成超像素归因热力图**。

## 功能概要

| 能力 | 说明 |
|------|------|
| 模型归因 | Captum IG 作用于 `joint_feat`，前向目标为 treatment=1 侧风险输出 |
| 超像素重要性 | 结合 IG 切片权重与 Step2 **节点特征 L2 范数**（`node_feat_joint_slice`，缺省回退 `node_feat_init_slice`） |
| 临床贡献 | Step5 `attribution_branch` 结构与近似临床分项（ajcc / age / youdao） |
| Step6 | **严格加载** `step6_model.pt`，缺失或损坏即报错终止 |
| 热力图 | 使用 Step1 **`pixel_coords`** 在底板图上按超像素涂色（OpenCV JET + ROI/黑底屏蔽） |

## 依赖与环境

```bash
pip install -r causalreasoning/requirements.txt
```

热力图依赖 **OpenCV**（`opencv-python`）；未安装时归因 JSON 仍可生成，可视化会被跳过并提示安装。

## 使用方法

前置：完成流水线至 Step6（需 `step6_model.pt` 与反事实结果）。

```bash
python run_step6.py --inference_all

# 归因 + 默认生成热力图（需 step1/step2、数据集底板图路径正确）
pip install -r causalreasoning/requirements.txt
python -m causalreasoning.run_attribution --patient 10071200 \
  --dataset_root ./dataset

# 仅 JSON，不要热力图
python -m causalreasoning.run_attribution --patient 10071200 --no_visualize
```

### 常用参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--step5_output` | `./output/step5` | 含 `step5_aligned_data.pkl` |
| `--step6_output` | `./output/step6` | 含 `step6_model.pt`、`step6_counterfactual_results.pkl` |
| `--step2_output` | `./output/step2` | 含 `step2_slice_graphs.pkl`，用于 per-superpixel 分数 |
| `--step1_output` | `./output/step1` | 含 `step1_slice_graphs.pkl`，热力图需要 `pixel_coords` |
| `--dataset_root` | `./dataset` | 底板图所在数据根目录 |
| `--image_pattern` | `{dataset_root}/{patient_id}/spixel_viz/{patient_id}_t1c_image_z_{slice_num:04d}_sPixel.png` | 原图路径模板 |
| `--alpha` | `0.6` | 热力图与原图混合权重（越大越突出热力） |
| `--bg_threshold` | `15` | `max(R,G,B) ≤` 该值视为黑底/空洞，不叠加热力 |
| `--no_visualize` | — | 关闭热力图，只写 JSON |

### 可选：Joint graph 流水线

若你走带 `--joint_graph` / `--joint_graph_upstream` 的 Step2–Step5，可与此模块衔接；归因命令不变，仅需保证 Step1–Step6 产物路径与上文参数一致。

```bash
python run_step1.py
python run_step2.py --clinical ./clinical.csv --joint_graph
python run_step3.py --joint_graph
python run_step4.py --joint_graph
python run_step5.py --clinical ./clinical.csv --joint_graph_upstream --step4_causal_output ./output/step4
python run_step6.py --inference_all
python -m causalreasoning.run_attribution --patient 10071200 --dataset_root ./dataset
```

## 输出目录

根目录：`output/causalreasoning_output/{patient_id}/`

- **`patient_{patient_id}_attribution.json`**  
  `ite_value`、`slice_importance`、`per_superpixel_importance`、`clinical_contribution`、`image_attribution_summary`、`method` 等。
- **`heatmaps/`**（默认开启可视化时）  
  `slice_{slice_id}_attribution.png`：按超像素着色的叠加图；**无文字**，数值以 JSON 为准。

## 热力图实现要点（`visualizer.py`）

- 每个超像素颜色对应该节点的归因分数；**切片内 min-max 归一化**，缺失节点用该片均值填充。
- 仅在 **超像素 ROI ∩ 非近黑背景** 像素上绘制，大图黑边与结节内空洞保持输出为黑，避免伪彩色泄漏到背景。

## 注意事项

- 必须先成功运行 Step6；否则归因会 **`FileNotFoundError` / `RuntimeError`**。
- Step2 未加载成功时，`per_superpixel_importance` 为空，热力图通常会无法生成或无有效图层，请确认 `step2_slice_graphs.pkl` 路径与患者键一致。
- JSON 中对用户展示 **`ite_value`**；与 Step6 pickle 字段 **`IC`** 语义一致（向后兼容）。

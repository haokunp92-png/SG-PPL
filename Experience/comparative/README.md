# comparative：对比实验（生存分析模块替换）

把主流水线的 **生存分析模块** 换成标准基线，前端超像素分支由 **输入路径** 决定，
指标以 **C-index** 为主。

## 设计约定

| 约定 | 说明 |
|------|------|
| 输入 | 多模态、**治疗无关**：`[图像特征(d_img), ajcc, age]`，不含 `youdao` |
| 为什么不含治疗 | 关联模型只能估计 as-treated 结局；保持治疗无关输入后，基线与 DH-CaS 用同一套协变量，判别力比较才是 apples-to-apples |
| 可比性 | Cox 部分似然与 C-index 直接复用 `pipeline.step6_causal_model`，与主流水线逐位一致 |
| 划分 | 传 `--step5_output` 时复用主流水线 `train/val/test`，保证逐例一致；否则用同样的 `seed → permutation → 70/15/15` 逻辑自建 |
| 训练配置 | AdamW + 梯度裁剪 + ReduceLROnPlateau + 按验证集 C-index 选优，与 step6 一致，使差异尽量只来自模型结构 |

**能报什么、不能报什么**：DeepSurv / Cox 是单一风险头且不含治疗，**无法产出 ITE / PITE / 反事实曲线**，
这一格在因果能力表里就是 `N/A by construction`。T-learner 与 DH-CaS 则**能**做反事实
（前者靠两臂各自的模型互相预测对方臂），只是本模块当前只报 C-index。

T-learner 的治疗变量同样不进特征：它只决定样本落到哪一臂，协变量仍是治疗无关的那一套。

## 前端如何切换

`--step4_output` 指向哪个前端的 step4 产物，就是哪一条前端线路：

| 线路 | `--step4_output` |
|------|------------------|
| 我的超像素（FCN） | `./output/step4` |
| SLIC | `./SLIC_function/output/step4`（需先跑 SLIC 分支的 Step1–4） |

SLIC 分支产物由 `SLIC_function/run_slic_pipeline.py` 生成，**需要原始图像**：

```bash
cd SLIC_function
pip install opencv-python scikit-image          # 本机缺这两个依赖
py run_slic_pipeline.py -i <原图根目录> -d ./dataset --run-output ./output -c ../clinical.csv
```

## 运行

```bash
# SLIC + DeepSurv
py -m comparative.run_deepsurv --step4_output ./SLIC_function/output/step4 \
    --step5_output ./output/step5 --tag slic_deepsurv

# SLIC + Cox（λ 在网格上按验证集 C-index 选优）
py -m comparative.run_cox --step4_output ./SLIC_function/output/step4 \
    --step5_output ./output/step5 --tag slic_cox

# SLIC + T-learner（两臂各一个 DeepSurv，互不共享参数）
py -m comparative.run_tlearner --step4_output ./SLIC_function/output/step4 \
    --step5_output ./output/step5 --tag slic_tlearner

# 评估集只有 34 例，噪声大；DeepSurv 可重复多个种子报 mean±std
py -m comparative.run_deepsurv --step4_output ./SLIC_function/output/step4 \
    --step5_output ./output/step5 --n_repeats 5 --tag slic_deepsurv
```

`--step5_output` 建议一律传主项目的 `./output/step5`：两条前端是同一批 217 例、
`sorted(pid)` 顺序一致，复用同一份 151/32/34 划分能保证各格逐例可比。

Windows 控制台若出现中文乱码：`$env:PYTHONIOENCODING='utf-8'`。

### 通用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--step4_output` | `./output/step4` | 患者级图像特征目录，决定前端 |
| `--step5_output` | 无 | 复用主流水线划分（强烈建议传） |
| `--input_mode` | `concat` | `concat`=图像⊕ajcc⊕age（基线标准喂法）；`joint_causal`=复用 step5 门控融合输出 |
| `--clinical` | `./clinical.csv` | 临床表 |
| `--output` | `./output/comparative` | 结果目录 |
| `--tag` | 自动 | 结果文件名前缀 |

### DeepSurv 专有

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hidden` | `64,32` | 隐藏层维度 |
| `--dropout` | `0.4` | Dropout |
| `--weight_decay` | `1e-4` | AdamW 权重衰减（原文的 ridge） |
| `--epochs` / `--lr` | `600` / `1e-3` | 与 step6 默认一致 |
| `--n_repeats` | `1` | 重复训练次数（种子递增，划分不变） |

### Cox 专有

| 参数 | 默认 | 说明 |
|------|------|------|
| `--solver` | `lbfgs` | `lbfgs`=解到收敛的 penalized MLE；`adamw`=与神经基线严格同训练协议 |
| `--l2_penalty` | 无 | 固定 λ；不设则在 `--l2_grid` 上按验证集 C-index 选优 |
| `--l2_grid` | `0,1,3,10,30,100,300` | λ 候选网格 |
| `--max_iter` | `500` | 单轮 L-BFGS 最大迭代（内部最多重复 8 轮至目标函数稳定） |
| `--top_k` | `10` | 打印系数绝对值最大的前 K 个特征 |

Cox 的目标函数 `−ℓ_Cox(w) + λ‖w‖²` 在 `w` 上是凸的，默认用 L-BFGS（strong-Wolfe）
解到收敛，即真正的 penalized partial-likelihood MLE。**这点不能省**：若用一阶优化器
跑固定轮数，线性 Cox 会明显欠拟合，基线被人为削弱，对比结论站不住脚。
零初始化 + 凸目标 ⇒ 解唯一、与随机种子无关，所以 Cox 没有 `--n_repeats`。

收敛判据用**相对**梯度范数 `‖g‖/max(1,|f|) < 1e-4`，因为部分似然是对事件求和的，
绝对阈值会随 n 失去意义。λ 很小时（0/1/3）会如实报告"未收敛"——66 维配 n_train=151，
弱惩罚下的 Cox 本就接近可分、病态，这些 λ 一般也不会被选中。

### T-learner 专有

| 参数 | 默认 | 说明 |
|------|------|------|
| `--base_learner` | `deepsurv` | 每臂的基学习器；`deepsurv` 与 DH-CaS 容量可比，是主要对照，`cox` 为线性版本 |
| `--hidden` / `--dropout` / `--epochs` / `--lr` | 同 DeepSurv | `base_learner=deepsurv` 时生效，每臂独立训练 |
| `--l2_penalty` | `10.0` | `base_learner=cox` 时每臂的**固定** ridge λ |
| `--no_center_risk` | 关 | 关闭各臂风险中心化，仅用于诊断 |
| `--n_repeats` | `1` | 同 DeepSurv |

`base_learner=cox` 不做逐臂 λ 网格搜索：每臂验证集只有 19/13 例，在这上面选 λ 纯属拟合噪声，
固定 λ 更稳也更诚实。

**跨臂风险尺度**（读结果前必须知道）：Cox 部分似然对风险分数的常数平移不变，两个**独立**
拟合的模型偏移量任意且互不相关，因此"把两臂混在一起算一个 C-index"本身没有定义。
默认按各臂训练集均值中心化，消掉这个任意常数，让合并指标至少良定义；但尺度差异消不掉，
所以同时报告各臂**组内** C-index（`c_index_by_arm`）——组内比较不涉及跨臂，才是无歧义的。
DH-CaS 没这个问题：两个头共用一个 pooled 部分似然，天然同尺度，这本身就是共享编码器的一项好处。

`--input_mode concat` 是默认值，因为基线不该借用主流水线的门控交互融合，
否则"我的融合"这一贡献会被让给基线；`joint_causal` 保留用于只想隔离生存头的补充实验。

## 输出

`--output`（默认 `./output/comparative`）下：

- `{tag}_metrics.json`：`c_index_train/val/test`（`--n_repeats>1` 时附 `_std`）、
  逐轮 loss 与验证 C-index、`data_meta`（输入模式、划分来源、事件率）、
  `supports_counterfactual`（DeepSurv/Cox 为 `false`，T-learner 为 `true`）。
  T-learner 另有 `c_index_by_arm`、`n_by_arm`、`risk_offset`、两臂各自的 `arm_histories`
- `{tag}_model.pt`

## 文件

| 文件 | 作用 |
|------|------|
| `data.py` | 装载 step4 图像特征 + clinical.csv，组装治疗无关多模态输入，复用/重建划分，训练集统计量标准化 |
| `trainer.py` | **各基线共用**的训练循环（AdamW + 梯度裁剪 + ReduceLROnPlateau + 按验证集 C-index 选优），以及 C-index 封装 |
| `report.py` | 结果汇总与落盘，保证各基线产出同一套 JSON 字段 |
| `deepsurv.py` | DeepSurv 模型（Linear→BN→SELU→Dropout ×L → Linear(1)） |
| `cox.py` | 线性 Cox + ridge，L-BFGS 精确求解与 λ 网格选优，系数汇总 |
| `tlearner.py` | 两臂子数据集切分（保持原划分归属）、分臂拟合、风险中心化、事实/组内 C-index |
| `run_deepsurv.py` / `run_cox.py` / `run_tlearner.py` | CLI 入口 |

训练循环之所以抽到 `trainer.py`，是为了让各格的差异**只来自风险函数的形式**，
而不来自优化器、学习率调度或选优准则的细节差异。

## 已知数据问题（会影响所有格子）

1. `clinical.csv` **无 `event` 列** → 217 例全部按 event=1 处理，即无删失。C-index 与 IBS 都依赖删失指示。
2. `output/step4` 的患者图像特征接近退化（患者两两相关 ≈ 1.0，95% 方差有效秩 2/64），
   前端差异在这种特征上难以体现。
3. `output/step5` 为旧版 schema（无 `joint_feat_causal`），用 `joint_feat` 可完全反推 `youdao`（AUC=1.000）。
   本模块默认 `concat`，不受该泄漏影响；划分仍可安全复用。
4. T-learner 分臂后训练集只剩 66（youdao=0）与 85（youdao=1）例，而 `d_in=66`，arm0 已是 n≈d 的病态情形；
   验证集更是只有 19/13 例。这是 T-learner 的固有代价（也正是共享编码器想解决的问题），
   但也意味着这一格的绝对数值噪声很大。

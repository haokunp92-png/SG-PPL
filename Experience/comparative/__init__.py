"""
对比实验（comparative baselines）

将主流水线的生存分析模块替换为标准基线（DeepSurv / Cox / T-learner），
前端超像素分支由输入路径决定（我的超像素 或 SLIC），指标以 C-index 为主。

基线均使用 **治疗无关** 的多模态输入（图像特征 + ajcc + age，不含 youdao），
与主流水线 DH-CaS 的 joint_feat_causal 保持同一套协变量。
"""

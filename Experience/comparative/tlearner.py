"""
T-learner 基线（Künzel et al., 2019 的分臂建模）：对两个治疗臂**各自独立**拟合一个
结局模型，互不共享参数，也不联合训练。

与主流水线 DH-CaS 的关键差异（即本对比想隔离的变量）：
- DH-CaS 是共享编码器 + 两个治疗条件头，用**同一个**在事实臂上的 Cox 部分似然联合训练；
- T-learner 的两臂完全独立，各自只看到本臂样本（本数据下训练集仅 66 / 85 例，
  而 d_in=66），因此数据效率明显更低。共享表示是否值得，就由这一格来回答。

治疗变量仍**不作为输入特征**：它只决定样本落到哪一臂，协变量本身是治疗无关的
（图像 + ajcc + age），与 Cox / DeepSurv / DH-CaS 用的是同一套。

**风险分数的跨臂可比性**（本实现最需要注意的一点）：Cox 部分似然对风险分数的常数
平移不变，两个**独立**拟合的模型各自的偏移量是任意且互不相关的。把两臂样本混在一起
算一个总的 C-index 时，跨臂比较 r₀(xᵢ) 与 r₁(xⱼ) 本身没有定义。DH-CaS 不存在这个问题
——它两个头共用一个 pooled 部分似然，天然处在同一尺度上。这里的处理是把每一臂的风险
按其**训练集均值**中心化，消掉任意偏移后再合并；这是让合并 C-index 有定义的最小修正，
但并不能对齐两臂的尺度差异，故同时报告各臂**组内** C-index（组内比较不受偏移影响，
是无歧义的）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .cox import DEFAULT_LBFGS_MAX_ITER, fit_cox_lbfgs
from .data import ComparativeDataset
from .deepsurv import DEFAULT_DROPOUT, DEFAULT_HIDDEN, DEFAULT_WEIGHT_DECAY, DeepSurv
from .trainer import SPLITS, compute_c_index, train_survival_model

ARMS = (0, 1)
BASE_LEARNERS = ("deepsurv", "cox")


def arm_dataset(dataset: ComparativeDataset, arm: int) -> ComparativeDataset:
    """取某治疗臂的子数据集，保持各患者原本的 train/val/test 归属。"""
    keep = np.where(dataset.youdao == arm)[0]
    pos = {int(old): new for new, old in enumerate(keep)}
    split = {
        name: np.array(
            [pos[int(i)] for i in dataset.split[name] if int(i) in pos], dtype=np.int64
        )
        for name in SPLITS
    }
    return ComparativeDataset(
        X=dataset.X[keep],
        time=dataset.time[keep],
        event=dataset.event[keep],
        youdao=dataset.youdao[keep],
        patient_ids=[dataset.patient_ids[i] for i in keep],
        split=split,
        feature_names=dataset.feature_names,
        meta={**dataset.meta, "arm": arm, "n_patients": int(len(keep))},
    )


class TLearner(nn.Module):
    """两臂独立模型的容器；offset 用于消掉各臂 Cox 风险的任意常数偏移。"""

    def __init__(self, model_0: nn.Module, model_1: nn.Module):
        super().__init__()
        self.model_0 = model_0
        self.model_1 = model_1
        self.register_buffer("offset", torch.zeros(2))

    def arm_risk(self, x: torch.Tensor, arm: int, centered: bool = True) -> torch.Tensor:
        """指定治疗臂下的 log-risk。"""
        model = self.model_0 if arm == 0 else self.model_1
        r = model(x)
        return r - self.offset[arm] if centered else r

    def factual_risk(
        self, x: torch.Tensor, youdao: torch.Tensor, centered: bool = True
    ) -> torch.Tensor:
        """事实风险：取与实际治疗一致的那一臂的模型输出。"""
        r0 = self.arm_risk(x, 0, centered)
        r1 = self.arm_risk(x, 1, centered)
        return torch.where(youdao.long() == 1, r1, r0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """默认返回 treatment=1 臂的风险（供通用工具调用）。"""
        return self.arm_risk(x, 1)


def _train_arm(
    arm_ds: ComparativeDataset,
    base_learner: str,
    arm: int,
    *,
    hidden: Tuple[int, ...],
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    l2_penalty: float,
    max_iter: int,
    seed: int,
    device: Optional[torch.device],
    verbose: bool,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """在单一治疗臂上拟合基学习器。"""
    if verbose:
        print(
            f"  --- arm {arm} (youdao={arm}): 训练 {len(arm_ds.split['train'])} 例, "
            f"验证 {len(arm_ds.split['val'])} 例, 测试 {len(arm_ds.split['test'])} 例 ---"
        )
    if base_learner == "deepsurv":
        return train_survival_model(
            lambda: DeepSurv(arm_ds.d_in, hidden=tuple(hidden), dropout=dropout),
            arm_ds,
            model_name=f"deepsurv_arm{arm}",
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            l2_penalty=0.0,
            seed=seed,
            device=device,
            verbose=verbose,
            extra_hyperparams={"hidden": list(hidden), "dropout": dropout, "arm": arm},
        )
    # cox 基学习器：各臂验证集仅十余例，不做逐臂 λ 网格搜索，用固定 λ 更稳
    model, info = fit_cox_lbfgs(arm_ds, l2_penalty, max_iter=max_iter, device=device)
    if verbose:
        print(
            f"      Cox λ={l2_penalty:g}, 目标值={info['final_objective']:.4f}, "
            f"相对梯度范数={info['final_rel_grad_norm']:.2e}"
            f"{'' if info['converged'] else '  [警告: 未收敛]'}"
        )
    history = {
        "model": f"cox_arm{arm}",
        "solver": "lbfgs",
        "c_index": info["c_index"],
        "final_objective": info["final_objective"],
        "final_rel_grad_norm": info["final_rel_grad_norm"],
        "converged": info["converged"],
        "hyperparams": {"solver": "lbfgs", "l2_penalty": l2_penalty, "arm": arm},
    }
    return model, history


def train_tlearner(
    dataset: ComparativeDataset,
    base_learner: str = "deepsurv",
    hidden: Tuple[int, ...] = DEFAULT_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    l2_penalty: float = 10.0,
    max_iter: int = DEFAULT_LBFGS_MAX_ITER,
    center_risk: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[TLearner, Dict[str, Any]]:
    """
    分臂拟合 T-learner，并按事实风险计算 C-index。

    Args:
        base_learner: 每臂的基学习器；deepsurv 与 DH-CaS 容量可比，是主要对照
        center_risk: 是否按各臂训练集均值中心化风险（合并 C-index 有定义的前提）
    """
    if base_learner not in BASE_LEARNERS:
        raise ValueError(f"base_learner 须为 {BASE_LEARNERS} 之一，收到 {base_learner}")
    if device is None:
        device = torch.device("cpu")

    arm_ds = {arm: arm_dataset(dataset, arm) for arm in ARMS}
    for arm in ARMS:
        if len(arm_ds[arm].split["train"]) < 2:
            raise ValueError(f"治疗臂 {arm} 的训练样本不足 2 例，无法拟合 T-learner")

    arm_models: Dict[int, nn.Module] = {}
    arm_hist: Dict[str, Any] = {}
    for arm in ARMS:
        model, hist = _train_arm(
            arm_ds[arm],
            base_learner,
            arm,
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            l2_penalty=l2_penalty,
            max_iter=max_iter,
            seed=seed + arm,
            device=device,
            verbose=verbose,
        )
        arm_models[arm] = model
        arm_hist[f"arm{arm}"] = hist

    tlearner = TLearner(arm_models[0], arm_models[1]).to(device)
    tlearner.eval()

    # 各臂用自己训练集样本的均值作为偏移量，消掉 Cox 风险的任意常数
    if center_risk:
        with torch.no_grad():
            for arm in ARMS:
                tr = arm_ds[arm].split["train"]
                x_tr = torch.from_numpy(arm_ds[arm].X[tr]).float().to(device)
                tlearner.offset[arm] = tlearner.arm_risk(x_tr, arm, centered=False).mean()

    X_all = torch.from_numpy(dataset.X).float().to(device)
    y_all = torch.from_numpy(dataset.youdao.astype(np.int64)).to(device)
    with torch.no_grad():
        risk_fact = tlearner.factual_risk(X_all, y_all, centered=center_risk)
    risk_fact = risk_fact.cpu().numpy().astype(float)

    def _c(idx: np.ndarray) -> float:
        if len(idx) < 2:
            return float("nan")
        return float(
            compute_c_index(risk_fact[idx], dataset.time[idx], dataset.event[idx])
        )

    c_index = {name: _c(dataset.split[name]) for name in SPLITS}
    c_index_by_arm: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        c_index_by_arm[f"arm{arm}"] = {}
        for name in SPLITS:
            idx = dataset.split[name]
            idx_arm = idx[dataset.youdao[idx] == arm]
            c_index_by_arm[f"arm{arm}"][name] = _c(idx_arm)

    n_by_arm = {
        f"arm{arm}": {
            name: int((dataset.youdao[dataset.split[name]] == arm).sum())
            for name in SPLITS
        }
        for arm in ARMS
    }

    if verbose:
        print(
            f"  风险偏移量 offset = [{float(tlearner.offset[0]):+.4f}, "
            f"{float(tlearner.offset[1]):+.4f}]"
            f"{'' if center_risk else '  (未中心化)'}"
        )

    history = {
        "model": "tlearner",
        "base_learner": base_learner,
        "c_index": c_index,
        "c_index_by_arm": c_index_by_arm,
        "n_by_arm": n_by_arm,
        "risk_offset": [float(tlearner.offset[0]), float(tlearner.offset[1])],
        "center_risk": bool(center_risk),
        "arm_histories": arm_hist,
        "hyperparams": {
            "base_learner": base_learner,
            "hidden": list(hidden) if base_learner == "deepsurv" else None,
            "dropout": dropout if base_learner == "deepsurv" else None,
            "epochs": epochs if base_learner == "deepsurv" else None,
            "lr": lr if base_learner == "deepsurv" else None,
            "weight_decay": weight_decay if base_learner == "deepsurv" else None,
            "l2_penalty": l2_penalty if base_learner == "cox" else None,
            "seed": seed,
        },
    }
    return tlearner, history

"""
Cox 比例风险基线：log-risk 为协变量的**线性**函数 r(x) = wᵀx，
用 Cox 部分似然（Breslow 处理并结）拟合，附 ridge 惩罚。

目标函数::

    min_w  −ℓ_Cox(w) + λ‖w‖²

其中 ℓ_Cox 与主流水线 step6 用的是同一个实现（对事件求和，非取均值），
因此有意义的 λ 在 O(1)–O(100) 量级，而非 glmnet 那种均值尺度下的 0.01–1。

**求解器**：目标函数在 w 上是凸的，默认用 L-BFGS（strong-Wolfe 线搜索）解到
收敛，即真正的 penalized partial-likelihood MLE。这一点很重要：若改用 AdamW
之类的一阶随机优化器 + 固定轮数，线性 Cox 会明显欠拟合，基线被人为削弱，
对比结果站不住脚。`--solver adamw` 保留下来，仅供需要与神经基线严格同协议时使用。

**λ 的选择**：d_in=66（图像 64 + ajcc + age）配 n_train≈151，无惩罚的 Cox 不稳定。
默认在 λ 网格上按**验证集 C-index** 选优（与 glmnet 用 CV 选 λ 同理），
并列时取较大的 λ（更强正则）。

与其它格子的关系：
- 相对 DeepSurv：唯一差别是风险函数由 MLP 退化为线性。
- 相对 DH-CaS：单一风险头、且不含治疗变量，**无法做反事实推理**，仅报告 C-index。

线性 Cox 零初始化 + 凸目标 ⇒ 解唯一，结果与随机种子无关。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .trainer import SPLITS, c_index_of, cox_partial_likelihood, train_survival_model

DEFAULT_L2_GRID: Tuple[float, ...] = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
DEFAULT_LBFGS_MAX_ITER = 500
# 每次 optimizer.step 最多走 max_iter 次，故重复若干轮直到目标函数稳定
LBFGS_ROUNDS = 8
# 收敛判据用相对梯度范数 ‖g‖/max(1,|f|)，因为部分似然是对事件求和的
REL_GRAD_TOL = 1e-4


class CoxLinear(nn.Module):
    """线性 Cox 风险函数 r(x) = wᵀx；无截距（部分似然对常数平移不变）。"""

    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (N,) 的 Cox log-risk 分数。"""
        return self.linear(x).squeeze(-1)

    @property
    def coef(self) -> np.ndarray:
        return self.linear.weight.detach().cpu().numpy().ravel()


def summarize_coefficients(
    model: CoxLinear, feature_names: Sequence[str], top_k: int = 10
) -> Dict[str, Any]:
    """
    汇总线性 Cox 的系数。系数正 = 风险升高 = 预后更差；
    输入已按训练集 z-score 标准化，故系数可直接横向比较重要性。
    """
    w = model.coef
    names = list(feature_names)[: len(w)]
    order = np.argsort(-np.abs(w))
    img_idx = [i for i, nm in enumerate(names) if nm.startswith("img_")]
    cli_idx = [i for i, nm in enumerate(names) if not nm.startswith("img_")]
    return {
        "n_coef": int(len(w)),
        "l2_norm": float(np.linalg.norm(w)),
        "l2_norm_image": float(np.linalg.norm(w[img_idx])) if img_idx else 0.0,
        "l2_norm_clinical": float(np.linalg.norm(w[cli_idx])) if cli_idx else 0.0,
        "top_by_abs": [
            {
                "feature": names[i],
                "coef": float(w[i]),
                "hazard_ratio": float(np.exp(w[i])),
            }
            for i in order[:top_k]
        ],
        "coefficients": {names[i]: float(w[i]) for i in range(len(w))},
    }


def _tensors(dataset, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(dataset.X[dataset.split[name]]).float().to(device)
        for name in SPLITS
    }


def fit_cox_lbfgs(
    dataset,
    l2_penalty: float,
    max_iter: int = DEFAULT_LBFGS_MAX_ITER,
    device: Optional[torch.device] = None,
) -> Tuple[CoxLinear, Dict[str, Any]]:
    """在训练集上把 ridge-Cox 解到收敛（凸问题，L-BFGS + strong-Wolfe）。"""
    if device is None:
        device = torch.device("cpu")  # 线性模型规模极小，CPU 更快且结果稳定
    X = _tensors(dataset, device)
    tr = dataset.split["train"]
    t = torch.from_numpy(dataset.time[tr]).float().to(device)
    e = torch.from_numpy(dataset.event[tr].astype(np.float32)).to(device)

    model = CoxLinear(dataset.d_in).to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad()
        loss = cox_partial_likelihood(model(X["train"]), t, e)
        if l2_penalty > 0:
            loss = loss + l2_penalty * (model.linear.weight ** 2).sum()
        loss.backward()
        return loss

    # 多轮 L-BFGS 直到目标函数不再改善（单次 step 有 max_iter 上限）
    final_loss = float("nan")
    for _ in range(LBFGS_ROUNDS):
        loss = float(optimizer.step(closure).item())
        if np.isfinite(final_loss) and abs(final_loss - loss) <= 1e-10 * max(
            1.0, abs(final_loss)
        ):
            final_loss = loss
            break
        final_loss = loss
    grad_norm = float(model.linear.weight.grad.detach().norm().item())
    # 部分似然是对事件求和的，梯度范数须按目标函数量级判定，不能用绝对阈值
    rel_grad_norm = grad_norm / max(1.0, abs(final_loss))

    c_index = {
        name: c_index_of(
            model,
            X[name],
            dataset.time[dataset.split[name]],
            dataset.event[dataset.split[name]],
        )
        for name in SPLITS
    }
    info = {
        "l2_penalty": float(l2_penalty),
        "final_objective": final_loss,
        "final_grad_norm": grad_norm,
        "final_rel_grad_norm": rel_grad_norm,
        "converged": bool(rel_grad_norm < REL_GRAD_TOL),
        "c_index": c_index,
    }
    return model, info


def select_l2_by_val(
    dataset,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
    max_iter: int = DEFAULT_LBFGS_MAX_ITER,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[CoxLinear, Dict[str, Any]]:
    """
    在 λ 网格上按验证集 C-index 选优；并列时取较大的 λ（更强正则）。
    验证集不足 2 例时退回网格中位数，并在结果里标注。
    """
    grid = sorted(float(v) for v in l2_grid)
    path: List[Dict[str, Any]] = []
    best = None
    has_val = len(dataset.split["val"]) >= 2

    for lam in grid:
        model, info = fit_cox_lbfgs(dataset, lam, max_iter=max_iter, device=device)
        path.append(
            {
                "l2_penalty": lam,
                "val_c_index": info["c_index"]["val"],
                "train_c_index": info["c_index"]["train"],
                "converged": info["converged"],
            }
        )
        if verbose:
            vc = info["c_index"]["val"]
            vc_s = "  n/a " if np.isnan(vc) else f"{vc:.4f}"
            print(
                f"    λ={lam:<7g} train_C={info['c_index']['train']:.4f} "
                f"val_C={vc_s}  |w|₂={np.linalg.norm(model.coef):.4f}"
                f"{'' if info['converged'] else '  [未收敛]'}"
            )
        if has_val and not np.isnan(info["c_index"]["val"]):
            score = info["c_index"]["val"]
            # >= 使并列时选到更大的 λ（grid 升序遍历）
            if best is None or score >= best[1]:
                best = (model, score, info)

    if best is None:
        lam = grid[len(grid) // 2]
        model, info = fit_cox_lbfgs(dataset, lam, max_iter=max_iter, device=device)
        selection = {
            "method": "fallback_median_lambda",
            "reason": "验证集不足 2 例，无法按 C-index 选 λ",
            "selected_l2_penalty": lam,
        }
    else:
        model, _, info = best
        selection = {
            "method": "val_c_index",
            "selected_l2_penalty": info["l2_penalty"],
        }
    selection["l2_grid"] = grid
    selection["path"] = path
    return model, {**info, "l2_selection": selection}


def train_cox(
    dataset,
    solver: str = "lbfgs",
    l2_penalty: Optional[float] = None,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
    max_iter: int = DEFAULT_LBFGS_MAX_ITER,
    epochs: int = 600,
    lr: float = 1e-2,
    lr_patience: int = 15,
    lr_factor: float = 0.5,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    拟合 Cox 基线。

    Args:
        solver: lbfgs（默认，解到收敛的 penalized MLE）或 adamw（与神经基线同协议）
        l2_penalty: 指定则用该 λ；为 None 且 solver=lbfgs 时在 l2_grid 上按验证集选优
    """
    if solver not in ("lbfgs", "adamw"):
        raise ValueError(f"solver 须为 'lbfgs' 或 'adamw'，收到 {solver}")

    if solver == "lbfgs":
        if l2_penalty is None:
            if verbose:
                print(f"  Cox (L-BFGS): 在 λ 网格上按验证集 C-index 选优, d_in={dataset.d_in}")
            model, info = select_l2_by_val(
                dataset, l2_grid=l2_grid, max_iter=max_iter, device=device, verbose=verbose
            )
        else:
            if verbose:
                print(f"  Cox (L-BFGS): λ={l2_penalty}, d_in={dataset.d_in}")
            model, info = fit_cox_lbfgs(
                dataset, float(l2_penalty), max_iter=max_iter, device=device
            )
            info["l2_selection"] = {
                "method": "fixed",
                "selected_l2_penalty": float(l2_penalty),
            }
        if verbose:
            sel = info["l2_selection"]
            print(
                f"  选定 λ={sel['selected_l2_penalty']:g} ({sel['method']}), "
                f"目标值={info['final_objective']:.4f}, "
                f"相对梯度范数={info['final_rel_grad_norm']:.2e}"
                f"{'' if info['converged'] else '  [警告: 未收敛]'}"
            )
        history: Dict[str, Any] = {
            "model": "cox",
            "solver": "lbfgs",
            "c_index": info["c_index"],
            "final_objective": info["final_objective"],
            "final_grad_norm": info["final_grad_norm"],
            "final_rel_grad_norm": info["final_rel_grad_norm"],
            "converged": info["converged"],
            "l2_selection": info["l2_selection"],
            "hyperparams": {
                "solver": "lbfgs",
                "l2_penalty": info["l2_selection"]["selected_l2_penalty"],
                "max_iter": max_iter,
            },
        }
    else:
        lam = DEFAULT_L2_GRID[3] if l2_penalty is None else float(l2_penalty)
        model, history = train_survival_model(
            lambda: CoxLinear(dataset.d_in),
            dataset,
            model_name="cox",
            epochs=epochs,
            lr=lr,
            weight_decay=0.0,
            l2_penalty=lam,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        history["solver"] = "adamw"
        history["l2_selection"] = {"method": "fixed", "selected_l2_penalty": lam}

    history["coefficients_summary"] = summarize_coefficients(model, dataset.feature_names)
    return model, history

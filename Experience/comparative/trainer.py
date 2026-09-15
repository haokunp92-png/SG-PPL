"""
对比实验的统一训练协议。

所有基线共用这一个训练循环，使各格之间的差异**只来自风险函数的形式**
（Cox 为线性、DeepSurv 为 MLP、DH-CaS 为共享编码器 + 双头），
而不来自优化器、学习率调度、选优准则等训练细节。

与主流水线 step6 对齐的部分：AdamW、梯度裁剪 max_norm=1.0、
ReduceLROnPlateau(mode=min, factor, patience)、按验证集 C-index 选最优权重。
Cox 部分似然与 C-index 直接复用 pipeline.step6_causal_model，保证指标逐位一致。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.step6_causal_model import (  # noqa: E402
    compute_c_index,
    cox_partial_likelihood,
)

SPLITS = ("train", "val", "test")


def risk_numpy(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """该批样本的 Cox log-risk 分数 (N,)。"""
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy().astype(float)


def c_index_of(
    model: nn.Module, X: torch.Tensor, time: np.ndarray, event: np.ndarray
) -> float:
    """Harrell C-index（风险越高预后越差）；样本不足 2 例时返回 nan。"""
    if len(time) < 2:
        return float("nan")
    return float(compute_c_index(risk_numpy(model, X), time, event))


def _l2_of(model: nn.Module) -> torch.Tensor:
    return sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)


def train_survival_model(
    model_factory: Callable[[], nn.Module],
    dataset,
    *,
    model_name: str,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    l2_penalty: float = 0.0,
    lr_patience: int = 15,
    lr_factor: float = 0.5,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    extra_hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    全批量训练一个生存风险模型（n≈150，无需 mini-batch）。

    Args:
        model_factory: 无参构造函数；在设定随机种子之后调用，保证初始化可复现
        weight_decay: 交给 AdamW 的解耦权重衰减
        l2_penalty: 显式加进目标函数的 L2 项（Cox 的 ridge 惩罚用这个，更贴近
                    "penalized Cox" 的定义，也便于在论文里写清目标函数）

    Returns:
        (model, history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    tensors = {
        name: torch.from_numpy(dataset.X[dataset.split[name]]).float().to(device)
        for name in SPLITS
    }
    tr = dataset.split["train"]
    t_train = torch.from_numpy(dataset.time[tr]).float().to(device)
    e_train = torch.from_numpy(dataset.event[tr].astype(np.float32)).to(device)

    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )

    has_val = len(dataset.split["val"]) >= 2
    best_val_c = -1.0
    best_state = None
    best_epoch = 0
    losses: List[float] = []
    val_hist: List[Dict[str, Any]] = []

    if verbose:
        print(
            f"  {model_name}: d_in={dataset.d_in}, 设备={device}, epochs={epochs}, "
            f"lr={lr}, weight_decay={weight_decay}, l2_penalty={l2_penalty}"
        )
        print(
            f"  训练 {len(tr)} 例, 验证 {len(dataset.split['val'])} 例, "
            f"测试 {len(dataset.split['test'])} 例"
        )
        if not has_val:
            print("  [warn] 验证集不足 2 例，无法按 C-index 选优，将使用最后一轮权重")

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        risk = model(tensors["train"])
        loss = cox_partial_likelihood(risk, t_train, e_train)
        if l2_penalty > 0:
            loss = loss + l2_penalty * _l2_of(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = float(loss.item())
        losses.append(train_loss)
        scheduler.step(train_loss)

        if has_val:
            val_c = c_index_of(
                model,
                tensors["val"],
                dataset.time[dataset.split["val"]],
                dataset.event[dataset.split["val"]],
            )
            val_hist.append({"epoch": ep + 1, "val_c_index": val_c})
            if val_c > best_val_c:
                best_val_c = val_c
                best_epoch = ep + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % max(1, epochs // 10) == 0:
            msg = (
                f"  Epoch {ep + 1}/{epochs}, loss={train_loss:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            if has_val:
                msg += f", val_C-index={val_hist[-1]['val_c_index']:.4f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(
                f"  已加载验证集 C-index 最优权重 "
                f"(epoch={best_epoch}, val_C-index={best_val_c:.4f})"
            )

    c_index = {
        name: c_index_of(
            model,
            tensors[name],
            dataset.time[dataset.split[name]],
            dataset.event[dataset.split[name]],
        )
        for name in SPLITS
    }

    hyperparams = {
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "l2_penalty": l2_penalty,
        "lr_patience": lr_patience,
        "lr_factor": lr_factor,
        "seed": seed,
    }
    if extra_hyperparams:
        hyperparams.update(extra_hyperparams)

    history = {
        "model": model_name,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "val_c_index_history": val_hist,
        "best_epoch": best_epoch if best_state is not None else None,
        "best_val_c_index": float(best_val_c) if best_state is not None else None,
        "c_index": c_index,
        "hyperparams": hyperparams,
    }
    return model, history

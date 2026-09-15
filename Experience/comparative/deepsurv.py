"""
DeepSurv 基线（Katzman et al., 2018）：以多层感知机替代 Cox 的线性风险函数，
仍用 Cox 部分似然监督，输出单一 log-risk 分数。

与主流水线 DH-CaS 的关键差异（即本对比想隔离的变量）：
- 单一风险头，不区分治疗臂；治疗变量不进入输入，故 **无法做反事实推理**，
  只能给出 as-treated 的风险排序，因此仅报告 C-index。

训练协议见 trainer.train_survival_model，与 Cox 基线完全共用。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .trainer import train_survival_model

DEFAULT_HIDDEN = (64, 32)
DEFAULT_DROPOUT = 0.4
DEFAULT_WEIGHT_DECAY = 1e-4


class DeepSurv(nn.Module):
    """
    DeepSurv 风险网络：[Linear → BatchNorm → SELU → Dropout] × L → Linear(→1)。

    输出层不加偏置：Cox 部分似然对风险分数的常数平移不变，偏置不可辨识。
    """

    def __init__(
        self,
        d_in: int,
        hidden: Tuple[int, ...] = DEFAULT_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_prev = d_in
        for d_h in hidden:
            layers += [
                nn.Linear(d_prev, d_h),
                nn.BatchNorm1d(d_h),
                nn.SELU(),
                nn.Dropout(dropout),
            ]
            d_prev = d_h
        layers.append(nn.Linear(d_prev, 1, bias=False))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (N,) 的 Cox log-risk 分数。"""
        return self.net(x).squeeze(-1)


def train_deepsurv(
    dataset,
    hidden: Tuple[int, ...] = DEFAULT_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    lr_patience: int = 15,
    lr_factor: float = 0.5,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """训练 DeepSurv；正则化走 AdamW 的权重衰减（原文的 ridge）。"""
    return train_survival_model(
        lambda: DeepSurv(dataset.d_in, hidden=tuple(hidden), dropout=dropout),
        dataset,
        model_name="deepsurv",
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        l2_penalty=0.0,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        seed=seed,
        device=device,
        verbose=verbose,
        extra_hyperparams={"hidden": list(hidden), "dropout": dropout},
    )

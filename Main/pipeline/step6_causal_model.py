"""
第六步：因果模型构建与反事实推理

基于 CXGNN 的因果模型（NNModel）作为共享编码层，
双头输出 head_0（youdao=0）/ head_1（youdao=1）用于生存风险预测。
Cox 部分似然损失，反事实推理，个体治疗效应（ITE，基于 RMST 的相对变化）及 bootstrap 置信区间。

ITE 计算公式：ITE = (RMST_counterfactual - RMST_factual) / RMST_factual（存为小数，×100 为百分比）
RMST = ∫₀^τ S(t) dt，限制性平均生存时间，τ=5 年（60 月）。
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    _HAS_TORCH = False

# 将 CXGNN 加入路径以导入其因果模型
_CXGNN_ROOT = Path(__file__).resolve().parent.parent / "CXGNN"
if _CXGNN_ROOT.exists() and str(_CXGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CXGNN_ROOT))

# 默认超参（与 CXGNN NNModel 一致：h_size=32, h_layers=2）
D_JOINT = 128
D_HIDDEN = 64
CXGNN_H_SIZE = 32
CXGNN_H_LAYERS = 2

# Landmark τ（月）：与 RMST / ITE 一致，默认 60 月 = 5 年
RMST_TAU_MONTHS = 60

# IBS（综合 Brier）：在 [0, τ] 上对 BS(t) 做梯形积分，IBS = (1/τ)∫_0^τ BS(t)dt
IBS_TRAPZ_N_POINTS = 121


def _first_n_unique_patient_ids_in_aligned_order(
    aligned: List[Dict[str, Any]], n: int
) -> set:
    """按 aligned 中出现顺序取前 n 个不同的 patient_id（用于子集实验）。"""
    if n <= 0:
        raise ValueError("n 须为正整数")
    ordered: List[Any] = []
    seen = set()
    for row in aligned:
        pid = row["patient_id"]
        if pid not in seen:
            ordered.append(pid)
            seen.add(pid)
            if len(ordered) >= n:
                break
    return set(ordered)


def _get_cxgnn_nnmodel():
    """导入 CXGNN 的 NNModel（需先安装 torch）"""
    try:
        from model import alg1
        return getattr(alg1, "NNModel", None)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


if _HAS_TORCH:
    _NNModelBase = _get_cxgnn_nnmodel()

    class DualHeadCausalSurvival(nn.Module):
        """
        双头因果生存模型，基于 CXGNN 的 NNModel 架构。
        共享编码层使用 NNModel 的 MLP 结构（Xavier 初始化、ReLU），
        双头分别预测 youdao=0/1 时的生存风险（Cox 风险分数）。
        head_hidden>0 时使用 2 层 head，增强治疗效应表达能力。
        """

        def __init__(self, d_in: int = D_JOINT, d_hidden: int = D_HIDDEN, head_hidden: int = 0):
            super().__init__()
            if _NNModelBase is not None:
                self._use_cxgnn = True
                self.nnmodel = _NNModelBase(
                    input_size=d_in,
                    output_size=d_hidden,
                    h_size=d_hidden,
                    h_layers=CXGNN_H_LAYERS,
                )
            else:
                self._use_cxgnn = False
                self.nnmodel = None
                self._encoder = nn.Sequential(
                    nn.Linear(d_in, d_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            if head_hidden > 0:
                self.head_0 = nn.Sequential(
                    nn.Linear(d_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, 1),
                )
                self.head_1 = nn.Sequential(
                    nn.Linear(d_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, 1),
                )
            else:
                self.head_0 = nn.Linear(d_hidden, 1)
                self.head_1 = nn.Linear(d_hidden, 1)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))

        def _get_encoder_output(self, x):
            if self._use_cxgnn and self.nnmodel is not None:
                return self.nnmodel.nn(x)  # 不含 sigmoid 的编码输出
            return self._encoder(x)

        def forward(self, x):
            h = self._get_encoder_output(x)
            risk_0 = self.head_0(h).squeeze(-1)
            risk_1 = self.head_1(h).squeeze(-1)
            return risk_0, risk_1

        @property
        def encoder(self):
            """供 balance_reg 使用：返回编码器模块"""
            if self._use_cxgnn and self.nnmodel is not None:
                return self.nnmodel.nn
            return self._encoder
else:
    DualHeadCausalSurvival = None


def cox_partial_likelihood(
    risk: "torch.Tensor",
    time: "torch.Tensor",
    event: "torch.Tensor",
) -> "torch.Tensor":
    """
    Cox 部分似然损失（负对数）。
    risk: (N,) 风险分数（log hazard ratio）
    time: (N,) 生存时间
    event: (N,) 事件指示器 1=发生 0=删失
    """
    if not _HAS_TORCH:
        raise ImportError("需要 torch")
    # 按时间降序排列（事件优先）
    order = torch.argsort(-time)
    risk = risk[order]
    time = time[order]
    event = event[order]

    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8)
    uncensored_likelihood = risk - log_risk
    return -torch.sum(uncensored_likelihood * event)


def _eval_cox_loss(
    model: "nn.Module",
    data: List[Dict[str, Any]],
    device: "torch.device",
    balance_reg: float = 0.0,
) -> float:
    """Compute Cox partial likelihood loss on a dataset (no grad)."""
    model.eval()
    feat_key = "joint_feat_causal" if any("joint_feat_causal" in s for s in data) else "joint_feat"
    X = torch.from_numpy(np.stack([s[feat_key] for s in data])).float().to(device)
    youdao_arr = np.array([s["youdao"] for s in data])
    time_arr = np.array([s["deadtime"] for s in data])
    event_arr = np.array([s["event"] for s in data], dtype=np.float32)
    with torch.no_grad():
        risk_0, risk_1 = model(X)
        risk_factual = torch.where(
            torch.from_numpy(youdao_arr).long().to(device) == 1,
            risk_1,
            risk_0,
        )
        time_t = torch.from_numpy(time_arr).float().to(device)
        event_t = torch.from_numpy(event_arr).float().to(device)
        loss = cox_partial_likelihood(risk_factual, time_t, event_t)
        if balance_reg > 0:
            h = model.encoder(X)
            mask0 = youdao_arr == 0
            mask1 = youdao_arr == 1
            if mask0.sum() > 0 and mask1.sum() > 0:
                h0 = h[torch.from_numpy(mask0).to(device)].mean(dim=0)
                h1 = h[torch.from_numpy(mask1).to(device)].mean(dim=0)
                loss = loss + balance_reg * torch.nn.functional.mse_loss(h0, h1)
    return loss.item()


def _eval_factual_risks_numpy(
    model: "nn.Module",
    data: List[Dict[str, Any]],
    device: "torch.device",
) -> np.ndarray:
    """Batch factual risk scores (N,)。No grad."""
    if not data:
        return np.array([], dtype=float)
    model.eval()
    feat_key = "joint_feat_causal" if any("joint_feat_causal" in s for s in data) else "joint_feat"
    X = torch.from_numpy(np.stack([s[feat_key] for s in data])).float().to(device)
    youdao_arr = np.array([s["youdao"] for s in data])
    with torch.no_grad():
        risk_0, risk_1 = model(X)
        risk_factual = torch.where(
            torch.from_numpy(youdao_arr).long().to(device) == 1,
            risk_1,
            risk_0,
        )
    return risk_factual.cpu().numpy().astype(float)


def train_causal_model(
    train_data: List[Dict[str, Any]],
    model: "nn.Module",
    device: "torch.device",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    balance_reg: float = 0.0,
    val_data: Optional[List[Dict[str, Any]]] = None,
    lr_patience: int = 15,
    lr_factor: float = 0.5,
    verbose: bool = True,
    checkpoint_metric: str = "c_index",
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Train dual-head causal survival model.

    checkpoint_metric: 有 val_data 时，用 "c_index"（默认，最大化验证集 C-index）或 "loss"（最小化验证集 Cox loss）选最优权重。
    返回 (losses, train_meta)。
    """
    if not _HAS_TORCH:
        raise ImportError("需要 torch")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )
    losses = []
    feat_key = "joint_feat_causal" if any("joint_feat_causal" in s for s in train_data) else "joint_feat"
    X = torch.from_numpy(np.stack([s[feat_key] for s in train_data])).float().to(device)
    youdao_arr = np.array([s["youdao"] for s in train_data])
    time_arr = np.array([s["deadtime"] for s in train_data])
    event_arr = np.array([s["event"] for s in train_data], dtype=np.float32)
    best_val_loss = float("inf")
    best_val_c_index = -1.0
    best_state = None
    best_epoch = 0
    val_c_index_history: List[Dict[str, Any]] = []
    use_cidx = bool(val_data) and checkpoint_metric == "c_index"
    use_loss_ckpt = bool(val_data) and checkpoint_metric == "loss"
    if val_data and checkpoint_metric not in ("c_index", "loss"):
        raise ValueError(f"checkpoint_metric 须为 'c_index' 或 'loss'，收到: {checkpoint_metric}")
    if verbose:
        print(f"  训练集样本数: {len(train_data)}, 设备: {device}, epochs: {epochs}, weight_decay: {weight_decay}")
        if val_data:
            print(
                f"  验证集样本数: {len(val_data)}, 最优权重依据: "
                f"{'验证集 C-index（最大化）' if use_cidx else '验证集 Cox loss（最小化）'}"
            )
    val_loss: Optional[float] = None
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        risk_0, risk_1 = model(X)
        risk_factual = torch.where(
            torch.from_numpy(youdao_arr).long().to(device) == 1,
            risk_1,
            risk_0,
        )
        time_t = torch.from_numpy(time_arr).float().to(device)
        event_t = torch.from_numpy(event_arr).float().to(device)
        loss = cox_partial_likelihood(risk_factual, time_t, event_t)
        if balance_reg > 0:
            h = model.encoder(X)
            mask0 = youdao_arr == 0
            mask1 = youdao_arr == 1
            if mask0.sum() > 0 and mask1.sum() > 0:
                h0 = h[torch.from_numpy(mask0).to(device)].mean(dim=0)
                h1 = h[torch.from_numpy(mask1).to(device)].mean(dim=0)
                loss = loss + balance_reg * torch.nn.functional.mse_loss(h0, h1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = loss.item()
        losses.append(train_loss)
        scheduler.step(train_loss)
        if val_data:
            val_loss = _eval_cox_loss(model, val_data, device, balance_reg)
            val_cidx = _eval_val_c_index(model, val_data, device)
            val_c_index_history.append({"epoch": ep + 1, "val_c_index": val_cidx, "val_loss": val_loss})
            model.train()
            if use_cidx:
                if val_cidx > best_val_c_index:
                    best_val_c_index = val_cidx
                    best_val_loss = val_loss
                    best_epoch = ep + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif use_loss_ckpt:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = ep + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % max(1, epochs // 10) == 0:
            msg = f"  Epoch {ep + 1}/{epochs}, loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}"
            if val_data and val_loss is not None:
                msg += f", val_loss={val_loss:.4f}, val_C-index={val_cidx:.4f}"
            print(msg)
    train_meta: Dict[str, Any] = {
        "checkpoint_metric": checkpoint_metric if val_data else None,
        "best_epoch": best_epoch if best_state is not None else None,
        "best_val_c_index": float(best_val_c_index) if use_cidx and best_state is not None else None,
        "best_val_loss": float(best_val_loss) if best_state is not None else None,
        "final_epoch": len(losses),
        "val_c_index_history": val_c_index_history if val_data else [],
    }
    if best_state is not None and val_c_index_history:
        for row in val_c_index_history:
            if row["epoch"] == best_epoch:
                train_meta["val_c_index_at_best_checkpoint"] = row["val_c_index"]
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            if use_cidx:
                print(
                    f"  已加载验证集 C-index 最优模型 (epoch={best_epoch}, val_C-index={best_val_c_index:.4f}, val_loss={best_val_loss:.4f})"
                )
            else:
                print(f"  已加载验证集最优模型 (epoch={best_epoch}, val_loss={best_val_loss:.4f})")
    if verbose:
        print(f"  训练完成, 最终 loss={losses[-1]:.4f}")
    return losses, train_meta


def breslow_baseline_hazard(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Breslow 估计基线累积风险 H0(t)。
    按时间升序，在每位事件发生时刻累加 1/sum(exp(risk) for j in R(t))
    返回 (sorted_unique_times, H0_at_times)
    """
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    risks = risks[order]
    hazard_ratio = np.exp(risks)
    # 对每个时间点 t，风险集 R(t) = {j: T_j >= t}，即 cumsum 从后往前
    # 时间升序时，R(t_i) = 从 i 到末尾
    n = len(times)
    H0 = np.zeros(n)
    cumsum_hr_rev = np.cumsum(hazard_ratio[::-1])[::-1]  # 从 i 到末尾的 exp(risk) 之和
    for i in range(n):
        if events[i] > 0:
            H0[i] = 1.0 / (cumsum_hr_rev[i] + 1e-8)
    H0_cum = np.cumsum(H0)
    return times, H0_cum


def risk_to_survival(
    risk: float,
    times: np.ndarray,
    H0: np.ndarray,
) -> np.ndarray:
    """
    S(t) = exp(-H0(t) * exp(risk))
    times 与 H0 一一对应
    """
    return np.exp(-np.clip(H0 * np.exp(risk), 0, 1e4))


def compute_rmst(times: np.ndarray, S: np.ndarray, tau: float = RMST_TAU_MONTHS) -> float:
    """
    限制性平均生存时间 RMST = ∫₀^τ S(t) dt，τ 单位为月（默认 60 即 5 年）。

    Breslow 基线 H₀(t) 在相邻事件时间之间为分段常数，故 S(t)=exp(-exp(r)H₀(t)) 亦为阶梯函数。
    此处按阶梯积分（区间上取常数 S），而非对 (t,S) 线性插值再梯形积分；后者会系统性扭曲 RMST，
    进而使 ITE 失真（甚至出现大量 0）。
    """
    times = np.asarray(times, dtype=float).ravel()
    S = np.asarray(S, dtype=float).ravel()
    if tau <= 0:
        return 0.0
    n = min(len(times), len(S))
    if n == 0:
        return 0.0
    times = times[:n]
    S = S[:n]
    tau = float(tau)
    area = 0.0
    t_prev = 0.0
    # 第一个事件时间之前 H₀=0，S≡1
    s_interval = 1.0
    for i in range(n):
        if t_prev >= tau:
            break
        t_i = float(times[i])
        t_seg_end = min(t_i, tau)
        if t_seg_end > t_prev:
            area += (t_seg_end - t_prev) * s_interval
        t_prev = t_seg_end
        if t_prev >= tau:
            break
        # 在 time[i] 处理完该时刻的跳跃后，区间 [time[i], time[i+1]) 上为常数 S[i]
        s_interval = float(S[i])
    if t_prev < tau:
        area += (tau - t_prev) * s_interval
    return float(area)


def _ite_relative_rmst(rmst_cf: float, rmst_fact: float, eps: float = 1e-6) -> float:
    """ITE = (RMST_cf − RMST_fact) / max(|RMST_fact|, eps)；相对变化，存为小数（×100 为百分比）。"""
    rf = float(rmst_fact)
    rc = float(rmst_cf)
    if not (np.isfinite(rf) and np.isfinite(rc)):
        return 0.0
    denom = max(abs(rf), eps)
    return float((rc - rf) / denom)


def run_counterfactual_inference(
    test_data: List[Dict[str, Any]],
    model: "nn.Module",
    device: "torch.device",
    train_times: np.ndarray,
    train_events: np.ndarray,
    train_risks: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    对给定患者列表进行反事实推理（常为验证集或测试集）。
    返回每例含 fact_risk，S_factual(t)，S_counterfactual(t)，ITE（RMST 相对增量，小数）等。
    """
    if not _HAS_TORCH:
        raise ImportError("需要 torch")
    model.eval()
    times_grid, H0 = breslow_baseline_hazard(train_times, train_events, train_risks)

    results = []
    with torch.no_grad():
        for sample in test_data:
            feat_key = "joint_feat_causal" if "joint_feat_causal" in sample else "joint_feat"
            x = torch.from_numpy(sample[feat_key]).float().unsqueeze(0).to(device)
            youdao = sample["youdao"]
            risk_0, risk_1 = model(x)
            r0 = risk_0.cpu().item()
            r1 = risk_1.cpu().item()
            fact_risk = r1 if youdao == 1 else r0
            counterfact_risk = r0 if youdao == 1 else r1
            S_fact = risk_to_survival(fact_risk, times_grid, H0)
            S_cf = risk_to_survival(counterfact_risk, times_grid, H0)
            delta_S = S_cf - S_fact
            # RMST 计算：τ = 5 年（60 月）
            rmst_fact = compute_rmst(times_grid, S_fact, RMST_TAU_MONTHS)
            rmst_cf = compute_rmst(times_grid, S_cf, RMST_TAU_MONTHS)
            ite = _ite_relative_rmst(rmst_cf, rmst_fact)
            results.append({
                "patient_id": sample["patient_id"],
                "youdao": youdao,
                "fact_risk": fact_risk,
                "counterfact_risk": counterfact_risk,
                "S_factual": S_fact,
                "S_counterfactual": S_cf,
                "delta_S": delta_S,
                "times": times_grid,
                "rmst_factual": rmst_fact,
                "rmst_counterfactual": rmst_cf,
                "rmst_tau_months": RMST_TAU_MONTHS,
                "ITE": ite,
            })
    return results


def _interp_survival_at_t(times: np.ndarray, S: np.ndarray, t: float) -> float:
    """在时刻 t（月）处线性插值预测生存概率 S(t)。"""
    if len(times) == 0:
        return 0.5
    return float(np.interp(t, np.asarray(times, dtype=float), np.asarray(S, dtype=float)))


def _brier_score_at_t(
    results: List[Dict],
    test_data: List[Dict],
    t: float,
) -> Tuple[Optional[float], int]:
    """时刻 t 的 landmark Brier：mean (Ŝ(t)−Y(t))²；Ŝ 为事实臂 S_factual；Y(t)=1{T>t}。"""
    sq_errs: List[float] = []
    for r, s in zip(results, test_data):
        T = float(s["deadtime"])
        ev = int(s["event"])
        if T <= float(t) and ev == 0:
            continue
        y = 1.0 if T > float(t) else 0.0
        s_hat = _interp_survival_at_t(
            np.asarray(r["times"], dtype=float),
            np.asarray(r["S_factual"], dtype=float),
            float(t),
        )
        sq_errs.append((s_hat - y) ** 2)
    n = len(sq_errs)
    if n == 0:
        return None, 0
    return float(np.mean(sq_errs)), n


def compute_integrated_brier_score(
    results: List[Dict],
    test_data: List[Dict],
    t_max_months: float = RMST_TAU_MONTHS,
    n_points: int = IBS_TRAPZ_N_POINTS,
) -> Dict[str, Any]:
    """
    Integrated Brier Score：IBS = (1/τ) ∫_0^τ BS(t) dt，τ = t_max_months。
    BS(t) 为同一 landmark 设定下的 Brier；数值积分用 numpy.trapz 在均匀网格上近似。
    """
    if float(t_max_months) <= 0:
        return {
            "integrated_brier_score": None,
            "ibs_t_min_months": 0.0,
            "ibs_t_max_months": float(t_max_months),
            "ibs_trapezoid_n_points": int(n_points),
            "ibs_note": "t_max_months must be > 0",
        }
    if n_points < 2:
        raise ValueError("IBS trapezoid integration requires n_points >= 2")
    tau = float(t_max_months)
    t_grid = np.linspace(0.0, tau, int(n_points))
    bs_curve: List[float] = []
    for t in t_grid:
        bs, _ = _brier_score_at_t(results, test_data, float(t))
        if bs is None:
            return {
                "integrated_brier_score": None,
                "ibs_t_min_months": 0.0,
                "ibs_t_max_months": tau,
                "ibs_trapezoid_n_points": int(n_points),
                "ibs_note": "no evaluable subjects at a time grid point",
            }
        bs_curve.append(bs)
    bs_arr = np.asarray(bs_curve, dtype=float)
    ibs = float(np.trapz(bs_arr, t_grid) / tau)
    return {
        "integrated_brier_score": ibs,
        "ibs_t_min_months": 0.0,
        "ibs_t_max_months": tau,
        "ibs_trapezoid_n_points": int(n_points),
    }


def compute_c_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """
    计算 C-index（一致性指数）。
    risk: 风险分数，越高表示预后越差
    time: 生存时间
    event: 事件指示器 1=发生 0=删失
    C=0.5 为随机，C>0.5 表示模型有区分能力
    """
    n = len(risk)
    concordant = 0
    comparable = 0
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            # 可比较对：i 发生事件且 T_i < T_j，或 j 发生事件且 T_j < T_i
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
            elif event[j] == 1 and time[j] < time[i]:
                comparable += 1
                if risk[j] > risk[i]:
                    concordant += 1
    if comparable == 0:
        return 0.5
    return concordant / comparable


def _eval_val_c_index(
    model: "nn.Module",
    val_data: List[Dict[str, Any]],
    device: "torch.device",
) -> float:
    """验证集 Harrell C-index（事实风险 vs 时间与事件）。"""
    risks = _eval_factual_risks_numpy(model, val_data, device)
    time_arr = np.array([s["deadtime"] for s in val_data], dtype=float)
    event_arr = np.array([s["event"] for s in val_data], dtype=np.int32)
    return float(compute_c_index(risks, time_arr, event_arr))


def compute_evaluation_metrics(
    results: List[Dict],
    test_data: List[Dict],
    model: "nn.Module",
    device: "torch.device",
) -> Dict[str, Any]:
    """
    计算模型评估指标：C-index、IBS（综合 Brier）、简要文字结论。
    """
    metrics = {}
    risk_factual = np.array([r["fact_risk"] for r in results])
    time_arr = np.array([s["deadtime"] for s in test_data])
    event_arr = np.array([s["event"] for s in test_data])
    metrics["c_index"] = float(compute_c_index(risk_factual, time_arr, event_arr))
    metrics["n_total"] = len(results)
    metrics.update(compute_integrated_brier_score(results, test_data))
    if metrics["c_index"] >= 0.7:
        metrics["summary"] = "模型区分能力较好 (C≥0.7)"
    elif metrics["c_index"] >= 0.6:
        metrics["summary"] = "模型有一定区分能力 (0.6≤C<0.7)"
    elif metrics["c_index"] >= 0.55:
        metrics["summary"] = "模型区分能力一般 (0.55≤C<0.6)"
    else:
        metrics["summary"] = "模型区分能力较弱 (C<0.55)"
    return metrics


def bootstrap_ic(
    results: List[Dict],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    对每例 ITE（个体治疗效应，RMST 相对增量）做 bootstrap 估计置信区间与 P 值。
    P 值：H0: ITE=0，双侧检验。
    """
    if n_bootstrap <= 0:
        return results
    np.random.seed(seed)
    n = len(results)
    for i, r in enumerate(results):
        ics = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            # 简化：对单例做 bootstrap 需重采样其 delta_S 曲线
            # 用 delta_S 的 bootstrap 近似 ITE（RMST）的抽样分布
            delta_S = r["delta_S"]
            if len(delta_S) > 0:
                bs_idx = np.random.choice(len(delta_S), len(delta_S), replace=True)
                delta_S_bs = delta_S[bs_idx]
                times = r["times"]
                S_fact = r["S_factual"]
                S_cf_bs = S_fact + delta_S_bs
                tau = r.get("rmst_tau_months", RMST_TAU_MONTHS)
                rmst_f = compute_rmst(times, S_fact, tau)
                rmst_c = compute_rmst(times, S_cf_bs, tau)
                ic_bs = _ite_relative_rmst(rmst_c, rmst_f)
                ics.append(ic_bs)
            else:
                ics.append(float(r.get("ITE", r.get("IC", 0.0))))
        ics = np.array(ics)
        ci_low = np.percentile(ics, 2.5)
        ci_high = np.percentile(ics, 97.5)
        p_val = 2 * min(np.mean(ics <= 0), np.mean(ics >= 0))
        r["ITE_ci_low"] = float(ci_low)
        r["ITE_ci_high"] = float(ci_high)
        r["ITE_pvalue"] = float(p_val)
    return results


def run_step6_causal(
    step5_pkl: str,
    output_dir: str,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    balance_reg: float = 0.0,
    n_bootstrap: int = 200,
    seed: int = 42,
    inference_all: bool = False,
    lr_patience: int = 15,
    lr_factor: float = 0.5,
    head_hidden: int = 32,
    checkpoint_metric: str = "c_index",
    max_patients: Optional[int] = None,
) -> Dict[str, Any]:
    """
    执行第六步：训练因果模型，反事实推理，计算 ITE（RMST 个体治疗效应）及 bootstrap。

    inference_all: 若为 True，对全部患者（train+val+test）进行反事实推理；
                   若为 False，对有验证集时仅对验证集推理，否则对测试集推理。
                   C-index 与 ITE 报告均基于上述默认评估集（有验证集则为验证集）。
    checkpoint_metric: 验证集选优依据 c_index（默认）或 loss。
    max_patients: 若为正整数，仅使用 step5 aligned 中按出现顺序的前 N 个不同患者，
                  并与原 train/val/test 划分求交（用于与全量数据对比的小样本实验）。
    """
    if not _HAS_TORCH:
        raise ImportError("需要 torch")

    with open(step5_pkl, "rb") as f:
        data = pickle.load(f)

    aligned = data["aligned"]
    subset_meta: Dict[str, Any] = {}
    if max_patients is not None and max_patients > 0:
        allowed = _first_n_unique_patient_ids_in_aligned_order(aligned, max_patients)
        aligned = [a for a in aligned if a["patient_id"] in allowed]
        train_ids = set(data["train_ids"]) & allowed
        val_ids = set(data["val_ids"]) & allowed
        test_ids = set(data["test_ids"]) & allowed
        subset_meta = {
            "subset_max_patients": int(max_patients),
            "subset_unique_patients": len(allowed),
        }
        print(
            f"子集模式: aligned 顺序前 {len(allowed)} 名患者 "
            f"(max_patients={max_patients})，划分已与该集合求交"
        )
    else:
        train_ids = set(data["train_ids"])
        val_ids = set(data["val_ids"])
        test_ids = set(data["test_ids"])
    d_joint = data["d_joint"]

    train_data = [a for a in aligned if a["patient_id"] in train_ids]
    val_data = [a for a in aligned if a["patient_id"] in val_ids]
    test_data = [a for a in aligned if a["patient_id"] in test_ids]

    if not train_data:
        raise ValueError("训练集为空")
    if not test_data:
        test_data = val_data if val_data else train_data
        if test_data is val_data:
            val_data = None  # avoid using val as both val and test
    if not val_data:
        val_data = None

    print(f"数据划分: 训练 {len(train_data)} 例, 验证 {len(val_data) if val_data else 0} 例, 测试 {len(test_data)} 例")
    eval_data = val_data if val_data else test_data
    eval_split = "validation" if val_data else "test"
    use_causal_feat = "joint_feat_causal" in (train_data[0] if train_data else {})
    if use_causal_feat:
        print("使用 joint_feat_causal（不含治疗 youdao）进行因果推断，避免信息泄露")
    else:
        print("使用 joint_feat（含治疗 youdao），建议重新运行 step5 以生成 joint_feat_causal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCausalSurvival(d_in=d_joint, d_hidden=D_HIDDEN, head_hidden=head_hidden).to(device)
    if getattr(model, "_use_cxgnn", False):
        print("已使用 CXGNN 的 NNModel 作为因果推理基线编码器")
    if head_hidden > 0:
        print(f"使用 2 层 head（head_hidden={head_hidden}）增强治疗效应表达")

    np.random.seed(seed)
    torch.manual_seed(seed)
    losses, train_meta = train_causal_model(
        train_data, model, device,
        epochs=epochs, lr=lr, weight_decay=weight_decay, balance_reg=balance_reg,
        val_data=val_data, lr_patience=lr_patience, lr_factor=lr_factor,
        checkpoint_metric=checkpoint_metric,
    )

    # 训练集风险（用于 Breslow）
    train_risks_all = []
    train_times_all = []
    train_events_all = []
    model.eval()
    with torch.no_grad():
        for s in train_data:
            feat_key = "joint_feat_causal" if "joint_feat_causal" in s else "joint_feat"
            x = torch.from_numpy(s[feat_key]).float().unsqueeze(0).to(device)
            r0, r1 = model(x)
            risk = r1.item() if s["youdao"] == 1 else r0.item()
            train_risks_all.append(risk)
            train_times_all.append(s["deadtime"])
            train_events_all.append(s["event"])
    train_risks_all = np.array(train_risks_all)
    train_times_all = np.array(train_times_all)
    train_events_all = np.array(train_events_all)

    inference_data = aligned if inference_all else eval_data
    if inference_all:
        print(f"正在进行反事实推理（全部 {len(inference_data)} 例患者）...")
    else:
        print(
            f"正在进行反事实推理（{'验证集' if eval_split == 'validation' else '测试集'}，"
            f"共 {len(inference_data)} 例）..."
        )
    results = run_counterfactual_inference(
        inference_data, model, device,
        train_times_all, train_events_all, train_risks_all,
    )
    if n_bootstrap > 0:
        print(f"正在进行 bootstrap (n={n_bootstrap}) 估计 ITE 置信区间与 P 值...")
        results = bootstrap_ic(results, n_bootstrap=n_bootstrap, seed=seed)
    else:
        print("跳过 bootstrap（n_bootstrap=0），不写入 ITE 置信区间与 P 值")

    # 评估指标：有验证集时为验证集上的 C-index
    eval_label = "验证集" if eval_split == "validation" else "测试集"
    print(f"正在计算评估指标（基于{eval_label}）...")
    result_by_pid = {r["patient_id"]: r for r in results}
    eval_rows = [s for s in eval_data if s["patient_id"] in result_by_pid]
    eval_results = [result_by_pid[s["patient_id"]] for s in eval_rows]
    metrics = compute_evaluation_metrics(eval_results, eval_rows, model, device)
    metrics["eval_split"] = eval_split
    metrics["final_loss"] = float(losses[-1])
    metrics["n_train"] = len(train_data)
    metrics["n_inference"] = len(results)  # 反事实推理的患者数（inference_all 时 = 全部）
    metrics["training_meta"] = train_meta
    if subset_meta:
        metrics["subset"] = subset_meta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_joint": d_joint}, output_dir / "step6_model.pt")
    with open(output_dir / "step6_counterfactual_results.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "step6_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir / "step6_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False, default=str)

    # 打印评估报告
    print("\n" + "=" * 50)
    print("第六步 评估报告")
    print("=" * 50)
    print(f"  C-index:        {metrics['c_index']:.4f}  ({metrics['summary']})")
    ibs = metrics.get("integrated_brier_score")
    ibs_tau = metrics.get("ibs_t_max_months", RMST_TAU_MONTHS)
    ibs_n = metrics.get("ibs_trapezoid_n_points", IBS_TRAPZ_N_POINTS)
    if ibs is not None:
        print(f"  IBS:            {ibs:.6f}  ((1/τ)∫₀^τ BS(t)dt, τ={ibs_tau:.0f}月, trapz n={ibs_n})")
    else:
        print(f"  IBS:            N/A  ({metrics.get('ibs_note', '')})")
    print(f"  训练集样本数:   {metrics['n_train']}")
    print(f"  评估集样本数:   {metrics['n_total']}（{eval_label}）")
    if metrics.get('n_inference', metrics['n_total']) != metrics['n_total']:
        print(f"  反事实推理数:   {metrics['n_inference']} (全部患者)")
    print(f"  最终 loss:      {metrics['final_loss']:.4f}")
    print("=" * 50)
    if train_meta.get("checkpoint_metric"):
        print(
            f"训练选优: metric={train_meta['checkpoint_metric']}, "
            f"best_epoch={train_meta.get('best_epoch')}"
        )
    print(f"已保存模型、反事实结果和评估指标到 {output_dir}")

    return {
        "model": model,
        "results": results,
        "losses": losses,
        "metrics": metrics,
        "train_data": train_data,
        "test_data": test_data,
        "train_meta": train_meta,
    }

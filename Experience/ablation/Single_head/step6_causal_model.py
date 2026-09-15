"""
第六步（单头消融）：因果模型构建与反事实推理

单头模型 f(x, t) -> risk：
- 训练时输入 factual treatment；
- 推理时强制翻转 treatment（1-t）得到反事实风险；
- 其余评估链路与双头版本一致（C-index / IBS / RMST-ITE / bootstrap）。
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

_CXGNN_ROOT = Path(__file__).resolve().parent.parent / "CXGNN"
if _CXGNN_ROOT.exists() and str(_CXGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CXGNN_ROOT))

D_JOINT = 128
D_HIDDEN = 64
CXGNN_H_LAYERS = 2
RMST_TAU_MONTHS = 60
IBS_TRAPZ_N_POINTS = 121


def _first_n_unique_patient_ids_in_aligned_order(
    aligned: List[Dict[str, Any]], n: int
) -> set:
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
    try:
        from model import alg1
        return getattr(alg1, "NNModel", None)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


if _HAS_TORCH:
    _NNModelBase = _get_cxgnn_nnmodel()

    class SingleHeadCausalSurvival(nn.Module):
        """
        单头因果生存模型（S-learner）：
        输入 concat([x, t])，输出单个 risk。
        """

        def __init__(self, d_in: int = D_JOINT, d_hidden: int = D_HIDDEN, head_hidden: int = 0):
            super().__init__()
            self.d_in = d_in
            self.total_in = d_in + 1  # + treatment indicator
            if _NNModelBase is not None:
                self._use_cxgnn = True
                self.nnmodel = _NNModelBase(
                    input_size=self.total_in,
                    output_size=d_hidden,
                    h_size=d_hidden,
                    h_layers=CXGNN_H_LAYERS,
                )
            else:
                self._use_cxgnn = False
                self.nnmodel = None
                self._encoder = nn.Sequential(
                    nn.Linear(self.total_in, d_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            if head_hidden > 0:
                self.head = nn.Sequential(
                    nn.Linear(d_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, 1),
                )
            else:
                self.head = nn.Linear(d_hidden, 1)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))

        def _get_encoder_output(self, xt):
            if self._use_cxgnn and self.nnmodel is not None:
                return self.nnmodel.nn(xt)
            return self._encoder(xt)

        def forward(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            xt = torch.cat([x, t.float()], dim=-1)
            h = self._get_encoder_output(xt)
            return self.head(h).squeeze(-1)

        @property
        def encoder(self):
            if self._use_cxgnn and self.nnmodel is not None:
                return self.nnmodel.nn
            return self._encoder
else:
    SingleHeadCausalSurvival = None


def cox_partial_likelihood(
    risk: "torch.Tensor",
    time: "torch.Tensor",
    event: "torch.Tensor",
) -> "torch.Tensor":
    if not _HAS_TORCH:
        raise ImportError("需要 torch")
    order = torch.argsort(-time)
    risk = risk[order]
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
    model.eval()
    feat_key = "joint_feat_causal" if any("joint_feat_causal" in s for s in data) else "joint_feat"
    X = torch.from_numpy(np.stack([s[feat_key] for s in data])).float().to(device)
    T = torch.from_numpy(np.array([s["youdao"] for s in data], dtype=np.float32)).to(device)
    time_t = torch.from_numpy(np.array([s["deadtime"] for s in data], dtype=np.float32)).to(device)
    event_t = torch.from_numpy(np.array([s["event"] for s in data], dtype=np.float32)).to(device)
    with torch.no_grad():
        risk = model(X, T)
        loss = cox_partial_likelihood(risk, time_t, event_t)
        if balance_reg > 0:
            h = model.encoder(torch.cat([X, T.unsqueeze(-1)], dim=-1))
            mask0 = T == 0
            mask1 = T == 1
            if mask0.sum() > 0 and mask1.sum() > 0:
                h0 = h[mask0].mean(dim=0)
                h1 = h[mask1].mean(dim=0)
                loss = loss + balance_reg * torch.nn.functional.mse_loss(h0, h1)
    return float(loss.item())


def _eval_factual_risks_numpy(
    model: "nn.Module",
    data: List[Dict[str, Any]],
    device: "torch.device",
) -> np.ndarray:
    if not data:
        return np.array([], dtype=float)
    model.eval()
    feat_key = "joint_feat_causal" if any("joint_feat_causal" in s for s in data) else "joint_feat"
    X = torch.from_numpy(np.stack([s[feat_key] for s in data])).float().to(device)
    T = torch.from_numpy(np.array([s["youdao"] for s in data], dtype=np.float32)).to(device)
    with torch.no_grad():
        risk = model(X, T)
    return risk.cpu().numpy().astype(float)


def compute_c_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    n = len(risk)
    concordant = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
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
    return float(concordant / comparable)


def _eval_val_c_index(
    model: "nn.Module",
    val_data: List[Dict[str, Any]],
    device: "torch.device",
) -> float:
    risks = _eval_factual_risks_numpy(model, val_data, device)
    time_arr = np.array([s["deadtime"] for s in val_data], dtype=float)
    event_arr = np.array([s["event"] for s in val_data], dtype=np.int32)
    return compute_c_index(risks, time_arr, event_arr)


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
    if not _HAS_TORCH:
        raise ImportError("需要 torch")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )
    feat_key = "joint_feat_causal" if any("joint_feat_causal" in s for s in train_data) else "joint_feat"
    X = torch.from_numpy(np.stack([s[feat_key] for s in train_data])).float().to(device)
    T = torch.from_numpy(np.array([s["youdao"] for s in train_data], dtype=np.float32)).to(device)
    time_t = torch.from_numpy(np.array([s["deadtime"] for s in train_data], dtype=np.float32)).to(device)
    event_t = torch.from_numpy(np.array([s["event"] for s in train_data], dtype=np.float32)).to(device)

    best_val_loss = float("inf")
    best_val_c_index = -1.0
    best_state = None
    best_epoch = 0
    val_history: List[Dict[str, Any]] = []
    use_cidx = bool(val_data) and checkpoint_metric == "c_index"
    use_loss_ckpt = bool(val_data) and checkpoint_metric == "loss"
    if val_data and checkpoint_metric not in ("c_index", "loss"):
        raise ValueError(f"checkpoint_metric 须为 'c_index' 或 'loss'，收到: {checkpoint_metric}")

    losses: List[float] = []
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        risk = model(X, T)
        loss = cox_partial_likelihood(risk, time_t, event_t)
        if balance_reg > 0:
            h = model.encoder(torch.cat([X, T.unsqueeze(-1)], dim=-1))
            mask0 = T == 0
            mask1 = T == 1
            if mask0.sum() > 0 and mask1.sum() > 0:
                h0 = h[mask0].mean(dim=0)
                h1 = h[mask1].mean(dim=0)
                loss = loss + balance_reg * torch.nn.functional.mse_loss(h0, h1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = float(loss.item())
        losses.append(train_loss)
        scheduler.step(train_loss)

        val_loss: Optional[float] = None
        if val_data:
            val_loss = _eval_cox_loss(model, val_data, device, balance_reg)
            val_cidx = _eval_val_c_index(model, val_data, device)
            val_history.append({"epoch": ep + 1, "val_c_index": float(val_cidx), "val_loss": float(val_loss)})
            model.train()
            if use_cidx and val_cidx > best_val_c_index:
                best_val_c_index = float(val_cidx)
                best_val_loss = float(val_loss)
                best_epoch = ep + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif use_loss_ckpt and val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_epoch = ep + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % max(1, epochs // 10) == 0:
            msg = f"  Epoch {ep + 1}/{epochs}, loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}"
            if val_data and val_loss is not None:
                msg += f", val_loss={val_loss:.4f}, val_C-index={val_cidx:.4f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    train_meta: Dict[str, Any] = {
        "checkpoint_metric": checkpoint_metric if val_data else None,
        "best_epoch": best_epoch if best_state is not None else None,
        "best_val_c_index": float(best_val_c_index) if use_cidx and best_state is not None else None,
        "best_val_loss": float(best_val_loss) if best_state is not None else None,
        "final_epoch": len(losses),
        "val_c_index_history": val_history if val_data else [],
    }
    return losses, train_meta


def breslow_baseline_hazard(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    risks = risks[order]
    hazard_ratio = np.exp(risks)
    n = len(times)
    H0 = np.zeros(n)
    cumsum_hr_rev = np.cumsum(hazard_ratio[::-1])[::-1]
    for i in range(n):
        if events[i] > 0:
            H0[i] = 1.0 / (cumsum_hr_rev[i] + 1e-8)
    return times, np.cumsum(H0)


def risk_to_survival(
    risk: float,
    times: np.ndarray,
    H0: np.ndarray,
) -> np.ndarray:
    return np.exp(-np.clip(H0 * np.exp(risk), 0, 1e4))


def compute_rmst(times: np.ndarray, S: np.ndarray, tau: float = RMST_TAU_MONTHS) -> float:
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
        s_interval = float(S[i])
    if t_prev < tau:
        area += (tau - t_prev) * s_interval
    return float(area)


def _ite_relative_rmst(rmst_cf: float, rmst_fact: float, eps: float = 1e-6) -> float:
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
    if not _HAS_TORCH:
        raise ImportError("需要 torch")
    model.eval()
    times_grid, H0 = breslow_baseline_hazard(train_times, train_events, train_risks)
    results = []
    with torch.no_grad():
        for sample in test_data:
            feat_key = "joint_feat_causal" if "joint_feat_causal" in sample else "joint_feat"
            x = torch.from_numpy(sample[feat_key]).float().unsqueeze(0).to(device)
            youdao = int(sample["youdao"])
            t_fact = torch.tensor([float(youdao)], dtype=torch.float32, device=device)
            t_cf = torch.tensor([float(1 - youdao)], dtype=torch.float32, device=device)
            fact_risk = float(model(x, t_fact).cpu().item())
            counterfact_risk = float(model(x, t_cf).cpu().item())
            S_fact = risk_to_survival(fact_risk, times_grid, H0)
            S_cf = risk_to_survival(counterfact_risk, times_grid, H0)
            delta_S = S_cf - S_fact
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
    if len(times) == 0:
        return 0.5
    return float(np.interp(t, np.asarray(times, dtype=float), np.asarray(S, dtype=float)))


def _brier_score_at_t(
    results: List[Dict],
    test_data: List[Dict],
    t: float,
) -> Tuple[Optional[float], int]:
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


def compute_evaluation_metrics(
    results: List[Dict],
    test_data: List[Dict],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
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
    if n_bootstrap <= 0:
        return results
    np.random.seed(seed)
    for r in results:
        ics = []
        delta_S = r["delta_S"]
        for _ in range(n_bootstrap):
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
                ics.append(float(r.get("ITE", 0.0)))
        ics = np.array(ics, dtype=float)
        r["ITE_ci_low"] = float(np.percentile(ics, 2.5))
        r["ITE_ci_high"] = float(np.percentile(ics, 97.5))
        r["ITE_pvalue"] = float(2 * min(np.mean(ics <= 0), np.mean(ics >= 0)))
    return results


def run_step6_single_head(
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
    else:
        train_ids = set(data["train_ids"])
        val_ids = set(data["val_ids"])
        test_ids = set(data["test_ids"])

    d_joint = int(data["d_joint"])
    train_data = [a for a in aligned if a["patient_id"] in train_ids]
    val_data = [a for a in aligned if a["patient_id"] in val_ids]
    test_data = [a for a in aligned if a["patient_id"] in test_ids]
    if not train_data:
        raise ValueError("训练集为空")
    if not test_data:
        test_data = val_data if val_data else train_data
        if test_data is val_data:
            val_data = None
    if not val_data:
        val_data = None

    print(f"数据划分: 训练 {len(train_data)} 例, 验证 {len(val_data) if val_data else 0} 例, 测试 {len(test_data)} 例")
    eval_data = val_data if val_data else test_data
    eval_split = "validation" if val_data else "test"
    use_causal_feat = "joint_feat_causal" in (train_data[0] if train_data else {})
    if use_causal_feat:
        print("使用 joint_feat_causal（不含治疗 youdao）进行单头消融")
    else:
        print("使用 joint_feat（含治疗 youdao），建议重新运行 step5 以生成 joint_feat_causal")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleHeadCausalSurvival(d_in=d_joint, d_hidden=D_HIDDEN, head_hidden=head_hidden).to(device)
    if getattr(model, "_use_cxgnn", False):
        print("已使用 CXGNN NNModel 作为单头编码器")

    np.random.seed(seed)
    torch.manual_seed(seed)
    losses, train_meta = train_causal_model(
        train_data=train_data,
        model=model,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        balance_reg=balance_reg,
        val_data=val_data,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        checkpoint_metric=checkpoint_metric,
    )

    train_risks_all = _eval_factual_risks_numpy(model, train_data, device)
    train_times_all = np.array([s["deadtime"] for s in train_data], dtype=float)
    train_events_all = np.array([s["event"] for s in train_data], dtype=float)

    inference_data = aligned if inference_all else eval_data
    results = run_counterfactual_inference(
        inference_data, model, device,
        train_times_all, train_events_all, train_risks_all,
    )
    if n_bootstrap > 0:
        results = bootstrap_ic(results, n_bootstrap=n_bootstrap, seed=seed)

    result_by_pid = {r["patient_id"]: r for r in results}
    eval_rows = [s for s in eval_data if s["patient_id"] in result_by_pid]
    eval_results = [result_by_pid[s["patient_id"]] for s in eval_rows]
    metrics = compute_evaluation_metrics(eval_results, eval_rows)
    metrics["eval_split"] = eval_split
    metrics["final_loss"] = float(losses[-1])
    metrics["n_train"] = len(train_data)
    metrics["n_inference"] = len(results)
    metrics["training_meta"] = train_meta
    metrics["model_type"] = "single_head_s_learner"
    if subset_meta:
        metrics["subset"] = subset_meta

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_joint": d_joint}, output_dir_p / "step6_model.pt")
    with open(output_dir_p / "step6_counterfactual_results.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir_p / "step6_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir_p / "step6_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 50)
    print("第六步（单头消融）评估报告")
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
    print(f"  评估集样本数:   {metrics['n_total']}（{'验证集' if eval_split == 'validation' else '测试集'}）")
    if metrics.get("n_inference", metrics["n_total"]) != metrics["n_total"]:
        print(f"  反事实推理数:   {metrics['n_inference']} (全部患者)")
    print(f"  最终 loss:      {metrics['final_loss']:.4f}")
    print("=" * 50)
    print(f"已保存模型、反事实结果和评估指标到 {output_dir_p}")

    return {
        "model": model,
        "results": results,
        "losses": losses,
        "metrics": metrics,
        "train_data": train_data,
        "test_data": test_data,
        "train_meta": train_meta,
    }

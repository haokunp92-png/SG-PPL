"""
Step 7: Individual and population PITE visualization

时间轴（Fig 1、3）统一截止 TIME_AXIS_MAX_MONTHS（默认 60 月）。
Fig 2：PITE 三分组后，组内按 ITE 降序、再按 PITE 降序排列患者；纵轴为 60 月处生存率。
Fig 4：横轴为患者序号（按 ITE 全局降序），与时间轴无关；柱色分组时 |ITE|%<阈值者一律视为 neutral。
Fig 2 与 Fig 4 统一分组口径：|ITE|% < PITE_NEUTRAL_ABS_ITE_PCT 者一律归为 ITE-neutral，其余再按符号+p 值判定。
（已移除整批治疗臂对比 KM 汇总图。）
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# 横轴时间统一截止（月），与 RMST/5 年生存率一致
TIME_AXIS_MAX_MONTHS = 60.0

# PITE 分组：|ITE| 百分比绝对值低于该阈值者一律归入 ITE-neutral（再按符号+p 值区分 preferred/opposed）
# Fig2、Fig4 统一使用该阈值
PITE_NEUTRAL_ABS_ITE_PCT = 5.0


def _ite_value(r: Dict) -> float:
    """个体治疗效应（RMST 相对增量，小数）；兼容旧结果键 IC。"""
    return float(r.get("ITE", r.get("IC", 0.0)))


def _ite_pvalue(r: Dict) -> float:
    return float(r.get("ITE_pvalue", r.get("IC_pvalue", 1.0)))


def load_step6_results(pkl_path: str) -> List[Dict[str, Any]]:
    """Load step 6 counterfactual inference results"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def filter_by_patients(
    results: List[Dict],
    patient_ids: Optional[List[str]] = None,
    patient_indices: Optional[List[int]] = None,
) -> List[Dict]:
    """Filter by patient ID or index."""
    if patient_ids is not None:
        id_set = set(str(p) for p in patient_ids)
        return [r for r in results if str(r["patient_id"]) in id_set]
    if patient_indices is not None:
        idx_set = set(patient_indices)
        return [r for i, r in enumerate(results) if i in idx_set]
    return results


def _interp_survival_at_t(times: np.ndarray, S: np.ndarray, t: float) -> float:
    """Interpolate survival probability at time t (months)."""
    if len(times) == 0:
        return 0.5
    return float(np.interp(t, times, S))


def _slice_curve_to_tmax(times: np.ndarray, S: np.ndarray, t_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """截取生存曲线至 t_max（月），末端点插值到 t_max。"""
    if len(times) == 0:
        return times, S
    times = np.asarray(times, dtype=float)
    S = np.asarray(S, dtype=float)
    mask = times <= t_max
    t_plot = list(times[mask])
    s_plot = list(S[mask])
    if len(times) > 0 and (not t_plot or t_plot[-1] < t_max):
        s_at = _interp_survival_at_t(times, S, t_max)
        if not t_plot or t_plot[-1] < t_max:
            t_plot.append(t_max)
            s_plot.append(s_at)
    return np.array(t_plot), np.array(s_plot)


def _median_survival_time(times: np.ndarray, S: np.ndarray) -> float:
    """Median survival time: time when S first drops below 0.5. If S always > 0.5, use max time."""
    if len(times) == 0:
        return 0.0
    below = np.where(S < 0.5)[0]
    if len(below) == 0:
        return float(np.max(times))
    idx = below[0]
    if idx == 0:
        return float(times[0])
    t0, t1 = times[idx - 1], times[idx]
    s0, s1 = S[idx - 1], S[idx]
    if s1 == s0:
        return float(t0)
    t_med = t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0)
    return float(t_med)


def _pval_floor(p: float, n_bootstrap: int) -> float:
    """Bootstrap p-value floor: avoid 0, use 2/(B+1) as minimum."""
    return max(p, 2.0 / (n_bootstrap + 1))


def _curve_comparison_pvalue(
    delta_S: np.ndarray,
    times: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> float:
    """Bootstrap-based P-value for curve comparison (mean(delta_S)=0)."""
    np.random.seed(seed)
    n = len(delta_S)
    if n == 0:
        return 1.0
    means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        means.append(np.mean(delta_S[idx]))
    means = np.array(means)
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    return _pval_floor(float(p), n_bootstrap)


def _classify_pite(
    results: List[Dict], min_abs_ite_pct: Optional[float] = None
) -> Dict[str, List[Dict]]:
    """按 ITE 将患者分为 ITE-preferred / ITE-neutral / ITE-opposed。

    默认（min_abs_ite_pct 为 None）：仅用 p<0.05 与 ITE 符号判定 preferred/opposed。
    若传入 min_abs_ite_pct（Fig2、Fig4 均使用 PITE_NEUTRAL_ABS_ITE_PCT），则 |ITE|% 低于该
    阈值者一律归为 ITE-neutral，否则再按符号与 p 值判定。
    """
    preferred = []
    neutral = []
    opposed = []
    for r in results:
        ite = _ite_value(r)
        pval = _ite_pvalue(r)
        ite_pct_abs = abs(ite * 100.0)
        if min_abs_ite_pct is not None and ite_pct_abs < min_abs_ite_pct:
            neutral.append(r)
        elif ite > 0 and pval < 0.05:
            preferred.append(r)
        elif ite < 0 and pval < 0.05:
            opposed.append(r)
        else:
            neutral.append(r)
    return {"ITE-preferred": preferred, "ITE-neutral": neutral, "ITE-opposed": opposed}


def plot_single_patient_survival_curves(
    result: Dict[str, Any],
    output_path: str,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> None:
    """
    Fig 1: Single patient counterfactual survival curves.
    Red solid=Factual, blue solid=Counterfactual. P-value in upper right.
    """
    if not _HAS_MPL:
        raise ImportError("Requires matplotlib")
    times = np.asarray(result["times"])
    S_fact = np.asarray(result["S_factual"])
    S_cf = np.asarray(result["S_counterfactual"])
    delta_S = np.asarray(result["delta_S"])
    pid = result["patient_id"]

    pval = _curve_comparison_pvalue(delta_S, times, n_bootstrap=n_bootstrap, seed=seed)
    t_plot_f, s_plot_f = _slice_curve_to_tmax(times, S_fact, TIME_AXIS_MAX_MONTHS)
    t_plot_c, s_plot_c = _slice_curve_to_tmax(times, S_cf, TIME_AXIS_MAX_MONTHS)
    pval_str = f"{pval:.4f}" if pval >= 0.0001 else f"{pval:.2e}"
    sig_mark = "*" if pval < 0.05 else ""
    pval_display = f"Log-rank test P = {pval_str}{sig_mark}"

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.plot(t_plot_f, s_plot_f * 100, color="#c0392b", linestyle="-", linewidth=2.5, label="Factual")
    ax.plot(t_plot_c, s_plot_c * 100, color="#2980b9", linestyle="-", linewidth=2.5, label="Counterfactual")
    ax.set_xlabel("Time (months)", fontsize=13)
    ax.set_ylabel("Survival probability (%)", fontsize=13)
    ax.set_title(f"{pid} Counterfactual Survival Analysis (0–{int(TIME_AXIS_MAX_MONTHS)} mo)", fontsize=14, pad=12)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95, title=pval_display, title_fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, TIME_AXIS_MAX_MONTHS + 2)
    ax.grid(True, alpha=0.4, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _plot_fig4_waterfall_ite(results: List[Dict], output_path: str) -> None:
    """Fig 4: 个体 ITE（RMST 相对变化 %）瀑布图；按 ITE 降序排列。"""
    if not _HAS_MPL:
        raise ImportError("Requires matplotlib")
    groups = _classify_pite(results, min_abs_ite_pct=PITE_NEUTRAL_ABS_ITE_PCT)
    group_order = ["ITE-preferred", "ITE-neutral", "ITE-opposed"]
    bar_colors = {"ITE-preferred": "#3498db", "ITE-neutral": "#95a5a6", "ITE-opposed": "#e74c3c"}

    all_ic = []
    all_group = []
    for g in group_order:
        for r in groups[g]:
            ic_pct = _ite_value(r) * 100
            all_ic.append(ic_pct)
            all_group.append(g)

    if not all_ic:
        all_ic, all_group = [0.0], ["ITE-neutral"]

    ic_arr = np.array(all_ic)
    order = np.argsort(-ic_arr)
    ic_arr = ic_arr[order]
    all_group = [all_group[i] for i in order]

    n = len(ic_arr)
    x = np.arange(n)
    colors = [bar_colors[g] for g in all_group]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.bar(x, ic_arr, color=colors, edgecolor="none", width=0.9)

    ax.axhline(y=0, color="#333", linewidth=1.5, linestyle="-")
    thr = PITE_NEUTRAL_ABS_ITE_PCT
    ax.axhline(y=thr, color="#bdc3c7", linewidth=1, linestyle=":", alpha=0.9)
    ax.axhline(y=-thr, color="#bdc3c7", linewidth=1, linestyle=":", alpha=0.9)

    thresh_upper = thresh_lower = None
    ic_preferred_vals = [ic_arr[i] for i in range(n) if all_group[i] == "ITE-preferred"]
    ic_opposed_vals = [ic_arr[i] for i in range(n) if all_group[i] == "ITE-opposed"]
    if ic_preferred_vals:
        thresh_upper = min(ic_preferred_vals)
    if ic_opposed_vals:
        thresh_lower = max(ic_opposed_vals)
    if thresh_upper is not None:
        ax.axhline(y=thresh_upper, color="#7f8c8d", linewidth=1, linestyle="--", alpha=0.8)
    if thresh_lower is not None:
        ax.axhline(y=thresh_lower, color="#7f8c8d", linewidth=1, linestyle="--", alpha=0.8)

    ax.set_xlabel("Patient index (sorted by ITE, high to low)", fontsize=12)
    ax.set_ylabel("Individual treatment effect ITE (%)", fontsize=12)
    ax.set_title("Individual Treatment Effect Waterfall (All Patients)", fontsize=14, pad=12)
    ic_min, ic_max = np.min(ic_arr), np.max(ic_arr)
    margin = max(5, (ic_max - ic_min) * 0.1) if ic_max != ic_min else 10
    ax.set_ylim(min(ic_min - margin, -5), max(ic_max + margin, 5))
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, axis="y", alpha=0.4, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="none", label="ITE-preferred (significant benefit)"),
        Patch(
            facecolor="#95a5a6",
            edgecolor="none",
            label=f"ITE-neutral (|ITE|<{thr}% or p≥0.05)",
        ),
        Patch(facecolor="#e74c3c", edgecolor="none", label="ITE-opposed (significant harm)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.95)

    n_pref = len(groups["ITE-preferred"])
    n_neut = len(groups["ITE-neutral"])
    n_opp = len(groups["ITE-opposed"])
    total = n_pref + n_neut + n_opp
    pct = lambda a, t: f"{100*a/t:.1f}%" if t > 0 else "0%"
    stats_text = (
        f"ITE-preferred: n={n_pref} ({pct(n_pref, total)})\n"
        f"ITE-neutral: n={n_neut} ({pct(n_neut, total)})\n"
        f"ITE-opposed: n={n_opp} ({pct(n_opp, total)})\n"
        f"Median ITE: {np.median(ic_arr):.2f}%\n"
        f"ITE range: [{ic_min:.2f}%, {ic_max:.2f}%]\n"
        f"Neutral rule: |ITE|<{thr}%"
    )
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="center", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.95))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _pite_and_survivals(r: Dict, landmark_months: float = 60.0) -> Tuple[float, float, float]:
    """PITE = S(治疗臂) − S(对照臂)（ landmark 月份处）；按 youdao 与事实/反事实曲线对应。"""
    times = np.asarray(r.get("times", []))
    S_fact = np.asarray(r.get("S_factual", []))
    S_cf = np.asarray(r.get("S_counterfactual", []))
    youdao = int(r.get("youdao", 0))
    if len(times) == 0 or len(S_fact) == 0:
        return 0.0, 0.5, 0.5
    t_max = min(landmark_months, float(np.max(times)))
    if youdao == 0:
        S_non_IC = _interp_survival_at_t(times, S_fact, t_max)
        S_IC = _interp_survival_at_t(times, S_cf, t_max)
    else:
        S_IC = _interp_survival_at_t(times, S_fact, t_max)
        S_non_IC = _interp_survival_at_t(times, S_cf, t_max)
    pite = S_IC - S_non_IC
    return pite, S_IC, S_non_IC


def _plot_bar_pite_individual_survival(results: List[Dict], output_path: str, header: str = "Establishment") -> None:
    """Fig 2: 按 ITE 三分组的 landmark 生存率柱状图（治疗臂 vs 对照臂曲线在 τ 处的值）。"""
    if not _HAS_MPL:
        raise ImportError("Requires matplotlib")
    groups = _classify_pite(results, min_abs_ite_pct=PITE_NEUTRAL_ABS_ITE_PCT)
    group_order = ["ITE-opposed", "ITE-neutral", "ITE-preferred"]

    all_x, all_s_ic, all_s_non_ic = [], [], []
    group_ranges = []
    gap = 1.0
    x_start = 0.0

    for g in group_order:
        pts = groups[g]
        if not pts:
            group_ranges.append((x_start, x_start, 0))
            x_start += gap
            continue
        ite_sort_vals = []
        pite_vals = []
        s_ic_vals = []
        s_non_ic_vals = []
        for r in pts:
            pite, s_ic, s_non_ic = _pite_and_survivals(r, landmark_months=TIME_AXIS_MAX_MONTHS)
            ite_sort_vals.append(_ite_value(r))
            pite_vals.append(pite)
            s_ic_vals.append(s_ic * 100)
            s_non_ic_vals.append(s_non_ic * 100)
        # 分组内排序：先按 ITE 降序，再按 PITE 降序
        ite_arr = np.array(ite_sort_vals)
        pite_arr = np.array(pite_vals)
        order = np.lexsort((-pite_arr, -ite_arr))
        n = len(pts)
        x_group = np.linspace(x_start, x_start + max(n - 1, 0.5), n) if n > 1 else np.array([x_start])
        for i in order:
            all_x.append(x_group[i])
            all_s_ic.append(s_ic_vals[i])
            all_s_non_ic.append(s_non_ic_vals[i])
        group_ranges.append((x_start, x_start + max(n - 1, 0.5), n))
        x_start += n + gap

    if not all_x:
        all_x, all_s_ic, all_s_non_ic = [0.0], [50.0], [50.0]
        group_ranges = [(0.0, 0.0, 1)]

    x_arr = np.array(all_x)
    s_ic_arr = np.array(all_s_ic)
    s_non_ic_arr = np.array(all_s_non_ic)

    boundaries = []
    for i in range(1, len(group_order)):
        if group_ranges[i - 1][2] > 0 and group_ranges[i][2] > 0:
            b = (group_ranges[i - 1][1] + group_ranges[i][0]) / 2
            if boundaries and b - boundaries[-1] < 0.5:
                b = boundaries[-1] + 0.5
            boundaries.append(b)

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    ax_header = fig.add_axes([0, 0.92, 1, 0.08])
    ax_header.set_facecolor("#5a5a5a")
    ax_header.text(0.5, 0.5, f"PITE classification: {header}", ha="center", va="center",
                   fontsize=14, color="white", fontweight="medium")
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis("off")

    ax = fig.add_axes([0.12, 0.22, 0.82, 0.68])
    ax.set_facecolor("#f5f5f5")
    ax.set_title(
        f"Patient-specific survival prediction (survival at {int(TIME_AXIS_MAX_MONTHS)} mo)",
        fontsize=14,
        pad=10,
    )
    bar_w = 0.35
    ax.bar(x_arr - bar_w / 2, s_ic_arr, width=bar_w, color="#f1c40f", alpha=0.9,
           label="Treatment arm S(τ)", zorder=2)
    ax.bar(x_arr + bar_w / 2, s_non_ic_arr, width=bar_w, color="#3498db", alpha=0.9,
           label="Control arm S(τ)", zorder=2)
    ax.set_ylabel("Survival probability (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, x_start - gap + 0.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.4, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    for b in boundaries:
        ax.axvline(x=b, color="#e74c3c", linewidth=1.5, linestyle="--", zorder=1)

    ax_bar = fig.add_axes([0.12, 0.08, 0.82, 0.12])
    ax_bar.set_facecolor("none")
    ax_bar.set_xlim(ax.get_xlim())
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.axis("off")

    bar_colors = {"ITE-opposed": "#e74c3c", "ITE-neutral": "#95a5a6", "ITE-preferred": "#3498db"}
    for i, g in enumerate(group_order):
        x0, x1, n = group_ranges[i]
        if n > 0:
            ax_bar.axvspan(x0, x1, ymin=0.25, ymax=0.75, facecolor=bar_colors[g], alpha=0.9)
            mid = (x0 + x1) / 2
            ax_bar.text(mid, 0.5, g, ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    fig.text(
        0.5,
        0.02,
        f"Within each PITE group: sort by ITE (desc) then PITE (desc). "
        f"Survival at {int(TIME_AXIS_MAX_MONTHS)} mo; PITE = S(treated) − S(control) at τ.",
        ha="center",
        fontsize=10,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_step7_plot(
    step6_output: str,
    output_dir: str,
    patient_ids: Optional[List[str]] = None,
    patient_indices: Optional[List[int]] = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> None:
    """
    Run step 7: Fig1 单患者曲线（可选）；Fig2 PITE 柱状图；Fig4 ITE 瀑布图。
    """
    if not _HAS_MPL:
        raise ImportError("Requires matplotlib. Install: pip install matplotlib")

    pkl6 = Path(step6_output) / "step6_counterfactual_results.pkl"
    if not pkl6.exists():
        raise FileNotFoundError(f"Not found {pkl6}, run step 6 first")

    results = load_step6_results(str(pkl6))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_bar_pite_individual_survival(
        results,
        str(output_dir / "step7_fig2_pite_classification.png"),
    )
    _plot_fig4_waterfall_ite(
        results,
        str(output_dir / "step7_fig4_waterfall_ite.png"),
    )

    filtered = filter_by_patients(results, patient_ids, patient_indices)
    if filtered:
        for r in filtered:
            pid = r["patient_id"]
            plot_single_patient_survival_curves(
                r, str(output_dir / f"step7_fig1_patient_{pid}_survival_curves.png"),
                n_bootstrap=n_bootstrap, seed=seed,
            )
        print("Step 7 complete. 3 figures saved (fig1 patient, fig2 PITE, fig4 waterfall).")
    else:
        print("Step 7 complete. 2 figures saved (fig2 PITE, fig4 waterfall). Use --patients for fig1.")

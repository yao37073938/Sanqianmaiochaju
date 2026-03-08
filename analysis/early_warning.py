"""
analysis/early_warning.py
=========================
从 Gillespie SSA 时间序列中提取三种早期预警信号（Early Warning Signals, EWS），
并用 Kendall τ 检验趋势显著性，验证临界减速（Critical Slowing Down, CSD）。

EWS 指标
--------
1. 滚动方差 (rolling variance)  ——窗口 = 时间序列长度的一半
2. 滞后-1 自相关系数 (AC1)     ——临界慢化的直接指标（AR(1) 主特征值）
3. 恢复时间 (recovery time)     ——τ = −Δt / ln(AC1)，AR(1) 近似导出；
                                  理论值由 ODE Jacobian 主特征值给出

临界减速预期
-----------
  sigma → sigma_c ≈ 2.035（鞍结分岔）时：
    - 方差升高   ← 能垒变浅，涨落增强
    - AC1 升高   ← 记忆变长，慢化
    - τ_r 延长   ← 恢复速率 |λ_1| → 0

模型
----
使用 bifurcation.py 的 BIFUR_PARAMS + k_dam_bifur（Hill n=2，修复阈值），
确保双稳态结构。SSA 中初始化到健康稳态（健康支：N ≈ 419），跟踪健康支的
统计量直到分岔。

统计检验
--------
对 sigma < sigma_c 区间用 Kendall τ 检验方差和 AC1 是否单调上升。
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kendalltau

from models.ode_model import MitoParams
from analysis.bifurcation import (
    BIFUR_PARAMS,
    K_REPAIR,
    PHI_C,
    k_dam_bifur,
    mito_ode_bifur,
    numerical_jacobian,
    find_all_fixed_points,
)

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

#: 鞍结分岔点（来自 bifurcation.py 分析结果）
SIGMA_C: float = 2.035

#: 采样间隔（小时）
DT_SAMPLE: float = 2.0

#: 用于区分健康支/损伤支的 N 阈值
N_BRANCH_THRESH: float = 250.0


# ---------------------------------------------------------------------------
# 化学计量矩阵（与 stochastic_model.py 相同，10 个反应）
# ---------------------------------------------------------------------------

_STOICH = np.array(
    [
        [+1,  0],   # 0  ∅ → H           生物合成
        [+1,  0],   # 1  H → 2H          H 裂变
        [-1,  0],   # 2  H+H → H         H-H 融合
        [-1,  0],   # 3  H+D → D         跨类融合（H 被 D 吸收）
        [ 0, -1],   # 4  D+H → H         跨类融合（D 被 H 吸收）
        [-1, +1],   # 5  H → D           ROS 损伤
        [-1,  0],   # 6  H → ∅           非选择性自噬（伴随 D）
        [ 0, -1],   # 7  D → ∅           选择性线粒体自噬
        [ 0, +1],   # 8  D → 2D          D 裂变
        [ 0, -1],   # 9  D+D → D         D-D 融合
    ],
    dtype=np.int64,
)
_N_REACTIONS: int = len(_STOICH)


# ---------------------------------------------------------------------------
# 命题函数（使用分岔版损伤率 k_dam_bifur）
# ---------------------------------------------------------------------------

def _propensities_bifur(H: int, D: int, sigma: float,
                        p: MitoParams) -> np.ndarray:
    """
    使用 k_dam_bifur (Hill n=2, phi-based) 计算 10 个反应命题。

    Parameters
    ----------
    H, D    : 当前状态（整数计数）
    sigma   : 外部应激强度
    p       : BIFUR_PARAMS 参数集

    Returns
    -------
    a : shape (10,) 命题向量（非负）
    """
    N = H + D
    kd = k_dam_bifur(sigma, float(D), float(N), p)
    a = np.empty(_N_REACTIONS)
    a[0] = p.k_bio
    a[1] = p.k_fis * H
    a[2] = p.k_fus * H * max(H - 1, 0)
    a[3] = p.k_fus * H * D
    a[4] = p.k_fus * D * H
    a[5] = kd * H
    a[6] = p.k_mit * H * D / (D + p.K_d) if D > 0 else 0.0
    a[7] = p.k_mit * D
    a[8] = p.k_fis * D
    a[9] = p.k_fus * D * max(D - 1, 0)
    return a


# ---------------------------------------------------------------------------
# 初始条件：ODE 稳态（健康支）
# ---------------------------------------------------------------------------

def _healthy_ic(sigma: float, p: MitoParams = BIFUR_PARAMS) -> tuple[int, int]:
    """
    返回健康支稳态的整数 (H, D) 初始条件。

    使用 find_all_fixed_points 找到所有稳态，选取 N 最大的稳态（健康支）。
    若无稳态（sigma > sigma_c），退而使用固定初值 [400, 2]。
    """
    fps = find_all_fixed_points(sigma, p, n_H=8, n_D=8)
    stable = [fp for fp in fps if fp["stable"]]
    if stable:
        # 取 N 最大的稳定不动点（健康支）
        best = max(stable, key=lambda fp: fp["N"])
        H_ss = max(1, int(round(best["H"])))
        D_ss = max(0, int(round(best["D"])))
        return H_ss, D_ss
    # 退化情况：无稳态（分岔后），从损伤态附近初始化
    return 80, 20


# ---------------------------------------------------------------------------
# 单次 Gillespie 直接法（分岔版）
# ---------------------------------------------------------------------------

def gillespie_bifur(
    sigma: float,
    p: MitoParams = BIFUR_PARAMS,
    t_end: float = 700.0,
    t_burn: float = 200.0,
    dt_sample: float = DT_SAMPLE,
    H0: int | None = None,
    D0: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    运行一次 Gillespie SSA（使用 k_dam_bifur），返回热化后的采样序列。

    Returns
    -------
    H_out, D_out : shape (n_samples,) 整数时间序列
    """
    rng = np.random.default_rng(seed)

    if H0 is None or D0 is None:
        H0, D0 = _healthy_ic(sigma, p)

    H: int = H0
    D: int = D0
    t: float = 0.0

    sample_times = np.arange(t_burn, t_end, dt_sample)
    n_samples = len(sample_times)
    H_out = np.zeros(n_samples, dtype=np.int64)
    D_out = np.zeros(n_samples, dtype=np.int64)
    s_idx = 0

    while t < t_end and s_idx < n_samples:
        a = _propensities_bifur(H, D, sigma, p)
        a0 = float(a.sum())

        if a0 < 1e-15:
            H_out[s_idx:] = H
            D_out[s_idx:] = D
            break

        tau = rng.exponential(1.0 / a0)
        t_next = t + tau

        # 记录落在 [t, t_next) 内的采样点
        while s_idx < n_samples and sample_times[s_idx] <= t_next:
            H_out[s_idx] = H
            D_out[s_idx] = D
            s_idx += 1

        # 选择反应
        r = rng.uniform(0.0, a0)
        rxn = int(np.searchsorted(np.cumsum(a), r))
        rxn = min(rxn, _N_REACTIONS - 1)

        H = max(H + int(_STOICH[rxn, 0]), 0)
        D = max(D + int(_STOICH[rxn, 1]), 0)
        t = t_next

    return H_out, D_out


# ---------------------------------------------------------------------------
# 多进程工作函数（顶层，支持 pickle）
# ---------------------------------------------------------------------------

def _ews_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    sigma, p_dict, t_end, t_burn, dt_sample, H0, D0, seed = args
    p = MitoParams(**p_dict)
    return gillespie_bifur(sigma, p, t_end, t_burn, dt_sample, H0, D0, seed)


# ---------------------------------------------------------------------------
# 集成运行
# ---------------------------------------------------------------------------

class EWSResult(NamedTuple):
    """单个 sigma 的 EWS 集成结果。"""
    sigma: float
    H_traj: np.ndarray   # shape (n_runs_valid, n_samples)
    D_traj: np.ndarray
    n_valid: int          # 留在健康支的模拟次数


def run_ews_ensemble(
    sigma: float,
    p: MitoParams = BIFUR_PARAMS,
    n_runs: int = 100,
    t_end: float = 700.0,
    t_burn: float = 200.0,
    dt_sample: float = DT_SAMPLE,
    n_processes: int | None = None,
) -> EWSResult:
    """
    在给定 sigma 并行运行 n_runs 次 Gillespie SSA（分岔版模型）。

    对 sigma < SIGMA_C，保留均值 N > N_BRANCH_THRESH 的轨迹（健康支）。
    对 sigma >= SIGMA_C，保留所有轨迹。
    """
    H0, D0 = _healthy_ic(sigma, p)
    p_dict = {f: getattr(p, f) for f in p.__dataclass_fields__}
    n_cpu = n_processes or min(mp.cpu_count(), n_runs)

    args_list = [
        (sigma, p_dict, t_end, t_burn, dt_sample, H0, D0, seed)
        for seed in range(n_runs)
    ]

    with mp.Pool(n_cpu) as pool:
        raw = pool.map(_ews_worker, args_list)

    H_all = np.array([r[0] for r in raw], dtype=np.float64)
    D_all = np.array([r[1] for r in raw], dtype=np.float64)

    # 按支过滤：在 sigma < sigma_c 时保留均值 N > (healthy_N + damaged_N)/2
    mean_N_per_run = (H_all + D_all).mean(axis=1)
    if sigma < SIGMA_C:
        # 动态阈值：健康态 N 可能随 sigma 降低，使用实际健康态 N 的一半
        healthy_N = float(H0 + D0)
        thresh = max(0.5 * healthy_N, N_BRANCH_THRESH)
        mask = mean_N_per_run > thresh
    else:
        mask = np.ones(n_runs, dtype=bool)

    n_valid = int(mask.sum())
    if n_valid == 0:
        # 退而求其次：保留均值最高的一半
        thresh = np.median(mean_N_per_run)
        mask = mean_N_per_run >= thresh
        n_valid = int(mask.sum())

    return EWSResult(
        sigma=sigma,
        H_traj=H_all[mask],
        D_traj=D_all[mask],
        n_valid=n_valid,
    )


# ---------------------------------------------------------------------------
# EWS 指标计算
# ---------------------------------------------------------------------------

def rolling_variance_mean(series: np.ndarray, window: int | None = None) -> float:
    """
    计算滚动方差的均值（窗口 = len//2）。

    在长度为 T 的时间序列上，用步长 1 的滑动窗口（大小 W = T//2）
    逐窗口计算方差，返回所有窗口方差的均值。

    Parameters
    ----------
    series : 1-D 时间序列 N(t)
    window : 窗口大小（默认 len(series)//2）

    Returns
    -------
    float : 平均滚动方差
    """
    T = len(series)
    W = window if window is not None else max(T // 2, 2)
    if T < W + 1:
        return float(np.var(series, ddof=1)) if T > 1 else 0.0
    vars_ = [
        float(np.var(series[i: i + W], ddof=1))
        for i in range(0, T - W + 1)
    ]
    return float(np.mean(vars_))


def lag1_autocorrelation(series: np.ndarray) -> float:
    """
    计算时间序列的滞后-1 自相关系数。

    使用 Pearson 相关系数定义：
        AC1 = corr(x[0:T-1], x[1:T])

    Returns
    -------
    float in [-1, 1]；若方差为零返回 0.0
    """
    if len(series) < 3:
        return 0.0
    x = series[:-1].astype(float)
    y = series[1:].astype(float)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def recovery_time_from_ac1(ac1: float, dt: float = DT_SAMPLE) -> float:
    """
    由 AR(1) 近似导出恢复时间：τ_r = −Δt / ln(|AC1|)。

    物理含义：系统受小扰动后，指数回归到稳态的特征时间。
    近分岔处 |AC1| → 1，τ_r → ∞（临界减速）。

    Returns
    -------
    float : τ_r（小时）；|AC1| 过小或 ≥ 1 时返回 NaN
    """
    a = abs(ac1)
    if a < 1e-6 or a >= 1.0:
        return np.nan
    return -dt / np.log(a)


def oDE_recovery_time(sigma: float, p: MitoParams = BIFUR_PARAMS) -> float:
    """
    由 ODE Jacobian 主特征值估计健康支的理论恢复时间。

    τ_theory = −1 / Re(λ_max)

    使用 find_all_fixed_points 找到所有稳态，选取健康支（N 最大的稳定不动点），
    然后计算其 Jacobian 主特征值。

    Returns
    -------
    float : τ_r（小时）；无健康稳态（sigma > sigma_c）返回 NaN
    """
    fps = find_all_fixed_points(sigma, p, n_H=10, n_D=10)
    stable = [fp for fp in fps if fp["stable"]]
    if not stable:
        return np.nan

    # 健康支 = N 最大的稳定不动点
    best = max(stable, key=lambda fp: fp["N"])
    H_s, D_s = best["H"], best["D"]

    J = numerical_jacobian(np.array([H_s, D_s]), sigma, p)
    eigs = np.linalg.eigvals(J)
    lambda_max = float(np.max(np.real(eigs)))
    if lambda_max >= 0:
        return np.nan
    return -1.0 / lambda_max


# ---------------------------------------------------------------------------
# Sigma 扫描：聚合所有 EWS 统计量
# ---------------------------------------------------------------------------

class ScanResult(NamedTuple):
    """sigma 扫描的完整 EWS 统计结果。"""
    sigmas: np.ndarray          # sigma 值
    rolling_var: np.ndarray     # 平均滚动方差（中位数跨 runs）
    rolling_var_std: np.ndarray # 标准差
    ac1: np.ndarray             # 中位数 AC1
    ac1_std: np.ndarray
    tau_rec: np.ndarray         # 由 AC1 导出的恢复时间（中位数）
    tau_theory: np.ndarray      # 由 ODE Jacobian 导出的理论恢复时间
    n_valid: np.ndarray         # 各 sigma 有效轨迹数


def compute_scan(
    sigmas: np.ndarray,
    p: MitoParams = BIFUR_PARAMS,
    n_runs: int = 100,
    t_end: float = 700.0,
    t_burn: float = 200.0,
    dt_sample: float = DT_SAMPLE,
    n_processes: int | None = None,
    verbose: bool = True,
) -> ScanResult:
    """
    对每个 sigma 运行集成 SSA，计算并返回三种 EWS 指标。

    Parameters
    ----------
    sigmas      : 应激强度数组
    p           : BIFUR_PARAMS（含 Hill n=2 ROS 反馈）
    n_runs      : 每 sigma 独立模拟次数
    t_end       : 总模拟时长（小时）
    t_burn      : 热化时间
    dt_sample   : 采样间隔
    n_processes : 并行进程数（None = CPU 核数）
    verbose     : 是否打印进度

    Returns
    -------
    ScanResult（含 rolling_var, ac1, tau_rec, tau_theory 等）
    """
    n = len(sigmas)
    rolling_var  = np.zeros(n)
    rolling_var_std = np.zeros(n)
    ac1          = np.zeros(n)
    ac1_std      = np.zeros(n)
    tau_rec      = np.zeros(n)
    tau_theory   = np.zeros(n)
    n_valid_arr  = np.zeros(n, dtype=int)

    for i, s in enumerate(sigmas):
        if verbose:
            print(f"  sigma = {s:.3f}  [{i+1}/{n}]", flush=True)

        result = run_ews_ensemble(
            s, p, n_runs=n_runs,
            t_end=t_end, t_burn=t_burn, dt_sample=dt_sample,
            n_processes=n_processes,
        )

        N_traj = result.H_traj + result.D_traj  # shape (n_valid, n_samples)
        T = N_traj.shape[1]
        W = T // 2

        # --- 1. 滚动方差（每条轨迹 → 中位数） ---
        rv_per_run = np.array([rolling_variance_mean(N_traj[j], W)
                                for j in range(result.n_valid)])
        rolling_var[i] = float(np.median(rv_per_run))
        rolling_var_std[i] = float(np.std(rv_per_run, ddof=1)) if len(rv_per_run) > 1 else 0.0

        # --- 2. 滞后-1 自相关（每条轨迹 → 中位数） ---
        ac1_per_run = np.array([lag1_autocorrelation(N_traj[j])
                                 for j in range(result.n_valid)])
        ac1[i] = float(np.median(ac1_per_run))
        ac1_std[i] = float(np.std(ac1_per_run, ddof=1)) if len(ac1_per_run) > 1 else 0.0

        # --- 3. 恢复时间（由 AC1 导出） ---
        tau_per_run = np.array([recovery_time_from_ac1(a, dt_sample)
                                 for a in ac1_per_run])
        tau_per_run = tau_per_run[np.isfinite(tau_per_run)]
        tau_rec[i] = float(np.median(tau_per_run)) if len(tau_per_run) > 0 else np.nan

        # --- 4. 理论恢复时间（ODE Jacobian） ---
        tau_theory[i] = oDE_recovery_time(s, p)

        n_valid_arr[i] = result.n_valid

        if verbose:
            print(
                f"    n_valid={result.n_valid}  "
                f"roll_var={rolling_var[i]:.1f}  "
                f"AC1={ac1[i]:.4f}  "
                f"τ_rec={tau_rec[i]:.1f} h  "
                f"τ_theory={tau_theory[i]:.1f} h",
                flush=True,
            )

    return ScanResult(
        sigmas=sigmas,
        rolling_var=rolling_var,
        rolling_var_std=rolling_var_std,
        ac1=ac1,
        ac1_std=ac1_std,
        tau_rec=tau_rec,
        tau_theory=tau_theory,
        n_valid=n_valid_arr,
    )


# ---------------------------------------------------------------------------
# Kendall τ 检验
# ---------------------------------------------------------------------------

def kendall_tau_test(
    sigmas: np.ndarray,
    metric: np.ndarray,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    n_valid: np.ndarray | None = None,
    min_valid: int = 10,
) -> dict:
    """
    对 metric vs sigma 做 Kendall τ 单调趋势检验。

    H₀：metric 与 sigma 无单调相关（τ = 0）。
    H₁：metric 随 sigma 单调上升（τ > 0）。

    Parameters
    ----------
    sigmas    : x 值（应激强度）
    metric    : EWS 指标序列
    sigma_min : 若给定，只用 sigma ≥ sigma_min 的区间
    sigma_max : 若给定，只用 sigma ≤ sigma_max 的区间
    n_valid   : 各 sigma 的有效轨迹数（用于过滤低质量点）
    min_valid : 有效轨迹数低于此值的点被排除（默认 10）

    Returns
    -------
    dict: {tau, p_value, n, significant}
    """
    mask = np.isfinite(metric)
    if sigma_min is not None:
        mask &= sigmas >= sigma_min
    if sigma_max is not None:
        mask &= sigmas <= sigma_max
    if n_valid is not None:
        mask &= n_valid >= min_valid
    x = sigmas[mask]
    y = metric[mask]
    if len(x) < 4:
        return {"tau": np.nan, "p_value": np.nan, "n": int(len(x)), "significant": False}
    tau_val, p_val = kendalltau(x, y)
    return {
        "tau": float(tau_val),
        "p_value": float(p_val),
        "n": int(len(x)),
        "significant": bool(p_val < 0.05),
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_ews(
    res: ScanResult,
    save_path: str = "figures/early_warning.pdf",
    sigma_c: float = SIGMA_C,
) -> None:
    """
    生成四面板 EWS 图（出版质量，300 dpi PDF）。

    Panels
    ------
    (a) 均值 N ± std（健康支均值）与损伤分数 E[φ]
    (b) 滚动方差（均值 ± std）
    (c) 滞后-1 自相关系数（中位数 ± std）
    (d) 恢复时间：AR(1) 导出（SSA）vs ODE Jacobian 理论值
    """
    sigmas = res.sigmas
    N_traj_means = []   # 用于重建均值 N（在 compute_scan 外）

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    ax_var, ax_ac1, ax_tau, ax_nv = axes[0], axes[1], axes[2], axes[3]

    # 共用颜色
    c_main   = "#1f4e79"   # 蓝色
    c_theory = "#c0392b"   # 红色

    # 分岔线标注
    def add_bifur_line(ax):
        ax.axvline(sigma_c, color="gray", ls=":", lw=1.5, zorder=1)
        ylo, yhi = ax.get_ylim()
        ax.text(sigma_c + 0.04, yhi - (yhi - ylo) * 0.05,
                rf"$\sigma_c={sigma_c:.2f}$",
                fontsize=8.5, color="gray", va="top")

    # (a) 滚动方差
    ax_var.fill_between(
        sigmas,
        res.rolling_var - res.rolling_var_std,
        res.rolling_var + res.rolling_var_std,
        alpha=0.25, color=c_main,
    )
    ax_var.plot(sigmas, res.rolling_var, "o-", color=c_main, lw=1.8,
                ms=4, label="Rolling variance (SSA median)")
    ax_var.set_ylabel(r"Rolling variance of $N(t)$", fontsize=10)
    ax_var.set_title("(a) Rolling Variance", fontsize=10)
    ax_var.legend(fontsize=8)
    ax_var.grid(True, ls="--", alpha=0.35)

    # (b) 滞后-1 自相关
    ax_ac1.fill_between(
        sigmas,
        res.ac1 - res.ac1_std,
        res.ac1 + res.ac1_std,
        alpha=0.25, color=c_main,
    )
    ax_ac1.plot(sigmas, res.ac1, "o-", color=c_main, lw=1.8,
                ms=4, label="Lag-1 AC (SSA median)")
    ax_ac1.set_ylabel(r"Lag-1 autocorrelation $\rho_1$", fontsize=10)
    ax_ac1.set_title("(b) Lag-1 Autocorrelation (AC1)", fontsize=10)
    ax_ac1.legend(fontsize=8)
    ax_ac1.grid(True, ls="--", alpha=0.35)

    # (c) 恢复时间
    tau_ssa    = res.tau_rec
    tau_theory = res.tau_theory

    ax_tau.plot(sigmas, tau_ssa, "o-", color=c_main, lw=1.8, ms=4,
                label=r"$\tau_r$ from AC1 (SSA)")

    theory_mask = np.isfinite(tau_theory)
    if theory_mask.sum() > 1:
        ax_tau.plot(
            sigmas[theory_mask], tau_theory[theory_mask],
            "--", color=c_theory, lw=1.6,
            label=r"$\tau_r$ from ODE Jacobian (theory)",
        )

    ax_tau.set_ylabel(r"Recovery time $\tau_r$ (h)", fontsize=10)
    ax_tau.set_title("(c) Recovery Time", fontsize=10)
    ax_tau.legend(fontsize=8)
    ax_tau.grid(True, ls="--", alpha=0.35)

    # (d) 有效轨迹数（质量监控）
    ax_nv.bar(sigmas, res.n_valid, width=np.diff(sigmas).mean() * 0.8,
              color=c_main, alpha=0.6, label="Valid runs (healthy branch)")
    ax_nv.set_ylabel("Valid trajectory count", fontsize=10)
    ax_nv.set_title("(d) Healthy-branch trajectory count", fontsize=10)
    ax_nv.legend(fontsize=8)
    ax_nv.grid(True, ls="--", alpha=0.35, axis="y")

    # 加分岔线（先完成绘图再加线）
    for ax in [ax_var, ax_ac1, ax_tau, ax_nv]:
        add_bifur_line(ax)
        ax.set_xlabel(r"Damage stress $\sigma$", fontsize=10)

    fig.suptitle(
        "Early Warning Signals of Critical Transition\n"
        r"Mitochondrial Network  —  Gillespie SSA  "
        rf"($\sigma_c \approx {sigma_c:.2f}$, Hill $n=2$ cooperative ROS)",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 打印 Kendall τ 结果
# ---------------------------------------------------------------------------

def print_kendall_results(res: ScanResult, sigma_c: float = SIGMA_C,
                          sigma_csd_min: float = 2.0,
                          sigma_csd_max: float = 2.022) -> None:
    """
    打印方差、AC1、恢复时间的 Kendall τ 检验结果。

    三个区间
    --------
    全区间         : 所有 sigma，n_valid ≥ 10
    分岔前 (σ<σ_c) : sigma ≤ sigma_c，n_valid ≥ 10
    CSD 窗口       : sigma ∈ [sigma_csd_min, sigma_csd_max]，n_valid ≥ 10
                     （修复阈值 sigma_0 之上至分岔前）
    """
    print("\n=== Kendall τ 趋势检验 ===")
    print(f"  分岔点 sigma_c = {sigma_c:.3f}")
    print(f"  CSD 窗口: sigma ∈ [{sigma_csd_min:.3f}, {sigma_csd_max:.3f}]"
          f"（有效轨迹数 ≥ 10）")
    print(f"\n  {'指标':<25} {'区间':<26} {'τ':>8} {'p-value':>12} {'n':>5} {'显著'}")
    print("-" * 88)

    tests = [
        # (label, metric, sigma_min, sigma_max, interval_str)
        ("rolling_var", res.rolling_var, None,            None,           "全区间 (n_valid≥10)"),
        ("rolling_var", res.rolling_var, None,            sigma_c,        f"sigma≤{sigma_c:.2f}"),
        ("rolling_var", res.rolling_var, sigma_csd_min,   sigma_csd_max,  f"CSD 窗口"),
        ("AC1",         res.ac1,         None,            None,           "全区间 (n_valid≥10)"),
        ("AC1",         res.ac1,         None,            sigma_c,        f"sigma≤{sigma_c:.2f}"),
        ("AC1",         res.ac1,         sigma_csd_min,   sigma_csd_max,  f"CSD 窗口"),
        ("tau_rec",     res.tau_rec,     None,            None,           "全区间 (n_valid≥10)"),
        ("tau_rec",     res.tau_rec,     sigma_csd_min,   sigma_csd_max,  f"CSD 窗口"),
    ]
    for name, metric, smin, smax, interval in tests:
        kt = kendall_tau_test(
            res.sigmas, metric,
            sigma_min=smin, sigma_max=smax,
            n_valid=res.n_valid, min_valid=10,
        )
        if np.isnan(kt["tau"]):
            print(f"  {name:<25} {interval:<26} {'N/A':>8} {'N/A':>12} {kt['n']:>5}  (样本不足)")
        else:
            sig_str = "*** 显著 ***" if kt["significant"] else "—"
            print(f"  {name:<25} {interval:<26} {kt['tau']:>8.4f} {kt['p_value']:>12.4f} "
                  f"{kt['n']:>5}  {sig_str}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # sigma 网格设计：
    #   [0, 2.0]       — 修复阈值之下，EWS 恒定；取少量代表点
    #   [2.0, 2.031]   — CSD 窗口，tau 在 sigma≈2.030 急剧升高；极密采样
    #   [2.1, 4.0]     — 分岔后单稳态；少量验证点
    sigmas_far  = np.array([0.0, 0.5, 1.0, 1.5, 1.9])
    sigmas_csd  = np.array([2.0, 2.005, 2.01, 2.015, 2.02, 2.025, 2.028, 2.030])
    sigmas_post = np.array([2.1, 2.5, 3.0, 4.0])
    SIGMAS = np.unique(np.concatenate([sigmas_far, sigmas_csd, sigmas_post]))

    N_RUNS      = 100     # 每 sigma 独立模拟次数
    T_END       = 1500.0  # 总时长（小时）— 需覆盖 CSD 窗口内 τ_r ≈ 100+ h
    T_BURN      = 400.0   # 热化时长
    DT          = DT_SAMPLE  # 采样间隔
    N_PROC      = None   # None = 全 CPU 核

    print("=" * 65)
    print("Early Warning Signals — Mitochondrial Network (Gillespie SSA)")
    print(f"  模型       : Hill n=2 cooperative ROS, BIFUR_PARAMS")
    print(f"  sigma_c    : {SIGMA_C:.3f} (鞍结分岔)")
    print(f"  sigma 范围 : [{SIGMAS[0]:.2f}, {SIGMAS[-1]:.2f}]  ({len(SIGMAS)} 点)")
    print(f"  n_runs     : {N_RUNS} per sigma")
    print(f"  t_end      : {T_END} h   t_burn : {T_BURN} h")
    print(f"  CPU 核     : {mp.cpu_count()}")
    print("=" * 65)

    t0 = time.time()
    scan = compute_scan(
        SIGMAS,
        p=BIFUR_PARAMS,
        n_runs=N_RUNS,
        t_end=T_END,
        t_burn=T_BURN,
        dt_sample=DT,
        n_processes=N_PROC,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f} s")

    print_kendall_results(scan)

    plot_ews(scan, save_path="figures/early_warning.pdf", sigma_c=SIGMA_C)

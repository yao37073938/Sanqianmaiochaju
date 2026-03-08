"""
analysis/sensitivity.py
========================
用 SALib 库对线粒体网络临界分岔点 sigma_c 做 Sobol 全局敏感性分析。

方法
----
- 采样方案：Saltelli 准随机序列（内部使用拉丁超立方增强均匀性），
  N_BASE = 1024，共生成 N_BASE*(D+2) ≈ 11 264 个参数组合（D=9 个参数）。
- 每个组合用二分搜索 + fsolve 数值求解 sigma_c（健康稳态消失的临界点）。
- 敏感性指数：一阶指数 S1（参数独立贡献）和全阶指数 ST（含交互效应）。
- 多进程加速（mp.Pool）。

分析的参数（及其生物学范围）
-----------------------------
  k_bio    : 生物合成速率        [1.0,  10.0]  /h
  k_fis    : 裂变速率            [0.01,  0.1]  /h
  k_fus    : 融合速率            [5e-5, 3e-4]  /h/线粒体
  k_mit    : 自噬速率            [0.01,  0.1]  /h
  K_d      : 自噬半饱和浓度      [10,   200 ]  线粒体数
  k_dam0   : 损伤基础速率系数    [0.01,  0.1]  /h/sigma
  alpha    : ROS 正反馈强度      [0.05,  0.5]  /h
  K_REPAIR : 抗氧化修复速率      [0.01,  0.2]  /h
  phi_c    : Hill 半饱和损伤分数 [0.1,   0.7]  无量纲

sigma_c 定义
------------
健康稳态（高 N）随 sigma 增大发生鞍结分岔而消失的临界外部损伤强度。
若在 [0, SIGMA_SEARCH_MAX] 内未找到（无双稳态），返回 NaN 并记录。
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
from scipy.optimize import fsolve
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze


# ---------------------------------------------------------------------------
# 搜索范围上限
# ---------------------------------------------------------------------------

SIGMA_SEARCH_MAX: float = 8.0   # 二分搜索上界
N_BISECT: int = 30              # 二分迭代次数（精度 ≈ 8/2^30 ≈ 7e-9）
RESIDUAL_TOL: float = 1e-5      # fsolve 残差容许值


# ---------------------------------------------------------------------------
# SALib 问题定义（9 个参数）
# ---------------------------------------------------------------------------

PROBLEM: dict = {
    "num_vars": 9,
    "names": [
        "k_bio",     # 0
        "k_fis",     # 1
        "k_fus",     # 2
        "k_mit",     # 3
        "K_d",       # 4
        "k_dam0",    # 5
        "alpha",     # 6
        "K_REPAIR",  # 7
        "phi_c",     # 8
    ],
    "bounds": [
        [1.0,    10.0   ],  # k_bio    /h
        [0.01,    0.1   ],  # k_fis    /h
        [5e-5,    3e-4  ],  # k_fus    /h/线粒体
        [0.01,    0.1   ],  # k_mit    /h
        [10.0,  200.0   ],  # K_d      线粒体数
        [0.01,    0.1   ],  # k_dam0   /h/sigma
        [0.05,    0.5   ],  # alpha    /h
        [0.01,    0.2   ],  # K_REPAIR /h
        [0.1,     0.7   ],  # phi_c    无量纲
    ],
}

#: 中文参数标签（用于图表）
PARAM_LABELS: list[str] = [
    r"$k_{\rm bio}$",
    r"$k_{\rm fis}$",
    r"$k_{\rm fus}$",
    r"$k_{\rm mit}$",
    r"$K_d$",
    r"$k_{\rm dam0}$",
    r"$\alpha$",
    r"$K_{\rm repair}$",
    r"$\phi_c$",
]

#: 参数名（英文，用于表格）
PARAM_NAMES: list[str] = PROBLEM["names"]


# ---------------------------------------------------------------------------
# 自包含 ODE 函数（不依赖全局 K_REPAIR / PHI_C）
# ---------------------------------------------------------------------------

def _k_dam(sigma: float, D: float, N: float,
           k_dam0: float, alpha: float, k_repair: float, phi_c: float) -> float:
    """Hill n=2 损伤率（基于损伤分数，含修复阈值）。"""
    phi = D / max(N, 1e-12)
    baseline = max(0.0, k_dam0 * sigma - k_repair)
    return baseline + alpha * phi ** 2 / (phi ** 2 + phi_c ** 2)


def _ode_rhs(y: np.ndarray, sigma: float,
             k_bio: float, k_fis: float, k_fus: float, k_mit: float,
             K_d: float, k_dam0: float, alpha: float,
             k_repair: float, phi_c: float) -> np.ndarray:
    """ODE 右端项（与 bifurcation.py 一致，参数完全自包含）。"""
    H = max(y[0], 1e-12)
    D = max(y[1], 1e-12)
    N = H + D
    kd = _k_dam(sigma, D, N, k_dam0, alpha, k_repair, phi_c)
    dH = k_bio + k_fis * H - k_fus * H * N - k_mit * H * D / (D + K_d) - kd * H
    dD = k_fis * D - k_fus * D * N - k_mit * D + kd * H
    return np.array([dH, dD])


def _numerical_jacobian(y: np.ndarray, sigma: float, params: tuple,
                        eps: float = 1e-5) -> np.ndarray:
    """2×2 中心差分 Jacobian。"""
    J = np.zeros((2, 2))
    for j in range(2):
        yp, ym = y.copy(), y.copy()
        yp[j] += eps
        ym[j] -= eps
        J[:, j] = (_ode_rhs(yp, sigma, *params) -
                   _ode_rhs(ym, sigma, *params)) / (2 * eps)
    return J


# ---------------------------------------------------------------------------
# 单参数组的 sigma_c 求解（二分搜索 + fsolve 延续法）
# ---------------------------------------------------------------------------

def _find_healthy_fp(sigma: float, y0: np.ndarray,
                     params: tuple) -> tuple[bool, np.ndarray]:
    """
    用 fsolve 从初值 y0 出发，寻找健康稳态不动点（高 N，稳定）。

    注意：要求找到的不动点 H_s > 0.1 * y0[0]（延续法健康支判断），
    防止 fsolve 跳跃到损伤支。

    Returns
    -------
    (found, fp) : found=True 表示找到稳定高-N 不动点；fp 为其坐标。
    """
    def eq(y):
        return _ode_rhs(np.abs(y), sigma, *params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol, _, ier, _ = fsolve(eq, y0, full_output=True)

    H_s, D_s = abs(sol[0]), abs(sol[1])
    if ier != 1 or H_s < 0.5:
        return False, y0

    res = np.max(np.abs(eq([H_s, D_s])))
    if res > RESIDUAL_TOL:
        return False, y0

    # 健康支延续判断：H_s 不能远低于初始猜测（防止跳跃到损伤支）
    H_min = max(0.1 * y0[0], 5.0)
    if H_s < H_min:
        return False, y0

    J = _numerical_jacobian(np.array([H_s, D_s]), sigma, params)
    eigs = np.linalg.eigvals(J)
    if not np.all(np.real(eigs) < 0):
        return False, y0

    return True, np.array([H_s, D_s])


def compute_sigma_c(row: np.ndarray) -> float:
    """
    对单个参数向量计算鞍结分岔点 sigma_c。

    使用二分搜索（N_BISECT 次迭代）+ fsolve 延续法追踪健康稳态。
    若健康稳态在 [0, SIGMA_SEARCH_MAX] 内从不存在或始终存在，返回 NaN。

    Parameters
    ----------
    row : shape (9,) 参数向量 [k_bio, k_fis, k_fus, k_mit, K_d,
                                k_dam0, alpha, K_REPAIR, phi_c]

    Returns
    -------
    float : sigma_c（/h），或 NaN
    """
    k_bio, k_fis, k_fus, k_mit, K_d, k_dam0, alpha, k_repair, phi_c = row
    params = (k_bio, k_fis, k_fus, k_mit, K_d, k_dam0, alpha, k_repair, phi_c)

    # 解析初值：sigma=0 时 D=0 的健康稳态
    disc = k_fis ** 2 + 4.0 * k_bio * k_fus
    if disc < 0:
        return np.nan
    H0 = (k_fis + np.sqrt(disc)) / (2.0 * k_fus)
    y0 = np.array([H0, 0.1])

    # 验证低 sigma 处存在健康稳态
    found_lo, fp_lo = _find_healthy_fp(0.0, y0, params)
    if not found_lo:
        return np.nan

    # 验证高 sigma 处健康稳态消失
    found_hi, _ = _find_healthy_fp(SIGMA_SEARCH_MAX, fp_lo.copy(), params)
    if found_hi:
        return np.nan  # sigma_c > SIGMA_SEARCH_MAX

    # 二分搜索
    sigma_lo = 0.0
    sigma_hi = SIGMA_SEARCH_MAX
    y_lo = fp_lo.copy()

    for _ in range(N_BISECT):
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        found_mid, fp_mid = _find_healthy_fp(sigma_mid, y_lo.copy(), params)
        if found_mid:
            sigma_lo = sigma_mid
            y_lo = fp_mid   # 延续：更新搜索起点
        else:
            sigma_hi = sigma_mid

    return 0.5 * (sigma_lo + sigma_hi)


# ---------------------------------------------------------------------------
# 多进程顶层工作函数（pickle 兼容）
# ---------------------------------------------------------------------------

def _worker_sigma_c(args: tuple) -> float:
    """计算单个 row 的 sigma_c（顶层函数，multiprocessing 兼容）。"""
    idx, row = args
    return compute_sigma_c(row)


# ---------------------------------------------------------------------------
# 批量计算 sigma_c
# ---------------------------------------------------------------------------

def evaluate_sigma_c(
    param_values: np.ndarray,
    n_processes: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    对 param_values 中每行参数并行计算 sigma_c。

    Parameters
    ----------
    param_values : shape (N_total, 9)
    n_processes  : 并行进程数（None = CPU 核数）
    verbose      : 打印进度

    Returns
    -------
    Y : shape (N_total,)，NaN 表示未找到分岔点
    """
    n_total = len(param_values)
    n_cpu = n_processes or mp.cpu_count()

    if verbose:
        print(f"  Computing sigma_c for {n_total} parameter sets "
              f"using {n_cpu} processes ...", flush=True)

    args = [(i, row) for i, row in enumerate(param_values)]

    with mp.Pool(n_cpu) as pool:
        results = pool.map(_worker_sigma_c, args)

    Y = np.array(results, dtype=float)

    n_nan = int(np.isnan(Y).sum())
    if verbose:
        print(f"  Done. NaN (no bistability): {n_nan}/{n_total} "
              f"({100*n_nan/n_total:.1f}%)", flush=True)

    return Y


# ---------------------------------------------------------------------------
# Sobol 敏感性分析
# ---------------------------------------------------------------------------

class SobolResult(NamedTuple):
    """Sobol 全局敏感性分析结果。"""
    S1: np.ndarray        # 一阶指数
    S1_conf: np.ndarray   # 95% 置信区间半宽
    ST: np.ndarray        # 全阶指数
    ST_conf: np.ndarray
    param_names: list[str]
    n_total: int          # 总模型评估次数
    n_nan: int            # NaN 数量
    Y_mean: float         # sigma_c 均值
    Y_std: float          # sigma_c 标准差


def run_sobol_analysis(
    N_base: int = 1024,
    n_processes: int | None = None,
    nan_fill: str = "mean",
    verbose: bool = True,
) -> tuple[SobolResult, np.ndarray, np.ndarray]:
    """
    完整 Sobol 全局敏感性分析流程。

    Parameters
    ----------
    N_base      : Saltelli 基础样本数（总样本 = N_base*(D+2)）
    n_processes : 并行进程数
    nan_fill    : NaN 填充策略 {'mean', 'max'}
    verbose     : 打印进度

    Returns
    -------
    result       : SobolResult NamedTuple
    param_values : 采样矩阵 shape (N_total, 9)
    Y            : sigma_c 向量（已填充 NaN）
    """
    D = PROBLEM["num_vars"]
    n_total = N_base * (D + 2)  # calc_second_order=False

    if verbose:
        print(f"\n{'='*60}")
        print("Sobol Global Sensitivity Analysis — sigma_c")
        print(f"  D = {D} 参数,  N_base = {N_base}")
        print(f"  总样本量 = {n_total}")
        print(f"  sigma_c 搜索范围: [0, {SIGMA_SEARCH_MAX}]")
        print(f"{'='*60}")

    # --- 1. Saltelli 采样 ---
    if verbose:
        print("Step 1: Saltelli sampling ...", flush=True)
    param_values = sobol_sample(PROBLEM, N=N_base, calc_second_order=False)

    # --- 2. 评估 sigma_c ---
    if verbose:
        print("Step 2: Evaluating sigma_c ...", flush=True)
    Y_raw = evaluate_sigma_c(param_values, n_processes=n_processes, verbose=verbose)

    n_nan = int(np.isnan(Y_raw).sum())
    Y_valid = Y_raw[~np.isnan(Y_raw)]

    # --- 3. NaN 填充 ---
    if nan_fill == "mean":
        fill_val = float(np.nanmean(Y_raw)) if n_nan < len(Y_raw) else SIGMA_SEARCH_MAX
    else:  # "max"
        fill_val = float(np.nanmax(Y_raw)) if n_nan < len(Y_raw) else SIGMA_SEARCH_MAX
    Y = np.where(np.isnan(Y_raw), fill_val, Y_raw)

    if verbose:
        print(f"\n  sigma_c 统计（有效值）:")
        print(f"    均值  = {np.mean(Y_valid):.4f}")
        print(f"    标准差 = {np.std(Y_valid):.4f}")
        print(f"    最小值 = {np.min(Y_valid):.4f}")
        print(f"    最大值 = {np.max(Y_valid):.4f}")
        print(f"    NaN 填充值 = {fill_val:.4f}")

    # --- 4. Sobol 分析 ---
    if verbose:
        print("\nStep 3: Sobol analysis ...", flush=True)

    Si = sobol_analyze(PROBLEM, Y, calc_second_order=False, print_to_console=False)

    result = SobolResult(
        S1=np.array(Si["S1"]),
        S1_conf=np.array(Si["S1_conf"]),
        ST=np.array(Si["ST"]),
        ST_conf=np.array(Si["ST_conf"]),
        param_names=PARAM_NAMES,
        n_total=n_total,
        n_nan=n_nan,
        Y_mean=float(np.nanmean(Y_raw)),
        Y_std=float(np.nanstd(Y_raw)),
    )

    return result, param_values, Y


# ---------------------------------------------------------------------------
# 打印摘要
# ---------------------------------------------------------------------------

def print_sobol_summary(result: SobolResult) -> None:
    """按全阶指数 ST 降序打印 Sobol 指数表。"""
    order = np.argsort(result.ST)[::-1]

    print(f"\n{'='*70}")
    print("Sobol 全局敏感性分析结果")
    print(f"  总样本量: {result.n_total}  |  NaN: {result.n_nan}")
    print(f"  sigma_c 均值: {result.Y_mean:.3f}  ±  {result.Y_std:.3f}")
    print(f"\n  {'参数':<12} {'S1':>8} {'±S1':>8} {'ST':>8} {'±ST':>8}  {'占总方差':>10}")
    print("-" * 70)

    total_ST = result.ST.sum()
    for i in order:
        name = result.param_names[i]
        s1   = result.S1[i]
        s1c  = result.S1_conf[i]
        st   = result.ST[i]
        stc  = result.ST_conf[i]
        frac = st / total_ST * 100 if total_ST > 0 else 0.0
        bar  = "█" * max(0, int(frac / 2))
        print(f"  {name:<12} {s1:>8.4f} {s1c:>8.4f} {st:>8.4f} {stc:>8.4f}  "
              f"{frac:>6.1f}%  {bar}")

    print(f"\n  最重要参数（ST 前三）：")
    for rank, i in enumerate(order[:3], 1):
        print(f"    {rank}. {result.param_names[i]}  (ST={result.ST[i]:.4f})")

    print(f"\n  交互效应显著的参数（ST - S1 > 0.05）：")
    for i in order:
        interaction = result.ST[i] - result.S1[i]
        if interaction > 0.05:
            print(f"    {result.param_names[i]}: ST-S1 = {interaction:.4f}")


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_sobol(
    result: SobolResult,
    save_path: str = "figures/sensitivity.pdf",
) -> None:
    """
    生成 Sobol 敏感性条形图（出版质量，300 dpi PDF）。

    图表布局
    --------
    - 主图：S1（一阶指数）和 ST（全阶指数）分组条形图，按 ST 降序排列
    - 误差条：95% 置信区间
    - 附图：sigma_c 直方图（有效值分布）
    """
    order = np.argsort(result.ST)[::-1]
    names_sorted  = [PARAM_LABELS[i] for i in order]
    S1_sorted     = result.S1[order]
    S1c_sorted    = result.S1_conf[order]
    ST_sorted     = result.ST[order]
    STc_sorted    = result.ST_conf[order]

    D = len(result.param_names)
    x = np.arange(D)
    width = 0.35

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    ax_bar, ax_hist = axes

    # ---- 条形图 ----
    c_s1 = "#1f4e79"   # 深蓝
    c_st = "#e67e22"   # 橙色

    bars_s1 = ax_bar.bar(
        x - width / 2, S1_sorted, width,
        label=r"$S_1$ (first-order)",
        color=c_s1, alpha=0.85,
        yerr=S1c_sorted, capsize=4, error_kw={"elinewidth": 1.2},
    )
    bars_st = ax_bar.bar(
        x + width / 2, ST_sorted, width,
        label=r"$S_T$ (total-order)",
        color=c_st, alpha=0.85,
        yerr=STc_sorted, capsize=4, error_kw={"elinewidth": 1.2},
    )

    # 标注数值（仅 ST > 0.05）
    for i, (st, bar) in enumerate(zip(ST_sorted, bars_st)):
        if st > 0.05:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + STc_sorted[i] + 0.01,
                f"{st:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=c_st,
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names_sorted, fontsize=10)
    ax_bar.set_ylabel("Sobol sensitivity index", fontsize=11)
    ax_bar.set_ylim(bottom=0.0)
    ax_bar.set_title(
        r"Sobol Global Sensitivity of $\sigma_c$ (Saddle-Node Bifurcation)"
        "\n"
        rf"$n = {result.n_total}$ evaluations  |  "
        rf"$\bar{{\sigma}}_c = {result.Y_mean:.3f} \pm {result.Y_std:.3f}$",
        fontsize=11,
    )
    ax_bar.legend(fontsize=10, loc="upper right")
    ax_bar.grid(True, ls="--", alpha=0.35, axis="y")
    ax_bar.axhline(0.05, color="gray", ls=":", lw=1.2,
                   label="threshold 0.05")

    # 在最重要参数旁标星
    for i_rank, i in enumerate(range(min(3, D))):
        stars = ["★", "☆", "☆"]
        ax_bar.text(
            i, -0.04,
            stars[i_rank] if i_rank == 0 else "▲",
            ha="center", va="top", fontsize=10,
            color="#c0392b" if i_rank == 0 else "#7f8c8d",
        )

    # ---- sigma_c 分布直方图 ----
    ax_hist.set_title(r"Distribution of $\sigma_c$", fontsize=10)
    ax_hist.set_xlabel(r"$\sigma_c$", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.grid(True, ls="--", alpha=0.35)
    ax_hist.text(
        0.05, 0.95,
        f"n = {result.n_total}\nNaN = {result.n_nan}",
        transform=ax_hist.transAxes,
        va="top", fontsize=9, color="gray",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    plt.close(fig)


def plot_sobol_with_hist(
    result: SobolResult,
    Y_raw: np.ndarray,
    save_path: str = "figures/sensitivity.pdf",
) -> None:
    """
    生成 Sobol 敏感性条形图 + sigma_c 分布直方图（出版质量）。

    Parameters
    ----------
    result  : SobolResult
    Y_raw   : 原始 sigma_c 向量（含 NaN）
    save_path : 保存路径
    """
    order = np.argsort(result.ST)[::-1]
    names_sorted  = [PARAM_LABELS[i] for i in order]
    S1_sorted     = result.S1[order]
    S1c_sorted    = result.S1_conf[order]
    ST_sorted     = result.ST[order]
    STc_sorted    = result.ST_conf[order]

    D = len(result.param_names)
    x = np.arange(D)
    width = 0.35

    fig = plt.figure(figsize=(14, 6))
    ax_bar = fig.add_axes([0.06, 0.12, 0.58, 0.78])
    ax_hist = fig.add_axes([0.70, 0.12, 0.27, 0.78])

    c_s1 = "#1f4e79"
    c_st = "#e67e22"

    # ---- 分组条形图 ----
    bars_s1 = ax_bar.bar(
        x - width / 2, S1_sorted, width,
        label=r"$S_1$ (first-order)",
        color=c_s1, alpha=0.85,
        yerr=S1c_sorted, capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#1a3a5c"},
    )
    bars_st = ax_bar.bar(
        x + width / 2, ST_sorted, width,
        label=r"$S_T$ (total-order)",
        color=c_st, alpha=0.85,
        yerr=STc_sorted, capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#a85700"},
    )

    # ST 数值标注
    for i, (st, stc, bar) in enumerate(zip(ST_sorted, STc_sorted, bars_st)):
        if st > 0.03:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + stc + 0.008,
                f"{st:.3f}",
                ha="center", va="bottom", fontsize=8, color="#a85700",
                fontweight="bold",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names_sorted, fontsize=11)
    ax_bar.set_ylabel("Sobol sensitivity index", fontsize=11)
    ymax = max(ST_sorted.max() + STc_sorted.max(), 0.15) * 1.25
    ax_bar.set_ylim(0, ymax)
    ax_bar.set_title(
        r"Global Sensitivity of $\sigma_c$ to Model Parameters"
        "\n"
        r"(Sobol indices, saddle-node bifurcation of mitochondrial network)",
        fontsize=11,
    )
    ax_bar.legend(fontsize=10, loc="upper right")
    ax_bar.grid(True, ls="--", alpha=0.3, axis="y")
    ax_bar.axhline(0.05, color="#27ae60", ls="--", lw=1.3, alpha=0.8,
                   label="significance threshold 0.05")

    # 重要参数标记（TOP-3）
    for rank_i in range(min(3, D)):
        color_ = ["#c0392b", "#e74c3c", "#e67e22"][rank_i]
        label_ = ["1st", "2nd", "3rd"][rank_i]
        ax_bar.annotate(
            label_,
            xy=(rank_i, ST_sorted[rank_i] + STc_sorted[rank_i] + 0.03),
            ha="center", va="bottom", fontsize=8.5,
            color=color_, fontweight="bold",
        )

    # ---- 直方图 ----
    Y_valid = Y_raw[np.isfinite(Y_raw)]
    ax_hist.hist(Y_valid, bins=40, color=c_s1, alpha=0.7, edgecolor="white", lw=0.4)
    ax_hist.axvline(np.mean(Y_valid), color="red", lw=1.5, ls="--",
                    label=rf"mean={np.mean(Y_valid):.2f}")
    ax_hist.axvline(np.median(Y_valid), color="orange", lw=1.5, ls=":",
                    label=rf"median={np.median(Y_valid):.2f}")
    ax_hist.set_xlabel(r"$\sigma_c$", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title(
        rf"$\sigma_c$ distribution"
        f"\n(n={len(Y_valid)} valid / {len(Y_raw)} total)",
        fontsize=10,
    )
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, ls="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # Saltelli N_base=1024 → 1024*(9+2)=11264 总评估次数
    N_BASE     = 1024
    N_PROC     = None  # None = 全部 CPU 核

    t0 = time.time()
    result, param_values, Y_filled = run_sobol_analysis(
        N_base=N_BASE,
        n_processes=N_PROC,
        nan_fill="mean",
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f} s")

    print_sobol_summary(result)

    # 重建原始 Y（含 NaN）用于直方图
    Y_raw = evaluate_sigma_c(param_values, n_processes=N_PROC, verbose=False)

    plot_sobol_with_hist(
        result, Y_raw,
        save_path="figures/sensitivity.pdf",
    )

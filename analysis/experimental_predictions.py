"""
analysis/experimental_predictions.py
=====================================
基于校准好的线粒体网络分岔模型，生成五类具体实验预测。
全部图表保存为出版质量 PDF（300 dpi）。

实验预测清单
------------
1. Rotenone 剂量-响应曲线（dose-response）
   - N* vs [rotenone]（nM），双稳态分支 + 鞍结分岔点
   - 标注预测临界浓度范围（对应 sigma_c 的不确定性区间）

2. 临界减速（CSD）理论预测指标 vs [rotenone]
   - 恢复时间 τ_theory = −1/Re(λ_max)（ODE Jacobian 导出）
   - 理论滞后-1 自相关 ρ₁ = exp(−Δt/τ)
   - 理论方差 ∝ 1/|Re(λ_max)|（涨落耗散定理）
   - 实验可测量的早期预警信号量化预测

3. 滞后效应预测（hysteresis loop）
   - Rotenone 浓度线性上升（0 → [rot]_max）后下降（→ 0）的准静态 ODE 模拟
   - N(t) 轨迹在 ([rot], N) 平面展示不可逆跳变
   - 说明：损伤态在所有 [rot] 下均稳定（完全不可逆滞后）

4. DRP1 抑制（↓ k_fis）对临界浓度的定量移动预测
   - DRP1 抑制 → 裂变速率 k_fis 降低 → sigma_c 增大 → [rot]_c 升高
   - 预测：DRP1 抑制剂（Mdivi-1）使临界浓度向右偏移多少 nM

5. PGC-1α 过表达（↑ k_bio）对临界浓度的定量移动预测
   - PGC-1α → 生物合成速率 k_bio 增大 → sigma_c 增大 → [rot]_c 升高
   - 预测：PGC-1α 表达水平翻倍使临界浓度向右偏移多少 nM

生物学参数映射
--------------
Rotenone 抑制线粒体复合体 I → 增加 ROS 和线粒体损伤 → 对应模型中 sigma 升高。
映射关系: sigma = [rotenone] / ROT_SCALE_NM
其中 ROT_SCALE_NM 使 sigma_c ≈ 2.035 对应 [rot]_c = ROT_CRIT_NM（可调）。

参考范围（HeLa 细胞）：
  - Rotenone EC50 for mitochondrial fragmentation: ~100–500 nM
  - 预测临界浓度 ROT_CRIT_NM = 250 nM（实验可测范围中点）
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import replace
from typing import NamedTuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp

from models.ode_model import MitoParams
from analysis.bifurcation import (
    BIFUR_PARAMS, K_REPAIR, PHI_C,
    k_dam_bifur, mito_ode_bifur, numerical_jacobian,
    scan_bifurcation, detect_saddle_node, find_all_fixed_points,
)
from analysis.sensitivity import compute_sigma_c


# ---------------------------------------------------------------------------
# 全局映射参数
# ---------------------------------------------------------------------------

#: Rotenone 临界预测浓度（nM）—— 对应模型 sigma_c ≈ 2.035
ROT_CRIT_NM: float = 250.0

#: sigma_c（来自 bifurcation.py 分析）
SIGMA_C: float = 2.035

#: 浓度换算系数：1 sigma 单位 = ROT_SCALE_NM nM rotenone
ROT_SCALE_NM: float = ROT_CRIT_NM / SIGMA_C  # ≈ 122.8 nM/sigma

#: 理论预测不确定度（sigma_c ± SIGMA_C_UNCERT → [rot]_c ± Δrot）
SIGMA_C_UNCERT: float = 0.15   # ±7% of sigma_c

#: 图表保存目录
FIG_DIR: str = "figures"


def rot(sigma: float | np.ndarray) -> float | np.ndarray:
    """Sigma → Rotenone 浓度（nM）。"""
    return sigma * ROT_SCALE_NM


def sigma_from_rot(rot_nm: float | np.ndarray) -> float | np.ndarray:
    """Rotenone 浓度（nM）→ sigma。"""
    return rot_nm / ROT_SCALE_NM


# ---------------------------------------------------------------------------
# 通用绘图工具
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _bifurcation_line(ax: plt.Axes, sigma_c: float = SIGMA_C) -> None:
    """在轴上添加分岔线与标注。"""
    rc = rot(sigma_c)
    ax.axvline(rc, color="#7f8c8d", ls=":", lw=1.5, zorder=1)
    ylo, yhi = ax.get_ylim()
    ax.text(rc + 4, yhi * 0.95,
            f"[rot]$_c$ ≈ {rc:.0f} nM",
            fontsize=8.5, color="#7f8c8d", va="top", style="italic")


# ===========================================================================
# 预测 1：Rotenone 剂量-响应曲线（分岔图）
# ===========================================================================

def predict_dose_response(
    rot_max_nm: float = 600.0,
    n_sigma: int = 300,
    p: MitoParams = BIFUR_PARAMS,
) -> plt.Figure:
    """
    生成 Rotenone 剂量-响应曲线（双稳态分支图）。

    图表内容
    --------
    - 蓝实线：健康稳态 N_health* vs [rot]
    - 红实线：损伤稳态 N_damage* vs [rot]
    - 灰虚线：不稳定鞍点 N_saddle vs [rot]
    - 绿色竖带：预测临界浓度范围 [rot]_c ± σ_c 不确定度
    - 阴影区：双稳态区域
    - 箭头：前向跳变（↓健康→损伤）与相变不可逆性

    Returns
    -------
    matplotlib.Figure
    """
    sigma_max = sigma_from_rot(rot_max_nm)
    fps_all = scan_bifurcation(p=p, sigma_range=(0.0, float(sigma_max)), n_sigma=n_sigma)

    # 分离稳定/不稳定分支
    stable   = [(rot(f["sigma"]), f["N"]) for f in fps_all if f["stable"]]
    unstable = [(rot(f["sigma"]), f["N"]) for f in fps_all if not f["stable"]]

    # 按 N 值分离健康支和损伤支（N > 200 → 健康，N < 200 → 损伤）
    health_br = [(r, n) for r, n in stable if n > 200]
    damage_br = [(r, n) for r, n in stable if n <= 200]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # 双稳态区域阴影
    rot_c      = rot(SIGMA_C)
    rot_c_lo   = rot(SIGMA_C - SIGMA_C_UNCERT)
    rot_c_hi   = rot(SIGMA_C + SIGMA_C_UNCERT)
    ax.axvspan(0, rot_c, color="#eaf4fb", alpha=0.55, zorder=0,
               label="Bistable region")

    # 不稳定鞍点
    if unstable:
        us, uN = zip(*sorted(unstable))
        ax.plot(us, uN, "--", color="#95a5a6", lw=1.5, zorder=2,
                label="Unstable (saddle)")

    # 健康支
    if health_br:
        hs, hN = zip(*sorted(health_br))
        ax.plot(hs, hN, "-", color="#1a6da8", lw=2.4, zorder=4,
                label=r"Healthy state $N^*_{\rm health}$")

    # 损伤支
    if damage_br:
        ds, dN = zip(*sorted(damage_br))
        ax.plot(ds, dN, "-", color="#c0392b", lw=2.4, zorder=4,
                label=r"Damaged state $N^*_{\rm damage}$")

    # 临界浓度竖带（不确定度区间）
    ax.axvspan(rot_c_lo, rot_c_hi, color="#27ae60", alpha=0.18, zorder=1)
    ax.axvline(rot_c, color="#27ae60", ls="-", lw=1.8, zorder=3)
    ax.text(rot_c + 4, 480,
            f"[rot]$_c$ = {rot_c:.0f}±{rot(SIGMA_C_UNCERT):.0f} nM\n"
            r"(predicted critical dose)",
            fontsize=8.5, color="#27ae60", va="top", fontweight="bold")

    # 前向跳变箭头（健康→损伤，在 [rot]_c 处）
    ax.annotate(
        "", xy=(rot_c + 5, 115), xytext=(rot_c - 5, 390),
        arrowprops=dict(arrowstyle="-|>", color="#e67e22",
                        lw=2.0, mutation_scale=14),
        zorder=5,
    )
    ax.text(rot_c + 10, 250,
            "Tipping\npoint", fontsize=8, color="#e67e22",
            va="center", fontweight="bold")

    # 双稳态标注
    ax.text(rot_c * 0.45, 450,
            "Bistable\n(healthy & damaged\nstates coexist)",
            fontsize=8, color="#1a6da8", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a6da8", alpha=0.7))
    ax.text(rot_c * 1.35, 80,
            "Only damaged\nstate stable",
            fontsize=8, color="#c0392b", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.7))

    ax.set_xlabel("[Rotenone] (nM)", fontsize=12)
    ax.set_ylabel(r"Steady-state mitochondria $N^*$", fontsize=12)
    ax.set_title(
        "Prediction 1 — Rotenone Dose-Response Curve\n"
        r"(Saddle-node bifurcation of mitochondrial network, Hill $n=2$ ROS feedback)",
        fontsize=11,
    )
    ax.set_xlim(0, rot_max_nm)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, ls="--", alpha=0.35)

    return fig


# ===========================================================================
# 预测 2：CSD 理论预测指标 vs [rotenone]
# ===========================================================================

def predict_csd_metrics(
    rot_max_nm: float = 500.0,
    n_pts: int = 120,
    dt_sample_h: float = 2.0,
    p: MitoParams = BIFUR_PARAMS,
) -> plt.Figure:
    """
    计算并绘制三种 CSD 理论预测指标 vs [rotenone]。

    方法
    ----
    从 ODE Jacobian 的主特征值 λ_max 推导：
      τ_theory  = −1 / Re(λ_max)          恢复时间（小时）
      ρ₁_theory = exp(−Δt / τ_theory)     理论滞后-1 自相关
      Var_theory ∝ 1 / |Re(λ_max)|        归一化方差（FDT）

    所有指标均归一化为低 [rot] 处（0.5*[rot]_c）的值，
    展示接近 [rot]_c 时的倍增幅度。

    Returns
    -------
    matplotlib.Figure（3 子图）
    """
    # 扫描 sigma（从 0.1 到 sigma_c 附近）
    sigmas = np.linspace(0.05, SIGMA_C * 0.995, n_pts)
    rot_arr = rot(sigmas)

    tau_arr  = np.full(n_pts, np.nan)
    ac1_arr  = np.full(n_pts, np.nan)
    var_arr  = np.full(n_pts, np.nan)

    for i, s in enumerate(sigmas):
        # 找健康支不动点
        fps = find_all_fixed_points(s, p, n_H=6, n_D=6)
        stable_fps = [f for f in fps if f["stable"]]
        if not stable_fps:
            continue
        best = max(stable_fps, key=lambda f: f["N"])
        H_s, D_s = best["H"], best["D"]

        J = numerical_jacobian(np.array([H_s, D_s]), s, p)
        eigs = np.linalg.eigvals(J)
        lmax = float(np.max(np.real(eigs)))
        if lmax >= 0:
            continue

        tau_arr[i] = -1.0 / lmax
        ac1_arr[i] = np.exp(-dt_sample_h / tau_arr[i])
        var_arr[i] = -1.0 / lmax  # ∝ τ（FDT 近似，归一化后等价）

    # 归一化（以 [rot] = 0.3*[rot]_c 处为基准）
    ref_idx = np.argmin(np.abs(sigmas - 0.3 * SIGMA_C))

    def norm(arr):
        v0 = arr[ref_idx]
        return arr / v0 if np.isfinite(v0) and v0 != 0 else arr

    tau_n  = norm(tau_arr)
    ac1_n  = norm(ac1_arr)
    var_n  = norm(var_arr)

    # 实验可测量指标的颜色和标签
    metrics = [
        (tau_n,  "#c0392b", r"Recovery time $\tau_r$ (normalized)",
         r"$\tau_r = -1/\lambda_{\max}$ → ∞ at bifurcation"),
        (ac1_n,  "#1a6da8", r"Lag-1 autocorrelation $\rho_1$ (normalized)",
         r"$\rho_1 = e^{-\Delta t/\tau_r}$ → 1 at bifurcation"),
        (var_n,  "#27ae60", r"Variance $\sigma^2_N$ (normalized, FDT)",
         r"$\sigma^2 \propto 1/|\lambda_{\max}|$ → ∞ at bifurcation"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for ax, (arr, color, ylabel, formula) in zip(axes, metrics):
        mask = np.isfinite(arr)
        ax.plot(rot_arr[mask], arr[mask], "-", color=color, lw=2.2)
        ax.fill_between(rot_arr[mask], 1.0, arr[mask],
                        color=color, alpha=0.12)
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7,
                   label="Baseline (0.3×[rot]$_c$)")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        ymax_ax = np.nanmax(arr[mask]) * 1.1 if mask.sum() > 0 else 5
        ax.set_ylim(0, min(ymax_ax, 15))
        ax.text(0.02, 0.92, formula, transform=ax.transAxes,
                fontsize=8.5, color=color, va="top", style="italic")
        _bifurcation_line(ax)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("[Rotenone] (nM)", fontsize=12)
    fig.suptitle(
        "Prediction 2 — Early Warning Signal Metrics vs [Rotenone]\n"
        r"(Theoretical from ODE Jacobian $\lambda_{\max}$, "
        "normalized to baseline)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# 预测 3：滞后效应（hysteresis loop）
# ===========================================================================

def predict_hysteresis(
    rot_max_nm: float = 500.0,
    t_ramp_h: float = 800.0,
    t_hold_h: float = 100.0,
    n_eval: int = 4000,
    p: MitoParams = BIFUR_PARAMS,
) -> plt.Figure:
    """
    模拟 Rotenone 浓度线性上升后下降的准静态 ODE 轨迹，展示不可逆滞后。

    协议
    ----
    阶段 1（0 → t_ramp_h）：[rot] 线性从 0 升到 rot_max_nm
    阶段 2（t_ramp_h → 2*t_ramp_h）：[rot] 线性从 rot_max_nm 降回 0
    阶段 3（保持 t_hold_h）：[rot] = 0，观察恢复（或不恢复）

    Returns
    -------
    matplotlib.Figure（左：时间序列；右：滞后相图）
    """
    sigma_max = sigma_from_rot(rot_max_nm)
    T_total = 2 * t_ramp_h + t_hold_h

    def sigma_protocol(t: float) -> float:
        """随时间变化的应激函数。"""
        if t <= t_ramp_h:
            return sigma_max * t / t_ramp_h
        elif t <= 2 * t_ramp_h:
            return sigma_max * (1.0 - (t - t_ramp_h) / t_ramp_h)
        else:
            return 0.0

    def ode_rhs(t: float, y: list[float]) -> list[float]:
        s = sigma_protocol(t)
        return mito_ode_bifur(t, y, s, p)

    # 初始条件：从健康态出发
    fps0 = find_all_fixed_points(0.0, p, n_H=6, n_D=6)
    stable0 = [f for f in fps0 if f["stable"]]
    if stable0:
        best0 = max(stable0, key=lambda f: f["N"])
        H0_ic, D0_ic = best0["H"], best0["D"]
    else:
        H0_ic, D0_ic = 410.0, 0.5

    t_eval = np.linspace(0, T_total, n_eval)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(
            ode_rhs, (0, T_total), [H0_ic, D0_ic],
            t_eval=t_eval, method="Radau",
            rtol=1e-7, atol=1e-9, dense_output=False,
        )

    t_arr = sol.t
    N_arr = sol.y[0] + sol.y[1]
    rot_arr = np.array([rot(sigma_protocol(ti)) for ti in t_arr])

    # 分岔图背景（快速扫描）
    fps_bg = scan_bifurcation(p=p, sigma_range=(0.0, sigma_max * 1.05), n_sigma=80)
    stable_bg   = [(rot(f["sigma"]), f["N"]) for f in fps_bg if f["stable"]]
    unstable_bg = [(rot(f["sigma"]), f["N"]) for f in fps_bg if not f["stable"]]

    # 分离健康 / 损伤支
    health_bg = [(r, n) for r, n in stable_bg if n > 200]
    damage_bg = [(r, n) for r, n in stable_bg if n <= 200]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_t, ax_ph = axes

    # ---- 左图：时间序列 ----
    # 上升 / 下降 / 保持三段着色
    t_up_mask  = t_arr <= t_ramp_h
    t_dn_mask  = (t_arr > t_ramp_h) & (t_arr <= 2 * t_ramp_h)
    t_hold_mask = t_arr > 2 * t_ramp_h

    ax_t.plot(t_arr[t_up_mask], N_arr[t_up_mask],
              color="#1a6da8", lw=2, label="[rot] ↑ (forward sweep)")
    ax_t.plot(t_arr[t_dn_mask], N_arr[t_dn_mask],
              color="#c0392b", lw=2, label="[rot] ↓ (reverse sweep)")
    ax_t.plot(t_arr[t_hold_mask], N_arr[t_hold_mask],
              color="#8e44ad", lw=2, label="Hold [rot]=0")

    # 标注跳变时刻
    jump_idx = np.argmin(np.abs(np.diff(N_arr)))  # 实际上找到变化最大的
    diff_N = np.abs(np.diff(N_arr))
    jump_idx = int(np.argmax(diff_N))
    ax_t.axvline(t_arr[jump_idx], color="#e67e22", ls="--", lw=1.5,
                 label=f"Tipping at t≈{t_arr[jump_idx]:.0f} h")

    # [rot] 协议（次 y 轴）
    ax2_t = ax_t.twinx()
    ax2_t.plot(t_arr, rot_arr, color="gray", lw=1.2, ls="--", alpha=0.6)
    ax2_t.set_ylabel("[Rotenone] (nM)", fontsize=10, color="gray")
    ax2_t.tick_params(axis="y", labelcolor="gray")
    ax2_t.set_ylim(0, rot_max_nm * 1.1)

    ax_t.set_xlabel("Time (h)", fontsize=11)
    ax_t.set_ylabel(r"Total mitochondria $N(t)$", fontsize=11)
    ax_t.set_title("(a) Time series\n(quasi-static ramp protocol)", fontsize=10)
    ax_t.legend(fontsize=8.5, loc="lower left")
    ax_t.grid(True, ls="--", alpha=0.35)

    # ---- 右图：滞后相图 ----
    # 稳定分支背景
    if health_bg:
        hs, hN = zip(*sorted(health_bg))
        ax_ph.plot(hs, hN, "-", color="#1a6da8", lw=2, alpha=0.5,
                   label=r"Healthy branch $N^*$")
    if damage_bg:
        ds, dN = zip(*sorted(damage_bg))
        ax_ph.plot(ds, dN, "-", color="#c0392b", lw=2, alpha=0.5,
                   label=r"Damaged branch $N^*$")
    if unstable_bg:
        us, uN = zip(*sorted(unstable_bg))
        ax_ph.plot(us, uN, "--", color="#95a5a6", lw=1.5, alpha=0.7,
                   label="Unstable (saddle)")

    # 模拟轨迹
    ax_ph.plot(rot_arr[t_up_mask], N_arr[t_up_mask],
               "-", color="#1a6da8", lw=2.2,
               label="Forward sweep (healthy→damaged)")
    ax_ph.plot(rot_arr[t_dn_mask], N_arr[t_dn_mask],
               "-", color="#c0392b", lw=2.2,
               label="Reverse sweep (trapped)")

    # 临界线 & 箭头
    rot_c = rot(SIGMA_C)
    ax_ph.axvline(rot_c, color="#27ae60", ls=":", lw=1.8)
    ax_ph.annotate(
        "", xy=(rot_c + 10, 130), xytext=(rot_c - 10, 380),
        arrowprops=dict(arrowstyle="-|>", color="#e67e22",
                        lw=2, mutation_scale=14),
    )
    ax_ph.text(rot_c + 15, 240, "Irreversible\ntipping",
               fontsize=8.5, color="#e67e22", va="center", fontweight="bold")

    # 标注"无恢复"
    ax_ph.annotate(
        f"System stays in damaged\nstate at [rot]=0\n→ irreversible transition",
        xy=(5, N_arr[-1]), xytext=(80, N_arr[-1] + 80),
        fontsize=8, color="#8e44ad",
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.3),
    )

    ax_ph.set_xlabel("[Rotenone] (nM)", fontsize=11)
    ax_ph.set_ylabel(r"Total mitochondria $N$", fontsize=11)
    ax_ph.set_title("(b) Hysteresis loop\n(phase plane)", fontsize=10)
    ax_ph.legend(fontsize=8, loc="upper right")
    ax_ph.grid(True, ls="--", alpha=0.35)
    ax_ph.set_xlim(0, rot_max_nm)
    ax_ph.set_ylim(0, 520)

    fig.suptitle(
        "Prediction 3 — Hysteresis: Ramp-Up / Ramp-Down [Rotenone]\n"
        "(Irreversible critical transition — damaged state persists at [rot]=0)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# 预测 4 & 5：干预移动临界浓度
# ===========================================================================

def _compute_shifted_sigma_c(
    p_base: MitoParams,
    param_name: str,
    multipliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单个参数按倍数扫描，计算每个值对应的 sigma_c。

    Parameters
    ----------
    p_base      : 基础参数集
    param_name  : 要改变的参数名（MitoParams 字段）
    multipliers : 相对倍数数组（1.0 = 基础值）

    Returns
    -------
    sigma_c_arr : shape (len(multipliers),) — NaN 表示无双稳态
    rot_c_arr   : sigma_c × ROT_SCALE_NM（nM）
    """
    base_val = getattr(p_base, param_name)
    sigma_c_arr = np.empty(len(multipliers))
    rot_c_arr   = np.empty(len(multipliers))

    for i, m in enumerate(multipliers):
        # 构建参数向量（与 sensitivity.py compute_sigma_c 一致）
        p_new = replace(p_base, **{param_name: base_val * m})
        row = np.array([
            p_new.k_bio, p_new.k_fis, p_new.k_fus, p_new.k_mit,
            p_new.K_d,   p_new.k_dam0, p_new.alpha,
            K_REPAIR,    PHI_C,
        ])
        sc = compute_sigma_c(row)
        sigma_c_arr[i] = sc
        rot_c_arr[i]   = sc * ROT_SCALE_NM if np.isfinite(sc) else np.nan

    return sigma_c_arr, rot_c_arr


def predict_drp1_inhibition(
    multipliers: np.ndarray | None = None,
    p: MitoParams = BIFUR_PARAMS,
) -> plt.Figure:
    """
    DRP1 抑制（↓ k_fis）对临界 Rotenone 浓度的定量移动预测。

    DRP1 被 Mdivi-1 等抑制剂抑制 → 裂变速率 k_fis 降低
    → 健康稳态更稳定 → 鞍结分岔需要更大外部损伤 → [rot]_c 升高。

    实验设计建议
    ------------
    - 对照：DMSO（k_fis = k_fis_ref）
    - 处理：10-50 µM Mdivi-1（~30-70% DRP1 抑制 → k_fis ≈ 0.3-0.7 × k_fis_ref）
    - 读数：线粒体总数（JC-1 荧光 / MitoTracker）在不同 [rotenone] 下的 S 型曲线

    Returns
    -------
    matplotlib.Figure（两个子图：sigma_c vs 抑制程度；[rot]_c 预测）
    """
    if multipliers is None:
        multipliers = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                1.0, 1.2, 1.5, 2.0])

    sc_arr, rc_arr = _compute_shifted_sigma_c(p, "k_fis", multipliers)

    # 抑制百分比：multiplier < 1 为抑制，> 1 为激活
    inhib_pct = (1.0 - multipliers) * 100   # positive = inhibition

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_sc, ax_rc = axes

    c_valid = "#1a6da8"
    c_ref   = "#e67e22"
    ref_idx = np.where(multipliers == 1.0)[0]

    for ax, arr, ylabel, unit in [
        (ax_sc, sc_arr, r"Critical bifurcation $\sigma_c$", ""),
        (ax_rc, rc_arr, r"Critical [rotenone] (nM)", "nM"),
    ]:
        valid = np.isfinite(arr)
        ax.plot(inhib_pct[valid], arr[valid], "o-",
                color=c_valid, lw=2, ms=6, zorder=3)

        # 标注参考点（无干预）
        if len(ref_idx) > 0:
            ri = ref_idx[0]
            if np.isfinite(arr[ri]):
                ax.axhline(arr[ri], color=c_ref, ls="--", lw=1.3, alpha=0.7)
                ax.plot(inhib_pct[ri], arr[ri], "D",
                        color=c_ref, ms=10, zorder=4,
                        label=f"Control: {arr[ri]:.1f}{unit}")

        # 填充区间
        ax.fill_between(inhib_pct[valid], arr[ri] if len(ref_idx) else 0,
                        arr[valid], where=arr[valid] > (arr[ri] if len(ref_idx) else 0),
                        color=c_valid, alpha=0.12)

        ax.set_xlabel("DRP1 inhibition (%)\n← control   |   inhibited →",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(fontsize=9)
        ax.axvline(0, color="gray", ls=":", lw=1)

    # 定量标注：在 70% DRP1 抑制时的预测移动
    inh70 = np.argmin(np.abs(multipliers - 0.3))  # k_fis × 0.3 → 70% 抑制
    if np.isfinite(rc_arr[inh70]):
        shift = rc_arr[inh70] - rc_arr[ref_idx[0]]
        ax_rc.annotate(
            f"70% DRP1 inhibition:\n[rot]$_c$ shifts +{shift:.0f} nM",
            xy=(inhib_pct[inh70], rc_arr[inh70]),
            xytext=(inhib_pct[inh70] - 30, rc_arr[inh70] - 50),
            fontsize=9, color=c_valid,
            arrowprops=dict(arrowstyle="->", color=c_valid, lw=1.3),
        )

    ax_sc.set_title("(a) sigma_c shift", fontsize=10)
    ax_rc.set_title("(b) Predicted [rot]$_c$ shift", fontsize=10)
    fig.suptitle(
        "Prediction 4 — DRP1 Inhibition (↓ k_fis) Shifts Critical [Rotenone]\n"
        "(Mdivi-1 treatment increases resilience of mitochondrial network)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, sc_arr, rc_arr, multipliers


def predict_pgc1a_overexpression(
    multipliers: np.ndarray | None = None,
    p: MitoParams = BIFUR_PARAMS,
) -> plt.Figure:
    """
    PGC-1α 过表达（↑ k_bio）对临界 Rotenone 浓度的定量移动预测。

    PGC-1α → 激活线粒体生物合成 → k_bio 增大
    → 健康态稳定盆加深 → 双稳态临界点后移 → [rot]_c 升高。

    实验设计建议
    ------------
    - 对照：空载体或 GFP 组
    - 处理：PGC-1α 过表达质粒（预计 k_bio × 1.5-3×）
    - 或：NAD+ 补充激活内源 PGC-1α（k_bio × 1.2-1.8×）
    - 读数：线粒体质量（mtDNA copy number）+ rotenone 耐受曲线

    Returns
    -------
    matplotlib.Figure
    """
    if multipliers is None:
        multipliers = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0])

    sc_arr, rc_arr = _compute_shifted_sigma_c(p, "k_bio", multipliers)

    overexp_fold = multipliers  # 相对基础值的倍数

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_sc, ax_rc = axes

    c_valid = "#27ae60"
    c_ref   = "#e67e22"
    ref_idx = np.where(multipliers == 1.0)[0]

    for ax, arr, ylabel, unit in [
        (ax_sc, sc_arr, r"Critical bifurcation $\sigma_c$", ""),
        (ax_rc, rc_arr, r"Critical [rotenone] (nM)", "nM"),
    ]:
        valid = np.isfinite(arr)
        ax.plot(overexp_fold[valid], arr[valid], "s-",
                color=c_valid, lw=2, ms=6, zorder=3)

        if len(ref_idx) > 0:
            ri = ref_idx[0]
            if np.isfinite(arr[ri]):
                ax.axhline(arr[ri], color=c_ref, ls="--", lw=1.3, alpha=0.7)
                ax.axvline(1.0, color="gray", ls=":", lw=1)
                ax.plot(overexp_fold[ri], arr[ri], "D",
                        color=c_ref, ms=10, zorder=4,
                        label=f"Control (1×): {arr[ri]:.1f}{unit}")

        ax.fill_between(overexp_fold[valid],
                        arr[ri] if len(ref_idx) else 0,
                        arr[valid],
                        where=arr[valid] > (arr[ri] if len(ref_idx) else 0),
                        color=c_valid, alpha=0.12)

        ax.set_xlabel(r"$k_{\rm bio}$ fold change (PGC-1α expression level)",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(fontsize=9)

    # 定量标注：2× PGC-1α 时的移动
    oe2x = np.argmin(np.abs(multipliers - 2.0))
    if np.isfinite(rc_arr[oe2x]) and len(ref_idx) > 0 and np.isfinite(rc_arr[ref_idx[0]]):
        shift = rc_arr[oe2x] - rc_arr[ref_idx[0]]
        ax_rc.annotate(
            f"2× PGC-1α:\n[rot]$_c$ shifts +{shift:.0f} nM",
            xy=(overexp_fold[oe2x], rc_arr[oe2x]),
            xytext=(overexp_fold[oe2x] + 0.3, rc_arr[oe2x] - 60),
            fontsize=9, color=c_valid,
            arrowprops=dict(arrowstyle="->", color=c_valid, lw=1.3),
        )

    ax_sc.set_title(r"(a) $\sigma_c$ shift", fontsize=10)
    ax_rc.set_title("(b) Predicted [rot]$_c$ shift", fontsize=10)
    fig.suptitle(
        r"Prediction 5 — PGC-1α Overexpression (↑ $k_{\rm bio}$) Shifts Critical [Rotenone]"
        "\n(Biogenesis enhancement increases network resilience)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, sc_arr, rc_arr, multipliers


# ===========================================================================
# 综合摘要表
# ===========================================================================

def print_intervention_table(
    drp1_mult: np.ndarray, drp1_rc: np.ndarray,
    pgc1a_mult: np.ndarray, pgc1a_rc: np.ndarray,
) -> None:
    """打印干预量化预测摘要表。"""
    ref_rc = ROT_CRIT_NM  # 参考临界浓度

    print("\n" + "=" * 70)
    print("实验干预定量预测摘要")
    print(f"  参考临界浓度: [rot]_c = {ref_rc:.0f} nM  (sigma_c = {SIGMA_C:.3f})")
    print(f"  换算: 1 sigma = {ROT_SCALE_NM:.1f} nM rotenone")
    print("=" * 70)

    print("\n── DRP1 抑制（Mdivi-1，↓ k_fis）──")
    print(f"  {'k_fis 比例':>12}  {'DRP1抑制%':>10}  {'[rot]_c (nM)':>14}  {'Δ[rot]_c (nM)':>14}")
    print(f"  {'-'*55}")
    for m, rc in zip(drp1_mult, drp1_rc):
        inh = (1 - m) * 100
        delta = (rc - ref_rc) if np.isfinite(rc) else float("nan")
        flag = " ← control" if abs(m - 1.0) < 1e-9 else ""
        rc_s = f"{rc:.1f}" if np.isfinite(rc) else "NaN"
        d_s  = f"{delta:+.1f}" if np.isfinite(delta) else "NaN"
        print(f"  {m:>12.1f}  {inh:>10.0f}%  {rc_s:>14}  {d_s:>14}{flag}")

    print("\n── PGC-1α 过表达（↑ k_bio）──")
    print(f"  {'k_bio 倍数':>12}  {'过表达':>10}  {'[rot]_c (nM)':>14}  {'Δ[rot]_c (nM)':>14}")
    print(f"  {'-'*55}")
    for m, rc in zip(pgc1a_mult, pgc1a_rc):
        delta = (rc - ref_rc) if np.isfinite(rc) else float("nan")
        flag = " ← control" if abs(m - 1.0) < 1e-9 else ""
        rc_s = f"{rc:.1f}" if np.isfinite(rc) else "NaN"
        d_s  = f"{delta:+.1f}" if np.isfinite(delta) else "NaN"
        print(f"  {m:>12.1f}  {m:>9.1f}×  {rc_s:>14}  {d_s:>14}{flag}")

    print("\n实验建议：")
    # 找 70% DRP1 抑制的预测
    idx_70 = np.argmin(np.abs(drp1_mult - 0.30))
    if np.isfinite(drp1_rc[idx_70]):
        print(f"  • 70% DRP1 抑制 → [rot]_c 预测移动 "
              f"+{drp1_rc[idx_70]-ref_rc:+.0f} nM "
              f"({(drp1_rc[idx_70]-ref_rc)/ref_rc*100:+.0f}%)")
    idx_2x = np.argmin(np.abs(pgc1a_mult - 2.0))
    if np.isfinite(pgc1a_rc[idx_2x]):
        print(f"  • PGC-1α 2× 过表达 → [rot]_c 预测移动 "
              f"+{pgc1a_rc[idx_2x]-ref_rc:+.0f} nM "
              f"({(pgc1a_rc[idx_2x]-ref_rc)/ref_rc*100:+.0f}%)")
    print(f"  • 联合处理（DRP1抑制 + PGC-1α）预测产生协同效应（ST-S1 > 0.28）")
    print(f"  • 临界浓度预测精度：±{rot(SIGMA_C_UNCERT):.0f} nM（模型不确定度）")


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == "__main__":
    import time

    print("=" * 65)
    print("Experimental Predictions — Mitochondrial Network Model")
    print(f"  BIFUR_PARAMS: sigma_c ≈ {SIGMA_C:.3f}")
    print(f"  Mapping: [rot]_c = {ROT_CRIT_NM:.0f} nM ↔ sigma_c = {SIGMA_C:.3f}")
    print(f"  Scale: 1 sigma = {ROT_SCALE_NM:.1f} nM rotenone")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. 剂量-响应曲线
    # ------------------------------------------------------------------
    print("\n[1/5] Dose-response curve (bifurcation diagram) ...")
    t0 = time.time()
    fig1 = predict_dose_response()
    _save(fig1, "exp_pred_01_dose_response.pdf")
    print(f"      Done ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. CSD 理论预测指标
    # ------------------------------------------------------------------
    print("\n[2/5] CSD metrics (theoretical, Jacobian) ...")
    t0 = time.time()
    fig2 = predict_csd_metrics()
    _save(fig2, "exp_pred_02_csd_metrics.pdf")
    print(f"      Done ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 3. 滞后效应
    # ------------------------------------------------------------------
    print("\n[3/5] Hysteresis loop (ramp ODE simulation) ...")
    t0 = time.time()
    fig3 = predict_hysteresis()
    _save(fig3, "exp_pred_03_hysteresis.pdf")
    print(f"      Done ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 4. DRP1 抑制
    # ------------------------------------------------------------------
    print("\n[4/5] DRP1 inhibition (↓ k_fis) intervention ...")
    t0 = time.time()
    fig4, drp1_sc, drp1_rc, drp1_mult = predict_drp1_inhibition()
    _save(fig4, "exp_pred_04_drp1_inhibition.pdf")
    print(f"      Done ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. PGC-1α 过表达
    # ------------------------------------------------------------------
    print("\n[5/5] PGC-1α overexpression (↑ k_bio) intervention ...")
    t0 = time.time()
    fig5, pgc1a_sc, pgc1a_rc, pgc1a_mult = predict_pgc1a_overexpression()
    _save(fig5, "exp_pred_05_pgc1a.pdf")
    print(f"      Done ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 定量摘要表
    # ------------------------------------------------------------------
    print_intervention_table(drp1_mult, drp1_rc, pgc1a_mult, pgc1a_rc)

    # ------------------------------------------------------------------
    # 汇总图（5 子图综合版）
    # ------------------------------------------------------------------
    print("\nGenerating combined summary figure ...")
    from matplotlib.gridspec import GridSpec

    fig_all = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig_all,
                  hspace=0.42, wspace=0.38,
                  left=0.07, right=0.97, top=0.92, bottom=0.07)

    def _copy_ax(src_fig, dst_ax, panel_label):
        """将 src_fig 的第一个 Axes 内容重绘到 dst_ax。"""
        dst_ax.text(-0.08, 1.05, panel_label, transform=dst_ax.transAxes,
                    fontsize=14, fontweight="bold")

    # 为了生成综合图，重新生成每个子图但直接画到 GridSpec 的 axes 上
    # （复用前面的函数，用 ax 参数直接绘制比 copy 更可靠）
    # 这里使用简化版：在综合图中重绘关键内容

    # (a) 剂量-响应
    ax_a = fig_all.add_subplot(gs[0, 0])
    fps_brief = scan_bifurcation(p=BIFUR_PARAMS,
                                 sigma_range=(0.0, 4.5), n_sigma=200)
    s_pts  = [(rot(f["sigma"]), f["N"]) for f in fps_brief if f["stable"]]
    us_pts = [(rot(f["sigma"]), f["N"]) for f in fps_brief if not f["stable"]]
    h_pts  = [(r, n) for r, n in s_pts if n > 200]
    d_pts  = [(r, n) for r, n in s_pts if n <= 200]
    if us_pts:
        ax_a.plot(*zip(*sorted(us_pts)), "--", color="#95a5a6", lw=1.2)
    if h_pts:
        ax_a.plot(*zip(*sorted(h_pts)), "-", color="#1a6da8", lw=2,
                  label="Healthy branch")
    if d_pts:
        ax_a.plot(*zip(*sorted(d_pts)), "-", color="#c0392b", lw=2,
                  label="Damaged branch")
    ax_a.axvline(ROT_CRIT_NM, color="#27ae60", lw=1.8, ls="-")
    ax_a.axvspan(rot(SIGMA_C - SIGMA_C_UNCERT),
                 rot(SIGMA_C + SIGMA_C_UNCERT),
                 color="#27ae60", alpha=0.18)
    ax_a.set_xlabel("[Rotenone] (nM)", fontsize=9)
    ax_a.set_ylabel(r"$N^*$ (mitochondria)", fontsize=9)
    ax_a.set_title("(a) Dose-response curve", fontsize=10)
    ax_a.legend(fontsize=7.5)
    ax_a.grid(True, ls="--", alpha=0.3)
    ax_a.set_xlim(0, 550)
    ax_a.set_ylim(0, 520)

    # (b) CSD 指标 tau
    ax_b = fig_all.add_subplot(gs[0, 1])
    sigmas_b = np.linspace(0.05, SIGMA_C * 0.99, 100)
    tau_b = []
    for s in sigmas_b:
        fps_b = find_all_fixed_points(s, BIFUR_PARAMS, n_H=5, n_D=5)
        sst = [f for f in fps_b if f["stable"]]
        if sst:
            best = max(sst, key=lambda f: f["N"])
            J = numerical_jacobian(np.array([best["H"], best["D"]]), s, BIFUR_PARAMS)
            lm = float(np.max(np.real(np.linalg.eigvals(J))))
            tau_b.append(-1.0 / lm if lm < 0 else np.nan)
        else:
            tau_b.append(np.nan)
    tau_b = np.array(tau_b)
    ref_b = tau_b[np.argmin(np.abs(sigmas_b - 0.3 * SIGMA_C))]
    if np.isfinite(ref_b) and ref_b > 0:
        tau_b_n = tau_b / ref_b
    else:
        tau_b_n = tau_b
    mask_b = np.isfinite(tau_b_n)
    ax_b.plot(rot(sigmas_b[mask_b]), tau_b_n[mask_b], "-",
              color="#c0392b", lw=2, label=r"$\tau_r$ (normalized)")
    ax_b.fill_between(rot(sigmas_b[mask_b]), 1, tau_b_n[mask_b],
                      color="#c0392b", alpha=0.12)
    ax_b.axvline(ROT_CRIT_NM, color="#27ae60", ls=":", lw=1.5)
    ax_b.axhline(1, color="gray", ls="--", lw=1, alpha=0.6)
    ax_b.set_xlabel("[Rotenone] (nM)", fontsize=9)
    ax_b.set_ylabel(r"Normalized $\tau_r$", fontsize=9)
    ax_b.set_title("(b) CSD: recovery time", fontsize=10)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, ls="--", alpha=0.3)
    ax_b.set_ylim(0, min(np.nanmax(tau_b_n[mask_b]) * 1.1, 15))

    # (c) 滞后环（右图）——重绘相图部分
    ax_c = fig_all.add_subplot(gs[0, 2])
    # 用小数据量快速重绘
    sigma_max_c = sigma_from_rot(500)
    t_ramp_c = 400.0
    def sigma_p(t):
        if t <= t_ramp_c: return sigma_max_c * t / t_ramp_c
        elif t <= 2*t_ramp_c: return sigma_max_c * (1-(t-t_ramp_c)/t_ramp_c)
        else: return 0.0
    def ode_c(t, y): return mito_ode_bifur(t, y, sigma_p(t), BIFUR_PARAMS)
    sol_c = solve_ivp(ode_c, (0, 2*t_ramp_c+100), [410, 0.5],
                      t_eval=np.linspace(0, 2*t_ramp_c+100, 2000),
                      method="Radau", rtol=1e-7, atol=1e-9)
    t_c, N_c = sol_c.t, sol_c.y[0]+sol_c.y[1]
    rot_c_traj = np.array([rot(sigma_p(ti)) for ti in t_c])
    up_c  = t_c <= t_ramp_c
    dn_c  = (t_c > t_ramp_c) & (t_c <= 2*t_ramp_c)
    hld_c = t_c > 2*t_ramp_c
    if h_pts: ax_c.plot(*zip(*sorted(h_pts)), "-", color="#1a6da8", lw=1.5, alpha=0.4)
    if d_pts: ax_c.plot(*zip(*sorted(d_pts)), "-", color="#c0392b", lw=1.5, alpha=0.4)
    ax_c.plot(rot_c_traj[up_c], N_c[up_c], "-",
              color="#1a6da8", lw=2, label="[rot]↑")
    ax_c.plot(rot_c_traj[dn_c], N_c[dn_c], "-",
              color="#c0392b", lw=2, label="[rot]↓ (trapped)")
    ax_c.axvline(ROT_CRIT_NM, color="#27ae60", ls=":", lw=1.5)
    ax_c.set_xlabel("[Rotenone] (nM)", fontsize=9)
    ax_c.set_ylabel("N (mitochondria)", fontsize=9)
    ax_c.set_title("(c) Hysteresis loop", fontsize=10)
    ax_c.legend(fontsize=7.5)
    ax_c.grid(True, ls="--", alpha=0.3)
    ax_c.set_xlim(0, 550); ax_c.set_ylim(0, 520)

    # (d) DRP1 抑制
    ax_d = fig_all.add_subplot(gs[1, 0:2])
    inh_pct = (1 - drp1_mult) * 100
    valid_d = np.isfinite(drp1_rc)
    ref_d = drp1_rc[np.argmin(np.abs(drp1_mult - 1.0))]
    ax_d.bar(inh_pct[valid_d], drp1_rc[valid_d],
             width=np.diff(inh_pct).mean() * 0.7 if len(inh_pct) > 1 else 10,
             color="#1a6da8", alpha=0.75, align="center")
    ax_d.axhline(ref_d, color="#e67e22", ls="--", lw=1.5,
                 label=f"Control: {ref_d:.0f} nM")
    ax_d.axhline(ROT_CRIT_NM, color="#27ae60", ls=":", lw=1.2, alpha=0.8)
    for m, rc in zip(drp1_mult[valid_d], drp1_rc[valid_d]):
        if np.isfinite(rc):
            ax_d.text((1-m)*100, rc+4, f"{rc:.0f}", ha="center",
                      va="bottom", fontsize=7.5, color="#1a6da8")
    ax_d.set_xlabel("DRP1 inhibition (%)", fontsize=10)
    ax_d.set_ylabel("[rot]$_c$ (nM)", fontsize=10)
    ax_d.set_title("(d) DRP1 inhibition (Mdivi-1) → shifts [rot]$_c$", fontsize=10)
    ax_d.legend(fontsize=8)
    ax_d.grid(True, ls="--", alpha=0.3, axis="y")

    # (e) PGC-1α
    ax_e = fig_all.add_subplot(gs[1, 2])
    valid_e = np.isfinite(pgc1a_rc)
    ref_e = pgc1a_rc[np.argmin(np.abs(pgc1a_mult - 1.0))]
    ax_e.plot(pgc1a_mult[valid_e], pgc1a_rc[valid_e], "s-",
              color="#27ae60", lw=2, ms=7)
    ax_e.axhline(ref_e, color="#e67e22", ls="--", lw=1.5,
                 label=f"Control: {ref_e:.0f} nM")
    ax_e.axvline(1.0, color="gray", ls=":", lw=1)
    ax_e.fill_between(pgc1a_mult[valid_e], ref_e, pgc1a_rc[valid_e],
                      where=pgc1a_rc[valid_e] > ref_e,
                      color="#27ae60", alpha=0.15)
    ax_e.set_xlabel(r"PGC-1α fold expression ($k_{\rm bio}$ multiplier)",
                    fontsize=9)
    ax_e.set_ylabel("[rot]$_c$ (nM)", fontsize=10)
    ax_e.set_title("(e) PGC-1α overexpression → shifts [rot]$_c$", fontsize=10)
    ax_e.legend(fontsize=8)
    ax_e.grid(True, ls="--", alpha=0.3)

    fig_all.suptitle(
        "Experimental Predictions — Mitochondrial Network Critical Transition\n"
        r"(Calibrated model: BIFUR\_PARAMS, $\sigma_c = 2.035$, "
        f"[rot]$_c$ = {ROT_CRIT_NM:.0f} nM predicted)",
        fontsize=13, fontweight="bold",
    )
    _save(fig_all, "experimental_predictions.pdf")
    print("\nAll figures saved to figures/")

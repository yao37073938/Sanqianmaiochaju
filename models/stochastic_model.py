"""
线粒体网络随机模拟 — Gillespie SSA
======================================
精确随机模拟算法（Gillespie 直接法, 1977），状态空间为整数 (H, D)。

反应列表（共 10 个，与 ode_model.py ODE 在均场极限下一致）
--------------------------------------------------------------
编号  反应方程           生物含义                   速率(命题)
  0   ∅ → H             生物合成                   k_bio
  1   H → 2H            裂变（H）                  k_fis · H
  2   H+H → H           融合（H-H）                k_fus · H·(H−1)
  3   H+D → D           跨类融合（H 被 D 吸收）    k_fus · H·D
  4   D+H → H           跨类融合（D 被 H 吸收）    k_fus · D·H
  5   H → D             ROS 损伤                   k_dam(σ,D) · H
  6   H → ∅             非选择性自噬（伴随 D）     k_mit · H·D/(D+K_d)
  7   D → ∅             选择性自噬（线粒体自噬）   k_mit · D
  8   D → 2D            裂变（D）                  k_fis · D
  9   D+D → D           融合（D-D）                k_fus · D·(D−1)

均场验证
--------
dH/dt = k_bio + k_fis·H − k_fus·H·(H−1) − k_fus·H·D
       − k_dam·H − k_mit·H·D/(D+K_d)
      ≈ k_bio + k_fis·H − k_fus·H·N − k_dam·H − k_mit·H·D/(D+K_d)  ✓

dD/dt = k_fis·D − k_fus·D·(D−1) − k_fus·D·H
       + k_dam·H − k_mit·D
      ≈ k_fis·D − k_fus·D·N + k_dam·H − k_mit·D                    ✓

统计量（每个 sigma，200 次独立模拟）
-------------------------------------
- E[N]   : 稳态总线粒体数均值
- Var[N] : 稳态方差（临界涨落指标）
- ρ₁[N]  : 拉格-1 自相关系数（慢化预警指标）
- E[φ]   : 损伤分数均值  φ = D/(H+D)
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import NamedTuple

import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ode_model import MitoParams, k_dam, simulate


# ---------------------------------------------------------------------------
# 反应化学计量矩阵  shape=(10, 2)   列顺序: [ΔH, ΔD]
# ---------------------------------------------------------------------------

STOICH: np.ndarray = np.array(
    [
        [+1,  0],   # 0  ∅ → H
        [+1,  0],   # 1  H → 2H
        [-1,  0],   # 2  H+H → H
        [-1,  0],   # 3  H+D → D
        [ 0, -1],   # 4  D+H → H
        [-1, +1],   # 5  H → D
        [-1,  0],   # 6  H → ∅
        [ 0, -1],   # 7  D → ∅
        [ 0, +1],   # 8  D → 2D
        [ 0, -1],   # 9  D+D → D
    ],
    dtype=np.int64,
)

N_REACTIONS: int = len(STOICH)


# ---------------------------------------------------------------------------
# 命题函数（propensity）
# ---------------------------------------------------------------------------

def propensities(H: int, D: int, sigma: float, p: MitoParams) -> np.ndarray:
    """
    计算所有 10 个反应的命题（瞬时速率），单位 [/小时]。

    Parameters
    ----------
    H, D    : 当前状态（整数计数）
    sigma   : 外部应激强度
    p       : MitoParams 参数集

    Returns
    -------
    a : shape (10,) 命题向量（非负）
    """
    kd = k_dam(sigma, float(D), p)
    a = np.empty(N_REACTIONS)
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
# 单次 Gillespie 直接法模拟
# ---------------------------------------------------------------------------

def gillespie_ssa(
    sigma: float,
    p: MitoParams,
    t_end: float = 800.0,
    t_burn: float = 300.0,
    dt_sample: float = 2.0,
    H0: int | None = None,
    D0: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    运行一次 Gillespie SSA，返回稳态采样序列。

    Parameters
    ----------
    sigma     : 应激强度
    p         : 参数集
    t_end     : 总模拟时长（小时）
    t_burn    : 热化时间（舍弃前段）
    dt_sample : 采样间隔（小时）
    H0, D0    : 初始条件（默认用 ODE 稳态）
    seed      : 随机种子

    Returns
    -------
    H_samples, D_samples : 热化后的采样时间序列，shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # --- 初始条件：使用 ODE 稳态近似 ---
    if H0 is None or D0 is None:
        p_ss = MitoParams(**{f: getattr(p, f) for f in p.__dataclass_fields__})
        p_ss.sigma = sigma
        try:
            sol = simulate(p_ss, t_span=(0.0, 500.0), H0=200.0, D0=80.0)
            H0 = max(1, int(round(sol["H"][-1])))
            D0 = max(0, int(round(sol["D"][-1])))
        except Exception:
            H0, D0 = 200, 80

    H: int = H0
    D: int = D0
    t: float = 0.0

    # 采样时刻（热化后均匀采样）
    sample_times: np.ndarray = np.arange(t_burn, t_end, dt_sample)
    n_samples: int = len(sample_times)
    H_out = np.zeros(n_samples, dtype=np.int64)
    D_out = np.zeros(n_samples, dtype=np.int64)
    s_idx: int = 0

    while t < t_end and s_idx < n_samples:
        a = propensities(H, D, sigma, p)
        a0: float = float(a.sum())

        if a0 < 1e-15:
            # 吸收态：用当前状态填充剩余
            H_out[s_idx:] = H
            D_out[s_idx:] = D
            break

        # 下一事件时间
        tau: float = rng.exponential(1.0 / a0)
        t_next: float = t + tau

        # 记录落在 [t, t_next) 内的采样点
        while s_idx < n_samples and sample_times[s_idx] <= t_next:
            H_out[s_idx] = H
            D_out[s_idx] = D
            s_idx += 1

        # 选择反应（二分查找累积命题）
        r: float = rng.uniform(0.0, a0)
        rxn: int = int(np.searchsorted(np.cumsum(a), r))
        rxn = min(rxn, N_REACTIONS - 1)

        H = max(H + int(STOICH[rxn, 0]), 0)
        D = max(D + int(STOICH[rxn, 1]), 0)
        t = t_next

    return H_out, D_out


# ---------------------------------------------------------------------------
# 多进程集成运行
# ---------------------------------------------------------------------------

class EnsembleResult(NamedTuple):
    sigma: float
    mean_N: float
    std_N: float
    var_N: float
    mean_phi: float
    ac1: float          # 拉格-1 自相关（慢化预警）
    H_traj: np.ndarray  # shape (n_runs, n_samples)
    D_traj: np.ndarray


def _worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """多进程工作函数（顶层，支持 pickle）。"""
    sigma, p_dict, t_end, t_burn, dt_sample, H0, D0, seed = args
    p = MitoParams(**p_dict)
    return gillespie_ssa(sigma, p, t_end, t_burn, dt_sample, H0, D0, seed)


def run_ensemble(
    sigma: float,
    p: MitoParams | None = None,
    n_runs: int = 200,
    t_end: float = 800.0,
    t_burn: float = 300.0,
    dt_sample: float = 2.0,
    H0: int | None = None,
    D0: int | None = None,
    n_processes: int | None = None,
) -> EnsembleResult:
    """
    在给定 sigma 下并行运行 n_runs 次独立 Gillespie SSA。

    Parameters
    ----------
    sigma       : 应激强度
    p           : 参数集（默认 MitoParams()）
    n_runs      : 独立模拟次数
    t_end       : 单次模拟总时长（小时）
    t_burn      : 热化时间
    dt_sample   : 采样间隔
    H0, D0      : 初始条件（None=ODE 稳态）
    n_processes : 进程数（None=CPU 核数）

    Returns
    -------
    EnsembleResult（含均值、方差、拉格-1 自相关等）
    """
    if p is None:
        p = MitoParams()

    n_cpu = n_processes or min(mp.cpu_count(), n_runs)
    p_dict = {f: getattr(p, f) for f in p.__dataclass_fields__}

    args_list = [
        (sigma, p_dict, t_end, t_burn, dt_sample, H0, D0, seed)
        for seed in range(n_runs)
    ]

    with mp.Pool(n_cpu) as pool:
        results = pool.map(_worker, args_list)

    Hs = np.array([r[0] for r in results], dtype=np.float64)  # (n_runs, n_samples)
    Ds = np.array([r[1] for r in results], dtype=np.float64)
    Ns = Hs + Ds

    # 防止 D/(H+D) 除零
    phi = Ds / np.where(Ns > 0, Ns, 1.0)

    # --- 统计量 ---
    N_flat = Ns.ravel()
    mean_N  = float(N_flat.mean())
    var_N   = float(N_flat.var(ddof=1))
    std_N   = float(N_flat.std(ddof=1))
    mean_phi = float(phi.ravel().mean())

    # 拉格-1 自相关：按时间序列逐条计算，取中位数（抗离群）
    ac1_vals = []
    for i in range(n_runs):
        ts = Ns[i]
        if ts.std() > 0.5 and len(ts) > 2:
            corr = float(np.corrcoef(ts[:-1], ts[1:])[0, 1])
            if np.isfinite(corr):
                ac1_vals.append(corr)
    ac1 = float(np.median(ac1_vals)) if ac1_vals else 0.0

    return EnsembleResult(
        sigma=sigma,
        mean_N=mean_N,
        std_N=std_N,
        var_N=var_N,
        mean_phi=mean_phi,
        ac1=ac1,
        H_traj=Hs,
        D_traj=Ds,
    )


# ---------------------------------------------------------------------------
# Sigma 扫描
# ---------------------------------------------------------------------------

def scan_sigma(
    sigmas: np.ndarray,
    p: MitoParams | None = None,
    n_runs: int = 200,
    t_end: float = 800.0,
    t_burn: float = 300.0,
    dt_sample: float = 2.0,
    n_processes: int | None = None,
    verbose: bool = True,
) -> list[EnsembleResult]:
    """
    对每个 sigma 值运行集成模拟，返回 EnsembleResult 列表。

    Parameters
    ----------
    sigmas      : 应激强度数组
    p           : 参数集
    n_runs      : 每个 sigma 的独立模拟次数
    t_end       : 单次模拟时长
    t_burn      : 热化时间
    dt_sample   : 采样间隔
    n_processes : 并行进程数
    verbose     : 是否打印进度

    Returns
    -------
    list[EnsembleResult]
    """
    if p is None:
        p = MitoParams()

    results: list[EnsembleResult] = []

    for i, s in enumerate(sigmas):
        if verbose:
            print(f"  sigma = {s:.2f}  [{i+1}/{len(sigmas)}]", flush=True)
        res = run_ensemble(
            sigma=s, p=p, n_runs=n_runs,
            t_end=t_end, t_burn=t_burn, dt_sample=dt_sample,
            n_processes=n_processes,
        )
        results.append(res)
        if verbose:
            print(
                f"    E[N]={res.mean_N:.1f}  Var[N]={res.var_N:.1f}"
                f"  E[phi]={res.mean_phi:.3f}  rho1={res.ac1:.4f}",
                flush=True,
            )

    return results


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_statistics(
    results: list[EnsembleResult],
    save_path: str = "figures/stochastic_ews.pdf",
    traj_sigmas: tuple[float, ...] = (0.5, 1.5, 3.0),
) -> None:
    """
    出版质量图（2×2 面板）：

    (a) 均值 E[N] ± std vs sigma       (b) 方差 Var[N] vs sigma
    (c) 拉格-1 自相关 ρ₁ vs sigma      (d) 代表性随机轨迹
    """
    sigmas  = np.array([r.sigma    for r in results])
    mean_N  = np.array([r.mean_N   for r in results])
    std_N   = np.array([r.std_N    for r in results])
    var_N   = np.array([r.var_N    for r in results])
    mean_phi = np.array([r.mean_phi for r in results])
    ac1     = np.array([r.ac1      for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_mean, ax_var, ax_ac1, ax_traj = axes.ravel()

    # ── (a) 均值 ──────────────────────────────────────────────
    ax_mean.fill_between(sigmas, mean_N - std_N, mean_N + std_N,
                         alpha=0.25, color="#1f4e79", label=r"$\pm 1\sigma$")
    ax_mean.plot(sigmas, mean_N, color="#1f4e79", lw=1.8, label=r"$\langle N \rangle$")
    ax2 = ax_mean.twinx()
    ax2.plot(sigmas, mean_phi, "--", color="#c0392b", lw=1.4, label=r"$\langle \phi \rangle$")
    ax2.set_ylabel(r"Mean damage fraction $\langle\phi\rangle$",
                   fontsize=9, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax_mean.set_ylabel(r"Mean total mitochondria $\langle N \rangle$", fontsize=10)
    ax_mean.set_title("(a) Mean steady-state population", fontsize=10)
    lines1, labels1 = ax_mean.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_mean.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax_mean.grid(True, ls="--", alpha=0.35)

    # ── (b) 方差（临界涨落）───────────────────────────────────
    ax_var.plot(sigmas, var_N, color="#27ae60", lw=1.8)
    ax_var.fill_between(sigmas, 0, var_N, alpha=0.2, color="#27ae60")
    ax_var.set_ylabel(r"Variance  Var$(N)$", fontsize=10)
    ax_var.set_title("(b) Critical fluctuations", fontsize=10)
    ax_var.grid(True, ls="--", alpha=0.35)
    # 标注最大方差处
    idx_max = np.argmax(var_N)
    ax_var.axvline(sigmas[idx_max], color="gray", ls=":", lw=1.2)
    ax_var.text(sigmas[idx_max] + 0.05, var_N[idx_max] * 0.9,
                rf"$\sigma^*\approx{sigmas[idx_max]:.2f}$",
                fontsize=8, color="gray")

    # ── (c) 拉格-1 自相关（慢化预警）───────────────────────────
    ax_ac1.plot(sigmas, ac1, color="#8e44ad", lw=1.8)
    ax_ac1.fill_between(sigmas, 0, ac1, alpha=0.15, color="#8e44ad")
    ax_ac1.axhline(0, color="black", lw=0.8)
    ax_ac1.set_ylabel(r"Lag-1 autocorrelation  $\rho_1$", fontsize=10)
    ax_ac1.set_title("(c) Critical slowing down", fontsize=10)
    ax_ac1.set_ylim(bottom=min(ac1.min() - 0.05, -0.05))
    ax_ac1.grid(True, ls="--", alpha=0.35)

    # ── (d) 代表性轨迹 ────────────────────────────────────────
    colors_traj = {"0.5": "#1f4e79", "1.5": "#e67e22", "3.0": "#c0392b"}
    sigma_res_map = {r.sigma: r for r in results}

    # 找最接近 traj_sigmas 的模拟结果
    n_traj_shown = 3
    for ts in traj_sigmas:
        idx = int(np.argmin(np.abs(sigmas - ts)))
        r = results[idx]
        n_samples = r.H_traj.shape[1]
        t_axis = np.arange(n_samples) * 2.0  # dt_sample=2h

        Ns_run = r.H_traj[:n_traj_shown] + r.D_traj[:n_traj_shown]
        color = list(colors_traj.values())[list(traj_sigmas).index(ts)]
        for j in range(n_traj_shown):
            lbl = rf"$\sigma={sigmas[idx]:.1f}$" if j == 0 else None
            ax_traj.plot(t_axis, Ns_run[j], lw=0.7, alpha=0.7,
                         color=color, label=lbl)

    ax_traj.set_xlabel("Time (h)", fontsize=10)
    ax_traj.set_ylabel(r"Total mitochondria $N(t)$", fontsize=10)
    ax_traj.set_title("(d) Representative SSA trajectories", fontsize=10)
    ax_traj.legend(fontsize=9)
    ax_traj.grid(True, ls="--", alpha=0.35)

    # 共用 x 轴标签
    for ax in [ax_mean, ax_var, ax_ac1]:
        ax.set_xlabel(r"Damage stress $\sigma$", fontsize=10)

    fig.suptitle(
        "Stochastic Early Warning Signals — Mitochondrial Network (Gillespie SSA)\n"
        rf"$n_{{runs}}={results[0].H_traj.shape[0]}$ per $\sigma$, "
        r"$\Delta t_{\rm sample}=2\,\rm h$",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 摘要打印
# ---------------------------------------------------------------------------

def print_summary(results: list[EnsembleResult]) -> None:
    """打印 sigma 扫描统计摘要表。"""
    hdr = (f"{'sigma':>7}  {'E[N]':>8}  {'std[N]':>8}  "
           f"{'Var[N]':>10}  {'E[phi]':>8}  {'rho1':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.sigma:>7.2f}  {r.mean_N:>8.1f}  {r.std_N:>8.1f}  "
              f"{r.var_N:>10.1f}  {r.mean_phi:>8.4f}  {r.ac1:>8.4f}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    N_RUNS       = 200
    N_SIGMA      = 12
    T_END        = 800.0
    T_BURN       = 300.0
    DT_SAMPLE    = 2.0
    N_PROCESSES  = None   # None = 全部 CPU 核

    sigmas = np.linspace(0.0, 5.0, N_SIGMA)
    p = MitoParams()

    print("=" * 60)
    print("Gillespie SSA — Mitochondrial Network")
    print(f"  n_runs    = {N_RUNS}  per sigma")
    print(f"  sigma     = [{sigmas[0]:.2f}, {sigmas[-1]:.2f}]  ({N_SIGMA} points)")
    print(f"  t_end     = {T_END} h    t_burn = {T_BURN} h")
    print(f"  processes = {mp.cpu_count()} CPU cores")
    print("=" * 60)

    t0 = time.time()
    results = scan_sigma(
        sigmas,
        p=p,
        n_runs=N_RUNS,
        t_end=T_END,
        t_burn=T_BURN,
        dt_sample=DT_SAMPLE,
        n_processes=N_PROCESSES,
        verbose=True,
    )
    print(f"\nTotal wall time: {time.time()-t0:.1f} s")

    print_summary(results)

    plot_statistics(
        results,
        save_path="figures/stochastic_ews.pdf",
        traj_sigmas=(0.5, 2.0, 4.0),
    )

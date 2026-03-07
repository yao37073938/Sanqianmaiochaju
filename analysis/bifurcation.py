"""
线粒体网络鞍结分岔分析
======================
分岔参数：外部应激强度 sigma（扫描 0 → 5）
状态变量：H（健康线粒体）、D（损伤线粒体）、N = H + D

模型扩展说明
------------
原始 ODE 中 k_dam = k_dam0*sigma + alpha*D/(D+K_r)（计数形式，
固定点方程退化为二次方程，乘积项恒为负，唯一正根，无法产生双稳态）。

本分析采用基于 **损伤分数** phi = D/N 的 ROS 正反馈（与文献 Bhatt et al.,
Vásárhelyi et al. 一致），配合 **基础修复率 k_repair** 使低应激下
健康态 (phi≈0, N≈N_health) 稳定：

    k_dam(sigma, phi) = max(0, k_dam0*sigma − k_repair)
                       + alpha * phi^2 / (phi^2 + phi_c^2)

其中：
  - max(0, k_dam0*sigma − k_repair)：净应激损伤（低于修复阈值时为 0）
  - alpha * phi^2 / (phi^2 + phi_c^2)：Hill n=2 协同 ROS 正反馈
    （phi_c ≈ 0.35：损伤分数超过 35% 时反馈显著增强）

分岔结构（sigma_0 = k_repair / k_dam0 = 2.0）：
  sigma < sigma_c ≈ 2.0：双稳态——健康态（N≈419）和损伤态（N≈100）共存
  sigma = sigma_c      ：鞍结分岔——健康态与鞍点合并消失
  sigma > sigma_c      ：单稳态——仅损伤态稳定
"""

from __future__ import annotations

import sys
import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
from scipy.linalg import eigvals

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ode_model import MitoParams

# ---------------------------------------------------------------------------
# 模型参数（分岔分析专用，与 ode_model.py 共用基础结构）
# ---------------------------------------------------------------------------

#: 基础修复速率 [/小时]：低于 k_repair/k_dam0 的应激被细胞抗氧化系统中和
K_REPAIR: float = 0.08
#: Hill n=2 的半饱和损伤分数（无量纲，phi = D/N）
PHI_C: float = 0.35

#: 分岔分析用参数（k_fis < k_mit 保证低应激下 D 被净清除）
BIFUR_PARAMS = MitoParams(
    k_bio=5.0,
    k_fis=0.03,
    k_fus=1e-4,
    k_mit=0.08,
    K_d=50.0,
    k_dam0=0.04,
    alpha=0.18,
    K_r=9999.0,   # 此参数在分岔模型中不使用（由 phi_c 替代）
)


# ---------------------------------------------------------------------------
# 核心：损伤率（协同 ROS 正反馈 + 修复阈值）
# ---------------------------------------------------------------------------

def k_dam_bifur(sigma: float, D: float, N: float,
                p: MitoParams = BIFUR_PARAMS) -> float:
    """
    分岔分析版损伤率（Hill n=2，基于损伤分数 phi=D/N）。

    k_dam = max(0, k_dam0*sigma - k_repair) + alpha*phi^2/(phi^2+phi_c^2)

    Parameters
    ----------
    sigma : 外部应激强度
    D, N  : 损伤线粒体数量，总量
    p     : 模型参数
    """
    phi = D / max(N, 1e-12)
    baseline = max(0.0, p.k_dam0 * sigma - K_REPAIR)
    ros_feedback = p.alpha * phi**2 / (phi**2 + PHI_C**2)
    return baseline + ros_feedback


def mito_ode_bifur(t: float, y: list[float],
                   sigma: float, p: MitoParams = BIFUR_PARAMS) -> list[float]:
    """分岔分析用 ODE 右端项（协同 ROS，基础修复）。"""
    H = max(y[0], 1e-12)
    D = max(y[1], 1e-12)
    N = H + D
    kd = k_dam_bifur(sigma, D, N, p)
    dH = (p.k_bio + p.k_fis * H - p.k_fus * H * N
          - p.k_mit * H * D / (D + p.K_d) - kd * H)
    dD = p.k_fis * D - p.k_fus * D * N - p.k_mit * D + kd * H
    return [dH, dD]


# ---------------------------------------------------------------------------
# 稳态求解 & Jacobian
# ---------------------------------------------------------------------------

def steady_state(sigma: float, y0: list[float],
                 p: MitoParams = BIFUR_PARAMS) -> np.ndarray | None:
    """
    用 fsolve 求解稳态不动点；若残差 > 1e-6 或物理不合理则返回 None。
    """
    def equations(y: np.ndarray) -> list[float]:
        H, D = abs(y[0]), abs(y[1])
        return mito_ode_bifur(0.0, [H, D], sigma, p)

    with np.errstate(all="ignore"):
        sol = fsolve(equations, y0, full_output=True)

    y_sol, _, ier, _ = sol
    H_s, D_s = abs(y_sol[0]), abs(y_sol[1])

    if ier != 1:
        return None
    residual = np.max(np.abs(equations([H_s, D_s])))
    if residual > 1e-6 or H_s < 1e-3 or D_s < 0:
        return None
    return np.array([H_s, D_s])


def numerical_jacobian(y: np.ndarray, sigma: float,
                       p: MitoParams = BIFUR_PARAMS,
                       eps: float = 1e-5) -> np.ndarray:
    """2×2 Jacobian（中心差分）。"""
    J = np.zeros((2, 2))
    for j in range(2):
        yp, ym = y.copy(), y.copy()
        yp[j] += eps
        ym[j] -= eps
        fp = mito_ode_bifur(0.0, list(yp), sigma, p)
        fm = mito_ode_bifur(0.0, list(ym), sigma, p)
        J[:, j] = (np.array(fp) - np.array(fm)) / (2 * eps)
    return J


def classify_fp(y: np.ndarray, sigma: float,
                p: MitoParams = BIFUR_PARAMS) -> tuple[bool, tuple]:
    """
    Returns (is_stable, eigenvalues)。
    稳定 ⟺ 所有特征值实部 < 0。
    """
    J = numerical_jacobian(y, sigma, p)
    eigs = eigvals(J)
    stable = bool(np.all(np.real(eigs) < 0))
    return stable, tuple(eigs)


# ---------------------------------------------------------------------------
# 多分支搜索
# ---------------------------------------------------------------------------

def find_all_fixed_points(
    sigma: float,
    p: MitoParams = BIFUR_PARAMS,
    n_H: int = 10,
    n_D: int = 10,
    tol: float = 2.0,
) -> list[dict]:
    """
    在 (H, D) 空间撒多初值，收集去重后的所有不动点。

    Returns
    -------
    list of dict: {H, D, N, stable, eigenvalues}
    """
    # --- 特殊处理：D=0 不动点（低 sigma 健康态）---
    results: list[dict] = []

    # D=0 fixed point: H satisfies k_bio + (k_fis-k_fus*H)*H = 0 approx
    if p.k_dam0 * sigma <= K_REPAIR:   # baseline k_dam = 0, D=0 may be stable
        # Solve H equation with D=0
        def h_eq(H):
            return p.k_bio + p.k_fis * H - p.k_fus * H * H
        try:
            H_healthy = brentq(h_eq, 10.0, 2000.0)
            y_h = np.array([H_healthy, 1e-6])  # near D=0
            fp = steady_state(sigma, list(y_h), p)
            if fp is not None and fp[1] < 1.0:
                stable, eigs = classify_fp(fp, sigma, p)
                results.append({
                    "H": fp[0], "D": 0.0, "N": fp[0],
                    "stable": stable, "eigenvalues": eigs
                })
        except Exception:
            pass

    # --- Interior fixed points ---
    H_grid = np.logspace(1, 3, n_H)
    D_grid = np.logspace(0.5, 3, n_D)

    for H0 in H_grid:
        for D0 in D_grid:
            fp = steady_state(sigma, [H0, D0], p)
            if fp is None:
                continue
            H_s, D_s = fp

            # De-duplication
            duplicate = any(
                np.linalg.norm(fp - np.array([r["H"], r["D"]])) < tol
                for r in results
            )
            if duplicate:
                continue

            stable, eigs = classify_fp(fp, sigma, p)
            results.append({
                "H": H_s, "D": D_s, "N": H_s + D_s,
                "stable": stable, "eigenvalues": eigs,
            })

    return results


# ---------------------------------------------------------------------------
# Sigma 扫描
# ---------------------------------------------------------------------------

def scan_bifurcation(
    p: MitoParams = BIFUR_PARAMS,
    sigma_range: tuple[float, float] = (0.0, 5.0),
    n_sigma: int = 200,
) -> list[dict]:
    """
    扫描 sigma，收集所有不动点，返回含 sigma 的记录列表。
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    all_fps: list[dict] = []

    for s in sigmas:
        fps = find_all_fixed_points(s, p)
        for fp in fps:
            fp["sigma"] = s
            all_fps.append(fp)

    return all_fps


# ---------------------------------------------------------------------------
# 鞍结分岔点检测
# ---------------------------------------------------------------------------

def detect_saddle_node(
    fps: list[dict],
    sigma_eps: float = 0.1,
) -> list[float]:
    """
    检测稳定不动点数目发生变化的 sigma，作为鞍结分岔候选点。
    """
    sigmas = sorted(set(fp["sigma"] for fp in fps))
    n_stable = {}
    for s in sigmas:
        n_stable[s] = sum(1 for fp in fps if fp["sigma"] == s and fp["stable"])

    sigma_c_list: list[float] = []
    prev_n = None
    for s in sigmas:
        curr_n = n_stable.get(s, 0)
        if prev_n is not None and curr_n != prev_n:
            sigma_c_list.append(s)
        prev_n = curr_n

    # 合并相近候选点
    merged: list[float] = []
    for sc in sigma_c_list:
        if merged and abs(sc - merged[-1]) < sigma_eps:
            merged[-1] = (merged[-1] + sc) / 2
        else:
            merged.append(sc)

    return merged


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_bifurcation(
    fps: list[dict],
    sigma_c_list: list[float],
    save_path: str = "figures/bifurcation.pdf",
) -> None:
    """
    绘制分岔图：N* vs sigma。

    - 稳定分支：实线（深蓝）
    - 不稳定分支（鞍点）：虚线（橙红）
    - 鞍结分岔点 sigma_c：灰色垂线 + 标注
    """
    stable_pts   = [(fp["sigma"], fp["N"]) for fp in fps if fp["stable"]]
    unstable_pts = [(fp["sigma"], fp["N"]) for fp in fps if not fp["stable"]]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # ---- 上图：N* vs sigma ----
    ax = axes[0]

    if stable_pts:
        ss, sN = zip(*sorted(stable_pts))
        ax.scatter(ss, sN, s=5, color="#1f4e79", zorder=3,
                   label="Stable fixed points")

    if unstable_pts:
        us, uN = zip(*sorted(unstable_pts))
        ax.scatter(us, uN, s=5, color="#c0392b", zorder=3, marker="s",
                   label="Unstable (saddle)")
        ax.plot(us, uN, "--", color="#c0392b", lw=0.9, alpha=0.7, zorder=2)

    # 标注鞍结分岔点
    for sc in sigma_c_list:
        ax.axvline(sc, color="gray", ls=":", lw=1.4, zorder=1)
        yhi = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500
        ax.text(sc + 0.05, yhi * 0.92,
                f"$\\sigma_c \\approx {sc:.2f}$",
                fontsize=9, color="gray", va="top")

    ax.set_ylabel(r"Steady-state total mitochondria $N^*$", fontsize=11)
    ax.set_title("Bifurcation Diagram: Mitochondrial Network\n"
                 r"(saddle-node with cooperative ROS feedback, Hill $n=2$)",
                 fontsize=12)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, ls="--", alpha=0.35)

    # ---- 下图：特征值实部 vs sigma ----
    ax2 = axes[1]
    for fp in fps:
        color = "#1f4e79" if fp["stable"] else "#c0392b"
        ls = "-" if fp["stable"] else "--"
        for ev in fp["eigenvalues"]:
            ax2.scatter(fp["sigma"], np.real(ev), s=3, color=color, alpha=0.6)

    ax2.axhline(0, color="black", lw=0.8, ls="-")
    for sc in sigma_c_list:
        ax2.axvline(sc, color="gray", ls=":", lw=1.4)

    ax2.set_xlabel(r"Damage stress $\sigma$", fontsize=11)
    ax2.set_ylabel(r"Re($\lambda$)  [eigenvalue real part]", fontsize=11)
    ax2.set_title("Jacobian eigenvalues vs. stress", fontsize=11)
    ax2.grid(True, ls="--", alpha=0.35)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Bifurcation diagram saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 摘要打印
# ---------------------------------------------------------------------------

def print_summary(fps: list[dict], sigma_c_list: list[float]) -> None:
    n_stable   = sum(fp["stable"] for fp in fps)
    n_unstable = sum(not fp["stable"] for fp in fps)
    bistable_sigmas = {
        fp["sigma"] for fp in fps
        if sum(1 for g in fps if g["sigma"] == fp["sigma"] and g["stable"]) > 1
    }

    print("\n=== Bifurcation Analysis Summary ===")
    print(f"  Model  : Hill n=2 cooperative ROS feedback")
    print(f"  sigma_0: {K_REPAIR / BIFUR_PARAMS.k_dam0:.2f}  "
          f"(transcritical threshold: k_repair/k_dam0)")
    print(f"  Fixed points found  : {len(fps)}")
    print(f"    Stable            : {n_stable}")
    print(f"    Unstable (saddle) : {n_unstable}")
    print(f"  Bistable sigma range: {len(bistable_sigmas)} scan points")
    if sigma_c_list:
        for sc in sigma_c_list:
            print(f"  Saddle-node sigma_c : {sc:.3f}")
    else:
        print("  No saddle-node detected in scanned range.")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Scanning sigma in [0, 5] for fixed points ...")
    print(f"  k_repair = {K_REPAIR}, phi_c = {PHI_C}")
    print(f"  sigma_0 = k_repair/k_dam0 = {K_REPAIR / BIFUR_PARAMS.k_dam0:.2f}")

    fps = scan_bifurcation(n_sigma=200)
    sigma_c_list = detect_saddle_node(fps)
    print_summary(fps, sigma_c_list)

    plot_bifurcation(fps, sigma_c_list, save_path="figures/bifurcation.pdf")

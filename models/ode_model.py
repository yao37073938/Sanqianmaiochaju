"""
线粒体网络动力学 ODE 模型
=========================
状态变量：
  H  - 健康线粒体数量
  D  - 损伤线粒体数量
  N  = H + D（总量）

方程：
  dH/dt = k_bio + k_fis*H - k_fus*H*N - k_mit*H*D/(D+K_d) - k_dam(sigma,D)*H
  dD/dt = k_fis*D - k_fus*D*N - k_mit*D      + k_dam(sigma,D)*H

损伤率（含 ROS 正反馈）：
  k_dam(sigma, D) = k_dam0*sigma + alpha*D/(D+K_r)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

@dataclass
class MitoParams:
    """线粒体网络动力学模型参数（生物学合理默认值）。"""

    # 生物合成速率 [线粒体/小时]
    k_bio: float = 5.0
    # 裂变速率 [/小时]
    k_fis: float = 0.05
    # 融合速率 [/线粒体/小时]
    k_fus: float = 1e-4
    # 线粒体自噬清除速率（健康介导）[/小时]
    k_mit: float = 0.05
    # 半饱和常数：自噬对损伤线粒体的选择性 [线粒体数]
    K_d: float = 50.0

    # 基础损伤率系数 [/小时/应激单位]
    k_dam0: float = 0.02
    # 外部应激强度（无量纲，0 = 无压力）
    sigma: float = 1.0

    # ROS 正反馈强度 [/小时]
    alpha: float = 0.08
    # ROS 反馈半饱和常数 [线粒体数]
    K_r: float = 100.0


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def k_dam(sigma: float, D: float, p: MitoParams) -> float:
    """
    损伤速率（应激 + ROS 正反馈）。

    k_dam(sigma, D) = k_dam0 * sigma + alpha * D / (D + K_r)

    Parameters
    ----------
    sigma : 外部应激强度
    D     : 当前损伤线粒体数量
    p     : 模型参数

    Returns
    -------
    有效损伤速率 [/小时]
    """
    return p.k_dam0 * sigma + p.alpha * D / (D + p.K_r)


def mito_ode(t: float, y: Sequence[float], p: MitoParams) -> list[float]:
    """
    线粒体网络 ODE 右端项。

    Parameters
    ----------
    t : 当前时间（供 solve_ivp 调用，方程本身不显含时间）
    y : [H, D]
    p : MitoParams 实例

    Returns
    -------
    [dH/dt, dD/dt]
    """
    H, D = y
    # 防止数值负值
    H = max(H, 0.0)
    D = max(D, 0.0)

    N = H + D
    kd = k_dam(p.sigma, D, p)

    dH = (p.k_bio
          + p.k_fis * H
          - p.k_fus * H * N
          - p.k_mit * H * D / (D + p.K_d)
          - kd * H)

    dD = (p.k_fis * D
          - p.k_fus * D * N
          - p.k_mit * D
          + kd * H)

    return [dH, dD]


def simulate(
    p: MitoParams | None = None,
    t_span: tuple[float, float] = (0.0, 200.0),
    t_eval_n: int = 2000,
    H0: float = 800.0,
    D0: float = 50.0,
) -> dict:
    """
    求解 ODE 并返回结果字典。

    Parameters
    ----------
    p        : 模型参数（默认 MitoParams()）
    t_span   : 积分时间区间 [小时]
    t_eval_n : 输出时间点数
    H0, D0   : 初始条件

    Returns
    -------
    dict with keys: t, H, D, N, params
    """
    if p is None:
        p = MitoParams()

    t_eval = np.linspace(t_span[0], t_span[1], t_eval_n)
    sol = solve_ivp(
        fun=mito_ode,
        t_span=t_span,
        y0=[H0, D0],
        t_eval=t_eval,
        args=(p,),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE 求解失败：{sol.message}")

    H_sol, D_sol = sol.y
    return {
        "t": sol.t,
        "H": H_sol,
        "D": D_sol,
        "N": H_sol + D_sol,
        "params": p,
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_dynamics(
    results: list[dict] | dict,
    labels: list[str] | None = None,
    save_path: str = "figures/mito_dynamics.pdf",
) -> None:
    """
    绘制 N(t)、H(t)、D(t) 时间演化曲线（出版质量）。

    Parameters
    ----------
    results   : simulate() 的返回值，或其列表（用于多条曲线对比）
    labels    : 各曲线图例标签
    save_path : 输出文件路径
    """
    if isinstance(results, dict):
        results = [results]
    if labels is None:
        labels = [f"σ={r['params'].sigma}" for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    colors = plt.cm.tab10.colors

    var_keys = ["N", "H", "D"]
    var_labels = [
        "N = H + D  (Total mitochondria)",
        "H  (Healthy mitochondria)",
        "D  (Damaged mitochondria)",
    ]

    for ax, key, ylabel in zip(axes, var_keys, var_labels):
        for i, (res, lbl) in enumerate(zip(results, labels)):
            ax.plot(res["t"], res[key], color=colors[i % 10],
                    linewidth=1.8, label=lbl)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(results[0]["t"][0], results[0]["t"][-1])

    axes[-1].set_xlabel("Time (hours)", fontsize=11)
    fig.suptitle("Mitochondrial Network Dynamics (ODE Model)", fontsize=13, y=1.01)
    fig.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存：{save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主程序：示例运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 单条曲线：默认参数 ---
    res_default = simulate()

    # --- 多条曲线：不同应激强度对比 ---
    sigmas = [0.5, 1.0, 2.0, 3.0]
    results_multi = [simulate(MitoParams(sigma=s)) for s in sigmas]
    labels_multi = [f"σ = {s}" for s in sigmas]

    plot_dynamics(
        results=results_multi,
        labels=labels_multi,
        save_path="figures/mito_dynamics.pdf",
    )

    # 打印终态摘要
    print(f"\n{'sigma':>6}  {'H_final':>10}  {'D_final':>10}  {'N_final':>10}  {'D/N':>8}")
    print("-" * 52)
    for s, res in zip(sigmas, results_multi):
        H_f, D_f, N_f = res["H"][-1], res["D"][-1], res["N"][-1]
        print(f"{s:>6.1f}  {H_f:>10.1f}  {D_f:>10.1f}  {N_f:>10.1f}  {D_f/N_f:>8.3f}")

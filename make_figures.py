# -*- coding: utf-8 -*-
"""
生成模型图：
    figures/fig1_fit_validation.png   11.5 m 拟合 + 留一法验证
    figures/fig2_model_structure.png  k(h) / b(h) / 不同高度的预测线
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import drift_model

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---- 设计参数（参考 dataviz 默认浅色调色板）----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RAMP = ["#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)


def load_all():
    model = drift_model.load_model(BASE / "drift_model.json")
    train = pd.read_csv(BASE / "data" / "measured_points.csv")
    v = train["风速(m/s)"].to_numpy(float)
    d = train["漂移距离(m)"].to_numpy(float)
    return model, v, d


def ols(v, d):
    X = np.column_stack([v, np.ones_like(v)])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    resid = d - X @ beta
    n = len(d)
    s = math.sqrt(resid @ resid / (n - 2))
    cov = s ** 2 * np.linalg.inv(X.T @ X)
    return beta, cov, s, resid


def loocv(v, d):
    n = len(d)
    pred = np.empty(n)
    lo = np.empty(n)
    hi = np.empty(n)
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        beta, cov, s, _ = ols(v[mask], d[mask])
        X = np.column_stack([v[mask], np.ones(mask.sum())])
        xtx_inv = np.linalg.inv(X.T @ X)
        x0 = np.array([v[i], 1.0])
        yhat = float(x0 @ beta)
        se = s * math.sqrt(1.0 + float(x0 @ xtx_inv @ x0))
        tcrit = stats.t.ppf(0.975, n - 3)
        pred[i] = yhat
        lo[i] = yhat - tcrit * se
        hi[i] = yhat + tcrit * se
    return pred, lo, hi


def fig1(model, v, d):
    beta, cov, sigma, resid = ols(v, d)
    k, b = beta
    n = len(d)
    sxx = ((v - v.mean()) ** 2).sum()
    vbar = v.mean()
    vv = np.linspace(0, 3.55, 300)
    dhat = k * vv + b
    tcrit = stats.t.ppf(0.975, n - 2)
    se_pred = sigma * np.sqrt(1 + 1 / n + (vv - vbar) ** 2 / sxx)
    lo = dhat - tcrit * se_pred
    hi = dhat + tcrit * se_pred

    pred, pi_lo, pi_hi = loocv(v, d)
    ss_res = ((pred - d) ** 2).sum()
    ss_tot = ((d - d.mean()) ** 2).sum()
    r2_cv = 1 - ss_res / ss_tot
    rmse_cv = math.sqrt(np.mean((pred - d) ** 2))
    coverage = np.mean((d >= pi_lo) & (d <= pi_hi))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    # --- A. 拟合 ---
    ax = axes[0]
    style_axis(ax)
    # 同一风速的 5 次重复点做很小水平抖动，便于看清重叠
    offsets = np.array([-0.055, -0.028, 0.0, 0.028, 0.055])
    xj = np.concatenate([np.full(5, u) + offsets for u in np.unique(v)])
    ax.fill_between(vv, lo, hi, color=ORANGE, alpha=0.10, linewidth=0, zorder=1,
                    label="95% 预测带")
    ax.plot(vv, dhat, color=ORANGE, lw=2.0, zorder=3, label="线性拟合")
    ax.scatter(xj, d, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.4,
               alpha=0.85, zorder=4, label="实测点（25 次）")
    ax.set_xlim(0, 3.55)
    ax.set_ylim(0, 14.2)
    ax.set_xlabel("风速 v (m/s)")
    ax.set_ylabel("漂移距离 d (m)")
    ax.set_title("11.5 m：漂移距离随风速线性增加", loc="left")
    ax.annotate(
        f"n = 25   R² = {model['metrics']['fit_11_5m']['r2']:.3f}\n"
        f"RMSE = {model['metrics']['fit_11_5m']['rmse']:.2f} m",
        xy=(0.03, 0.96), xycoords="axes fraction", va="top", ha="left",
        fontsize=9.5, color=INK2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor="none", alpha=0.9),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82), labelcolor=INK2)
    ax.text(0.99, 0.02, "同一风速的 5 次重复点做了轻微水平抖动",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=MUTED)

    # --- B. 留一法 ---
    ax = axes[1]
    style_axis(ax)
    lim = (0, 14.2)
    ax.plot(lim, lim, color=AXIS, lw=1.0, zorder=1)
    ax.scatter(d, pred, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.4,
               alpha=0.85, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("实测漂移距离 (m)")
    ax.set_ylabel("留一法预测距离 (m)")
    ax.set_title("留一法验证：预测值与实测值", loc="left")
    ax.annotate(
        f"LOOCV R² = {r2_cv:.3f}\nRMSE = {rmse_cv:.2f} m\n95% 区间覆盖率 = {coverage:.0%}",
        xy=(0.03, 0.96), xycoords="axes fraction", va="top", ha="left",
        fontsize=9.5, color=INK2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor="none", alpha=0.9),
    )
    ax.text(0.97, 0.04, "1:1 参考线", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=MUTED)

    fig.suptitle("图 1  11.5 m 实测数据的拟合与交叉验证", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.savefig(FIG_DIR / "fig1_fit_validation.png", dpi=220)
    plt.close(fig)


def fig2(model):
    p = model["parameters"]
    h = np.linspace(1.3, 16.5, 400)
    k = p["a_k"] * h ** p["p_k"]
    b = p["a_b"] * h ** p["p_b"]

    fig = plt.figure(figsize=(13.6, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # --- A. k(h) ---
    ax = axes[0]
    style_axis(ax)
    ax.axvspan(11.5, 16.5, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(h, k, color=BLUE, lw=2.0, zorder=3)
    ax.scatter([1.55, 11.5], [0.4215, 2.9639], s=55, color=BLUE,
               edgecolor=SURFACE, linewidth=1.6, zorder=4)
    ax.set_xlim(1.2, 16.5)
    ax.set_ylim(0, 5.3)
    ax.set_xlabel("释放高度 h (m)")
    ax.set_ylabel("斜率 k (s)")
    ax.set_title("k(h)：风速每增加 1 m/s 的漂移增量", loc="left", fontsize=11)
    ax.annotate("实测锚点", xy=(11.5, 2.9639), xytext=(9.2, 4.55),
                fontsize=8.8, color=INK2, ha="right",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.text(14.0, 0.28, "外推区", ha="center", va="bottom", fontsize=8.5, color=MUTED)

    # --- B. b(h) ---
    ax = axes[1]
    style_axis(ax)
    ax.axvspan(11.5, 16.5, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(h, b, color=BLUE, lw=2.0, zorder=3)
    ax.scatter([1.55, 11.5], [0.2633, 0.7614], s=55, color=BLUE,
               edgecolor=SURFACE, linewidth=1.6, zorder=4)
    ax.set_xlim(1.2, 16.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("释放高度 h (m)")
    ax.set_ylabel("截距 b (m)")
    ax.set_title("b(h)：零风速时的偏移量", loc="left", fontsize=11)
    ax.annotate("实测锚点", xy=(11.5, 0.7614), xytext=(9.4, 0.98),
                fontsize=8.8, color=INK2, ha="right",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.text(14.0, 0.05, "外推区", ha="center", va="bottom", fontsize=8.5, color=MUTED)

    # --- C. 不同高度的预测线 ---
    ax = axes[2]
    style_axis(ax)
    heights = [1.55, 7.5, 11.5, 13.5, 15.5]
    vv = np.linspace(0, 3.5, 200)
    for hgt, color in zip(heights, RAMP):
        kk = p["a_k"] * hgt ** p["p_k"]
        bb = p["a_b"] * hgt ** p["p_b"]
        ax.plot(vv, kk * vv + bb, color=color, lw=2.0, zorder=3,
                label=f"{hgt:g} m")
    # 11.5 m 实测均值点
    train = pd.read_csv(BASE / "data" / "measured_points.csv")
    means = train.groupby("风速(m/s)")["漂移距离(m)"].mean().reset_index()
    ax.scatter(means["风速(m/s)"], means["漂移距离(m)"], s=38, color="#2a78d6",
               edgecolor=SURFACE, linewidth=1.4, zorder=5, label="11.5 m 实测均值")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 15.5)
    ax.set_xlabel("风速 v (m/s)")
    ax.set_ylabel("漂移距离 d (m)")
    ax.set_title("不同释放高度的预测线", loc="left", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", ncol=2, labelcolor=INK2,
              handlelength=1.8, columnspacing=1.0)
    # 只给最外侧两条线直接标注，其余靠图例
    k_lo = p["a_k"] * 1.55 ** p["p_k"]
    b_lo = p["a_b"] * 1.55 ** p["p_b"]
    k_hi = p["a_k"] * 15.5 ** p["p_k"]
    b_hi = p["a_b"] * 15.5 ** p["p_b"]
    ax.annotate("1.55 m", xy=(3.5, k_lo * 3.5 + b_lo), xytext=(3.42, 1.55),
                ha="right", va="bottom", fontsize=8.8, color=INK2)
    ax.annotate("15.5 m", xy=(3.5, k_hi * 3.5 + b_hi), xytext=(3.42, 14.9),
                ha="right", va="bottom", fontsize=8.8, color=INK2)

    fig.suptitle("图 2  高度模型的结构：k(h)、b(h) 与不同高度的预测线", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.savefig(FIG_DIR / "fig2_model_structure.png", dpi=220)
    plt.close(fig)


def main():
    model, v, d = load_all()
    fig1(model, v, d)
    fig2(model)
    print("已生成：")
    print(" ", FIG_DIR / "fig1_fit_validation.png")
    print(" ", FIG_DIR / "fig2_model_structure.png")


if __name__ == "__main__":
    main()

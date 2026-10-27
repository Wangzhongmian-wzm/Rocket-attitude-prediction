# -*- coding: utf-8 -*-
"""
生成离线网页：漂移距离预测.html（以及 index.html 副本）。

读取 drift_model.json 中的模型参数与 5000 组 bootstrap 抽样，
预计算高度 1–20 m、风速 0–5 m/s 的网格（含 95% 区间），
再注入 web_template.html。

使用：
    python build_web.py
然后双击 index.html 或 漂移距离预测.html。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import drift_model

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "web_template.html"
MODEL = BASE / "drift_model.json"
OUT_MAIN = BASE / "漂移距离预测.html"
OUT_INDEX = BASE / "index.html"

H_MIN, H_MAX, H_STEP = 1.0, 20.0, 0.25
V_MIN, V_MAX, V_STEP = 0.0, 5.0, 0.1
SEED = 20261020


def build_grid(model: dict) -> tuple[dict, dict]:
    p = model["parameters"]
    draws = np.asarray(model["uncertainty"]["anchor_draws"], dtype=float)
    if draws.ndim != 2 or draws.shape[1] != 4:
        raise ValueError("anchor_draws 形状不正确，请重新运行 train_model.py")

    k1, b1, k2, b2 = draws.T
    h1 = float(model["anchors"]["h1"])
    h2 = float(model["anchors"]["h2"])
    log_ratio = np.log(h2 / h1)
    pk = np.log(k2 / k1) / log_ratio
    pb = np.log(b2 / b1) / log_ratio

    heights = np.round(np.arange(H_MIN, H_MAX + 1e-9, H_STEP), 4)
    winds = np.round(np.arange(V_MIN, V_MAX + 1e-9, V_STEP), 4)

    # 固定噪声矩阵：让区间随高度/风速平滑变化，而不是每个点重新抽签
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(len(draws), len(winds)))

    data = []
    for h in heights:
        kd = k1 * (h / h1) ** pk
        bd = b1 * (h / h1) ** pb
        sigma = drift_model._sigma_at(float(h), model)
        dd = kd[:, None] * winds[None, :] + bd[:, None] + sigma * z

        lo = np.maximum(0.0, np.percentile(dd, 2.5, axis=0))
        hi = np.percentile(dd, 97.5, axis=0)
        d = p["a_k"] * h ** p["p_k"] * winds + p["a_b"] * h ** p["p_b"]
        t = p["a_t"] * h ** p["p_t"]

        lo = np.minimum(lo, d)
        hi = np.maximum(hi, d)

        d_r = np.round(d, 3)
        lo_r = np.round(np.minimum(lo, d), 3)
        hi_r = np.round(np.maximum(hi, d), 3)
        # 舍入后再次保证点预测落在区间内
        lo_r = np.minimum(lo_r, d_r)
        hi_r = np.maximum(hi_r, d_r)

        row = [
            [float(d_r[j]), float(lo_r[j]), float(hi_r[j]), float(round(t, 3))]
            for j in range(len(winds))
        ]
        data.append(row)

        if not np.isfinite(row).all():
            raise ValueError(f"高度 {h} 的网格出现非有限值")

    grid = {
        "h0": float(H_MIN),
        "dh": float(H_STEP),
        "nh": int(len(heights)),
        "v0": float(V_MIN),
        "dv": float(V_STEP),
        "nv": int(len(winds)),
        "data": data,
    }

    metrics = model["metrics"]
    fit = metrics["fit_11_5m"]
    loocv = metrics["loocv_11_5m"]
    train_df = pd.read_csv(BASE / "data" / "measured_points.csv")
    train_points = [
        {
            "h": float(row["释放高度(m)"]),
            "v": float(row["风速(m/s)"]),
            "d": float(row["漂移距离(m)"]),
        }
        for _, row in train_df.iterrows()
    ]

    # 给“下载固化模型”准备一份精简的锚点抽样（500 组，足够做区间估计）
    base_draws = np.asarray(model["uncertainty"]["anchor_draws"], dtype=float)
    rng = np.random.default_rng(SEED)
    take = min(500, len(base_draws))
    idx = rng.choice(len(base_draws), size=take, replace=False)
    base_anchor_draws = np.round(base_draws[idx], 6).tolist()

    meta = {
        "a_k": p["a_k"], "p_k": p["p_k"],
        "a_b": p["a_b"], "p_b": p["p_b"],
        "a_t": p["a_t"], "p_t": p["p_t"],
        "fit_r2": fit["r2"],
        "fit_rmse": fit["rmse"],
        "loocv_r2": loocv["r2"],
        "loocv_rmse": loocv["rmse"],
        "coverage": loocv["pi95_coverage"],
        "n_train": fit["n"],
        "measured_heights": [1.55, 11.5],
        "measured_wind": [0.2, 3.3],
        "sigma_h1": model["uncertainty"]["sigma_h1"],
        "sigma_h2": model["uncertainty"]["sigma_h2"],
        "h1": model["anchors"]["h1"],
        "h2": model["anchors"]["h2"],
        "base_anchors": model["anchors"],
        "base_anchor_draws": base_anchor_draws,
        "train_points": train_points,
        "date": "2026-10-20",
    }
    return grid, meta


def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"找不到模板：{TEMPLATE}")
    model = drift_model.load_model(MODEL)
    grid, meta = build_grid(model)

    template = TEMPLATE.read_text(encoding="utf-8")
    if "__GRID_JSON__" not in template or "__META_JSON__" not in template:
        raise SystemExit("模板缺少 __GRID_JSON__ 或 __META_JSON__ 占位符")

    grid_json = json.dumps(grid, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    html = template.replace("__GRID_JSON__", grid_json).replace("__META_JSON__", meta_json)

    OUT_MAIN.write_text(html, encoding="utf-8")
    OUT_INDEX.write_text(html, encoding="utf-8")

    # 质量检查
    data = np.asarray(grid["data"], dtype=float)
    d, lo, hi = data[..., 0], data[..., 1], data[..., 2]
    assert (lo <= d + 1e-9).all() and (d <= hi + 1e-9).all()
    assert np.isfinite(data).all()
    assert (lo >= 0).all()

    print("网页已生成：")
    print(f"  {OUT_MAIN.name}  ({OUT_MAIN.stat().st_size / 1024:.0f} KB)")
    print(f"  {OUT_INDEX.name}  ({OUT_INDEX.stat().st_size / 1024:.0f} KB)")
    print(f"网格：{grid['nh']} 个高度 × {grid['nv']} 个风速 = {grid['nh'] * grid['nv']} 点")
    print(f"高度范围：{H_MIN}–{H_MAX} m（步长 {H_STEP}）")
    print(f"风速范围：{V_MIN}–{V_MAX} m/s（步长 {V_STEP}）")
    print(f"区间检查：lo ≤ d ≤ hi 全部通过；最小下限 {lo.min():.3f} m，最大上限 {hi.max():.3f} m")
    print(f"示例：h=10, v=2.0 → 网格插值 d≈{data[36, 20, 0]:.2f} m "
          f"[{data[36, 20, 1]:.2f}, {data[36, 20, 2]:.2f}]")


if __name__ == "__main__":
    main()

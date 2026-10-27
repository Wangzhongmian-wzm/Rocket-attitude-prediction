# -*- coding: utf-8 -*-
"""
漂移距离预测模型 —— 共享加载与预测逻辑。

模型形式
--------
    d(h, v) = k(h) * v + b(h)
    k(h) = a_k * h ** p_k
    b(h) = a_b * h ** p_b

其中 h 为释放高度 (m)，v 为风速 (m/s)，d 为漂移距离 (m)。
模型参数由 train_model.py 从 楼层抛伞记录.xlsx 的实测数据训练得到，
保存为 drift_model.json。

用法
----
    from drift_model import load_model, predict
    model = load_model()
    result = predict(12.0, 2.0, model)
    print(result["drift_m"], result["lo95"], result["hi95"])
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).with_name("drift_model.json")


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """读取训练好的模型 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sigma_at(h: float, model: dict) -> float:
    """残差标准差 sigma(h)：在两个实测锚点之间按对数线性插值。"""
    u = model["uncertainty"]
    h1, h2 = u["h1"], u["h2"]
    s1, s2 = u["sigma_h1"], u["sigma_h2"]
    # 用幂律等价形式：sigma ∝ h^q
    q = math.log(s2 / s1) / math.log(h2 / h1)
    return s1 * (h / h1) ** q


def _applicability(h: float, v: float, model: dict) -> tuple[str, list[str]]:
    """
    判断输入的适用性。

    注意：高度落在 1.55–11.5 m 之间，并不等于该高度实测过。
    只有 1.55 m 与 11.5 m 是实测锚点；7.5 / 9.5 m 属于内插，
    13.5 / 15.5 m 属于外推。
    """
    rng = model["applicable_range"]
    hmin, hmax = rng["height"]
    vmin, vmax = rng["wind"]
    measured_h = model.get("data_provenance", {}).get("measured_heights", [hmin, hmax])

    h_out = not (hmin <= h <= hmax)
    v_out = not (vmin <= v <= vmax)
    is_anchor = any(abs(h - mh) < 1e-9 for mh in measured_h)
    is_interp = (not h_out) and (not is_anchor)

    notes: list[str] = []
    if h_out:
        notes.append(
            f"释放高度 {h:g} m 超出实测锚点范围 {hmin:g}–{hmax:g} m，"
            "属于高度外推，误差会明显增大。"
        )
    elif is_interp:
        notes.append(
            f"释放高度 {h:g} m 介于两个实测锚点之间，属于高度内插；"
            "该高度本身没有实测数据，可信度低于 1.55 / 11.5 m 锚点。"
        )
    if v_out:
        if v == 0:
            notes.append("风速 0 m/s 略低于实测风速下限 0.2 m/s，此时预测值等于截距 b(h)。")
        else:
            notes.append(
                f"风速 {v:g} m/s 超出实测范围 {vmin:g}–{vmax:g} m/s，"
                "属于风速外推，误差会明显增大。"
            )

    if h_out and v_out:
        status = "高度+风速双重外推"
    elif h_out:
        status = "高度外推"
    elif is_interp and v_out:
        status = "高度内插+风速外推"
    elif is_interp:
        status = "高度内插（未实测）"
    elif v_out:
        status = "风速外推"
    else:
        status = "实测高度范围内"
    return status, notes


def predict(
    height_m: float,
    wind_mps: float,
    model: dict | None = None,
    with_interval: bool = True,
    seed: int | None = 20261020,
) -> dict:
    """
    预测漂移距离。

    参数
    ----
    height_m : 释放高度，单位 m，必须 > 0。
    wind_mps : 风速，单位 m/s，必须 >= 0。
    model    : load_model() 返回的字典；省略时自动读取 drift_model.json。
    with_interval : 是否计算 95% 预测区间。
    seed     : 区间抽样随机种子；固定种子使结果可复现。

    返回
    ----
    字典，包含 drift_m（点预测）、lo95 / hi95（95% 预测区间）、
    k / b（该高度下的线性系数）、hang_time_s（估计留空时间）、
    applicability（适用性）和 warnings（中文提示列表）。
    """
    if model is None:
        model = load_model()

    h = float(height_m)
    v = float(wind_mps)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("释放高度必须是大于 0 的有限数值。")
    if not math.isfinite(v) or v < 0:
        raise ValueError("风速必须是不小于 0 的有限数值。")

    p = model["parameters"]
    k = p["a_k"] * h ** p["p_k"]
    b = p["a_b"] * h ** p["p_b"]
    d = k * v + b
    t = p["a_t"] * h ** p["p_t"]

    # 可选：网页训练模式固化下来的现场校准层
    cal = model.get("calibration")
    corr = 0.0
    var_corr = 0.0
    zh = zv = 0.0
    if cal:
        ref = cal.get("reference", {"h": 10.0, "v": 2.0, "hs": 10.0, "vs": 2.0})
        zh = (h - float(ref["h"])) / float(ref["hs"])
        zv = (v - float(ref["v"])) / float(ref["vs"])
        c = cal.get("c")
        if c is None:
            c = [cal.get("c0", 0.0), cal.get("c1", 0.0), cal.get("c2", 0.0)]
        c = [float(x) for x in c]
        corr = c[0] + c[1] * zh + c[2] * zv
        d += corr
        k += c[2] / float(ref["vs"])
        b += c[0] + c[1] * zh - c[2]
        cov = np.asarray(cal.get("cov", np.zeros((3, 3))), dtype=float)
        x = np.array([1.0, zh, zv])
        var_corr = float(max(0.0, x @ cov @ x))

    status, notes = _applicability(h, v, model)
    result = {
        "height_m": h,
        "wind_mps": v,
        "drift_m": float(d),
        "k_s": float(k),
        "b_m": float(b),
        "hang_time_s": float(t),
        "applicability": status,
        "warnings": notes,
        "calibrated": bool(cal),
        "calibration_correction": float(corr) if cal else 0.0,
        "calibration_n": int(cal.get("n_samples", 0)) if cal else 0,
    }

    if with_interval:
        draws = np.asarray(model["uncertainty"]["anchor_draws"], dtype=float)
        if draws.ndim != 2 or draws.shape[1] != 4 or len(draws) < 100:
            raise ValueError("模型文件中的 anchor_draws 不完整，请重新运行 train_model.py。")
        k1, b1, k2, b2 = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3]
        h1 = model["anchors"]["h1"]
        h2 = model["anchors"]["h2"]
        log_ratio = math.log(h2 / h1)

        # 对每个锚点抽样，反解该高度对应的幂律指数
        p_k_draw = np.log(k2 / k1) / log_ratio
        p_b_draw = np.log(b2 / b1) / log_ratio
        k_draw = k1 * (h / h1) ** p_k_draw
        b_draw = b1 * (h / h1) ** p_b_draw

        sigma = _sigma_at(h, model)
        rng = np.random.default_rng(seed)
        d_draw = k_draw * v + b_draw + rng.normal(0.0, sigma, size=len(draws))

        widen = 1.96 * math.sqrt(var_corr)
        # 漂移距离不可能为负：区间下限按物理约束截断到 0
        result["lo95"] = float(max(0.0, np.percentile(d_draw, 2.5) + corr - widen))
        result["hi95"] = float(np.percentile(d_draw, 97.5) + corr + widen)
        result["sigma_m"] = float(sigma)
        if cal:
            result["calibration_sigma_m"] = float(math.sqrt(var_corr))

    return result


def format_result(result: dict) -> str:
    """把 predict() 的结果格式化为便于终端阅读的中文文本。"""
    lines = [
        f"释放高度：      {result['height_m']:.2f} m",
        f"风速：          {result['wind_mps']:.2f} m/s",
        f"预测漂移距离：  {result['drift_m']:.2f} m",
    ]
    if "lo95" in result:
        lines.append(f"95% 预测区间：  {result['lo95']:.2f} – {result['hi95']:.2f} m")
    lines.extend(
        [
            f"该高度的 k：    {result['k_s']:.4f} s",
            f"该高度的 b：    {result['b_m']:.4f} m",
            f"估计留空时间：  {result['hang_time_s']:.2f} s",
            f"适用性：        {result['applicability']}",
        ]
    )
    if result.get("calibrated"):
        lines.append(f"模型：          已固化（含 {result.get('calibration_n', 0)} 条本地数据）")
    for w in result["warnings"]:
        lines.append(f"提示：{w}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_result(predict(12.0, 2.0)))

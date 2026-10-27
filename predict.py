# -*- coding: utf-8 -*-
"""
漂移距离预测命令行工具。

单点预测：
    python predict.py --height 10 --wind 2.0

批量预测：
    python predict.py --batch 待预测.csv --output 批量预测结果.csv

输入 CSV / Excel 需包含两列：
    释放高度(m)   风速(m/s)
也接受 height / wind、height_m / wind_mps 等英文列名。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from drift_model import format_result, load_model, predict

BASE = Path(__file__).resolve().parent

HEIGHT_ALIASES = {"释放高度(m)", "释放高度", "高度(m)", "高度", "height", "height_m", "height m"}
WIND_ALIASES = {"风速(m/s)", "风速", "wind", "wind_mps", "wind m/s", "v"}


def _find_column(df: pd.DataFrame, aliases: set[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def run_batch(input_path: Path, output_path: Path, model: dict) -> None:
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    hcol = _find_column(df, HEIGHT_ALIASES)
    vcol = _find_column(df, WIND_ALIASES)
    if hcol is None or vcol is None:
        raise SystemExit(
            "批量输入缺少列。需要包含“释放高度(m)”和“风速(m/s)”两列；"
            f"当前列为：{list(df.columns)}"
        )

    rows = []
    for _, row in df.iterrows():
        try:
            h = float(row[hcol])
            v = float(row[vcol])
        except (TypeError, ValueError):
            continue
        r = predict(h, v, model=model)
        rows.append(
            {
                "释放高度(m)": r["height_m"],
                "风速(m/s)": r["wind_mps"],
                "预测漂移距离(m)": round(r["drift_m"], 4),
                "95%下限(m)": round(r["lo95"], 4),
                "95%上限(m)": round(r["hi95"], 4),
                "留空时间(s)": round(r["hang_time_s"], 4),
                "适用性": r["applicability"],
                "提示": "；".join(r["warnings"]) if r["warnings"] else "",
            }
        )

    if not rows:
        raise SystemExit("批量输入中没有可解析的数值行。")

    out = pd.DataFrame(rows)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        out.to_excel(output_path, index=False)
    else:
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已输出 {len(out)} 行预测结果：{output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="根据释放高度与风速预测降落伞/水火箭的漂移距离。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--height", type=float, help="释放高度，单位 m")
    parser.add_argument("--wind", type=float, help="风速，单位 m/s")
    parser.add_argument("--batch", type=Path, help="批量预测输入文件（CSV 或 Excel）")
    parser.add_argument("--output", type=Path, help="批量预测输出文件（CSV 或 Excel）")
    parser.add_argument("--model", type=Path, default=BASE / "drift_model.json",
                        help="模型 JSON 路径（默认 drift_model.json）")
    parser.add_argument("--no-interval", action="store_true", help="不计算 95%% 预测区间")
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"找不到模型文件：{args.model}\n请先运行 python train_model.py", file=sys.stderr)
        return 2

    model = load_model(args.model)

    if args.batch is not None:
        if args.output is None:
            args.output = args.batch.with_name(args.batch.stem + "_预测结果.csv")
        run_batch(args.batch, args.output, model)
        return 0

    if args.height is None or args.wind is None:
        parser.print_help()
        return 1

    if args.no_interval:
        # 预测函数仍会抽样；这里只展示点预测
        r = predict(args.height, args.wind, model=model)
        print(f"释放高度：      {r['height_m']:.2f} m")
        print(f"风速：          {r['wind_mps']:.2f} m/s")
        print(f"预测漂移距离：  {r['drift_m']:.2f} m")
        print(f"适用性：        {r['applicability']}")
        for w in r["warnings"]:
            print(f"提示：{w}")
        return 0

    r = predict(args.height, args.wind, model=model)
    print(format_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

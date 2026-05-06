#!/usr/bin/env python3
"""
Find where high Average scores (>= threshold) concentrate by (n, unit_idx).

Reads Excel files from data/eval_files/teler and data/eval_files/reasoning.
Reports: per model and across models, which (n, unit_idx) pairs have the most high-score rows.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR


def to_native(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def process_method(eval_dir: Path, threshold: float, top_k: int) -> dict:
    # model -> list of {n, unit_idx, high_count}
    by_model: dict[str, list] = {}
    aggregated: dict[tuple, int] = {}

    for xlsx in sorted(eval_dir.glob("*.xlsx")):
        model = xlsx.stem
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
            continue

        if "Average" not in df.columns:
            print(f"Warning: skip {model} (no Average column)", file=sys.stderr)
            continue
        if "n" not in df.columns or "unit_idx" not in df.columns:
            print(f"Warning: skip {model} (missing n or unit_idx)", file=sys.stderr)
            continue

        high_df = df[df["Average"] >= threshold].copy()
        if high_df.empty:
            by_model[model] = []
            continue

        grp = high_df.groupby(["n", "unit_idx"]).size().reset_index(name="high_count")
        grp = grp.sort_values("high_count", ascending=False)

        rows = []
        for _, r in grp.head(top_k).iterrows():
            n_val, u_val, cnt = r["n"], r["unit_idx"], int(r["high_count"])
            rows.append({"n": to_native(n_val), "unit_idx": to_native(u_val), "high_count": cnt})
            key = (n_val, u_val)
            aggregated[key] = aggregated.get(key, 0) + cnt

        by_model[model] = rows

    agg_list = []
    for (n, u), c in sorted(aggregated.items(), key=lambda x: -x[1])[:top_k]:
        agg_list.append({"n": to_native(n), "unit_idx": to_native(u), "high_count": int(c)})

    return {"by_model": by_model, "aggregated": agg_list}


def print_report(method: str, data: dict, threshold: float, top_k: int):
    print(f"\n{'='*70}")
    print(f"  {method.upper()} — Top (n, unit_idx) with Average >= {threshold}")
    print("=" * 70)

    print(f"\n--- Aggregated across all models (top {top_k}) ---")
    if not data["aggregated"]:
        print("  (none)")
    else:
        for r in data["aggregated"]:
            print(f"  n={r['n']}, unit_idx={r['unit_idx']}: {r['high_count']} high-score rows")

    print(f"\n--- Per model (top {top_k} each) ---")
    for model in sorted(data["by_model"].keys()):
        rows = data["by_model"][model]
        if not rows:
            print(f"  {model}: (none)")
        else:
            print(f"  {model}:")
            for r in rows:
                print(f"    n={r['n']}, unit_idx={r['unit_idx']}: {r['high_count']}")


def main():
    parser = argparse.ArgumentParser(description="Find (n, unit_idx) with most high Average scores")
    parser.add_argument("--threshold", "-t", type=float, default=4.0, help="Threshold (>=) (default: 4)")
    parser.add_argument("--top", "-k", type=int, default=20, help="Top K (default: 20)")
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="both")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--csv", help="Output CSV path (aggregated only)")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"

    results = {}
    for method in (["teler", "reasoning"] if args.method == "both" else [args.method]):
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue
        results[method] = process_method(eval_dir, args.threshold, args.top)
        print_report(method, results[method], args.threshold, args.top)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "n", "unit_idx", "high_count"])
            for method, data in results.items():
                for r in data["aggregated"]:
                    w.writerow([method, r["n"], r["unit_idx"], r["high_count"]])
        print(f"Wrote {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Find the pattern:
  Fluency >= fluency_high  AND  (dims_low) <= low_threshold  (all dims)

Runs over data/eval_files/<method>/*.xlsx.
Reports per model and aggregated across models, including top (n, unit_idx) hotspots.

Requires: pandas, openpyxl
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR


DEFAULT_DIMS_LOW = [
    "Narrative_Consistency",
    "Informativeness",
    "Bidirectional_Coherence",
    "Context_Faithfulness",
]


def to_native(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def analyze_method(eval_dir: Path, fluency_high: float, low_threshold: float, dims_low: list[str], top_k: int) -> dict:
    per_model: dict[str, dict] = {}
    hotspots: dict[tuple, int] = {}

    agg_total_rows = 0
    agg_fluent_rows = 0
    agg_pattern_rows = 0
    n_skipped = 0

    for xlsx in sorted(eval_dir.glob("*.xlsx")):
        model = xlsx.stem
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        required = ["Fluency", "n", "unit_idx", *dims_low]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Warning: skip {model} (missing columns: {missing})", file=sys.stderr)
            n_skipped += 1
            continue

        fluent = df["Fluency"] >= fluency_high
        others_low = (df[dims_low] <= low_threshold).all(axis=1)
        pattern = fluent & others_low

        total_rows = int(len(df))
        fluent_rows = int(fluent.sum())
        pattern_rows = int(pattern.sum())

        # Hotspots across models: count how many pattern rows per (n, unit_idx)
        if pattern_rows:
            hot = df.loc[pattern, ["n", "unit_idx"]].groupby(["n", "unit_idx"]).size()
            for (n, u), c in hot.items():
                key = (n, u)
                hotspots[key] = hotspots.get(key, 0) + int(c)

        per_model[model] = {
            "total_rows": total_rows,
            "fluent_rows": fluent_rows,
            "pattern_rows": pattern_rows,
            "p_others_low_given_fluent": (pattern_rows / fluent_rows) if fluent_rows else 0.0,
            "p_pattern_overall": (pattern_rows / total_rows) if total_rows else 0.0,
        }

        agg_total_rows += total_rows
        agg_fluent_rows += fluent_rows
        agg_pattern_rows += pattern_rows

    agg_hotspots = [
        {"n": to_native(n), "unit_idx": to_native(u), "count": int(c)}
        for (n, u), c in sorted(hotspots.items(), key=lambda x: -x[1])[:top_k]
    ]

    return {
        "per_model": per_model,
        "aggregate": {
            "total_rows": agg_total_rows,
            "fluent_rows": agg_fluent_rows,
            "pattern_rows": agg_pattern_rows,
            "p_others_low_given_fluent": (agg_pattern_rows / agg_fluent_rows) if agg_fluent_rows else 0.0,
            "p_pattern_overall": (agg_pattern_rows / agg_total_rows) if agg_total_rows else 0.0,
            "top_n_unitidx_hotspots": agg_hotspots,
            "skipped_files": n_skipped,
        },
    }


def print_report(method: str, result: dict, fluency_high: float, low_threshold: float, dims_low: list[str], top_k: int):
    print("\n" + "=" * 72)
    print(f"  {method.upper()} — Fluency >= {fluency_high} AND {dims_low} <= {low_threshold}")
    print("=" * 72)

    agg = result["aggregate"]
    print(
        f"\nAGGREGATE: pattern={agg['pattern_rows']}  fluent={agg['fluent_rows']}  total={agg['total_rows']}  "
        f"P(others_low|fluent)={agg['p_others_low_given_fluent']:.3f}  P(pattern)={agg['p_pattern_overall']:.3f}"
    )

    print(f"\nTop hotspots (n, unit_idx) (top {top_k}):")
    if not agg["top_n_unitidx_hotspots"]:
        print("  (none)")
    else:
        for r in agg["top_n_unitidx_hotspots"]:
            print(f"  n={r['n']}, unit_idx={r['unit_idx']}: {r['count']}")

    print("\nPer model:")
    for model in sorted(result["per_model"].keys()):
        m = result["per_model"][model]
        print(
            f"  {model}: pattern={m['pattern_rows']}  fluent={m['fluent_rows']}  total={m['total_rows']}  "
            f"P(others_low|fluent)={m['p_others_low_given_fluent']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Find rows where Fluency is high but other dimensions are low"
    )
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="teler")
    parser.add_argument("--fluency-high", type=float, default=4.0, help="Fluency threshold (>=). Default: 4")
    parser.add_argument("--low-threshold", type=float, default=2.0, help="Low threshold (<=). Default: 2")
    parser.add_argument(
        "--dims-low",
        default=",".join(DEFAULT_DIMS_LOW),
        help="Comma-separated list of dimensions that must be <= low-threshold",
    )
    parser.add_argument("--top", "-k", type=int, default=20, help="Top K hotspots to show. Default: 20")
    parser.add_argument("--output", "-o", default=None, help="Write full JSON results to this path")
    parser.add_argument("--csv", default=None, help="Write per-model summary CSV to this path")
    parser.add_argument("--base-dir", default=None, help="Base directory (default: Narrative-Infilling project dir)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    dims_low = [d.strip() for d in str(args.dims_low).split(",") if d.strip()]
    eval_root = base_dir / "data" / "eval_files"

    results = {}
    for method in (["teler", "reasoning"] if args.method == "both" else [args.method]):
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue
        results[method] = analyze_method(eval_dir, args.fluency_high, args.low_threshold, dims_low, args.top)
        print_report(method, results[method], args.fluency_high, args.low_threshold, dims_low, args.top)

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
            w.writerow(
                [
                    "method",
                    "model",
                    "pattern_rows",
                    "fluent_rows",
                    "total_rows",
                    "p_others_low_given_fluent",
                    "p_pattern_overall",
                ]
            )
            for method, res in results.items():
                for model, m in res["per_model"].items():
                    w.writerow(
                        [
                            method,
                            model,
                            m["pattern_rows"],
                            m["fluent_rows"],
                            m["total_rows"],
                            f"{m['p_others_low_given_fluent']:.6f}",
                            f"{m['p_pattern_overall']:.6f}",
                        ]
                    )
        print(f"Wrote {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


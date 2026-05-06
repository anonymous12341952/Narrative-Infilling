#!/usr/bin/env python3
"""
Count rows where each LLM rubric dimension >= threshold (default: 4).

Reads Excel files from data/eval_files/<method>/. For each dimension, reports:
- Aggregate over all LLMs: total high count, top (n, unit_idx) hotspots
- Per LLM: high count, total rows, %, and top (n, unit_idx) hotspots

Requires: pandas, openpyxl
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # Narrative-Infilling (data/eval_files/ lives here)

DIMENSIONS = [
    "Fluency",
    "Context_Faithfulness",
    "Bidirectional_Coherence",
    "Narrative_Consistency",
    "Informativeness",
]


def getModelsInDir(eval_dir: Path) -> set[str]:
    """Return set of model names (xlsx stems) in eval_dir."""
    return {f.stem for f in eval_dir.glob("*.xlsx")}


def to_native(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def process_method(eval_dir: Path, threshold: float, top_k: int, models_filter: set[str] | None = None) -> dict:
    """Process Excel files in eval_dir. If models_filter is set, only process models in that set."""
    by_dim: dict[str, dict] = {d: {"aggregate": {"total": 0, "by_n_unitidx": []}, "by_model": {}} for d in DIMENSIONS}
    agg_n_unit: dict[str, dict[tuple, int]] = {d: {} for d in DIMENSIONS}

    for xlsx in sorted(eval_dir.glob("*.xlsx")):
        model = xlsx.stem
        if models_filter is not None and model not in models_filter:
            continue
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
            continue

        has_n_unit = "n" in df.columns and "unit_idx" in df.columns

        for dim in DIMENSIONS:
            if dim not in df.columns:
                continue

            high_mask = df[dim] >= threshold
            high_count = int(high_mask.sum())

            by_dim[dim]["by_model"][model] = {
                "total": high_count,
                "total_rows": len(df),
                "by_n_unitidx": [],
            }
            by_dim[dim]["aggregate"]["total"] += high_count

            if has_n_unit and high_count > 0:
                high_df = df.loc[high_mask, ["n", "unit_idx"]].copy()
                grp = high_df.groupby(["n", "unit_idx"]).size().reset_index(name="count")
                grp = grp.sort_values("count", ascending=False)
                for _, r in grp.head(top_k).iterrows():
                    n_val, u_val, cnt = r["n"], r["unit_idx"], int(r["count"])
                    by_dim[dim]["by_model"][model]["by_n_unitidx"].append(
                        {"n": to_native(n_val), "unit_idx": to_native(u_val), "count": cnt}
                    )
                    key = (n_val, u_val)
                    agg_n_unit[dim][key] = agg_n_unit[dim].get(key, 0) + cnt

    for dim in DIMENSIONS:
        items = sorted(agg_n_unit[dim].items(), key=lambda x: -x[1])[:top_k]
        by_dim[dim]["aggregate"]["by_n_unitidx"] = [
            {"n": to_native(n), "unit_idx": to_native(u), "count": int(c)} for (n, u), c in items
        ]

    return by_dim


def print_report(method: str, data: dict, threshold: float, top_k: int):
    print(f"\n{'='*70}")
    print(f"  {method.upper()} — Dimension scores >= {threshold}")
    print("=" * 70)

    for dim in DIMENSIONS:
        d = data.get(dim, {})
        agg = d.get("aggregate", {})
        by_model = d.get("by_model", {})

        print(f"\n--- {dim} ---")
        print(f"  Aggregate high count: {agg.get('total', 0)}")

        agg_nu = agg.get("by_n_unitidx", [])
        if agg_nu:
            print(f"  Top (n, unit_idx) aggregated (top {top_k}):")
            for r in agg_nu[:10]:
                print(f"    n={r['n']}, unit_idx={r['unit_idx']}: {r['count']}")

        print("  Per model:")
        for model in sorted(by_model.keys()):
            m = by_model[model]
            total = m["total"]
            total_rows = m.get("total_rows", 0)
            pct = 100 * total / total_rows if total_rows else 0
            print(f"    {model}: {total} / {total_rows} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Count rows where dimension scores >= threshold in evaluation Excel files")
    parser.add_argument("--threshold", "-t", type=float, default=4.0, help="High score threshold (>=) (default: 4)")
    parser.add_argument("--top", "-k", type=int, default=20, help="Top K (n, unit_idx) per dimension (default: 20)")
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="both")
    parser.add_argument(
        "--common-models-only",
        action="store_true",
        help="When method=both, restrict to models present in both teler and reasoning",
    )
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--csv", help="Output CSV path (per-model summary)")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"

    models_filter: set[str] | None = None
    if args.common_models_only and args.method == "both":
        teler_dir = eval_root / "teler"
        reasoning_dir = eval_root / "reasoning"
        if teler_dir.exists() and reasoning_dir.exists():
            models_filter = getModelsInDir(teler_dir) & getModelsInDir(reasoning_dir)
            print(f"Restricting to {len(models_filter)} models common in both: {sorted(models_filter)}", file=sys.stderr)
        else:
            print("Warning: --common-models-only set but teler/reasoning dirs not found", file=sys.stderr)

    results = {}
    for method in (["teler", "reasoning"] if args.method == "both" else [args.method]):
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue
        results[method] = process_method(eval_dir, args.threshold, args.top, models_filter=models_filter)
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
            w.writerow(["method", "dimension", "model", "high_count", "total_rows", "pct"])
            for method, data in results.items():
                for dim in DIMENSIONS:
                    agg = data.get(dim, {}).get("aggregate", {})
                    w.writerow([method, dim, "(aggregate)", agg.get("total", 0), "", ""])
                    for model, m in data.get(dim, {}).get("by_model", {}).items():
                        total = m.get("total", 0)
                        tr = m.get("total_rows", 0)
                        pct = f"{100*total/tr:.1f}%" if tr else ""
                        w.writerow([method, dim, model, total, tr, pct])
        print(f"Wrote {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


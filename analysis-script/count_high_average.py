#!/usr/bin/env python3
"""
Count rows where Average score >= threshold in evaluation Excel files.
Reads from data/eval_files/teler and data/eval_files/reasoning.
Reports: per model, per template, and across templates.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # Narrative-Infilling (data/eval_files/ lives here)


def getModelsInDir(eval_dir: Path) -> set[str]:
    """Return set of model names (xlsx stems) in eval_dir."""
    return {f.stem for f in eval_dir.glob("*.xlsx")}


REASONING_TEMPLATE_LABELS = {
    0: "abductive_reasoning",
    1: "causal_reasoning",
    2: "counterfactual_thinking",
    3: "chain_of_thought",
}


def process_method(
    eval_dir: Path,
    method: str,
    threshold: float,
    models_filter: set[str] | None = None,
) -> dict:
    """Process all Excel files in eval_dir. Return counts by model and template.
    If models_filter is set, only process models whose name is in that set."""
    group_col = "template_name" if method == "teler" else "template_id"

    # model -> { total_rows, high_count, by_template: { template: { total, high } } }
    by_model: dict[str, dict] = {}
    # template -> { total_rows, high_count, by_model: { model: high_count } }
    by_template: dict[str, dict] = {}

    for xlsx in sorted(eval_dir.glob("*.xlsx")):
        model = xlsx.stem
        if models_filter is not None and model not in models_filter:
            continue
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
            continue

        if "Average" not in df.columns:
            print(f"Warning: skip {model} (no Average column)", file=sys.stderr)
            continue
        if group_col not in df.columns:
            print(f"Warning: skip {model} (no {group_col})", file=sys.stderr)
            continue

        total = len(df)
        high = int((df["Average"] >= threshold).sum())

        by_model[model] = {
            "total_rows": total,
            "high_count": high,
            "by_template": {},
        }

        for tpl_val, grp in df.groupby(group_col):
            if method == "reasoning":
                try:
                    tid = int(tpl_val) if pd.notna(tpl_val) else tpl_val
                    tpl_key = REASONING_TEMPLATE_LABELS.get(tid, str(tpl_val))
                except (ValueError, TypeError):
                    tpl_key = str(tpl_val)
            else:
                tpl_key = str(tpl_val)

            t_total = len(grp)
            t_high = int((grp["Average"] >= threshold).sum())

            by_model[model]["by_template"][tpl_key] = {"total": t_total, "high_count": t_high}

            if tpl_key not in by_template:
                by_template[tpl_key] = {"total_rows": 0, "high_count": 0, "by_model": {}}
            by_template[tpl_key]["total_rows"] += t_total
            by_template[tpl_key]["high_count"] += t_high
            by_template[tpl_key]["by_model"][model] = t_high

    return {"by_model": by_model, "by_template": by_template}


def print_report(method: str, data: dict, threshold: float):
    print(f"\n{'='*60}")
    print(f"  {method.upper()} — Rows with Average >= {threshold}")
    print("=" * 60)

    by_model = data["by_model"]
    by_template = data["by_template"]

    print("\n--- Per model ---")
    for model in sorted(by_model.keys()):
        m = by_model[model]
        pct = 100 * m["high_count"] / m["total_rows"] if m["total_rows"] else 0
        print(f"  {model}: {m['high_count']} / {m['total_rows']} ({pct:.1f}%)")

    print("\n--- Per template (across all models) ---")
    for tpl in sorted(by_template.keys()):
        t = by_template[tpl]
        pct = 100 * t["high_count"] / t["total_rows"] if t["total_rows"] else 0
        print(f"  {tpl}: {t['high_count']} / {t['total_rows']} ({pct:.1f}%)")

    print("\n--- Per model × template (high count only) ---")
    for model in sorted(by_model.keys()):
        m = by_model[model]
        high_tpls = [(t, v["high_count"]) for t, v in m["by_template"].items() if v["high_count"] > 0]
        if high_tpls:
            print(f"  {model}:")
            for tpl, cnt in sorted(high_tpls, key=lambda x: -x[1]):
                print(f"    {tpl}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Count rows with Average >= threshold in evaluation Excel files")
    parser.add_argument("--threshold", "-t", type=float, default=4.0, help="Threshold (default: 4)")
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="both")
    parser.add_argument(
        "--common-models-only",
        action="store_true",
        help="When method=both, restrict to models present in both teler and reasoning",
    )
    parser.add_argument("--output", "-o", help="Output JSON path")
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
        results[method] = process_method(eval_dir, method, args.threshold, models_filter=models_filter)
        print_report(method, results[method], args.threshold)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


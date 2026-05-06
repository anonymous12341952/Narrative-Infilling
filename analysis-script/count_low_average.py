#!/usr/bin/env python3
"""
Count rows where Average score < threshold in evaluation Excel files.
Reads from data/eval_files/teler and data/eval_files/reasoning.
Reports: per model, per template, and across templates.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

REASONING_TEMPLATE_LABELS = {
    0: "abductive_reasoning",
    1: "causal_reasoning",
    2: "counterfactual_thinking",
    3: "chain_of_thought",
}


def getModelsInDir(eval_dir: Path) -> set[str]:
    """Return set of model names (xlsx stems) in eval_dir."""
    return {f.stem for f in eval_dir.glob("*.xlsx")}


def process_method(
    eval_dir: Path,
    method: str,
    threshold: float,
    models_filter: set[str] | None = None,
) -> dict:
    """Process all Excel files in eval_dir. Return counts by model and template.
    If models_filter is set, only process models whose name is in that set."""
    if method == "teler":
        group_col = "template_name"
    else:
        group_col = "template_id"

    # model -> { total_rows, low_count, by_template: { template: { total, low } } }
    by_model: dict[str, dict] = {}
    # template -> { total_rows, low_count, by_model: { model: low_count } }
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
        low = int((df["Average"] < threshold).sum())

        by_model[model] = {
            "total_rows": total,
            "low_count": low,
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
            t_low = int((grp["Average"] < threshold).sum())

            by_model[model]["by_template"][tpl_key] = {"total": t_total, "low_count": t_low}

            if tpl_key not in by_template:
                by_template[tpl_key] = {"total_rows": 0, "low_count": 0, "by_model": {}}
            by_template[tpl_key]["total_rows"] += t_total
            by_template[tpl_key]["low_count"] += t_low
            by_template[tpl_key]["by_model"][model] = t_low

    return {"by_model": by_model, "by_template": by_template}


def print_report(method: str, data: dict, threshold: float, per_model: bool = True):
    """Print human-readable report."""
    print(f"\n{'='*60}")
    print(f"  {method.upper()} — Rows with Average < {threshold}")
    print("=" * 60)

    by_model = data["by_model"]
    by_template = data["by_template"]

    if per_model:
        print("\n--- Per model ---")
        for model in sorted(by_model.keys()):
            m = by_model[model]
            pct = 100 * m["low_count"] / m["total_rows"] if m["total_rows"] else 0
            print(f"  {model}: {m['low_count']} / {m['total_rows']} ({pct:.1f}%)")

    print("\n--- Per template (across all models) ---")
    for tpl in sorted(by_template.keys()):
        t = by_template[tpl]
        pct = 100 * t["low_count"] / t["total_rows"] if t["total_rows"] else 0
        print(f"  {tpl}: {t['low_count']} / {t['total_rows']} ({pct:.1f}%)")

    if per_model:
        print("\n--- Per model × template (low count only) ---")
        for model in sorted(by_model.keys()):
            m = by_model[model]
            low_tpls = [(t, v["low_count"]) for t, v in m["by_template"].items() if v["low_count"] > 0]
            if low_tpls:
                print(f"  {model}:")
                for tpl, cnt in sorted(low_tpls, key=lambda x: -x[1]):
                    print(f"    {tpl}: {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Count rows with Average < threshold in evaluation Excel files"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=2.0, help="Threshold (default: 2)"
    )
    parser.add_argument(
        "--method", "-m", choices=["teler", "reasoning", "both"], default="both"
    )
    parser.add_argument("--no-per-model", action="store_true", help="Omit per-model and per-model×template sections")
    parser.add_argument(
        "--filter-teler-to-reasoning",
        action="store_true",
        help="Restrict teler to only the LLMs present in reasoning (fair comparison: 10 models)",
    )
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"

    teler_models_filter: set[str] | None = None
    if args.filter_teler_to_reasoning:
        reasoning_dir = eval_root / "reasoning"
        if reasoning_dir.exists():
            teler_models_filter = getModelsInDir(reasoning_dir)
            print(f"Filtering teler to {len(teler_models_filter)} models from reasoning: {sorted(teler_models_filter)}", file=sys.stderr)
        else:
            print("Warning: --filter-teler-to-reasoning set but reasoning dir not found", file=sys.stderr)

    results = {}

    for method in (["teler", "reasoning"] if args.method == "both" else [args.method]):
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue
        models_filter = teler_models_filter if method == "teler" else None
        results[method] = process_method(eval_dir, method, args.threshold, models_filter=models_filter)
        print_report(method, results[method], args.threshold, per_model=not args.no_per_model)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize for JSON (convert any non-JSON types)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

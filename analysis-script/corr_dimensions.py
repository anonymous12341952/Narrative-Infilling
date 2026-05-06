#!/usr/bin/env python3
"""
Compute correlation matrix across evaluation dimensions per model and per method.

Reads Excel files from data/eval_files/<method>/*.xlsx and computes correlations across:
  Fluency, Context_Faithfulness, Bidirectional_Coherence, Narrative_Consistency, Informativeness
Optionally includes automatic metrics (e.g., bert_f1, rouge1/R-1, meteor, chrf, sem_f1).

Outputs:
- Prints a correlation matrix per model (and optional aggregate across models)
- Optionally writes JSON and/or CSV (long-form) outputs

Requires: pandas, openpyxl

Commands:
  # Default (rubric dimensions, spearman, both methods)
  python corr_dimensions.py --method teler

  # All metrics (rubric + auto)
  python corr_dimensions.py --method teler --preset all

  # Aggregate only (no per-model matrices)
  python corr_dimensions.py --method teler --aggregate

  # Kendall correlation
  python corr_dimensions.py --method teler --corr kendall

  # Save outputs
  python corr_dimensions.py --method both --preset all --aggregate --output-json corr.json --output-csv corr.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

DEFAULT_DIMS = [
    "Fluency",
    "Context_Faithfulness",
    "Bidirectional_Coherence",
    "Narrative_Consistency",
    "Informativeness",
]

DEFAULT_AUTO_METRICS = [
    "bert_f1",
    "rouge1",
    "meteor",
    "chrf",
    "sem_f1",
]

SHORT_NAME_MAP: dict[str, str] = {
    # rubric dimensions
    "Fluency": "F",
    "Context_Faithfulness": "CF",
    "Bidirectional_Coherence": "BC",
    "Narrative_Consistency": "NC",
    "Informativeness": "I",
    # auto metrics
    "bert_f1": "bert",
    "rouge1": "R1",
    "meteor": "MET",
    "chrf": "CHRF",
    "sem_f1": "SemF1",
}


_ALIASES: dict[str, str] = {
    # ROUGE-1 aliases
    "r-1": "rouge1",
    "rouge-1": "rouge1",
    "r1": "rouge1",
    "rouge1": "rouge1",
    # Semantic F1 aliases
    "sem-f1": "sem_f1",
    "semf1": "sem_f1",
    "sem_f1": "sem_f1",
    # Other auto metric aliases (case/format variants)
    "meteor": "meteor",
    "chrf": "chrf",
    "bert_f1": "bert_f1",
    "bert-f1": "bert_f1",
}


def resolve_metric_name(name: str) -> str:
    """
    Normalize metric names from CLI (e.g., 'R-1' -> 'rouge1', 'Sem-F1' -> 'sem_f1').
    Keeps unknown names unchanged.
    """
    s = str(name).strip()
    key = s.lower().replace(" ", "").replace("/", "").replace("\\", "")
    return _ALIASES.get(key, s)


def matrix_to_nested_dict(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for r in df.index:
        out[str(r)] = {}
        for c in df.columns:
            v = df.loc[r, c]
            out[str(r)][str(c)] = None if pd.isna(v) else float(v)
    return out


def pairwise_non_nan_counts(x: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise overlap counts:
      count[i,j] = number of rows where both dim i and dim j are non-null.
    """
    mask = x.notna().astype(int)
    return mask.T.dot(mask)


def maybe_shorten(df: pd.DataFrame, use_short_names: bool) -> pd.DataFrame:
    if not use_short_names:
        return df
    rename = {c: SHORT_NAME_MAP.get(c, c) for c in df.columns}
    # df is square; index and columns are same labels
    out = df.rename(columns=rename, index=rename)
    return out


def print_matrix(
    title: str,
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    decimals: int = 3,
    show_counts: bool = False,
    use_short_names: bool = True,
):
    print("\n" + title)
    print("-" * len(title))
    # correlation matrix
    print("corr:")
    corr_disp = maybe_shorten(corr, use_short_names)
    print(corr_disp.round(decimals).to_string())
    if show_counts:
        print("\nN (pairwise non-null rows):")
        counts_disp = maybe_shorten(counts, use_short_names)
        print(counts_disp.to_string())


def compute_for_one_df(df: pd.DataFrame, dims: list[str], corr_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[dims].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method=corr_method)
    counts = pairwise_non_nan_counts(x)
    return corr, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Correlation matrix across evaluation dimensions per model")
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="both")
    parser.add_argument(
        "--corr",
        choices=["spearman", "kendall", "pearson"],
        default="kendall",
        help="Correlation type (default: kendall)",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Print the N (pairwise non-null rows) table under each correlation matrix",
    )
    parser.add_argument(
        "--long-names",
        action="store_true",
        help="Use full metric names in printed matrices (default: short names)",
    )
    parser.add_argument(
        "--preset",
        choices=["dimensions", "auto", "all"],
        default="dimensions",
        help="Metric preset: rubric dimensions, auto metrics, or all (default: dimensions)",
    )
    parser.add_argument(
        "--dims",
        default=None,
        help="Override metrics with a comma-separated list (supports aliases like R-1, Sem-F1)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Also compute aggregate correlation by concatenating rows across all models",
    )
    parser.add_argument(
        "--show-per-model",
        action="store_true",
        help="When using --aggregate, also print per-model matrices (default: off)",
    )
    parser.add_argument("--output-json", default=None, help="Write results to JSON path")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Write long-form CSV: method,model,dim_i,dim_j,corr,N",
    )
    parser.add_argument("--base-dir", default=None, help="Base directory (default: Narrative-Infilling project dir)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR

    if args.dims:
        dims = [resolve_metric_name(d) for d in str(args.dims).split(",") if str(d).strip()]
    else:
        if args.preset == "dimensions":
            dims = list(DEFAULT_DIMS)
        elif args.preset == "auto":
            dims = list(DEFAULT_AUTO_METRICS)
        else:
            dims = list(DEFAULT_DIMS) + list(DEFAULT_AUTO_METRICS)

    eval_root = base_dir / "data" / "eval_files"
    methods = ["teler", "reasoning"] if args.method == "both" else [args.method]

    results: dict[str, dict] = {}

    for method in methods:
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue

        print("\n" + "=" * 72)
        print(f"  {method.upper()} — correlation={args.corr}")
        print("=" * 72)

        method_res: dict[str, dict] = {"per_model": {}}
        agg_frames: list[pd.DataFrame] = []

        # If aggregate is requested, default to NOT printing per-model matrices
        print_per_model = (not args.aggregate) or args.show_per_model
        # Only compute per-model correlations if we need them for output (or printing)
        need_per_model_stats = print_per_model or bool(args.output_csv) or bool(args.output_json)

        use_short_names = not args.long_names

        for xlsx in sorted(eval_dir.glob("*.xlsx")):
            model = xlsx.stem
            try:
                df = pd.read_excel(xlsx)
            except Exception as e:
                print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
                continue

            missing = [d for d in dims if d not in df.columns]
            if missing:
                print(f"Warning: {model} missing dims: {missing} (skipping)", file=sys.stderr)
                continue

            if args.aggregate:
                agg_frames.append(df[dims].copy())

            if need_per_model_stats:
                corr, counts = compute_for_one_df(df, dims, args.corr)
                if print_per_model:
                    print_matrix(
                        f"{model}",
                        corr,
                        counts,
                        show_counts=args.show_counts,
                        use_short_names=use_short_names,
                    )
                method_res["per_model"][model] = {
                    "corr": matrix_to_nested_dict(corr),
                    "counts": matrix_to_nested_dict(counts),
                    "rows": int(len(df)),
                }

        if args.aggregate and agg_frames:
            agg_df = pd.concat(agg_frames, ignore_index=True)
            corr, counts = compute_for_one_df(agg_df, dims, args.corr)
            print_matrix(
                "(aggregate across models)",
                corr,
                counts,
                show_counts=args.show_counts,
                use_short_names=use_short_names,
            )
            method_res["aggregate"] = {
                "corr": matrix_to_nested_dict(corr),
                "counts": matrix_to_nested_dict(counts),
                "rows": int(len(agg_df)),
            }

        results[method] = method_res

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dims": dims,
                    "corr_method": args.corr,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {out_path}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for method, mres in results.items():
            for model, s in mres.get("per_model", {}).items():
                corr = s.get("corr", {})
                counts = s.get("counts", {})
                for di in corr.keys():
                    for dj in corr.get(di, {}).keys():
                        rows.append(
                            {
                                "method": method,
                                "model": model,
                                "dim_i": di,
                                "dim_j": dj,
                                "corr": corr[di][dj],
                                "N": counts.get(di, {}).get(dj),
                            }
                        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


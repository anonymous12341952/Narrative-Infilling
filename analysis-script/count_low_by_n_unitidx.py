#!/usr/bin/env python3
"""
Find where low Average scores (< threshold) concentrate by (n, unit_idx).

Reads Excel files from data/eval_files/teler and data/eval_files/reasoning.
Reports: per model and across models, which (n, unit_idx) pairs have the most low scores.
Shows percentage (low_count / total_low_scores_in_dataset) — fraction of all low scores from each hotspot.
doc_length = number of sentences (spacy) when --doc-length is used.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

# Lazy-load spacy for sentence counting (only when --doc-length)
_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Warning: spacy not available for doc_length ({e}). Run: pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)
    return _nlp


def count_sentences(text: str) -> int:
    nlp = get_nlp()
    if nlp is None:
        return 0
    doc = nlp(str(text or "").strip())
    return len(list(doc.sents)) if doc.sents else (1 if doc.text.strip() else 0)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

REASONING_TEMPLATE_LABELS = {
    0: "abductive_reasoning",
    1: "causal_reasoning",
    2: "counterfactual_thinking",
    3: "chain_of_thought",
}


def to_native(x):
    """Convert to JSON-serializable native Python type."""
    if pd.isna(x):
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def process_method(
    eval_dir: Path,
    method: str,
    threshold: float,
    top_k: int,
    *,
    include_doc_length: bool = False,
) -> dict:
    """Process all Excel files. Return low counts by (n, unit_idx) per model and aggregated."""
    if method == "teler":
        group_col = "template_name"
    else:
        group_col = "template_id"

    by_model: dict[str, list] = {}
    aggregated: dict[tuple, int] = {}
    # (n, unit_idx) -> doc_length (num sentences via spacy), from first file with source_text column
    doc_length_per_key: dict[tuple, int] = {}
    n_models = 0

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

        n_models += 1

        if include_doc_length and "source_text" in df.columns and not doc_length_per_key:
            for _, row in df[["n", "unit_idx", "source_text"]].drop_duplicates(["n", "unit_idx"]).iterrows():
                key = (row["n"], row["unit_idx"])
                doc_length_per_key[key] = count_sentences(row.get("source_text", "") or "")

        low_df = df[df["Average"] < threshold].copy()
        if low_df.empty:
            by_model[model] = []
            continue

        grp = low_df.groupby(["n", "unit_idx"]).size().reset_index(name="low_count")
        grp = grp.sort_values("low_count", ascending=False)

        rows = []
        for _, r in grp.head(top_k).iterrows():
            n_val, u_val, cnt = r["n"], r["unit_idx"], int(r["low_count"])
            rows.append({
                "n": to_native(n_val),
                "unit_idx": to_native(u_val),
                "low_count": cnt,
            })
            key = (n_val, u_val)
            aggregated[key] = aggregated.get(key, 0) + cnt

        by_model[model] = rows

    total_low_all = sum(aggregated.values())
    agg_list = []
    for (n, u), c in sorted(aggregated.items(), key=lambda x: -x[1])[:top_k]:
        pct = 100 * c / total_low_all if total_low_all else 0
        row = {
            "n": to_native(n),
            "unit_idx": to_native(u),
            "low_count": int(c),
            "total": int(total_low_all),
            "pct": round(pct, 2),
        }
        if (n, u) in doc_length_per_key:
            row["doc_length"] = doc_length_per_key[(n, u)]
        agg_list.append(row)

    return {
        "by_model": by_model,
        "aggregated": agg_list,
        "n_models": n_models,
        "doc_length_per_key": doc_length_per_key if include_doc_length else {},
    }


def print_report(
    method: str,
    data: dict,
    threshold: float,
    top_k: int,
    *,
    show_per_model: bool = True,
    show_doc_length: bool = False,
):
    """Print human-readable report."""
    print(f"\n{'='*70}")
    print(f"  {method.upper()} — Top (n, unit_idx) with Average < {threshold}")
    print("=" * 70)

    by_model = data["by_model"]
    aggregated = data["aggregated"]

    print(f"\n--- Aggregated across all models (top {top_k}) ---")
    if not aggregated:
        print("  (none)")
    else:
        for r in aggregated:
            total = r.get("total", 0)
            pct = r.get("pct", 0)
            line = f"  n={r['n']}, unit_idx={r['unit_idx']}: {r['low_count']}/{total} ({pct}%)"
            if show_doc_length and "doc_length" in r:
                line += f"  sents={r['doc_length']}"
            print(line)

    if show_doc_length and data.get("doc_length_per_key"):
        print(f"\n--- Document length (sentences) vs position (unit_idx) vs n vs low count ---")
        rows = []
        for r in aggregated:
            if "doc_length" in r:
                rows.append((r["doc_length"], r["unit_idx"], r["n"], r["low_count"], r["total"], r["pct"]))
        if rows:
            print(f"  {'sents':>6} {'unit_idx':>8} {'n':>6} {'low':>6} {'total':>6} {'pct':>6}%")
            for dl, ui, n, low, tot, pct in sorted(rows, key=lambda x: (-x[3], x[0]))[:top_k]:
                print(f"  {dl:>6} {ui:>8} {n:>6} {low:>6} {tot:>6} {pct:>5.1f}%")

    if show_per_model:
        print(f"\n--- Per model (top {top_k} each) ---")
        for model in sorted(by_model.keys()):
            rows = by_model[model]
            if not rows:
                print(f"  {model}: (none)")
            else:
                print(f"  {model}:")
                for r in rows:
                    print(f"    n={r['n']}, unit_idx={r['unit_idx']}: {r['low_count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Find (n, unit_idx) with most low Average scores in evaluation Excel files"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=2.0, help="Threshold (default: 2)"
    )
    parser.add_argument(
        "--top", "-k", type=int, default=20, help="Top K (n, unit_idx) per model (default: 20)"
    )
    parser.add_argument(
        "--method", "-m", choices=["teler", "reasoning", "both"], default="both"
    )
    parser.add_argument(
        "--no-per-model",
        action="store_true",
        help="Do not print per-model section (aggregate only)",
    )
    parser.add_argument(
        "--doc-length",
        action="store_true",
        help="Show doc length (num sentences, spacy) vs position vs n vs low count (requires source_text column)",
    )
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
        results[method] = process_method(
            eval_dir,
            method,
            args.threshold,
            args.top,
            include_doc_length=args.doc_length,
        )
        print_report(
            method,
            results[method],
            args.threshold,
            args.top,
            show_per_model=not args.no_per_model,
            show_doc_length=args.doc_length,
        )

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
            headers = ["method", "n", "unit_idx", "low_count", "total", "pct"]
            if args.doc_length and any(data.get("doc_length_per_key") for data in results.values()):
                headers.append("doc_length")
            w.writerow(headers)
            for method, data in results.items():
                for r in data["aggregated"]:
                    row = [method, r["n"], r["unit_idx"], r["low_count"], r.get("total", ""), r.get("pct", "")]
                    if "doc_length" in headers and "doc_length" in r:
                        row.append(r["doc_length"])
                    w.writerow(row)
        print(f"Wrote {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

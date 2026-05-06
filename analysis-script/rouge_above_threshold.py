#!/usr/bin/env python3
"""
Report how many rows in evaluation xlsx file(s) have ROUGE above a threshold (default 0.7).

Use --method teler|reasoning|both to run every *.xlsx in that folder (all LLMs). With more than
one file, the default is a single summary table; add --detail for per-file sections (and --list).

Columns: by default both rouge1 (ROUGE-1) and rougeL (ROUGE-L) when present — numeric 0–1.
Use --output/-o to mirror the report into a UTF-8 .txt file (still prints to the terminal).

Examples:
  python rouge_above_threshold.py -f data/eval_files/teler/deepseek-qwen-32B.xlsx
  python rouge_above_threshold.py -m teler --threshold 0.7
  python rouge_above_threshold.py -m reasoning
  python rouge_above_threshold.py -m both --detail
  python rouge_above_threshold.py -f some.xlsx --list 5
  python rouge_above_threshold.py -f some.xlsx -c rouge1
  python rouge_above_threshold.py -m teler -o data/eval_files/rouge_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import math

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


class TeeIO:
    """Write the same text to multiple streams (e.g. console + report file)."""

    __slots__ = ("_streams",)

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def safe_float(x) -> float | None:
    if pd.isna(x):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def example_columns(df: pd.DataFrame, rouge_col: str) -> list[str]:
    """Columns to print for --list: ids plus rouge1/rougeL when available."""
    order = ("dataset", "ref_id", "n", "unit_idx", "rouge1", "rougeL")
    return [c for c in order if c in df.columns] or [
        c for c in ("dataset", "ref_id", "n", "unit_idx", rouge_col) if c in df.columns
    ]


def metric_heading(rouge_col: str) -> str:
    if rouge_col == "rouge1":
        return f"{rouge_col} (ROUGE-1)"
    if rouge_col == "rougeL":
        return f"{rouge_col} (ROUGE-L)"
    return rouge_col


def print_one_metric(
    df: pd.DataFrame,
    rouge_col: str,
    threshold: float,
    list_k: int,
) -> int:
    """Print stats for one ROUGE column. Returns 1 if column missing or error, else 0."""
    res = analyze_df(df, rouge_col, threshold)
    print(f"\n  --- {metric_heading(rouge_col)} ---")

    if "error" in res:
        print(f"  {res['error']}")
        return 1

    print(f"  Threshold: > {threshold}")
    print(f"  Rows with numeric {rouge_col}:   {res['n_valid_rouge']}")
    print(f"  Rows with {rouge_col} > {threshold}:      {res['n_above']}")
    print(f"  % of all rows:             {res['pct_of_all_rows']}%")
    print(f"  % of rows w/ valid score:  {res['pct_of_valid_rouge']}%")
    if res["n_above"] > 0:
        print(f"  → At least one response has {rouge_col} > {threshold}.")
    else:
        print(f"  → No row with {rouge_col} > {threshold} (among valid scores).")

    if list_k > 0 and res["n_above"] > 0:
        sub = df.loc[res["above_mask"]].head(list_k)
        cols = example_columns(sub, rouge_col)
        print(f"\n  Examples for {rouge_col} (columns: {', '.join(cols)}):")
        for _, row in sub.iterrows():
            bits = [f"{c}={row.get(c)}" for c in cols]
            print("    " + "  ".join(bits))
    return 0


def print_summary_table(
    rows: list[dict],
    rouge_cols: list[str],
    threshold: float,
) -> None:
    """One row per model file: % of all rows with score > threshold, plus counts."""
    if not rows:
        return
    print("\n" + "=" * 72)
    print(f"  Summary — all models (ROUGE > {threshold}, % of all rows)")
    print("=" * 72)

    subcols = []
    for c in rouge_cols:
        subcols.extend([f"{c} %", f"{c} #"])
    headers = ["method", "model", "n_rows"] + subcols
    cells: list[list[str]] = [headers]
    for r in rows:
        line = [
            str(r.get("method", "")),
            str(r.get("model", "")),
            str(r.get("n_total", "")),
        ]
        for c in rouge_cols:
            err = r.get(f"{c}_err")
            if err:
                line.extend(["—", "—"])
            else:
                p = r.get(f"{c}_pct")
                n = r.get(f"{c}_n")
                line.append(f"{p}%" if p is not None else "—")
                line.append(str(n) if n is not None else "—")
        cells.append(line)

    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    for i, row in enumerate(cells):
        print("  " + "  ".join(row[j].ljust(widths[j]) for j in range(len(headers))))
        if i == 0:
            print("  " + sep)

    print_method_subtotals(rows, rouge_cols)


def print_method_subtotals(rows: list[dict], rouge_cols: list[str]) -> None:
    """Sum per-model % and counts within each method (teler vs reasoning)."""
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        m = str(r.get("method", ""))
        by_method.setdefault(m, []).append(r)

    print("\n  Per-method (sum across LLMs in that method: Σ % = sum of each model’s «% of all rows»; Σ # = sum of row counts)")
    subcols: list[str] = []
    for c in rouge_cols:
        subcols.extend([f"Σ {c} %", f"Σ {c} #"])
    h = ["method", "models"] + subcols
    lines: list[list[str]] = [h]
    for method in sorted(by_method.keys()):
        group = by_method[method]
        line = [method, str(len(group))]
        for c in rouge_cols:
            sum_pct = 0.0
            sum_n = 0
            any_ok = False
            for r in group:
                if r.get(f"{c}_err"):
                    continue
                p = r.get(f"{c}_pct")
                n = r.get(f"{c}_n")
                if p is not None:
                    sum_pct += float(p)
                    any_ok = True
                if n is not None:
                    sum_n += int(n)
                    any_ok = True
            if any_ok:
                line.append(f"{round(sum_pct, 2)}%")
                line.append(str(sum_n))
            else:
                line.extend(["—", "—"])
        lines.append(line)

    w = [max(len(lines[i][j]) for i in range(len(lines))) for j in range(len(h))]
    sep2 = "  ".join("-" * x for x in w)
    for i, row in enumerate(lines):
        print("  " + "  ".join(row[j].ljust(w[j]) for j in range(len(h))))
        if i == 0:
            print("  " + sep2)


def analyze_df(df: pd.DataFrame, rouge_col: str, threshold: float) -> dict:
    if rouge_col not in df.columns:
        return {"error": f"missing column {rouge_col!r}"}
    vals = df[rouge_col].map(safe_float)
    valid = vals.notna()
    n_total = len(df)
    n_valid = int(valid.sum())
    above = valid & (vals > threshold)
    n_above = int(above.sum())
    pct_of_all = 100 * n_above / n_total if n_total else 0.0
    pct_of_valid = 100 * n_above / n_valid if n_valid else 0.0
    return {
        "n_total": n_total,
        "n_valid_rouge": n_valid,
        "n_above": n_above,
        "pct_of_all_rows": round(pct_of_all, 2),
        "pct_of_valid_rouge": round(pct_of_valid, 2),
        "above_mask": above,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Count eval rows with ROUGE above a threshold")
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="Single evaluation .xlsx path",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["teler", "reasoning", "both"],
        default=None,
        help="Scan all xlsx under data/eval_files/<method> (use instead of --file)",
    )
    parser.add_argument(
        "--column",
        "-c",
        action="append",
        dest="columns",
        metavar="NAME",
        default=None,
        help="ROUGE column (rouge1, rougeL). Repeat -c for several. Default: both rouge1 and rougeL.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Strictly greater than this value (default: 0.7)",
    )
    parser.add_argument(
        "--list",
        "-l",
        type=int,
        default=0,
        metavar="K",
        help="Per metric: print up to K example rows (ids + rouge1/rougeL when present) where that score > threshold",
    )
    parser.add_argument("--base-dir", default=None, help="Project root (default: Narrative-Infilling)")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="With multiple files: print per-file ROUGE sections (default: summary table only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Also write the same report (stdout) to this .txt file (UTF-8); still prints to terminal",
    )
    args = parser.parse_args()

    rouge_cols: list[str]
    if args.columns:
        seen: set[str] = set()
        rouge_cols = []
        for c in args.columns:
            if c not in seen:
                seen.add(c)
                rouge_cols.append(c)
    else:
        rouge_cols = ["rouge1", "rougeL"]

    base = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base / "data" / "eval_files"

    paths: list[Path] = []
    if args.file:
        p = Path(args.file)
        paths = [p if p.is_absolute() else base / p]
    elif args.method:
        for sub in (["teler", "reasoning"] if args.method == "both" else [args.method]):
            d = eval_root / sub
            if d.is_dir():
                paths.extend(sorted(d.glob("*.xlsx")))
    else:
        print("Give --file PATH.xlsx or --method teler|reasoning|both", file=sys.stderr)
        return 1

    if not paths:
        print("No xlsx files matched.", file=sys.stderr)
        return 1

    multi = len(paths) > 1
    summary_only = multi and not args.detail

    out_fp = None
    old_stdout = sys.stdout
    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

    any_error = 0
    summary_rows: list[dict] = []

    try:
        if out_path is not None:
            out_fp = open(out_path, "w", encoding="utf-8")
            sys.stdout = TeeIO(sys.__stdout__, out_fp)

        for path in paths:
            if not path.exists():
                print(f"\nMissing: {path}", file=sys.stderr)
                any_error = 1
                continue
            try:
                df = pd.read_excel(path)
            except Exception as e:
                print(f"\nSkip {path}: {e}", file=sys.stderr)
                any_error = 1
                continue

            row: dict = {
                "method": path.parent.name,
                "model": path.stem,
                "n_total": len(df),
            }
            for col in rouge_cols:
                res = analyze_df(df, col, args.threshold)
                if "error" in res:
                    row[f"{col}_err"] = res["error"]
                    row[f"{col}_pct"] = None
                    row[f"{col}_n"] = None
                    any_error = 1
                else:
                    row[f"{col}_pct"] = res["pct_of_all_rows"]
                    row[f"{col}_n"] = res["n_above"]
            summary_rows.append(row)

            if summary_only:
                continue

            print(f"\n{'='*72}\n  {path}\n{'='*72}")
            print(f"  Total rows: {len(df)}")
            print(f"  Metrics: {', '.join(rouge_cols)}   threshold: > {args.threshold}")

            for col in rouge_cols:
                err = print_one_metric(df, col, args.threshold, args.list)
                any_error |= err

        if multi and summary_rows:
            print_summary_table(summary_rows, rouge_cols, args.threshold)

        return any_error
    finally:
        if out_fp is not None:
            sys.stdout = old_stdout
            out_fp.close()


if __name__ == "__main__":
    sys.exit(main())

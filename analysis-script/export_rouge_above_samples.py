#!/usr/bin/env python3
"""
Export evaluation rows where ROUGE is above a threshold from one model's .xlsx.

Includes problem, gold (ground truth), extracted answer, plus dataset / ids / ROUGE scores /
model name. Full model response, source text, and other eval fields are omitted. UTF-8 text file.

Examples:
  python export_rouge_above_samples.py -f data/eval_files/teler/deepseek-qwen-32B.xlsx -t 0.7 -o rouge_high.txt
  python export_rouge_above_samples.py -f data/eval_files/reasoning/deepseek-llama-70B.xlsx -t 0.9 -o high.txt
  python export_rouge_above_samples.py -f eval.xlsx --match all -t 0.7 -o out.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import math

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def safe_float(x) -> float | None:
    if pd.isna(x):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def mask(
    df: pd.DataFrame,
    *,
    threshold: float,
    match: str,
) -> tuple[pd.Series, str | None]:
    """Return boolean mask and error message if match mode is invalid."""
    n = len(df)
    empty = pd.Series([False] * n, index=df.index)

    def col_series(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series([None] * n, index=df.index)
        return df[name].map(safe_float)

    r1 = col_series("rouge1")
    rL = col_series("rougeL")
    ok1 = r1.notna() & (r1 > threshold)
    okL = rL.notna() & (rL > threshold)

    if match == "any":
        return ok1 | okL, None
    if match == "all":
        if "rouge1" not in df.columns or "rougeL" not in df.columns:
            return empty, "need both rouge1 and rougeL columns for --match all"
        return ok1 & okL, None
    if match == "rouge1":
        if "rouge1" not in df.columns:
            return empty, "missing column rouge1"
        return ok1, None
    if match == "rougeL":
        if "rougeL" not in df.columns:
            return empty, "missing column rougeL"
        return okL, None
    return empty, f"unknown match mode {match!r}"


def write_txt(fp, sub: pd.DataFrame, *, src_path: Path, threshold: float, match: str) -> None:
    fp.write(
        f"# source: {src_path}\n"
        f"# threshold: > {threshold}\n"
        f"# match: {match}\n"
        f"# rows: {len(sub)}\n\n"
    )
    meta_cols = [
        "dataset",
        "ref_id",
        "n",
        "unit_idx",
        "template_name",
        "template_id",
        "rouge1",
        "rougeL",
        "model",
        "temperature",
    ]
    text_cols = [
        ("problem", "Problem"),
        ("gold_answer", "Gold answer (ground truth)"),
        ("extracted_answer", "Extracted answer"),
    ]
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        fp.write("=" * 80 + "\n")
        fp.write(f"Sample {i} / {len(sub)}\n")
        for c in meta_cols:
            if c in sub.columns:
                fp.write(f"  {c}: {row.get(c)}\n")
        fp.write("-" * 80 + "\n")
        for col, title in text_cols:
            if col not in sub.columns:
                continue
            fp.write(f"{title}:\n{row.get(col)}\n\n")
        fp.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Export rows with ROUGE above threshold from one eval xlsx")
    p.add_argument("--file", "-f", required=True, help="Path to one model evaluation .xlsx")
    p.add_argument("--threshold", "-t", type=float, default=0.7, help="Strictly greater than (default: 0.7)")
    p.add_argument(
        "--match",
        choices=["any", "all", "rouge1", "rougeL"],
        default="any",
        help="any = rouge1>t or rougeL>t; all = both; or single metric (default: any)",
    )
    p.add_argument("--output", "-o", required=True, metavar="PATH", help="Output .txt path (UTF-8)")
    p.add_argument("--base-dir", default=None, help="Project root for relative --file (default: Narrative-Infilling)")
    p.add_argument("--max-rows", type=int, default=None, help="Cap exported rows (default: no limit)")
    args = p.parse_args()

    base = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    xlsx = Path(args.file)
    if not xlsx.is_absolute():
        xlsx = base / xlsx
    out = Path(args.output)

    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)

    if not xlsx.exists():
        print(f"Missing file: {xlsx}", file=sys.stderr)
        return 1

    try:
        df = pd.read_excel(xlsx)
    except Exception as e:
        print(f"Failed to read {xlsx}: {e}", file=sys.stderr)
        return 1

    mask, err = mask(df, threshold=args.threshold, match=args.match)
    if err:
        print(err, file=sys.stderr)
        return 1

    sub = df.loc[mask].copy()
    if args.max_rows is not None and len(sub) > args.max_rows:
        sub = sub.iloc[: args.max_rows]

    if sub.empty:
        print("No rows match the ROUGE criterion.", file=sys.stderr)
        with open(out, "w", encoding="utf-8") as fp:
            fp.write(
                f"# source: {xlsx}\n# threshold: > {args.threshold}\n# match: {args.match}\n# rows: 0\n"
            )
        return 0

    with open(out, "w", encoding="utf-8") as fp:
        write_txt(fp, sub, src_path=xlsx, threshold=args.threshold, match=args.match)

    print(f"Wrote {len(sub)} row(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

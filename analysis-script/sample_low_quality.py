#!/usr/bin/env python3
"""
Sample random low-quality evaluation rows (Average <= threshold) and write to a log file.

Mirrors sample_high_quality.py but selects low composite Average. Same output format.


Example usage:
python analysis-script/sample_low_quality.py -m both -n 30
python analysis-script/sample_low_quality.py --max-avg 2.0 -o data/eval_files/low_q_examples.log
python analysis-script/sample_low_quality.py --max-problem-len 0   # no length cap

Output default: low_quality_samples.log
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DIMENSIONS = [
    "Fluency",
    "Context_Faithfulness",
    "Bidirectional_Coherence",
    "Narrative_Consistency",
    "Informativeness",
]


def extract_score(val) -> float | None:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if 1 <= v <= 5 else None
    if isinstance(val, dict):
        s = val.get("score")
        return float(s) if s is not None and 1 <= float(s) <= 5 else None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Sample low-quality (Average <= --max-avg) evaluation rows"
    )
    parser.add_argument("-n", "--num", type=int, default=20, help="Number of samples (default: 20)")
    parser.add_argument(
        "--max-avg",
        type=float,
        default=2.5,
        help="Maximum Average (inclusive); rows with Average <= this are candidates (default: 2.5)",
    )
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="teler")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output log path (default: low_quality_samples.log)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-files", type=int, default=None, help="Max Excel files per method (default: all)")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    parser.add_argument(
        "--max-problem-len",
        type=int,
        default=600,
        help="Only include samples where problem length <= N chars (0=no limit, default: 600)",
    )
    parser.add_argument(
        "--min-problem-len",
        type=int,
        default=0,
        help="Only include samples where problem length >= N chars (default: 0)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"
    out_path = Path(args.output) if args.output else base_dir / "low_quality_samples.log"
    if not out_path.is_absolute():
        out_path = base_dir / out_path

    random.seed(args.seed)

    all_rows = []
    for method in (["teler", "reasoning"] if args.method == "both" else [args.method]):
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found", file=sys.stderr)
            continue
        xlsx_list = sorted(eval_dir.glob("*.xlsx"))
        if args.max_files:
            xlsx_list = xlsx_list[: args.max_files]
        for xlsx in xlsx_list:
            try:
                df = pd.read_excel(xlsx)
            except Exception as e:
                print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
                continue
            if "Average" not in df.columns or "problem" not in df.columns:
                continue
            if "gold_answer" not in df.columns:
                df["gold_answer"] = ""
            if "extracted_answer" not in df.columns:
                df["extracted_answer"] = ""
            if "dataset" not in df.columns:
                df["dataset"] = ""

            for _, row in df.iterrows():
                avg = extract_score(row.get("Average"))
                if avg is None or avg > args.max_avg:
                    continue
                problem_text = str(row.get("problem", "") or "")
                plen = len(problem_text)
                if args.max_problem_len and plen > args.max_problem_len:
                    continue
                if args.min_problem_len and plen < args.min_problem_len:
                    continue
                all_rows.append((method, row))

    if not all_rows:
        print(f"No rows with Average <= {args.max_avg} found", file=sys.stderr)
        return 1

    samples = random.sample(all_rows, min(args.num, len(all_rows)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        len_desc = f", problem len <= {args.max_problem_len} chars" if args.max_problem_len else ""
        f.write(
            f"Low-quality samples (Average <= {args.max_avg}{len_desc}), n={len(samples)}\n"
        )
        f.write("=" * 80 + "\n\n")

        for i, (method, row) in enumerate(samples, 1):
            problem = str(row.get("problem", "") or "")
            gold = str(row.get("gold_answer", "") or "")
            extracted = str(row.get("extracted_answer", "") or "")
            dataset = str(row.get("dataset", "") or "").strip() or "N/A"

            dim_scores = {}
            for d in DIMENSIONS:
                s = extract_score(row.get(d))
                dim_scores[d] = s if s is not None else "N/A"

            avg_val = extract_score(row.get("Average")) or "N/A"

            f.write("-" * 80 + "\n")
            f.write(
                f"Sample {i} [{method}]  Dataset: {dataset}  Average: {avg_val}  (problem: {len(problem)} chars)\n"
            )
            f.write("-" * 80 + "\n\n")
            f.write(f"DATASET: {dataset}\n\n")
            f.write("PROBLEM:\n")
            f.write(problem + "\n\n")
            f.write("GOLD ANSWER:\n")
            f.write(gold + "\n\n")
            f.write("EXTRACTED ANSWER:\n")
            f.write(extracted + "\n\n")
            f.write("DIMENSION SCORES:\n")
            for d in DIMENSIONS:
                f.write(f"  {d}: {dim_scores[d]}\n")
            f.write("\n")

    print(f"Wrote {len(samples)} samples to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Common models only (stems in both data/eval_files/teler and data/eval_files/reasoning).

For samples where composite Average meets the threshold (>= by default, or > with
--strict-gt), report the mean of each rubric dimension across those samples only,
and the % of those dimension scores that are >= --dim-threshold (default: same as -t).

Also reports what fraction of rows exceed the Average threshold. Use --by-template
to stratify by template_name (teler) or reasoning paradigm (template_id).

Output goes to the terminal; use --output to also write the same report to a .txt file.

Uses the same score parsing as count_by_position.py (numeric / dict / string).

Requires: pandas, openpyxl (for read_excel only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from count_by_position import (  # noqa: E402
    DIMENSIONS,
    commonModelStems,
    extractAverage,
    extractScore,
)

PROJECT_DIR = SCRIPT_DIR.parent


class StdoutTee:
    """Mirror stdout to multiple text streams (e.g. console + file)."""

    __slots__ = ("streams",)

    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


REASONING_TEMPLATE_LABELS = {
    0: "abductive_reasoning",
    1: "causal_reasoning",
    2: "counterfactual_thinking",
    3: "chain_of_thought",
}


def rowTemplateLabel(row, method: str) -> str:
    if method == "teler":
        if "template_name" not in row.index:
            return "(no_template_name)"
        v = row.get("template_name")
        if pd.isna(v):
            return "(missing_template_name)"
        s = str(v).strip()
        return s if s else "(empty_template_name)"
    if "template_id" not in row.index:
        return "(no_template_id)"
    v = row.get("template_id")
    if pd.isna(v):
        return "(missing_template_id)"
    try:
        tid = int(v)
        return REASONING_TEMPLATE_LABELS.get(tid, str(tid))
    except (ValueError, TypeError):
        return str(v)


def emptyTemplateState() -> dict:
    return {
        "n_total": 0,
        "n_valid": 0,
        "n_qual": 0,
        "buckets": {d: [] for d in DIMENSIONS},
    }


def accumulateRowIntoTemplate(
    st: dict,
    row,
    df: pd.DataFrame,
    threshold: float,
    strict_gt: bool,
) -> None:
    st["n_total"] += 1
    if "Average" not in df.columns:
        return
    av = extractAverage(row.get("Average"))
    if av is None:
        return
    st["n_valid"] += 1
    ok = av > threshold if strict_gt else av >= threshold
    if not ok:
        return
    st["n_qual"] += 1
    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        s = extractScore(row.get(dim))
        if s is not None:
            st["buckets"][dim].append(s)


def templateStatsFromDataFrame(
    df: pd.DataFrame,
    method: str,
    threshold: float,
    strict_gt: bool,
) -> dict[str, dict]:
    """Per template label: counts + dimension lists for Average-high rows only."""
    by_tpl: dict[str, dict] = {}
    for _, row in df.iterrows():
        tpl = rowTemplateLabel(row, method)
        if tpl not in by_tpl:
            by_tpl[tpl] = emptyTemplateState()
        accumulateRowIntoTemplate(by_tpl[tpl], row, df, threshold, strict_gt)
    return by_tpl


def mergeTemplateStats(dst: dict[str, dict], src: dict[str, dict]) -> None:
    for tpl, st in src.items():
        if tpl not in dst:
            dst[tpl] = emptyTemplateState()
        d = dst[tpl]
        d["n_total"] += st["n_total"]
        d["n_valid"] += st["n_valid"]
        d["n_qual"] += st["n_qual"]
        for dim in DIMENSIONS:
            d["buckets"][dim].extend(st["buckets"][dim])


def sheetRowStats(df: pd.DataFrame, threshold: float, strict_gt: bool) -> tuple[int, int, int]:
    """Return (n_total_rows, n_valid_average_rows, n_qualifying_rows)."""
    n_total = len(df)
    if "Average" not in df.columns:
        return n_total, 0, 0
    n_valid = 0
    n_qual = 0
    for _, row in df.iterrows():
        av = extractAverage(row.get("Average"))
        if av is None:
            continue
        n_valid += 1
        ok = av > threshold if strict_gt else av >= threshold
        if ok:
            n_qual += 1
    return n_total, n_valid, n_qual


def collectHighAverageRows(
    df: pd.DataFrame,
    threshold: float,
    strict_gt: bool,
) -> dict[str, list[float]]:
    """Rows where Average > (or >=) threshold; dim -> list of parsed scores."""
    out: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    if "Average" not in df.columns:
        return out

    for _, row in df.iterrows():
        av = extractAverage(row.get("Average"))
        if av is None:
            continue
        if strict_gt:
            if not (av > threshold):
                continue
        else:
            if not (av >= threshold):
                continue
        for dim in DIMENSIONS:
            if dim not in df.columns:
                continue
            s = extractScore(row.get(dim))
            if s is not None:
                out[dim].append(s)

    return out


def formatPercent(num: float, den: int) -> str:
    if den <= 0:
        return "—"
    return f"{100 * num / den:.1f}%"


def percentScoresAtLeast(vals: list[float], dim_threshold: float) -> str:
    if not vals:
        return "—"
    n = sum(1 for v in vals if v >= dim_threshold)
    return f"{100 * n / len(vals):.1f}%"


def printDimensionMeans(
    title: str,
    buckets: dict[str, list[float]],
    *,
    n_total_rows: int,
    n_valid_avg_rows: int,
    n_qualifying: int,
    threshold: float,
    strict_gt: bool,
    dim_threshold: float,
) -> None:
    op = ">" if strict_gt else ">="
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)
    print(f"  Total rows:              {n_total_rows}")
    print(f"  Valid Average rows:      {n_valid_avg_rows}")
    print(
        f"  Rows Average {op} {threshold}: {n_qualifying}  "
        f"({formatPercent(n_qualifying, n_total_rows)} of all rows, "
        f"{formatPercent(n_qualifying, n_valid_avg_rows)} of valid-Average rows)"
    )
    print(f"  Per dimension (only rows in Average-high set; dim score >= {dim_threshold}):")
    print(f"  {'dimension':<28} {'mean':>8}  {'pct_ge':>8}  {'n_scores':>8}")
    for dim in DIMENSIONS:
        vals = buckets[dim]
        if not vals:
            mean_s = "—"
            pct_s = "—"
            ns = 0
        else:
            mean_s = f"{sum(vals) / len(vals):.4f}"
            pct_s = percentScoresAtLeast(vals, dim_threshold)
            ns = len(vals)
        print(f"  {dim:<28} {mean_s:>8}  {pct_s:>8}  {ns:>8}")


def run(
    eval_root: Path,
    common: set[str],
    threshold: float,
    strict_gt: bool,
    dim_threshold: float,
    limit: int | None,
    per_model: bool,
    by_template: bool,
) -> None:
    for method in ("teler", "reasoning"):
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue

        method_buckets: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
        sum_total = 0
        sum_valid_avg = 0
        sum_qualifying = 0
        template_pooled: dict[str, dict] = {}

        for model in sorted(common):
            path = eval_dir / f"{model}.xlsx"
            if not path.exists():
                continue
            try:
                df = pd.read_excel(path, nrows=limit)
            except Exception as e:
                print(f"Warning: skip {path.name}: {e}", file=sys.stderr)
                continue

            n_total, n_valid, n_qual = sheetRowStats(df, threshold, strict_gt)
            sum_total += n_total
            sum_valid_avg += n_valid
            sum_qualifying += n_qual

            row_data = collectHighAverageRows(df, threshold, strict_gt)

            if by_template:
                local_tpl = templateStatsFromDataFrame(df, method, threshold, strict_gt)
                mergeTemplateStats(template_pooled, local_tpl)

            if per_model:
                if by_template:
                    for tpl in sorted(local_tpl.keys()):
                        st = local_tpl[tpl]
                        label = "template" if method == "teler" else "reasoning_type"
                        printDimensionMeans(
                            f"{method.upper()} — {model} — {label}: {tpl}",
                            st["buckets"],
                            n_total_rows=st["n_total"],
                            n_valid_avg_rows=st["n_valid"],
                            n_qualifying=st["n_qual"],
                            threshold=threshold,
                            strict_gt=strict_gt,
                            dim_threshold=dim_threshold,
                        )
                else:
                    printDimensionMeans(
                        f"{method.upper()} — {model}",
                        row_data,
                        n_total_rows=n_total,
                        n_valid_avg_rows=n_valid,
                        n_qualifying=n_qual,
                        threshold=threshold,
                        strict_gt=strict_gt,
                        dim_threshold=dim_threshold,
                    )

            for dim in DIMENSIONS:
                method_buckets[dim].extend(row_data[dim])

        if by_template:
            label = "TELER template" if method == "teler" else "reasoning type"
            for tpl in sorted(template_pooled.keys()):
                st = template_pooled[tpl]
                printDimensionMeans(
                    f"{method.upper()} — ALL COMMON MODELS — by {label}: {tpl}",
                    st["buckets"],
                    n_total_rows=st["n_total"],
                    n_valid_avg_rows=st["n_valid"],
                    n_qualifying=st["n_qual"],
                    threshold=threshold,
                    strict_gt=strict_gt,
                    dim_threshold=dim_threshold,
                )

        printDimensionMeans(
            f"{method.upper()} — ALL COMMON MODELS (pooled)",
            method_buckets,
            n_total_rows=sum_total,
            n_valid_avg_rows=sum_valid_avg,
            n_qualifying=sum_qualifying,
            threshold=threshold,
            strict_gt=strict_gt,
            dim_threshold=dim_threshold,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mean of each dimension for samples with high composite Average (common models only)."
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=4.0,
        help="Keep samples with composite Average above this (default: 4; see --strict-gt)",
    )
    parser.add_argument(
        "--strict-gt",
        action="store_true",
        help="Use Average > threshold instead of >= threshold",
    )
    parser.add_argument(
        "--dim-threshold",
        type=float,
        default=None,
        help="Count dimension scores >= this for pct_ge column (default: same as --threshold)",
    )
    parser.add_argument("--base-dir", default=None, help="Project root (default: parent of this script = Narrative-Infilling)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max rows read per workbook (testing only)")
    parser.add_argument("--per-model", action="store_true", help="Also print one table per model")
    parser.add_argument(
        "--by-template",
        action="store_true",
        help="Stratify by template_name (teler) or reasoning paradigm (template_id); pooled across common models; with --per-model, also split each model file by template",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Also write the full report to this .txt file (UTF-8); still prints to terminal",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"
    common = commonModelStems(eval_root)
    if not common:
        print("No common models found (need data/eval_files/teler ∩ data/eval_files/reasoning xlsx stems).", file=sys.stderr)
        return 1

    print("dimension_drivers_by_average.py", file=sys.stderr)
    print(f"Common models ({len(common)}): {', '.join(sorted(common))}", file=sys.stderr)
    op = ">" if args.strict_gt else ">="
    print(f"Filter: Average {op} {args.threshold}", file=sys.stderr)

    dim_t = args.dim_threshold if args.dim_threshold is not None else args.threshold
    print(f"Dimension pct_ge: share of scores >= {dim_t} (within Average-high rows)", file=sys.stderr)
    if args.by_template:
        print(
            "By-template: teler uses template_name; reasoning uses template_id (paradigm labels).",
            file=sys.stderr,
        )

    out_path: Path | None = None
    out_f = None
    saved_stdout = sys.stdout
    try:
        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = base_dir / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_f = open(out_path, "w", encoding="utf-8")
            out_f.write("dimension_drivers_by_average.py\n")
            out_f.write(f"Common models ({len(common)}): {', '.join(sorted(common))}\n")
            out_f.write(f"Filter: Average {op} {args.threshold}\n")
            out_f.write(
                f"Dimension pct_ge: share of scores >= {dim_t} (within Average-high rows)\n"
            )
            if args.by_template:
                out_f.write(
                    "By-template: teler uses template_name; reasoning uses template_id (paradigm labels).\n"
                )
            out_f.write("\n")
            out_f.flush()
            sys.stdout = StdoutTee(saved_stdout, out_f)

        run(
            eval_root,
            common,
            args.threshold,
            args.strict_gt,
            dim_t,
            args.limit,
            args.per_model,
            args.by_template,
        )
    finally:
        if out_f is not None:
            sys.stdout = saved_stdout
            out_f.close()
            print(f"Wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

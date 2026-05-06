#!/usr/bin/env python3
"""
Analyze evaluation scores by blank position: opening, middle, closing.

Divides the problem text into three equal parts by sentence count (spacy),
then classifies which part the blank falls into. Reports:
- Average score per position
- Percentage of samples with Average >= threshold per position
- Percentage of samples where at most 1 dimension has low score (<= low_threshold)

Options: --show-avg, --show-high-pct, --show-low-count-pct, --common-models-only, etc.
Also reports by `dataset` when present in eval Excels (--no-dataset-report to hide print).

CSV exports: --csv (same scope as main run), --csv-common (models in both dirs),
--csv-all (teler+reasoning with every xlsx, no intersection filter).
"""

DIMENSIONS = [
    "Fluency",
    "Context_Faithfulness",
    "Bidirectional_Coherence",
    "Narrative_Consistency",
    "Informativeness",
]

REASONING_TEMPLATE_LABELS = {
    0: "abductive_reasoning",
    1: "causal_reasoning",
    2: "counterfactual_thinking",
    3: "chain_of_thought",
}

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # Narrative-Infilling (data/eval_files/ lives here)

# Lazy-load spacy for sentence counting
_nlp = None
_nlp_warned = False


def get_nlp():
    global _nlp, _nlp_warned
    if _nlp is None and not _nlp_warned:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            _nlp_warned = True
            print(
                f"Warning: spacy not available ({e}). Using simple sentence split. Run: pip install spacy && python -m spacy download en_core_web_sm for better accuracy.",
                file=sys.stderr,
            )
    return _nlp


_sentence_cache: dict[str, int] = {}


def count_sentences_simple(text: str) -> int:
    """Fallback when spacy is not available: split by sentence-ending punctuation."""
    import re
    s = str(text or "").strip()
    if not s:
        return 0
    # Split on . ! ? followed by space or end
    parts = re.split(r"[.!?]+\s+|[.!?]+$", s)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts) if parts else 1


def count_sentences(text: str) -> int:
    nlp = get_nlp()
    key = str(text or "").strip()
    if key in _sentence_cache:
        return _sentence_cache[key]
    if nlp is not None:
        doc = nlp(key)
        n = len(list(doc.sents)) if doc.sents else (1 if doc.text.strip() else 0)
    else:
        n = count_sentences_simple(key)
    _sentence_cache[key] = n
    return n


def extractScore(val) -> float | None:
    """Extract numeric score from dimension value (int, float, dict, or dict string)."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if 1 <= v <= 5 else None
    if isinstance(val, dict):
        s = val.get("score")
        return float(s) if s is not None and 1 <= float(s) <= 5 else None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            import ast
            obj = ast.literal_eval(val)
            if isinstance(obj, dict):
                s = obj.get("score")
                return float(s) if s is not None and 1 <= float(s) <= 5 else None
            if isinstance(obj, (int, float)):
                v = float(obj)
                return v if 1 <= v <= 5 else None
        except (ValueError, SyntaxError):
            pass
    return None


def extractAverage(val) -> float | None:
    """Extract numeric Average from value (int, float, dict, or dict string)."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if 1 <= v <= 5 else None
    if isinstance(val, dict):
        s = val.get("score")
        return float(s) if s is not None and 1 <= float(s) <= 5 else None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            import ast
            obj = ast.literal_eval(val)
            if isinstance(obj, dict):
                s = obj.get("score")
                return float(s) if s is not None and 1 <= float(s) <= 5 else None
            if isinstance(obj, (int, float)):
                v = float(obj)
                return v if 1 <= v <= 5 else None
        except (ValueError, SyntaxError):
            pass
    return None


def classify_position(total_sentences: int, unit_idx: int) -> str | None:
    """Classify blank position: opening, middle, or closing.
    Divides the story into three equal parts by sentence count.
    """
    if total_sentences <= 0 or unit_idx < 1 or unit_idx > total_sentences:
        return None
    first_end = math.ceil(total_sentences / 3)
    second_end = math.ceil(2 * total_sentences / 3)
    if unit_idx <= first_end:
        return "opening"
    if unit_idx <= second_end:
        return "middle"
    return "closing"


def getModelsInDir(eval_dir: Path) -> set[str]:
    """Return set of model names (xlsx stems) in eval_dir."""
    return {f.stem for f in eval_dir.glob("*.xlsx")}


def commonModelStems(eval_root: Path) -> set[str]:
    teler_dir = eval_root / "teler"
    reasoning_dir = eval_root / "reasoning"
    if not teler_dir.exists() or not reasoning_dir.exists():
        return set()
    return getModelsInDir(teler_dir) & getModelsInDir(reasoning_dir)


def collect_results(
    eval_root: Path,
    methods: list[str],
    models_filter: set[str] | None,
    *,
    threshold: float,
    limit: int | None,
    low_threshold: float,
) -> dict:
    results = {}
    for method in methods:
        eval_dir = eval_root / method
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} not found, skipping", file=sys.stderr)
            continue
        results[method] = process_method(
            eval_dir,
            method,
            threshold,
            models_filter=models_filter,
            limit=limit,
            low_threshold=low_threshold,
        )
    return results


def write_position_csv(csv_path: Path, results: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "dataset",
                "template",
                "position",
                "total",
                "high_count",
                "high_pct",
                "low_count_leq_1",
                "low_count_leq_1_pct",
                "avg",
            ]
        )
        for method, data in results.items():
            by_tpl = data["by_template"]
            by_pos = data["by_position"]
            by_ds = data.get("by_dataset", {})
            by_dst = data.get("by_dataset_template", {})
            for tpl in sorted(by_tpl.keys()):
                for pos in ["opening", "middle", "closing"]:
                    p = by_tpl[tpl][pos]
                    total = p["total"]
                    avg = round(p["sum"] / total, 4) if total else 0
                    pct = round(100 * p["high_count"] / total, 2) if total else 0
                    low_leq_1 = p.get("low_count_leq_1", 0)
                    low_pct = round(100 * low_leq_1 / total, 2) if total else 0
                    w.writerow([method, "*", tpl, pos, total, p["high_count"], pct, low_leq_1, low_pct, avg])
            for pos in ["opening", "middle", "closing"]:
                p = by_pos[pos]
                total = p["total"]
                avg = round(p["sum"] / total, 4) if total else 0
                pct = round(100 * p["high_count"] / total, 2) if total else 0
                low_leq_1 = p.get("low_count_leq_1", 0)
                low_pct = round(100 * low_leq_1 / total, 2) if total else 0
                w.writerow([method, "*", "(aggregate)", pos, total, p["high_count"], pct, low_leq_1, low_pct, avg])
            for ds in sorted(by_ds.keys()):
                for pos in ["opening", "middle", "closing"]:
                    p = by_ds[ds][pos]
                    total = p["total"]
                    avg = round(p["sum"] / total, 4) if total else 0
                    pct = round(100 * p["high_count"] / total, 2) if total else 0
                    low_leq_1 = p.get("low_count_leq_1", 0)
                    low_pct = round(100 * low_leq_1 / total, 2) if total else 0
                    w.writerow([method, ds, "(dataset_aggregate)", pos, total, p["high_count"], pct, low_leq_1, low_pct, avg])
            for ds in sorted(by_dst.keys()):
                for tpl in sorted(by_dst[ds].keys()):
                    for pos in ["opening", "middle", "closing"]:
                        p = by_dst[ds][tpl][pos]
                        total = p["total"]
                        avg = round(p["sum"] / total, 4) if total else 0
                        pct = round(100 * p["high_count"] / total, 2) if total else 0
                        low_leq_1 = p.get("low_count_leq_1", 0)
                        low_pct = round(100 * low_leq_1 / total, 2) if total else 0
                        w.writerow([method, ds, tpl, pos, total, p["high_count"], pct, low_leq_1, low_pct, avg])
    print(f"Wrote {csv_path}")


def empty_pos_stats() -> dict:
    return {
        "opening": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
        "middle": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
        "closing": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
    }


def dataset_key(row) -> str:
    if "dataset" not in row.index:
        return "(unknown)"
    v = row.get("dataset")
    if pd.isna(v):
        return "(unknown)"
    s = str(v).strip()
    return s if s else "(unknown)"


def process_method(
    eval_dir: Path,
    method: str,
    threshold: float,
    models_filter: set[str] | None = None,
    limit: int | None = None,
    low_threshold: float = 2.0,
) -> dict:
    """Process all Excel files. Return stats by position (opening, middle, closing)
    per prompt level (teler: template_name) or reasoning paradigm (reasoning: template_id).
    If models_filter is set, only process models in that set.
    """
    group_col = "template_name" if method == "teler" else "template_id"
    by_template: dict[str, dict[str, dict]] = {}
    by_position: dict[str, dict] = {
        "opening": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
        "middle": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
        "closing": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
    }
    by_model: dict[str, dict[str, dict]] = {}
    by_dataset: dict[str, dict[str, dict]] = {}
    by_dataset_template: dict[str, dict[str, dict[str, dict]]] = {}

    for xlsx in sorted(eval_dir.glob("*.xlsx")):
        model = xlsx.stem
        if models_filter is not None and model not in models_filter:
            continue
        try:
            df = pd.read_excel(xlsx, nrows=limit)
        except Exception as e:
            print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
            continue

        if "Average" not in df.columns:
            print(f"Warning: skip {model} (no Average column)", file=sys.stderr)
            continue
        if "unit_idx" not in df.columns:
            print(f"Warning: skip {model} (no unit_idx)", file=sys.stderr)
            continue
        if group_col not in df.columns:
            print(f"Warning: skip {model} (no {group_col})", file=sys.stderr)
            continue

        text_col = "problem" if "problem" in df.columns else "source_text" if "source_text" in df.columns else None
        if text_col is None:
            print(f"Warning: skip {model} (no problem or source_text column)", file=sys.stderr)
            continue

        dim_cols = [c for c in DIMENSIONS if c in df.columns]

        by_model[model] = {
            "opening": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
            "middle": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
            "closing": {"total": 0, "sum": 0.0, "high_count": 0, "low_count_leq_1": 0},
        }

        for _, row in df.iterrows():
            text = row.get(text_col, "") or ""
            unit_idx = row.get("unit_idx")
            if pd.isna(unit_idx):
                continue
            try:
                unit_idx = int(unit_idx)
            except (ValueError, TypeError):
                continue

            tpl_val = row.get(group_col)
            if method == "reasoning":
                try:
                    tid = int(tpl_val) if pd.notna(tpl_val) else tpl_val
                    tpl_key = REASONING_TEMPLATE_LABELS.get(tid, str(tpl_val))
                except (ValueError, TypeError):
                    tpl_key = str(tpl_val)
            else:
                tpl_key = str(tpl_val)

            ds_key = dataset_key(row)

            total_sentences = count_sentences(text)
            if total_sentences <= 0:
                continue

            pos = classify_position(total_sentences, unit_idx)
            if pos is None:
                continue

            avg_val = extractAverage(row.get("Average"))
            if avg_val is None:
                continue

            low_count = 0
            if dim_cols:
                for dim in dim_cols:
                    s = extractScore(row.get(dim))
                    if s is not None and s <= low_threshold:
                        low_count += 1

            if tpl_key not in by_template:
                by_template[tpl_key] = empty_pos_stats()
            by_template[tpl_key][pos]["total"] += 1
            by_template[tpl_key][pos]["sum"] += avg_val
            if avg_val >= threshold:
                by_template[tpl_key][pos]["high_count"] += 1
            if low_count <= 1:
                by_template[tpl_key][pos]["low_count_leq_1"] += 1

            by_position[pos]["total"] += 1
            by_position[pos]["sum"] += avg_val
            if avg_val >= threshold:
                by_position[pos]["high_count"] += 1
            if low_count <= 1:
                by_position[pos]["low_count_leq_1"] += 1

            by_model[model][pos]["total"] += 1
            by_model[model][pos]["sum"] += avg_val
            if avg_val >= threshold:
                by_model[model][pos]["high_count"] += 1
            if low_count <= 1:
                by_model[model][pos]["low_count_leq_1"] += 1

            if ds_key not in by_dataset:
                by_dataset[ds_key] = empty_pos_stats()
            by_dataset[ds_key][pos]["total"] += 1
            by_dataset[ds_key][pos]["sum"] += avg_val
            if avg_val >= threshold:
                by_dataset[ds_key][pos]["high_count"] += 1
            if low_count <= 1:
                by_dataset[ds_key][pos]["low_count_leq_1"] += 1

            if ds_key not in by_dataset_template:
                by_dataset_template[ds_key] = {}
            if tpl_key not in by_dataset_template[ds_key]:
                by_dataset_template[ds_key][tpl_key] = empty_pos_stats()
            by_dataset_template[ds_key][tpl_key][pos]["total"] += 1
            by_dataset_template[ds_key][tpl_key][pos]["sum"] += avg_val
            if avg_val >= threshold:
                by_dataset_template[ds_key][tpl_key][pos]["high_count"] += 1
            if low_count <= 1:
                by_dataset_template[ds_key][tpl_key][pos]["low_count_leq_1"] += 1

    return {
        "by_template": by_template,
        "by_position": by_position,
        "by_model": by_model,
        "by_dataset": by_dataset,
        "by_dataset_template": by_dataset_template,
    }


def format_pos_line(pos: str, p: dict, show_avg: bool, show_high_pct: bool, show_low_count_pct: bool) -> str:
    total = p["total"]
    if total == 0:
        return f"  {pos:8}: (no samples)"
    high = p["high_count"]
    avg = p["sum"] / total
    pct = 100 * high / total
    low_leq_1 = p.get("low_count_leq_1", 0)
    low_pct = 100 * low_leq_1 / total
    parts = []
    if show_avg:
        parts.append(f"avg={avg:.2f}")
    if show_high_pct:
        parts.append(f"high%={pct:.1f} ({high:,} / {total:,})")
    if show_low_count_pct:
        parts.append(f"low≤1%={low_pct:.1f} ({low_leq_1:,} / {total:,})")
    return f"  {pos:8}: " + ", ".join(parts)


def to_out_json(p: dict) -> dict:
    return {
        "total": p["total"],
        "high_count": p["high_count"],
        "low_count_leq_1": p.get("low_count_leq_1", 0),
        "avg": round(p["sum"] / p["total"], 4) if p["total"] else 0,
        "high_pct": round(100 * p["high_count"] / p["total"], 2) if p["total"] else 0,
        "low_count_leq_1_pct": round(100 * p.get("low_count_leq_1", 0) / p["total"], 2) if p["total"] else 0,
    }


def print_report(
    method: str,
    data: dict,
    threshold: float,
    *,
    show_avg: bool = True,
    show_high_pct: bool = True,
    show_low_count_pct: bool = True,
    show_per_model: bool = False,
    show_dataset: bool = True,
    show_dataset_template_detail: bool = False,
):
    """Print human-readable report."""
    tpl_label = "prompt level" if method == "teler" else "reasoning paradigm"
    print(f"\n{'='*70}")
    print(f"  {method.upper()} — By blank position per {tpl_label}")
    print("=" * 70)

    by_template = data["by_template"]
    by_pos = data["by_position"]
    by_model = data["by_model"]

    for tpl in sorted(by_template.keys()):
        print(f"\n--- {tpl} ---")
        for pos in ["opening", "middle", "closing"]:
            p = by_template[tpl][pos]
            print(format_pos_line(pos, p, show_avg, show_high_pct, show_low_count_pct))

    print(f"\n--- Aggregate (all {tpl_label}s) ---")
    for pos in ["opening", "middle", "closing"]:
        p = by_pos[pos]
        print(format_pos_line(pos, p, show_avg, show_high_pct, show_low_count_pct))

    if show_dataset and data.get("by_dataset"):
        print("\n--- By dataset (aggregate over templates & models in this run) ---")
        for ds in sorted(data["by_dataset"].keys()):
            print(f"\n  [{ds}]")
            for pos in ["opening", "middle", "closing"]:
                p = data["by_dataset"][ds][pos]
                print(format_pos_line(pos, p, show_avg, show_high_pct, show_low_count_pct))

    if show_dataset_template_detail and data.get("by_dataset_template"):
        print(f"\n--- By dataset × {tpl_label} ---")
        for ds in sorted(data["by_dataset_template"].keys()):
            for tpl in sorted(data["by_dataset_template"][ds].keys()):
                print(f"\n  [{ds}] | {tpl} ---")
                for pos in ["opening", "middle", "closing"]:
                    p = data["by_dataset_template"][ds][tpl][pos]
                    print(format_pos_line(pos, p, show_avg, show_high_pct, show_low_count_pct))

    if show_per_model and by_model:
        print("\n--- Per model ---")
        for model in sorted(by_model.keys()):
            m = by_model[model]
            lines = []
            for pos in ["opening", "middle", "closing"]:
                p = m[pos]
                total = p["total"]
                if total == 0:
                    continue
                avg = p["sum"] / total
                high = p["high_count"]
                pct = 100 * high / total
                low_leq_1 = p.get("low_count_leq_1", 0)
                low_pct = 100 * low_leq_1 / total if total else 0
                parts = []
                if show_avg:
                    parts.append(f"avg={avg:.2f}")
                if show_high_pct:
                    parts.append(f"high%={pct:.1f}")
                if show_low_count_pct:
                    parts.append(f"low≤1%={low_pct:.1f}")
                lines.append(f"{pos}={','.join(parts)}")
            if lines:
                print(f"  {model}: " + " | ".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation scores by blank position (opening/middle/closing)"
    )
    parser.add_argument("--threshold", "-t", type=float, default=4.0, help="High score threshold (default: 4)")
    parser.add_argument("--method", "-m", choices=["teler", "reasoning", "both"], default="both")
    parser.add_argument(
        "--common-models-only",
        action="store_true",
        help="Restrict to stems that appear in both data/eval_files/teler and data/eval_files/reasoning (for any -m)",
    )
    parser.add_argument("--show-avg", action="store_true", default=True, help="Show average score (default: on)")
    parser.add_argument("--no-show-avg", action="store_false", dest="show_avg", help="Do not show average")
    parser.add_argument("--show-high-pct", action="store_true", default=True, help="Show %% with Average>=threshold (default: on)")
    parser.add_argument("--no-show-high-pct", action="store_false", dest="show_high_pct", help="Do not show high %%")
    parser.add_argument("--show-low-count-pct", action="store_true", default=True, help="Show %% where at most 1 dimension has low score (default: on)")
    parser.add_argument("--no-show-low-count-pct", action="store_false", dest="show_low_count_pct", help="Do not show low-count %%")
    parser.add_argument("--low-threshold", type=float, default=2.0, help="Dimension score <= this is 'low' for low-count metric (default: 2)")
    parser.add_argument("--per-model", action="store_true", help="Show per-model breakdown")
    parser.add_argument(
        "--no-dataset-report",
        action="store_true",
        help="Do not print by-dataset sections",
    )
    parser.add_argument(
        "--dataset-template-detail",
        action="store_true",
        help="Print dataset × template grid (verbose)",
    )
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--csv", help="Output CSV path (same model scope as this run)")
    parser.add_argument(
        "--csv-common",
        metavar="PATH",
        help="CSV: teler+reasoning using only models present in both dirs",
    )
    parser.add_argument(
        "--csv-all",
        metavar="PATH",
        help="CSV: teler+reasoning using all *.xlsx in each dir (no model intersection)",
    )
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit rows per file (for testing)")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"

    models_filter: set[str] | None = None
    if args.common_models_only:
        common = commonModelStems(eval_root)
        if not common:
            print("Warning: --common-models-only but teler/reasoning dirs missing or empty intersection", file=sys.stderr)
        else:
            models_filter = common
            print(f"Restricting to {len(models_filter)} models common in both: {sorted(models_filter)}", file=sys.stderr)

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
            models_filter=models_filter,
            limit=args.limit,
            low_threshold=args.low_threshold,
        )
        print_report(
            method,
            results[method],
            args.threshold,
            show_avg=args.show_avg,
            show_high_pct=args.show_high_pct,
            show_low_count_pct=args.show_low_count_pct,
            show_per_model=args.per_model,
            show_dataset=not args.no_dataset_report,
            show_dataset_template_detail=args.dataset_template_detail,
        )

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {}
        for method, data in results.items():
            by_tpl = data["by_template"]
            by_pos = data["by_position"]
            by_ds = data.get("by_dataset", {})
            by_dst = data.get("by_dataset_template", {})

            out_data[method] = {
                "by_template": {
                    tpl: {pos: to_out_json(p) for pos, p in by_tpl[tpl].items()}
                    for tpl in by_tpl
                },
                "aggregate": {pos: to_out_json(p) for pos, p in by_pos.items()},
                "by_dataset": {
                    ds: {pos: to_out_json(p) for pos, p in by_ds[ds].items()}
                    for ds in by_ds
                },
                "by_dataset_template": {
                    ds: {
                        tpl: {pos: to_out_json(p) for pos, p in by_dst[ds][tpl].items()}
                        for tpl in by_dst[ds]
                    }
                    for ds in by_dst
                },
            }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nWrote {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
        write_position_csv(csv_path, results)

    common_stems = commonModelStems(eval_root)
    if args.csv_common:
        if not common_stems:
            print("Warning: --csv-common skipped (no common models / missing dirs)", file=sys.stderr)
        else:
            p = Path(args.csv_common)
            if not p.is_absolute():
                p = base_dir / p
            r = collect_results(
                eval_root,
                ["teler", "reasoning"],
                common_stems,
                threshold=args.threshold,
                limit=args.limit,
                low_threshold=args.low_threshold,
            )
            write_position_csv(p, r)

    if args.csv_all:
        p = Path(args.csv_all)
        if not p.is_absolute():
            p = base_dir / p
        r = collect_results(
            eval_root,
            ["teler", "reasoning"],
            None,
            threshold=args.threshold,
            limit=args.limit,
            low_threshold=args.low_threshold,
        )
        write_position_csv(p, r)

    return 0


if __name__ == "__main__":
    sys.exit(main())

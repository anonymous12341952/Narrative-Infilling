#!/usr/bin/env python3
"""
Aggregate evaluation scores by template (teler: template_name; reasoning: template_id) and LLM.

Teler: groups by template_name (fitb_l0, fitb_l1, etc.)
Reasoning: groups by template_id -> 0=abductive_reasoning, 1=causal_reasoning,
           2=counterfactual_thinking, 3=chain_of_thought

Use -m both to write one JSON and one TXT containing both methods (keys: teler, reasoning).

Metrics: bert_f1, rouge1, meteor, chrf, sem_f1,
         auto_average (mean of available AUTO_METRICS),
         Fluency, Context_Faithfulness, Bidirectional_Coherence, Narrative_Consistency, Informativeness, Average, QAvg.

--by-dataset: also stratify by the dataset column (dataset -> template -> LLM -> metrics).
  If dataset is missing in a file, rows are grouped under (no_dataset_col).

--output-txt: write the same structure as a UTF-8 text report (in addition to optional JSON).

--common-models-only: only include LLMs whose xlsx stem exists in both data/eval_files/teler and
  data/eval_files/reasoning (for fair comparison). Default output filenames use a _common suffix.

With --by-dataset, the text report starts with an EXECUTIVE SUMMARY: (A) per-dataset tables
(template×LLM and macro across LLMs); (B) same metrics pooled over datasets. Use --txt-no-detail
to omit the long per-dataset detail from the .txt only (JSON is always full).

# Summary + full detail (new summary at top)
python aggregate_eval_by_template.py -m both --by-dataset --common-models-only \
  -o data/eval_files/my_combined.json -T data/eval_files/my_combined.txt

# Summary only (small .txt)
python aggregate_eval_by_template.py -m both --by-dataset --common-models-only \
  -o data/eval_files/my_combined.json -T data/eval_files/my_combined_summary.txt --txt-no-detail

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# template_id -> label for reasoning method
REASONING_TEMPLATE_LABELS = {
    0: "abductive_reasoning",
    1: "causal_reasoning",
    2: "counterfactual_thinking",
    3: "chain_of_thought",
}

AUTO_METRICS = ["bert_f1", "rouge1", "meteor", "chrf", "sem_f1"]
LLM_METRICS = [
    "Fluency",
    "Context_Faithfulness",
    "Bidirectional_Coherence",
    "Narrative_Consistency",
    "Informativeness",
    "Average",
    "QAvg",
]
ALL_METRICS = AUTO_METRICS + LLM_METRICS


def getModelsInDir(eval_dir: Path) -> set[str]:
    return {p.stem for p in eval_dir.glob("*.xlsx")}


def commonModelStems(eval_root: Path) -> set[str]:
    teler_dir = eval_root / "teler"
    reasoning_dir = eval_root / "reasoning"
    if not teler_dir.exists() or not reasoning_dir.exists():
        return set()
    return getModelsInDir(teler_dir) & getModelsInDir(reasoning_dir)


def normalizeDatasetColumn(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with normalized string column dataset_key."""
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset_key"] = "(no_dataset_col)"
        return out

    def norm(v) -> str:
        if pd.isna(v):
            return "(unknown)"
        s = str(v).strip()
        return s if s else "(unknown)"

    out["dataset_key"] = out["dataset"].map(norm)
    return out


def templateKeyForMethod(val, method: str) -> str:
    if method == "reasoning":
        try:
            tid = int(val) if pd.notna(val) else val
            return REASONING_TEMPLATE_LABELS.get(tid, str(val))
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def rowFromGroup(grp: pd.DataFrame, avail: list[str]) -> dict[str, float]:
    means = grp[avail].mean(numeric_only=True)
    row = {c: round(float(means[c]), 4) for c in avail if c in means}
    auto_avail = [c for c in AUTO_METRICS if c in row]
    if auto_avail:
        row["auto_average"] = round(sum(row[c] for c in auto_avail) / len(auto_avail), 3)
    return row


def safeMean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def appendOneDatasetSummary(lines: list[str], dataset_name: str, tpl_map: object) -> None:
    """One dataset: template×LLM values + macro across LLMs per template."""
    if not isinstance(tpl_map, dict):
        return

    w_tpl, w_llm = 22, 28
    lines.append("\n" + "-" * 72 + "\n")
    lines.append(f"DATASET: {dataset_name}\n")
    lines.append("-" * 72 + "\n")

    lines.append("\n(a) template × LLM — auto_average & qualitative Average\n\n")
    hdr = (
        f"{'template':<{w_tpl}} {'llm':<{w_llm}} {'auto_avg':>10} {'Avg':>10}\n"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr.rstrip()) + "\n")

    tpl_auto_macro: dict[str, list[float]] = {}
    tpl_qual_macro: dict[str, list[float]] = {}

    for tpl in sorted(tpl_map.keys()):
        llm_map = tpl_map[tpl]
        if not isinstance(llm_map, dict):
            continue
        for llm in sorted(llm_map.keys()):
            metrics = llm_map[llm]
            if not isinstance(metrics, dict):
                continue
            auto_v: float | None = None
            qual_v: float | None = None
            if "auto_average" in metrics and metrics["auto_average"] is not None:
                try:
                    auto_v = float(metrics["auto_average"])
                except (TypeError, ValueError):
                    pass
            if "Average" in metrics and metrics["Average"] is not None:
                try:
                    qual_v = float(metrics["Average"])
                except (TypeError, ValueError):
                    pass
            auto_s = f"{round(auto_v, 4)}" if auto_v is not None else "—"
            qual_s = f"{round(qual_v, 4)}" if qual_v is not None else "—"
            lines.append(
                f"{tpl:<{w_tpl}} {llm:<{w_llm}} {auto_s:>10} {qual_s:>10}\n"
            )
            if auto_v is not None:
                tpl_auto_macro.setdefault(str(tpl), []).append(auto_v)
            if qual_v is not None:
                tpl_qual_macro.setdefault(str(tpl), []).append(qual_v)

    lines.append("\n(b) Per template — macro mean across LLMs (this dataset only)\n\n")
    hdr2 = (
        f"{'template':<{w_tpl}} {'n_llm_auto':>11} {'macro_auto':>11} "
        f"{'n_llm_Avg':>10} {'macro_Avg':>10}\n"
    )
    lines.append(hdr2)
    lines.append("-" * len(hdr2.rstrip()) + "\n")
    for tpl in sorted(set(tpl_auto_macro.keys()) | set(tpl_qual_macro.keys())):
        las = tpl_auto_macro.get(tpl, [])
        lqs = tpl_qual_macro.get(tpl, [])
        ma_m = safeMean(las)
        mq_m = safeMean(lqs)
        ma_ms = f"{round(ma_m, 4)}" if ma_m is not None else "—"
        mq_ms = f"{round(mq_m, 4)}" if mq_m is not None else "—"
        lines.append(
            f"{tpl:<{w_tpl}} {len(las):>11} {ma_ms:>11} {len(lqs):>10} {mq_ms:>10}\n"
        )


def appendSummaryTables(lines: list[str], nested: dict, *, block_title: str) -> None:
    """nested: dataset -> template -> llm -> metrics."""
    lines.append("\n" + "=" * 72 + "\n")
    lines.append(f"{block_title}\n")
    lines.append("=" * 72 + "\n")

    lines.append(
        "\n### A) PER DATASET (compact tables)\n"
    )
    for ds in sorted(nested.keys()):
        appendOneDatasetSummary(lines, ds, nested[ds])

    lines.append("\n" + "=" * 72 + "\n")
    lines.append("### B) POOLED ACROSS ALL DATASETS\n")
    lines.append("=" * 72 + "\n")

    tpl_llm_auto: dict[tuple[str, str], list[float]] = {}
    tpl_llm_qual: dict[tuple[str, str], list[float]] = {}
    for dataset_name, tpl_map in nested.items():
        if not isinstance(tpl_map, dict):
            continue
        for tpl, llm_map in tpl_map.items():
            if not isinstance(llm_map, dict):
                continue
            for llm, metrics in llm_map.items():
                if not isinstance(metrics, dict):
                    continue
                if "auto_average" in metrics and metrics["auto_average"] is not None:
                    try:
                        tpl_llm_auto.setdefault((str(tpl), str(llm)), []).append(
                            float(metrics["auto_average"])
                        )
                    except (TypeError, ValueError):
                        pass
                if "Average" in metrics and metrics["Average"] is not None:
                    try:
                        tpl_llm_qual.setdefault((str(tpl), str(llm)), []).append(
                            float(metrics["Average"])
                        )
                    except (TypeError, ValueError):
                        pass

    keys = sorted(set(tpl_llm_auto.keys()) | set(tpl_llm_qual.keys()))
    w_tpl, w_llm = 22, 28

    lines.append(
        "\n(B1) Per template × LLM — mean of auto_average and qualitative Average over datasets\n\n"
    )
    hdr = (
        f"{'template':<{w_tpl}} {'llm':<{w_llm}} {'n_ds_auto':>9} {'mean_auto':>10} "
        f"{'n_ds_Avg':>8} {'mean_Avg':>10}\n"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr.rstrip()) + "\n")

    tpl_auto_means: dict[str, list[float]] = {}
    tpl_qual_means: dict[str, list[float]] = {}

    for tpl, llm in keys:
        autos = tpl_llm_auto.get((tpl, llm), [])
        avgs = tpl_llm_qual.get((tpl, llm), [])
        ma = safeMean(autos)
        mq = safeMean(avgs)
        ma_s = f"{round(ma, 4)}" if ma is not None else "—"
        mq_s = f"{round(mq, 4)}" if mq is not None else "—"
        lines.append(
            f"{tpl:<{w_tpl}} {llm:<{w_llm}} {len(autos):>9} {ma_s:>10} {len(avgs):>8} {mq_s:>10}\n"
        )
        if ma is not None:
            tpl_auto_means.setdefault(tpl, []).append(ma)
        if mq is not None:
            tpl_qual_means.setdefault(tpl, []).append(mq)

    lines.append(
        "\n(B2) Per template — macro mean across LLMs (mean of the per-LLM values from (B1))\n\n"
    )
    hdr2 = (
        f"{'template':<{w_tpl}} {'n_llm_auto':>11} {'macro_auto':>11} "
        f"{'n_llm_Avg':>10} {'macro_Avg':>10}\n"
    )
    lines.append(hdr2)
    lines.append("-" * len(hdr2.rstrip()) + "\n")
    for tpl in sorted(set(tpl_auto_means.keys()) | set(tpl_qual_means.keys())):
        las = tpl_auto_means.get(tpl, [])
        lqs = tpl_qual_means.get(tpl, [])
        ma_m = safeMean(las)
        mq_m = safeMean(lqs)
        ma_ms = f"{round(ma_m, 4)}" if ma_m is not None else "—"
        mq_ms = f"{round(mq_m, 4)}" if mq_m is not None else "—"
        lines.append(
            f"{tpl:<{w_tpl}} {len(las):>11} {ma_ms:>11} {len(lqs):>10} {mq_ms:>10}\n"
        )


def appendExecutiveSummary(
    lines: list[str],
    data: dict,
    *,
    method: str,
    by_dataset: bool,
    txt_no_detail: bool,
) -> None:
    if not by_dataset:
        lines.append(
            "\n(No executive summary tables: need --by-dataset for cross-dataset aggregation.)\n"
        )
        if not txt_no_detail:
            lines.append("\n" + "*" * 72 + "\nDETAIL (full breakdown)\n" + "*" * 72 + "\n")
        return

    lines.append("\n" + "*" * 72 + "\n")
    lines.append("EXECUTIVE SUMMARY\n")
    lines.append("*" * 72 + "\n")

    if method == "both":
        for key in ("teler", "reasoning"):
            sub = data.get(key, {})
            if isinstance(sub, dict) and sub:
                appendSummaryTables(
                    lines,
                    sub,
                    block_title=f"METHOD: {key.upper()}",
                )
    else:
        if isinstance(data, dict) and data:
            appendSummaryTables(lines, data, block_title=f"METHOD: {method.upper()}")

    if not txt_no_detail:
        lines.append("\n" + "*" * 72 + "\n")
        lines.append("DETAIL (full breakdown)\n")
        lines.append("*" * 72 + "\n")


def appendTxtSection(
    lines: list[str],
    data: dict,
    *,
    by_dataset: bool,
    method_title: str,
) -> None:
    lines.append("\n" + "#" * 72 + "\n")
    lines.append(f"METHOD: {method_title.upper()}\n")
    lines.append("#" * 72 + "\n")

    if by_dataset:
        for ds in sorted(data.keys()):
            lines.append("\n" + "=" * 72 + "\n")
            lines.append(f"DATASET: {ds}\n")
            lines.append("=" * 72 + "\n")
            for tpl in sorted(data[ds].keys()):
                lines.append(f"\n--- Template: {tpl} ---\n")
                for llm in sorted(data[ds][tpl].keys()):
                    lines.append(f"  [{llm}]\n")
                    row = data[ds][tpl][llm]
                    for k in sorted(row.keys(), key=lambda x: (x != "auto_average", x)):
                        lines.append(f"    {k}: {row[k]}\n")
    else:
        for tpl in sorted(data.keys()):
            lines.append("\n" + "=" * 72 + "\n")
            lines.append(f"TEMPLATE: {tpl}\n")
            lines.append("=" * 72 + "\n")
            for llm in sorted(data[tpl].keys()):
                lines.append(f"\n  [{llm}]\n")
                row = data[tpl][llm]
                for k in sorted(row.keys(), key=lambda x: (x != "auto_average", x)):
                    lines.append(f"    {k}: {row[k]}\n")


def writeTxtReport(
    path: Path,
    data: dict,
    *,
    by_dataset: bool,
    method: str,
    common_models_only: bool,
    txt_no_detail: bool,
) -> None:
    lines: list[str] = []
    lines.append("aggregate_eval_by_template.py\n")
    lines.append(
        f"method={method}  by_dataset={by_dataset}  common_models_only={common_models_only}"
        f"  txt_no_detail={txt_no_detail}\n\n"
    )

    appendExecutiveSummary(
        lines, data, method=method, by_dataset=by_dataset, txt_no_detail=txt_no_detail
    )

    if not txt_no_detail:
        if method == "both":
            for key in ("teler", "reasoning"):
                sub = data.get(key, {})
                if sub:
                    appendTxtSection(lines, sub, by_dataset=by_dataset, method_title=key)
        else:
            appendTxtSection(lines, data, by_dataset=by_dataset, method_title=method)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def aggregate_one_method(
    eval_dir: Path,
    method: str,
    *,
    by_dataset: bool,
    common_stems: set[str] | None,
    llm_name: str | None,
) -> dict:
    """Return nested dict for one method (same shape as before: by_dataset or by template)."""
    by_group: dict = {}

    xlsx_paths = sorted(eval_dir.glob("*.xlsx"))
    if llm_name:
        xlsx_paths = [eval_dir / f"{llm_name}.xlsx"]

    if common_stems is not None:
        xlsx_paths = [p for p in xlsx_paths if p.stem in common_stems]

    for xlsx in xlsx_paths:
        if not xlsx.exists():
            print(f"Warning: skip missing file: {xlsx}", file=sys.stderr)
            continue
        llm_stem = xlsx.stem
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"Warning: skip {xlsx.name}: {e}", file=sys.stderr)
            continue

        if method == "teler":
            if "template_name" not in df.columns:
                print(f"Warning: skip {llm_stem} (no template_name)", file=sys.stderr)
                continue
            group_col = "template_name"
        else:
            if "template_id" not in df.columns:
                print(f"Warning: skip {llm_stem} (no template_id)", file=sys.stderr)
                continue
            group_col = "template_id"

        avail = [c for c in ALL_METRICS if c in df.columns]
        if not avail:
            continue

        if by_dataset:
            df_w = normalizeDatasetColumn(df)
            if "dataset" not in df.columns:
                print(f"Note: {llm_stem} has no dataset column; using (no_dataset_col).", file=sys.stderr)
            for (ds_key, val), grp in df_w.groupby(["dataset_key", group_col], sort=False):
                tpl_key = templateKeyForMethod(val, method)
                row = rowFromGroup(grp, avail)
                by_group.setdefault(ds_key, {}).setdefault(tpl_key, {})[llm_stem] = row
        else:
            for val, grp in df.groupby(group_col, sort=False):
                tpl_key = templateKeyForMethod(val, method)
                row = rowFromGroup(grp, avail)
                by_group.setdefault(tpl_key, {})[llm_stem] = row

    return by_group


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation scores by template and LLM; optional stratification by dataset."
    )
    parser.add_argument(
        "--method",
        "-m",
        default="teler",
        choices=["teler", "reasoning", "both"],
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default=None,
        help="Evaluation files dir (default: data/eval_files/<method>). Ignored when -m both.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON path (defaults under data/eval_files/; see --method both for combined path)",
    )
    parser.add_argument(
        "--output-txt",
        "-T",
        default=None,
        metavar="PATH",
        help="Also write a UTF-8 text report. If omitted: defaults beside JSON when --by-dataset, --common-models-only, or -m both.",
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Stratify by dataset column then template (structure: dataset -> template -> llm -> metrics)",
    )
    parser.add_argument(
        "--llm-name",
        "-l",
        default=None,
        help="Only aggregate this single LLM (xlsx stem). Prints JSON to stdout and does not write files unless -T/-o set.",
    )
    parser.add_argument(
        "--common-models-only",
        action="store_true",
        help="Only xlsx stems present in both data/eval_files/teler and data/eval_files/reasoning",
    )
    parser.add_argument(
        "--txt-no-detail",
        action="store_true",
        help="In the .txt report, write only the executive summary (omit long per-dataset detail)",
    )
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else PROJECT_DIR
    eval_root = base_dir / "data" / "eval_files"

    if args.method == "both":
        if args.input_dir:
            print("Note: --input-dir ignored when -m both (using data/eval_files/teler and reasoning).", file=sys.stderr)
        teler_dir = eval_root / "teler"
        reasoning_dir = eval_root / "reasoning"
        if not teler_dir.exists():
            print(f"Error: {teler_dir} not found", file=sys.stderr)
            return 1
        if not reasoning_dir.exists():
            print(f"Error: {reasoning_dir} not found", file=sys.stderr)
            return 1
    else:
        eval_dir = Path(args.input_dir) if args.input_dir else eval_root / args.method
        if not eval_dir.exists():
            print(f"Error: {eval_dir} not found", file=sys.stderr)
            return 1

    common_stems: set[str] | None = None
    if args.common_models_only:
        common_stems = commonModelStems(eval_root)
        if not common_stems:
            teler_d = eval_root / "teler"
            reasoning_d = eval_root / "reasoning"
            if not teler_d.exists() or not reasoning_d.exists():
                print(
                    "Error: --common-models-only requires data/eval_files/teler and data/eval_files/reasoning",
                    file=sys.stderr,
                )
            else:
                print("Error: no overlapping xlsx stems between teler and reasoning.", file=sys.stderr)
            return 1
        print(
            f"Common models only ({len(common_stems)}): {', '.join(sorted(common_stems))}",
            file=sys.stderr,
        )
        if args.llm_name and args.llm_name not in common_stems:
            print(
                f"Warning: --llm-name {args.llm_name} is not in the common set; no data.",
                file=sys.stderr,
            )

    suffix = "_common" if args.common_models_only else ""

    if args.method == "both":
        by_group = {
            "teler": aggregate_one_method(
                teler_dir,
                "teler",
                by_dataset=args.by_dataset,
                common_stems=common_stems,
                llm_name=args.llm_name,
            ),
            "reasoning": aggregate_one_method(
                reasoning_dir,
                "reasoning",
                by_dataset=args.by_dataset,
                common_stems=common_stems,
                llm_name=args.llm_name,
            ),
        }
        default_json_name = (
            f"evaluation_by_dataset_template_both{suffix}.json"
            if args.by_dataset
            else f"evaluation_by_template_both{suffix}.json"
        )
        default_json = eval_root / default_json_name
    else:
        by_group = aggregate_one_method(
            eval_dir,
            args.method,
            by_dataset=args.by_dataset,
            common_stems=common_stems,
            llm_name=args.llm_name,
        )
        default_json = (
            eval_root / args.method / f"evaluation_by_dataset_template{suffix}.json"
            if args.by_dataset
            else eval_root / args.method / f"evaluation_by_template{suffix}.json"
        )

    if args.llm_name:
        print(json.dumps(by_group, indent=2))
        if args.output_txt:
            txt_path = Path(args.output_txt)
            if not txt_path.is_absolute():
                txt_path = base_dir / txt_path
            writeTxtReport(
                txt_path,
                by_group,
                by_dataset=args.by_dataset,
                method=args.method,
                common_models_only=args.common_models_only,
                txt_no_detail=args.txt_no_detail,
            )
            print(f"Wrote {txt_path}", file=sys.stderr)
        return 0

    out_path = Path(args.output) if args.output else base_dir / default_json
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(by_group, f, indent=2)
    print(f"Wrote {out_path}")

    txt_path: Path | None = None
    if args.output_txt is not None:
        txt_path = Path(args.output_txt)
    elif args.by_dataset or args.common_models_only or args.method == "both":
        txt_path = out_path.with_suffix(".txt")
    if txt_path is not None:
        if not txt_path.is_absolute():
            txt_path = base_dir / txt_path
        writeTxtReport(
            txt_path,
            by_group,
            by_dataset=args.by_dataset,
            method=args.method,
            common_models_only=args.common_models_only,
            txt_no_detail=args.txt_no_detail,
        )
        print(f"Wrote {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

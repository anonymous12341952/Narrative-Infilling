import argparse
import json, os
import pandas as pd
import numpy as np
from pathlib import Path

from metrics_engine import Evaluator

# Path setup: Narrative-Infilling root (parent of metrics/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
METRICS_SUMMARY_DIR = PROJECT_DIR / "data" / "eval_files" / "metrics_summary"
METRICS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDED_METRICS = {"rougeL", "sem_precision", "sem_recall"}


# -------------------------------------------------
# JSON leaderboard update
# -------------------------------------------------

def update_json(method, llm_name, avg_scores, summary_dir=METRICS_SUMMARY_DIR):
    path = summary_dir / "evaluation_summary.json"

    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {}

    data.setdefault(method, {})
    data[method][llm_name] = avg_scores

    path.write_text(json.dumps(data, indent=2))
    print(f"Updated evaluation summary: {path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def evaluate_one_file(
    *,
    method: str,
    llm_name: str,
    file_path: Path,
    batch_size: int,
    single_sample_index: int | None,
    limit: int | None,
    save_excel: bool,
    output_dir: Path,
):
    df = pd.read_excel(file_path)

    if limit is not None and single_sample_index is None:
        df = df.iloc[:limit]

    preds = df["extracted_answer"].fillna("").astype(str).tolist()
    refs = df["gold_answer"].fillna("").astype(str).tolist()

    # single sample mode
    if single_sample_index is not None:
        idx = single_sample_index
        preds = [preds[idx]]
        refs = [refs[idx]]
        print(f"Running SINGLE sample evaluation for index {idx}")

    evaluator = Evaluator(batch_size=batch_size)
    raw_scores = evaluator.evaluate(preds, refs)
    scores = {k: v for k, v in raw_scores.items() if k not in EXCLUDED_METRICS}

    # add columns back to dataframe (only when full run)
    if single_sample_index is None and save_excel:
        rounded_scores = {}
        for metric, values in scores.items():
            # Store per-row metric values at 4 decimal places in Excel output.
            rounded = [
                round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                for v in values
            ]
            rounded_scores[metric] = rounded
            df[metric] = rounded

        # AtAvg: per-row average across all kept automatic metrics.
        metric_names = list(rounded_scores.keys())
        row_count = len(df)
        at_avg = []
        for row_idx in range(row_count):
            row_vals = []
            for metric_name in metric_names:
                v = rounded_scores[metric_name][row_idx]
                if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                    row_vals.append(float(v))
            at_avg.append(round(float(np.mean(row_vals)), 4) if row_vals else np.nan)
        df["AtAvg"] = at_avg

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_path.name
        df.to_excel(output_path, index=False)
        print(f"Updated Excel saved: {output_path}")

    # compute averages
    avg_scores = {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    # AtAvg summary: mean of per-row average across kept metrics.
    if scores:
        metric_names = list(scores.keys())
        row_count = len(next(iter(scores.values())))
        at_avg_vals = []
        for row_idx in range(row_count):
            row_vals = []
            for metric_name in metric_names:
                v = scores[metric_name][row_idx]
                if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                    row_vals.append(float(v))
            if row_vals:
                at_avg_vals.append(float(np.mean(row_vals)))
        if at_avg_vals:
            avg_scores["AtAvg"] = round(float(np.mean(at_avg_vals)), 4)

    print("\nAVERAGE SCORES:")
    for k, v in avg_scores.items():
        print(k, ":", v)

    update_json(method, llm_name, avg_scores, METRICS_SUMMARY_DIR)


def main(args):

    # Step 1 input defaults to generated response files.
    response_dir = Path(args.input_dir) if args.input_dir else (PROJECT_DIR / "data" / "responses" / args.method)
    # Step 1 output defaults to eval_files (auto metrics written to same filename).
    eval_dir = Path(args.eval_dir) if args.eval_dir else (PROJECT_DIR / "data" / "eval_files" / args.method)
    response_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        xlsx_paths = sorted(response_dir.glob("*.xlsx"))
        if not xlsx_paths:
            print(f"No .xlsx files found in {response_dir}", file=os.sys.stderr)
            return

        total = len(xlsx_paths)
        for i, file_path in enumerate(xlsx_paths, start=1):
            llm_name = file_path.stem
            print("\n" + "=" * 60)
            print(f"[{i}/{total}] Evaluating {llm_name} (method={args.method})")
            print("=" * 60)
            evaluate_one_file(
                method=args.method,
                llm_name=llm_name,
                file_path=file_path,
                batch_size=args.batch_size,
                single_sample_index=args.single_sample_index,
                limit=args.limit,
                save_excel=not args.no_save_excel,
                output_dir=eval_dir,
            )

            print(f"\nCompleted {i}/{total}: {llm_name}")
            if not args.no_save_excel and args.single_sample_index is None:
                saved = len(list(eval_dir.glob("*.xlsx"))) if eval_dir.exists() else 0
                print(f"Saved results in {eval_dir}: {saved}/{total}")
        return

    if not args.input_excel:
        raise ValueError("--input_excel is required unless --all is set")

    file_path = response_dir / args.input_excel
    if not file_path.exists():
        raise FileNotFoundError(f"Excel not found: {file_path}")

    llm_name = args.llm_name or Path(args.input_excel).stem
    print("Loading Excel:", args.input_excel)
    evaluate_one_file(
        method=args.method,
        llm_name=llm_name,
        file_path=file_path,
        batch_size=args.batch_size,
        single_sample_index=args.single_sample_index,
        limit=args.limit,
        save_excel=not args.no_save_excel,
        output_dir=eval_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--all", action="store_true", help="Evaluate all .xlsx files in input dir")
    parser.add_argument("--input_excel", default=None, help="Single Excel filename under input dir")
    parser.add_argument("--llm_name", default=None, help="Override LLM name (default: stem of input_excel)")
    parser.add_argument("--input_dir", default=None, help="Input dir for response files (default: data/responses/<method>)")
    parser.add_argument("--eval_dir", default=None, help="Output dir for auto-eval files (default: data/eval_files/<method>)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--single_sample_index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (ignored in single-sample mode)")
    parser.add_argument("--no_save_excel", action="store_true", help="Do not write updated Excel to eval dir")

    args = parser.parse_args()
    main(args)

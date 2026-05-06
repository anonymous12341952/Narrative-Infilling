#!/usr/bin/env python3
"""
Batch narrative infilling evaluation: load one LLM's Excel, call evaluator API with
rate limiting, write Excel with per-dimension {score, reason} and a JSON stats file.

Supports two methods: teler and reasoning. Each method has a list of LLMs (Excel files).
Uses asyncio + batch processing with configurable requests-per-minute rate limit.

Environment / credentials:
- This script requires evaluator API credentials in environment variables.
- It auto-loads env files from:
  1) metrics/run_evaluation.env
  2) metrics/.env
  3) ~/.env
- For provider=azure (default): set AZURE_OPENAI_API_KEY (and typically AZURE_OPENAI_ENDPOINT).
- For provider=openai: set OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Script dir first so we can load env before any API imports.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent


def _load_env_file(path: Path) -> None:
    """Load KEY=value lines into os.environ. Skips comments and empty lines."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if m:
                key, value = m.group(1), m.group(2).strip()
                if value.startswith(('"', "'")) and value[0] == value[-1]:
                    value = value[1:-1].replace("\\n", "\n")
                os.environ[key] = value


def _load_dotenv() -> None:
    """Load env from run_evaluation.env or .env so api_main has vars when imported."""
    for name in ("run_evaluation.env", ".env"):
        _load_env_file(SCRIPT_DIR / name)
    _load_env_file(Path.home() / ".env")


_load_dotenv()


def ensure_provider_credentials(provider_name: str) -> None:
    """Fail fast with clear guidance when API credentials are missing."""
    provider = str(provider_name or "").strip().lower()
    if provider == "azure":
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            raise RuntimeError(
                "Missing AZURE_OPENAI_API_KEY for provider=azure.\n"
                "Add it to one of: metrics/run_evaluation.env, metrics/.env, or ~/.env.\n"
                "You will usually also need AZURE_OPENAI_ENDPOINT."
            )
    elif provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Missing OPENAI_API_KEY for provider=openai.\n"
                "Add it to one of: metrics/run_evaluation.env, metrics/.env, or ~/.env."
            )

# Ensure API and project root are on path
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "API"))

import pandas as pd
import yaml
from tqdm import tqdm

from api_main import run
from pydantic import BaseModel, Field, conint

# ---------- Pydantic models for evaluator response ----------
# Prompts that expect scores only (no reason); schema must match
PROMPTS_SCORES_ONLY = {"narrative_infilling_evaluation_v4"}


class ScoreReason(BaseModel):
    score: conint(ge=1, le=5)
    reason: str


class NarrativeInfillingEvaluation(BaseModel):
    Fluency: ScoreReason
    Context_Faithfulness: ScoreReason
    Bidirectional_Coherence: ScoreReason
    Narrative_Consistency: ScoreReason
    Informativeness: ScoreReason


class NarrativeInfillingScoresOnly(BaseModel):
    """Scores only, no reasons (for v4 and similar prompts)."""
    Fluency: conint(ge=1, le=5)
    Context_Faithfulness: conint(ge=1, le=5)
    Bidirectional_Coherence: conint(ge=1, le=5)
    Narrative_Consistency: conint(ge=1, le=5)
    Informativeness: conint(ge=1, le=5)


def get_response_model(prompt_name: str) -> type:
    """Return the Pydantic model for structured output based on prompt."""
    if prompt_name in PROMPTS_SCORES_ONLY:
        return NarrativeInfillingScoresOnly
    return NarrativeInfillingEvaluation


DIMENSIONS = [
    "Fluency",
    "Context_Faithfulness",
    "Bidirectional_Coherence",
    "Narrative_Consistency",
    "Informativeness",
]

# ---------- Config: method -> input dir (relative to repo root). ----------
METHOD_INPUT_DIRS = {
    "teler": "data/eval_files/teler",
    "reasoning": "data/eval_files/reasoning",
}


def load_prompts(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_api_prompt(prompts: dict, prompt_name: str, **format_kwargs) -> dict:
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in YAML")
    p = prompts[prompt_name]
    system_parts = [p["system"]]
    if "evaluation_procedure" in p:
        system_parts.append(p["evaluation_procedure"])
    # if "automatic_penalties" in p:
    #     system_parts.append(p["automatic_penalties"])
    if "rubric" in p:
        system_parts.append(p["rubric"])
    user_tpl = p.get("user") or p.get("human", "")
    user = user_tpl.format(**format_kwargs) if format_kwargs else user_tpl
    return {"system": "\n\n".join(system_parts), "user": user}


def estimate_tokens_for_request(
    prompts: dict,
    prompt_name: str,
    story_with_blank: str,
    candidate_infill: str,
    max_output_tokens: int = 1024,
    chars_per_token: float = 4.0,
) -> float:
    """Estimate total tokens (input + output) for this evaluation request. Used for TPM limiting."""
    try:
        d = build_api_prompt(prompts, prompt_name, story_with_blank=story_with_blank, candidate_infill=candidate_infill)
        input_chars = len(d["system"]) + len(d["user"])
        input_tokens = input_chars / chars_per_token
        return input_tokens + max_output_tokens
    except Exception:
        return 8000.0  # fallback if prompt build fails (conservative)


def eval_to_row_dict(e: Optional[NarrativeInfillingEvaluation]) -> dict[str, dict]:
    """Turn one evaluation into dict of dimension -> {score, reason} for Excel."""
    row = {}
    if e is None:
        for dim in DIMENSIONS:
            row[dim] = {"score": None, "reason": ""}
        return row
    for dim in DIMENSIONS:
        val = getattr(e, dim, None)
        if val is None:
            row[dim] = {"score": None, "reason": ""}
        elif isinstance(val, int):
            row[dim] = {"score": val, "reason": ""}
        else:
            row[dim] = {"score": val.score, "reason": val.reason or ""}
    return row


# ---------- Rate limiter: RPM + optional TPM (tokens per minute) ----------
# Estimated tokens per request when TPM is used (input + output; conservative for narrative eval)
ESTIMATED_TOKENS_PER_REQUEST = 4000


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: float,
        concurrency: int = 1,
        tokens_per_minute: float = 0,
    ):
        self.rpm = max(1.0, requests_per_minute)
        self.tpm = max(0.0, tokens_per_minute)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.last_times: list[float] = []
        self.token_times: list[tuple[float, float]] = []  # (timestamp, tokens)
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: float = ESTIMATED_TOKENS_PER_REQUEST):
        await self.semaphore.acquire()
        loop = asyncio.get_event_loop()
        now = loop.time()
        async with self._lock:
            # Enforce RPM
            self.last_times = [t for t in self.last_times if now - t < 60.0]
            if len(self.last_times) >= self.rpm:
                wait = 60.0 - (now - self.last_times[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                now = loop.time()
                self.last_times = [t for t in self.last_times if now - t < 60.0]
            self.last_times.append(loop.time())

            # Enforce TPM (tokens per minute) if set
            if self.tpm > 0:
                self.token_times = [(t, n) for t, n in self.token_times if now - t < 60.0]
                tokens_in_window = sum(n for _, n in self.token_times)
                while tokens_in_window + estimated_tokens > self.tpm:
                    if not self.token_times:
                        break
                    wait = 60.0 - (now - self.token_times[0][0])
                    if wait > 0:
                        await asyncio.sleep(wait)
                    now = loop.time()
                    self.token_times = [(t, n) for t, n in self.token_times if now - t < 60.0]
                    tokens_in_window = sum(n for _, n in self.token_times)
                self.token_times.append((loop.time(), estimated_tokens))

    def release(self):
        self.semaphore.release()


# ---------- Run one evaluation (sync) in thread ----------
def _run_one(
    provider_name: str,
    model: str,
    prompts: dict,
    prompt_name: str,
    story_with_blank: str,
    candidate_infill: str,
    project: Optional[str],
    response_model: type,
) -> Optional[NarrativeInfillingEvaluation]:
    prompt_dict = build_api_prompt(prompts, prompt_name, story_with_blank=story_with_blank, candidate_infill=candidate_infill)
    return run(
        provider_name,
        model,
        prompt_dict,
        response_model=response_model,
        project=project,
    )


async def evaluate_one_async(
    loop: asyncio.AbstractEventLoop,
    executor: Any,
    rate_limiter: RateLimiter,
    provider_name: str,
    model: str,
    prompts: dict,
    prompt_name: str,
    story_with_blank: str,
    candidate_infill: str,
    project: Optional[str],
    response_model: type,
) -> Optional[NarrativeInfillingEvaluation]:
    """Runs one evaluation in executor; raises on error."""
    estimated_tokens = estimate_tokens_for_request(prompts, prompt_name, story_with_blank, candidate_infill)
    await rate_limiter.acquire(estimated_tokens=estimated_tokens)
    try:
        return await loop.run_in_executor(
            executor,
            _run_one,
            provider_name,
            model,
            prompts,
            prompt_name,
            story_with_blank,
            candidate_infill,
            project,
            response_model,
        )
    finally:
        rate_limiter.release()


async def run_batch(
    df: pd.DataFrame,
    provider_name: str,
    model: str,
    prompts: dict,
    prompt_name: str,
    rate_limiter: RateLimiter,
    batch_size: int,
    project: Optional[str],
) -> tuple[list[dict], dict]:
    """
    Run evaluations for all rows in df. Returns (list of row dicts for dimensions, stats).
    """
    loop = asyncio.get_event_loop()
    executor = None  # default executor
    stats: dict = {
        "total": len(df),
        "success": 0,
        "error_count": 0,
        "error_breakdown": defaultdict(int),
        "no_score_count": 0,
        "scores_sum": {d: 0 for d in DIMENSIONS},
        "scores_count": {d: 0 for d in DIMENSIONS},
    }
    response_model = get_response_model(prompt_name)
    results: list[Optional[NarrativeInfillingEvaluation]] = [None] * len(df)
    errors_by_index: list[tuple[int, str, str]] = []  # (idx, error_type, error_message)

    async def task_for_row(idx: int):
        row = df.iloc[idx]
        story = row.get("problem", "")
        candidate = row.get("extracted_answer", "")
        if pd.isna(story):
            story = ""
        if pd.isna(candidate):
            candidate = ""
        try:
            ev = await evaluate_one_async(
                loop, executor, rate_limiter,
                provider_name, model, prompts, prompt_name,
                story, candidate, project,
                response_model,
            )
            return idx, ev, None, None
        except Exception as e:
            return idx, None, type(e).__name__, str(e)

    first_error_message: Optional[str] = None

    # Process in batches of batch_size concurrent tasks
    batch_starts = range(0, len(df), batch_size)
    pbar = tqdm(batch_starts, total=len(batch_starts), unit="batch", desc="Evaluating")
    for start in pbar:
        end = min(start + batch_size, len(df))
        tasks = [task_for_row(i) for i in range(start, end)]
        out = await asyncio.gather(*tasks)
        for x in out:
            idx, ev, err, err_msg = x
            if err:
                errors_by_index.append((idx, err, err_msg or ""))
                stats["error_breakdown"][err] += 1
                stats["error_count"] += 1
                results[idx] = None
                if first_error_message is None and err_msg:
                    first_error_message = err_msg
            else:
                results[idx] = ev
                stats["success"] += 1
                if ev:
                    for dim in DIMENSIONS:
                        val = getattr(ev, dim, None)
                        if val is not None:
                            score = val if isinstance(val, int) else val.score
                            stats["scores_sum"][dim] += score
                            stats["scores_count"][dim] += 1
        pbar.set_postfix(ok=stats["success"], err=stats["error_count"])

    # no_score: row has no valid evaluation (result is None or all dimensions missing)
    no_score = 0
    for i, ev in enumerate(results):
        if ev is None:
            no_score += 1
        else:
            has_any = any(getattr(ev, d, None) is not None for d in DIMENSIONS)
            if not has_any:
                no_score += 1
    stats["no_score_count"] = no_score
    stats["error_breakdown"] = dict(stats["error_breakdown"])

    # Build row dicts for Excel
    rows_out = []
    for ev in results:
        rows_out.append(eval_to_row_dict(ev))

    # Averages
    stats["average_scores"] = {}
    for dim in DIMENSIONS:
        n = stats["scores_count"][dim]
        if n > 0:
            stats["average_scores"][dim] = round(stats["scores_sum"][dim] / n, 4)
        else:
            stats["average_scores"][dim] = None
    del stats["scores_sum"]
    del stats["scores_count"]

    # Keep backward compatibility: errors_by_index as list of [idx, type]; add message sample
    stats["errors_by_index"] = [[e[0], e[1]] for e in errors_by_index[:500]]
    stats["errors_by_index_total"] = len(errors_by_index)
    if first_error_message:
        stats["first_error_message"] = first_error_message
        stats["error_messages_sample"] = [e[2] for e in errors_by_index[:5] if e[2]]
    return rows_out, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Batch narrative infilling evaluation with rate limiting")
    parser.add_argument("--method", choices=["teler", "reasoning"], required=True, help="Method: teler or reasoning")
    parser.add_argument("--llm", type=str, default=None, help="LLM name (e.g. deepseek-llama-70B), filename without .xlsx; required unless --all")
    parser.add_argument("--provider", type=str, default="azure", help="Evaluator API provider (default: azure)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Evaluator model name")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt key in YAML (default: first key in prompt file)")
    parser.add_argument("--rpm", type=float, default=30.0, help="Max requests per minute (default: 30)")
    parser.add_argument("--tpm", type=float, default=0, help="Max tokens per minute (0 = disabled). Set to your Azure tier limit to avoid 429 (e.g. 60000).")
    parser.add_argument("--batch-size", type=int, default=5, help="Concurrent requests per batch (default: 5)")
    parser.add_argument("--input-dir", type=str, default=None, help="Override input dir for method (default: data/eval_files/<method>)")
    parser.add_argument("--inplace", dest="inplace", action="store_true", default=True, help="Save GPT-eval columns in the same input Excel file (default)")
    parser.add_argument("--no-inplace", dest="inplace", action="store_false", help="Write GPT-eval output to --output-dir instead")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional separate output base dir when not using --inplace")
    parser.add_argument("--stats-file", type=str, default=None, help="Optional JSON stats path (disabled by default)")
    parser.add_argument("--prompt-file", type=str, default=None, help="Prompt YAML path (default: cfg/prompts/eval_prompt.yaml)")
    parser.add_argument("--project", type=str, default="text-infilling", help="Usage tracker project name")
    parser.add_argument("--all", action="store_true", default=False, help="Run for all LLMs in the given method (ignores --llm)")
    args = parser.parse_args()
    ensure_provider_credentials(args.provider)

    base_dir = PROJECT_DIR
    method = args.method
    input_subdir = args.input_dir or METHOD_INPUT_DIRS.get(method)
    if not input_subdir:
        print(f"Error: unknown method '{method}'. Known: {list(METHOD_INPUT_DIRS)}", file=sys.stderr)
        sys.exit(1)
    input_dir = Path(base_dir) / input_subdir
    if args.all:
        llm_list = sorted(p.stem for p in input_dir.glob("*.xlsx"))
        if not llm_list:
            print(f"No .xlsx files in {input_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.llm:
            print("Error: pass --llm <name> or --all", file=sys.stderr)
            sys.exit(1)
        llm_list = [args.llm]

    out_base = None
    if not args.inplace:
        if not args.output_dir:
            raise ValueError("--output-dir is required when --inplace is disabled")
        out_base = Path(base_dir) / args.output_dir
        out_base.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_file) if args.stats_file else None
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if stats_path.exists():
            with open(stats_path) as f:
                all_stats = json.load(f)
        else:
            all_stats = {"teler": [], "reasoning": []}
    else:
        all_stats = None

    for llm in llm_list:
        in_path = input_dir / f"{llm}.xlsx"
        if not in_path.exists():
            print(f"Skipping {llm}: input file not found: {in_path}", file=sys.stderr)
            continue

        if args.inplace:
            out_excel = in_path
        else:
            method_out = out_base / method
            method_out.mkdir(parents=True, exist_ok=True)
            out_excel = method_out / f"{llm}.xlsx"

        prompts_path = Path(args.prompt_file) if args.prompt_file else (PROJECT_DIR / "cfg" / "prompts" / "eval_prompt.yaml")
        if not prompts_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompts_path}. Pass --prompt-file explicitly."
            )
        prompts = load_prompts(prompts_path)
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError(f"Prompt file is empty or invalid: {prompts_path}")
        if args.prompt:
            if args.prompt not in prompts:
                raise ValueError(
                    f"Prompt key '{args.prompt}' not found in {prompts_path}. "
                    f"Available keys: {list(prompts.keys())}"
                )
            prompt_name = args.prompt
        else:
            prompt_name = next(iter(prompts.keys()))
        print(f"Using prompt key: {prompt_name}")

        print(f"Loading {in_path} ...")
        df = pd.read_excel(in_path)
        df = df.iloc[:10000]
        if "problem" not in df.columns or "extracted_answer" not in df.columns:
            print(f"Error: {in_path} must have columns 'problem' and 'extracted_answer'", file=sys.stderr)
            continue

        rate_limiter = RateLimiter(
            requests_per_minute=args.rpm,
            concurrency=args.batch_size,
            tokens_per_minute=args.tpm,
        )
        tpm_info = f", tpm={args.tpm}" if args.tpm > 0 else ""
        print(f"Running evaluation: method={method}, llm={llm}, rpm={args.rpm}, batch_size={args.batch_size}{tpm_info} ...")
        rows_out, stats = asyncio.run(run_batch(
            df,
            args.provider,
            args.model,
            prompts,
            prompt_name,
            rate_limiter,
            args.batch_size,
            args.project,
        ))
        if stats.get("error_count", 0) > 0 and stats.get("first_error_message"):
            print(f"First error ({stats['error_count']} total): {stats['first_error_message']}", file=sys.stderr)

        for dim in DIMENSIONS:
            df[dim] = [r[dim] for r in rows_out]

        # QAvg: per-row average across GPT dimension scores.
        qavg_vals: list[float | None] = []
        for row in rows_out:
            row_scores = []
            for dim in DIMENSIONS:
                score_val = row.get(dim, {}).get("score")
                if isinstance(score_val, (int, float)):
                    row_scores.append(float(score_val))
            if row_scores:
                qavg_vals.append(round(float(sum(row_scores) / len(row_scores)), 4))
            else:
                qavg_vals.append(None)
        df["QAvg"] = qavg_vals

        df.to_excel(out_excel, index=False)
        print(f"Wrote {out_excel}")

        valid_qavg = [x for x in qavg_vals if isinstance(x, (int, float))]
        avg_qavg = round(float(sum(valid_qavg) / len(valid_qavg)), 4) if valid_qavg else None
        print("\nAVERAGE GPT SCORES:")
        for dim in DIMENSIONS:
            print(f"{dim}: {stats['average_scores'].get(dim)}")
        print(f"AvgQAvg: {avg_qavg}")

        stats_entry = {
            "model_name": llm,
            "method": method,
            "total": stats["total"],
            "success": stats["success"],
            "error_count": stats["error_count"],
            "error_breakdown": stats["error_breakdown"],
            "no_score_count": stats["no_score_count"],
            "average_scores": stats["average_scores"],
            "AvgQAvg": avg_qavg,
            "errors_by_index_total": stats["errors_by_index_total"],
            "timestamp": datetime.now().isoformat(),
        }
        if stats.get("first_error_message"):
            stats_entry["first_error_message"] = stats["first_error_message"]
        if stats.get("errors_by_index"):
            stats_entry["errors_by_index_sample"] = stats["errors_by_index"][:50]
        if stats.get("error_messages_sample"):
            stats_entry["error_messages_sample"] = stats["error_messages_sample"]

        if all_stats is not None:
            key = method
            existing = [x for x in all_stats[key] if x.get("model_name") != llm]
            existing.append(stats_entry)
            all_stats[key] = existing

        print("Stats:", json.dumps(stats_entry, indent=2))

    if stats_path is not None and all_stats is not None:
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()

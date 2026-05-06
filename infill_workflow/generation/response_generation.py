from __future__ import annotations

from pathlib import Path
import os
import re

import pandas as pd
from tqdm import tqdm

from infill_workflow.llm.generators.vllm import VLLM
from infill_workflow.llm.session import Session
from infill_workflow.export import loadExcel, writeExcel


def visibleGpuCountFromEnv() -> int | None:
    """Count visible GPUs from CUDA_VISIBLE_DEVICES when set."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    tokens = [t.strip() for t in str(raw).split(",")]
    devices = [t for t in tokens if t and t != "-1"]
    return len(devices)


def iterBatches(df: pd.DataFrame, batch_size: int):
    """Yield dataframe slices as (start_index, batch_df)."""
    n = len(df)
    for start in range(0, n, batch_size):
        yield start, df.iloc[start : start + batch_size].copy()


def extractAnswerText(response: str) -> tuple[str, str]:
    """Extract answer from tags when present, otherwise fallback."""
    text = str(response or "")
    m = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), "success"
    return text.strip(), "fallback_full_response"


def buildSessions(batch: pd.DataFrame, supports_sys: bool) -> list[Session]:
    """Build Session objects for one generation batch."""
    return [
        Session(system_role=str(sys_txt), supports_sys=supports_sys)
        for sys_txt in batch["sys_text"].fillna("").tolist()
    ]


def collectResponsesToExcel(
    *,
    input_excel: str | Path,
    output_excel: str | Path,
    model: str,
    temperature: float,
    batch_size: int,
    max_rows: int | None,
    no_sys: bool,
    model_cache: str | None,
    model_args: dict | None,
) -> Path:
    """Fill response/extracted columns and write output Excel."""
    df = loadExcel(input_excel)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    df["model"] = str(model)
    df["temperature"] = float(temperature)
    if "response" not in df.columns:
        df["response"] = ""
    else:
        df["response"] = df["response"].astype("object")
    if "extracted_answer" not in df.columns:
        df["extracted_answer"] = ""
    else:
        df["extracted_answer"] = df["extracted_answer"].astype("object")
    if "extraction_status" not in df.columns:
        df["extraction_status"] = ""
    else:
        df["extraction_status"] = df["extraction_status"].astype("object")

    kwargs = dict(model_args or {})
    if model_cache is not None:
        kwargs["model_cache"] = model_cache
    # Keep vLLM output quiet so terminal stays focused on outer tqdm.
    kwargs.setdefault("disable_log_stats", True)
    visible_gpus = visibleGpuCountFromEnv()
    requested_tp = kwargs.get("tensor_parallel_size")
    if (
        visible_gpus is not None
        and isinstance(requested_tp, int)
        and requested_tp > visible_gpus
    ):
        # Keep tensor parallel within currently visible GPU count.
        print(
            "[info] reducing tensor_parallel_size from "
            f"{requested_tp} to {visible_gpus} based on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        kwargs["tensor_parallel_size"] = visible_gpus
    generator = None
    init_error = None
    # Auto-tune startup memory target when GPU is currently busy.
    for attempt_idx in range(3):
        try:
            generator = VLLM(model, **kwargs)
            init_error = None
            break
        except Exception as e:
            init_error = e
            msg = str(e)
            mem_match = re.search(
                r"Free memory on device \(([\d.]+)/([\d.]+) GiB\).*desired GPU memory utilization \(([\d.]+),",
                msg,
                flags=re.DOTALL,
            )
            if not mem_match:
                break
            free_gib = float(mem_match.group(1))
            total_gib = float(mem_match.group(2))
            current_util = float(mem_match.group(3))
            suggested_util = max(0.05, min(0.95, (free_gib / total_gib) * 0.9))
            if suggested_util >= current_util:
                suggested_util = max(0.05, current_util * 0.8)
            kwargs["gpu_memory_utilization"] = round(suggested_util, 3)
            print(
                "[info] vLLM init failed due to low free VRAM; retrying with "
                f"gpu_memory_utilization={kwargs['gpu_memory_utilization']} "
                f"(attempt {attempt_idx + 2}/3)."
            )
    if generator is None:
        raise RuntimeError(
            "Failed to initialize vLLM after retrying lower GPU memory utilization."
        ) from init_error
    supports_sys = not no_sys

    for start_idx, batch in tqdm(
        iterBatches(df, batch_size),
        total=(len(df) + batch_size - 1) // batch_size,
        desc=f"generating responses ({model})",
    ):
        sessions = buildSessions(batch, supports_sys=supports_sys)
        prompts = batch["prompt_text"].fillna("").astype(str).tolist()
        try:
            responses = generator.generate(
                sessions,
                prompts,
                temp=float(temperature),
                use_tqdm=False,
            )[1]
        except ValueError as e:
            if (
                supports_sys
                and "System role not supported" in str(e)
            ):
                # Some chat templates reject system role; merge and retry.
                print(
                    "[info] model chat template does not support system role; "
                    "retrying with merged system+prompt."
                )
                supports_sys = False
                sessions = buildSessions(batch, supports_sys=supports_sys)
                responses = generator.generate(
                    sessions,
                    prompts,
                    temp=float(temperature),
                    use_tqdm=False,
                )[1]
            else:
                raise
        df.loc[start_idx : start_idx + len(batch) - 1, "response"] = responses
        extracted = [extractAnswerText(r) for r in responses]
        df.loc[start_idx : start_idx + len(batch) - 1, "extracted_answer"] = [x[0] for x in extracted]
        df.loc[start_idx : start_idx + len(batch) - 1, "extraction_status"] = [x[1] for x in extracted]

    total_rows = len(df)
    success_rows = int((df["extraction_status"] == "success").sum())
    non_success_rows = total_rows - success_rows
    fallback_rows = int((df["extraction_status"] == "fallback_full_response").sum())
    print(
        f"extraction summary: success={success_rows}/{total_rows}, "
        f"non_success={non_success_rows}, fallback_full_response={fallback_rows}"
    )

    return writeExcel(df, output_excel)


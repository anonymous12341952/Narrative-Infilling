from __future__ import annotations

from pathlib import Path

import pandas as pd

preEvalColumns = [
    "dataset",
    "ref_id",
    "source_text",
    "problem",
    "gold_answer",
    "unit",
    "n",
    "unit_idx",
    "template_name",
    "template_id",
    "sys_id",
    "sys_text",
    "prompt_text",
    "prompt_id",
    "model",
    "temperature",
    "response",
]


def validatePreEvalSchema(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    """Check required pre-eval columns and report extras."""
    missing = [c for c in preEvalColumns if c not in df.columns]
    extras = [c for c in df.columns if c not in preEvalColumns]
    return (len(missing) == 0, missing, extras)


def writeExcel(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write DataFrame to Excel while preserving expected column order."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for col in preEvalColumns:
        if col not in df.columns:
            df[col] = ""
    extra_cols = [c for c in df.columns if c not in preEvalColumns]
    df = df[preEvalColumns + extra_cols].copy()
    df.to_excel(out, index=False)
    return out


def loadExcel(path: str | Path) -> pd.DataFrame:
    """Read an Excel file into a DataFrame."""
    return pd.read_excel(path)


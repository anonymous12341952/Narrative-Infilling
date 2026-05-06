from __future__ import annotations

from pathlib import Path

import pandas as pd

requiredInputColumns = [
    "dataset",
    "ref_id",
    "source_text",
    "problem",
    "gold_answer",
    "unit",
    "n",
    "unit_idx",
]


def loadInfillingCsv(
    csv_path: str | Path,
    *,
    limit: int | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Load and normalize required columns from infilling CSV."""
    df = pd.read_csv(csv_path)
    missing = [c for c in requiredInputColumns if c not in df.columns]
    if missing:
        raise ValueError(
            f"dataset CSV is missing required columns: {missing}. "
            f"expected at least: {requiredInputColumns}"
        )

    out = df[requiredInputColumns].copy()
    if datasets:
        wanted = set(datasets)
        out = out[out["dataset"].astype(str).isin(wanted)].copy()
    if limit is not None and limit > 0:
        out = out.head(limit).copy()

    out["dataset"] = out["dataset"].astype(str)
    out["ref_id"] = pd.to_numeric(out["ref_id"], errors="coerce").fillna(-1).astype(int)
    out["n"] = pd.to_numeric(out["n"], errors="coerce").fillna(1).astype(int)
    out["unit_idx"] = pd.to_numeric(out["unit_idx"], errors="coerce").fillna(-1).astype(int)
    for c in ("source_text", "problem", "gold_answer", "unit"):
        out[c] = out[c].fillna("").astype(str)

    return out.reset_index(drop=True)


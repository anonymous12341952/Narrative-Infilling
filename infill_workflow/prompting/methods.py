from __future__ import annotations

from pathlib import Path

import pandas as pd

from infill_workflow.utils import strings
from infill_workflow.utils.files import load_yaml

methodPromptSources = {
    # TELeR prompt set.
    "teler": {
        "sys": "cfg/prompts/fitb_system.yaml",
        "levels": {
            "fitb_l0": "cfg/prompts/fitb_l0.yaml",
            "fitb_l1": "cfg/prompts/fitb_l1.yaml",
            "fitb_l2": "cfg/prompts/fitb_l2.yaml",
            "fitb_l3": "cfg/prompts/fitb_l3.yaml",
            "fitb_l4": "cfg/prompts/fitb_l4.yaml",
        },
    },
    # Reasoning prompt set.
    "reasoning": {
        "sys": "cfg/prompts/fitb_system.yaml",
        "levels": {
            "reasoning_paradigms": "cfg/prompts/paradigms.yaml",
        },
    },
}


def resolvePath(root: str | Path, rel_path: str) -> Path:
    """Resolve a project-relative path into an absolute path."""
    return Path(root).joinpath(rel_path).resolve()


def loadMethodTemplates(project_root: str | Path, method: str) -> dict:
    """Load system + level prompt templates for one method."""
    if method not in methodPromptSources:
        raise ValueError(
            f"invalid method '{method}'. choose from {sorted(methodPromptSources)}"
        )

    src = methodPromptSources[method]
    sys_list = load_yaml(str(resolvePath(project_root, src["sys"]))) or []
    if not isinstance(sys_list, list) or not sys_list:
        raise ValueError(f"system template list is empty for method '{method}'")

    levels: dict[str, list[dict]] = {}
    for level_name, rel_path in src["levels"].items():
        prompts = load_yaml(str(resolvePath(project_root, rel_path))) or []
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"prompt list is empty for {method}:{level_name}")
        levels[level_name] = [
            {"template_id": idx, "prompt_text": str(prompt)}
            for idx, prompt in enumerate(prompts)
        ]
    return {"sys": sys_list, "levels": levels}


def buildPromptRows(
    base_df: pd.DataFrame,
    *,
    templates: dict,
    model: str,
    temperature: float,
    prompt_choice: int | None = None,
) -> pd.DataFrame:
    """Expand base dataset rows into prompt-level/model-ready rows."""
    rows: list[dict] = []
    prompt_id = 0
    sys_list: list[str] = templates["sys"]
    levels: dict[str, list[dict]] = templates["levels"]

    for src in base_df.to_dict(orient="records"):
        for template_name, prompt_variants in levels.items():
            candidates = prompt_variants
            if prompt_choice is not None:
                target_idx = prompt_choice - 1
                candidates = [
                    p for p in prompt_variants if int(p["template_id"]) == target_idx
                ]
                if not candidates:
                    # Keep level coverage stable: if requested variant does not
                    # exist for a level (e.g., fitb_l0 has only id=0), fall
                    # back to the first available variant for that level.
                    candidates = [prompt_variants[0]]

            for prompt_variant in candidates:
                template_id = int(prompt_variant["template_id"])
                prompt_template = str(prompt_variant["prompt_text"])
                for sys_id, sys_text in enumerate(sys_list):
                    entry = dict(src)
                    entry["template_name"] = template_name
                    entry["template_id"] = template_id
                    entry["sys_id"] = sys_id
                    entry["sys_text"] = str(sys_text)
                    entry["prompt_text"] = strings.replace_slots(
                        str(prompt_template),
                        entry,
                    )
                    entry["prompt_id"] = prompt_id
                    entry["model"] = model
                    entry["temperature"] = float(temperature)
                    entry["response"] = ""
                    rows.append(entry)
                    prompt_id += 1

    return pd.DataFrame(rows)


# Narrative-Infilling

An end-to-end framework for benchmarking **narrative nfiiling** task using large language models.

The pipeline covers dataset ingestion, prompt construction, LLM response collection, automatic metrics, GPT-based qualitative scoring, and result aggregation — all stored in Excel files for reproducibility and auditability.

---

## Table of Contents

1. [Repository structure](#repository-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick start (full pipeline)](#quick-start-full-pipeline)
5. [Step-by-step guide](#step-by-step-guide)
   - [Step 1 — Prepare prompt spreadsheet](#step-1--prepare-prompt-spreadsheet)
   - [Step 2 — Collect LLM responses](#step-2--collect-llm-responses)
   - [Step 3 — Automatic evaluation](#step-3--automatic-evaluation)
   - [Step 4 — GPT-based qualitative evaluation](#step-4--gpt-based-qualitative-evaluation)
   - [Step 5 — Aggregate results by template](#step-5--aggregate-results-by-template)
6. [Data paths](#data-paths)
7. [Configuration](#configuration)
8. [Prompt methods](#prompt-methods)
9. [Output schema](#output-schema)
10. [Analysis scripts](#analysis-scripts)


---

## Repository structure

```
Narrative-Infilling/
├── cfg/
│   ├── config.yaml               # Model parameters and runtime defaults
│   ├── supported_models.yaml     # Known model registry
│   └── prompts/
│       ├── fitb_l0.yaml          # TELeR level-0 prompt template
│       ├── fitb_l1.yaml          # TELeR level-1 prompt template
│       ├── fitb_l2.yaml          # TELeR level-2 prompt template
│       ├── fitb_l3.yaml          # TELeR level-3 prompt template
│       ├── fitb_l4.yaml          # TELeR level-4 prompt template
│       ├── fitb_system.yaml      # System-role template (TELeR)
│       ├── paradigms.yaml        # Reasoning-method prompt templates
│       └── eval_prompt.yaml      # GPT evaluation scoring prompt
├── dataset/
│   └── infilling_dataset.csv     # Source dataset (input to pipeline)
├── infill_workflow/              # Core Python package
│   ├── arguments/                # CLI argument definitions
│   ├── cfg_reader/               # YAML config loader
│   ├── data_loading/             # CSV ingestion and validation
│   ├── export/                   # Excel schema and I/O helpers
│   ├── generation/               # vLLM response collection
│   ├── llm/                      # Model/session/generator abstractions
│   ├── pipeline/                 # High-level workflow commands
│   ├── prompting/                # Method-specific prompt expansion
│   └── utils/                    # Shared helpers
├── metrics/
│   ├── metrics_engine.py         # BERTScore, ROUGE, METEOR, ChrF, SemF1
│   ├── auto_eval.py              # Automatic evaluation runner
│   └── gpt_eval.py               # GPT-based qualitative evaluation runner
├── result_scripts/
│   └── result_by_template.py     # Per-template score aggregation
├── analysis-script/              # Standalone post-hoc analysis utilities
│   └── README.md                 # Full guide for all analysis scripts
├── scripts/                      # Bash convenience wrappers
│   ├── run_prepare.sh
│   ├── run_collect.sh
│   ├── run_auto_eval.sh
│   ├── run_gpt_eval.sh
│   └── run_aggregate_eval.sh
├── data/                         # Pipeline outputs (git-ignored)
│   ├── prepared/<method>/        # Pre-response Excel files
│   ├── responses/<method>/       # Response Excel files
│   └── eval_files/<method>/      # Scored Excel files
├── requirements.txt
└── setup.py
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | `typing.Self` workaround included for 3.10 |
| CUDA GPU | Required for vLLM response collection |
| 24 GB+ VRAM | For 7–8 B models; 70 B models require 2× 40 GB+ with quantization |
| OpenAI / Azure OpenAI API key | Required for GPT-based evaluation only |

---

## Installation

```bash
cd /home/hossain/Narrative-Infilling

# create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install the infill_workflow package in editable mode
pip install -e .

# verify
infill_workflow --help
```

> **Note**: `vllm` is pinned in `requirements.txt`. If you are on a different CUDA version, install the matching vLLM wheel manually before running `pip install -r requirements.txt`.

---

## Quick start (full pipeline)

Run these five commands from the repo root to reproduce results end-to-end for a single model:

```bash
# 1. Build prompt spreadsheet
./scripts/run_prepare.sh teler google/gemma-2-2b-it

# 2. Collect responses (GPU required)
./scripts/run_collect.sh teler google/gemma-2-2b-it

# 3. Automatic evaluation (ROUGE, BERTScore, METEOR, ChrF, SemF1)
./scripts/run_auto_eval.sh teler

# 4. GPT-based qualitative evaluation  (API key required)
AZURE_OPENAI_API_KEY=<key> ./scripts/run_gpt_eval.sh teler

# 5. Aggregate scores by template
./scripts/run_aggregate_eval.sh
```

All intermediate and final files are written to `data/` as Excel workbooks.

---

## Step-by-step guide

### Step 1 — Prepare prompt spreadsheet

Reads `dataset/infilling_dataset.csv`, expands prompt templates for the chosen method, and writes an Excel file containing all pre-response metadata columns.

```bash
infill_workflow prepare_excel \
  --method teler \
  --dataset-csv dataset/infilling_dataset.csv \
  --model google/gemma-2-2b-it \
  --temperature 0.3
```

| Option | Default | Description |
|---|---|---|
| `--method` | required | `teler` or `reasoning` |
| `--dataset-csv` | required | Path to source CSV |
| `--model` | required | HuggingFace model ID |
| `--temperature` | `0.3` | Sampling temperature written into the sheet |
| `--prompt-choice` | `all` | `all` keeps every variant; `1`–`4` keeps one level |
| `--limit N` | — | Cap number of dataset rows before expansion |
| `--datasets d1 d2` | — | Filter by dataset name(s) in the CSV |
| `--output PATH` | auto | Override default output path |

**Default output**: `data/prepared/<method>/<model_slug>.xlsx`

Using the wrapper (fixed dataset path + defaults):

```bash
./scripts/run_prepare.sh teler google/gemma-2-2b-it
```

---

### Step 2 — Collect LLM responses

Reads the prepared Excel and fills the `response` column by running the model via vLLM. Answer text between `<ANSWER>…</ANSWER>` tags is extracted automatically into `extracted_answer`.

```bash
infill_workflow collect_responses \
  --method teler \
  --model google/gemma-2-2b-it \
  --temperature 0.3 \
  --batch_size 256
```

| Option | Default | Description |
|---|---|---|
| `--method` | required | `teler` or `reasoning` |
| `--model` | required | HuggingFace model ID |
| `--temperature` | `0.3` | Generation temperature |
| `--batch_size` | `256` | Prompts per vLLM batch |
| `--gpu-memory-utilization` | `0.9` | vLLM GPU memory fraction (lower if OOM) |
| `--no_sys` | off | Merge system + user content into one user message |
| `--limit N` | — | Process only first N rows |
| `--input-excel PATH` | auto | Override prepared sheet path |
| `--output-excel PATH` | auto | Override responses sheet path |

**Default paths**:
- Input: `data/prepared/<method>/<model_slug>.xlsx`
- Output: `data/responses/<method>/<model_slug>.xlsx`

Using the wrapper:

```bash
./scripts/run_collect.sh teler google/gemma-2-2b-it
```

> **Multi-GPU**: Set `tensor_parallel_size` in `cfg/config.yaml` per model entry. The pipeline automatically clamps to the number of GPUs visible via `CUDA_VISIBLE_DEVICES`.

---

### Step 3 — Automatic evaluation

Reads response files from `data/responses/<method>/` and writes scored Excel files to `data/eval_files/<method>/`. Metrics computed: `rouge1`, `bert_precision`, `bert_recall`, `bert_f1`, `meteor`, `chrf`, `sem_f1`. Each row also receives an `AtAvg` column (mean of all kept metrics). Per-model averages are printed in the terminal.

```bash
python metrics/auto_eval.py --method teler --all
```

Using the wrapper (defaults to `--all`):

```bash
./scripts/run_auto_eval.sh teler
```

| Option | Description |
|---|---|
| `--method` | `teler` or `reasoning` |
| `--all` | Evaluate every file in the method directory |
| `--input_excel FILE` | Evaluate a single file |
| `--input_dir DIR` | Override input directory |
| `--eval_dir DIR` | Override output directory |
| `--batch_size N` | BERTScore batch size (default 32) |

---

### Step 4 — GPT-based qualitative evaluation

Reads scored Excel files from `data/eval_files/<method>/` and appends GPT-scored columns in-place: `Fluency`, `Coherence`, `Consistency`, `Relevance`, and `QAvg` (mean of the four dimensions). Per-model averages and an `AvgQAvg` summary are printed in the terminal.

**API credentials** — set one before running:

```bash
# Azure OpenAI
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_ENDPOINT=<your-endpoint>   # if required by your deployment

# or standard OpenAI
export OPENAI_API_KEY=<your-key>
```

Alternatively, place credentials in any of these files (auto-loaded at startup):

```
metrics/run_evaluation.env
metrics/.env
~/.env
```

```bash
python metrics/gpt_eval.py --method teler --all --inplace
```

Using the wrapper (defaults to `--all --inplace`):

```bash
./scripts/run_gpt_eval.sh teler
```

| Option | Description |
|---|---|
| `--method` | `teler` or `reasoning` |
| `--all` | Evaluate every file in the method directory |
| `--llm FILE_STEM` | Evaluate a single model file by stem name |
| `--inplace` / `--no-inplace` | Update input file vs. write separate output |
| `--prompt-file PATH` | Scoring prompt YAML (default: `cfg/prompts/eval_prompt.yaml`) |
| `--model MODEL` | Evaluator model ID (default: `gpt-4o`) |
| `--rpm N` | Requests per minute rate limit |
| `--batch-size N` | Concurrent requests per batch |

---

### Step 5 — Aggregate results by template

Reads all scored Excel files in `data/eval_files/<method>/` and produces a JSON + TXT summary of mean scores grouped by prompt template and LLM.

Edit the config block at the top of `scripts/run_aggregate_eval.sh` to select method, model filter, and output paths, then run:

```bash
./scripts/run_aggregate_eval.sh
```

Or call directly:

```bash
python result_scripts/result_by_template.py \
  --method teler \
  --output data/eval_files/aggregate/evaluation_by_template.json \
  --output-txt data/eval_files/aggregate/evaluation_by_template.txt \
  --by-dataset
```

| Option | Description |
|---|---|
| `--method` | `teler`, `reasoning`, or `both` |
| `--llm-name STEM` | Filter to one model |
| `--by-dataset` | Add dataset-level breakdown |
| `--common-models-only` | Restrict to models present in both methods |
| `--txt-no-detail` | Compact TXT without per-template rows |
| `--input-dir DIR` | Override input directory |
| `--output PATH` | JSON output path |
| `--output-txt PATH` | TXT output path |

---

## Data paths

| Stage | Input | Output |
|---|---|---|
| `prepare_excel` | `dataset/infilling_dataset.csv` | `data/prepared/<method>/<model_slug>.xlsx` |
| `collect_responses` | `data/prepared/<method>/<model_slug>.xlsx` | `data/responses/<method>/<model_slug>.xlsx` |
| `auto_eval` | `data/responses/<method>/<model_slug>.xlsx` | `data/eval_files/<method>/<model_slug>.xlsx` |
| `gpt_eval` | `data/eval_files/<method>/<model_slug>.xlsx` | same file (in-place) |
| `aggregate` | `data/eval_files/<method>/` | `data/eval_files/aggregate/` |

---

## Configuration

**`cfg/config.yaml`** controls vLLM model loading parameters.

```yaml
model_params:
  model_cache: "{{project_root}}/data/llm_cache"
  default:
    max_model_len: 4096
    trust_remote_code: true
    tensor_parallel_size: 1       # increase for multi-GPU
  meta-llama/Llama-3.3-70B-Instruct:
    tensor_parallel_size: 2
    quantization: "bitsandbytes"
    load_format: "bitsandbytes"
```

Add a new model entry to override any `default` key for that specific model.

---

## Prompt methods

| Method | Flag | Template files | Description |
|---|---|---|---|
| TELeR | `--method teler` | `cfg/prompts/fitb_l0.yaml` … `fitb_l4.yaml` | Five prompt complexity levels (L0–L4) following the TELeR taxonomy |
| Reasoning | `--method reasoning` | `cfg/prompts/paradigms.yaml` | Chain-of-thought and reasoning-style prompt variants |

---

## Output schema

Every Excel file produced by the pipeline shares a common column contract:

| Column | Stage added | Description |
|---|---|---|
| `dataset` | prepare | Source dataset name |
| `ref_id` | prepare | Row reference ID |
| `source_text` | prepare | Original narrative passage |
| `problem` | prepare | Fill-in-the-blank problem statement |
| `gold_answer` | prepare | Ground-truth answer |
| `unit` | prepare | Linguistic unit type |
| `n` | prepare | Unit index in passage |
| `unit_idx` | prepare | Sentence position |
| `template_name` | prepare | Prompt template identifier |
| `template_id` | prepare | Numeric template ID |
| `sys_id` | prepare | System-role ID |
| `sys_text` | prepare | System-role text |
| `prompt_text` | prepare | Full prompt sent to LLM |
| `prompt_id` | prepare | Prompt hash/ID |
| `model` | prepare | Model HuggingFace ID |
| `temperature` | prepare | Sampling temperature |
| `response` | collect | Raw LLM response |
| `extracted_answer` | collect | Text between `<ANSWER>…</ANSWER>` tags |
| `extraction_status` | collect | `ok` / `missing` / `empty` |
| `rouge1` | auto_eval | ROUGE-1 F1 |
| `bert_f1` | auto_eval | BERTScore F1 |
| `meteor` | auto_eval | METEOR |
| `chrf` | auto_eval | ChrF |
| `sem_f1` | auto_eval | Semantic F1 |
| `AtAvg` | auto_eval | Row-mean of all automatic metrics |
| `Fluency` | gpt_eval | GPT fluency score (1–5) |
| `Coherence` | gpt_eval | GPT coherence score (1–5) |
| `Consistency` | gpt_eval | GPT consistency score (1–5) |
| `Relevance` | gpt_eval | GPT relevance score (1–5) |
| `QAvg` | gpt_eval | Row-mean of GPT dimensions |

---

## Analysis scripts

See [`analysis-script/README.md`](analysis-script/README.md) for full documentation.

---


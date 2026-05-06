# Analysis Scripts Guide

This folder contains standalone analysis utilities for evaluated Excel files.

## Common Conventions

- Default input root is `data/eval_files/` under project root.
- Methods are typically `teler`, `reasoning`, or `both`.
- Most scripts print to terminal and can also write JSON/CSV/TXT.
- `--base-dir` is available in most scripts to override project root.
- For full details any time: `python analysis-script/<script>.py --help`

---

## `rouge_above_threshold.py`
- **What it does**: Counts rows above ROUGE threshold (`rouge1` and/or `rougeL`) for one file or all files in a method.
- **How to run**:
```bash
python analysis-script/rouge_above_threshold.py -f data/eval_files/teler/google_gemma-2-2b-it.xlsx
python analysis-script/rouge_above_threshold.py -m teler -t 0.7 -o data/eval_files/rouge_summary.txt
```
- **Output**: Terminal summary; optional TXT.
- **Flags**:
  - `--file, -f`: analyze one Excel file.
  - `--method, -m`: analyze all files under `data/eval_files/<method>/`.
  - `--column, -c`: metric column(s), can repeat (`rouge1`, `rougeL`).
  - `--threshold, -t`: strict cutoff (`score > threshold`).
  - `--list, -l`: print up to K matching example rows per metric.
  - `--detail`: include per-file sections in method-wide mode.
  - `--output, -o`: mirror terminal output to TXT.
  - `--base-dir`: override project root.

## `export_rouge_above_samples.py`
- **What it does**: Exports rows above ROUGE threshold from one model file to readable TXT.
- **How to run**:
```bash
python analysis-script/export_rouge_above_samples.py -f data/eval_files/teler/google_gemma-2-2b-it.xlsx -t 0.7 -o rouge_high.txt
```
- **Output**: TXT sample report.
- **Flags**:
  - `--file, -f`: input model Excel.
  - `--threshold, -t`: threshold.
  - `--match`: `any` / `all` / single metric mode.
  - `--output, -o`: required output TXT path.
  - `--max-rows`: cap exported rows.
  - `--base-dir`: resolve relative file path.

## `corr_dimensions.py`
- **What it does**: Computes correlation matrices across evaluation dimensions per model and/or pooled.
- **How to run**:
```bash
python analysis-script/corr_dimensions.py -m both
python analysis-script/corr_dimensions.py -m teler --output-json data/eval_files/corr_teler.json
```
- **Output**: Terminal matrix tables; optional JSON/CSV.
- **Flags**:
  - `--method, -m`: `teler` / `reasoning` / `both`.
  - `--corr`: correlation type (default `kendall`).
  - `--preset`: metric preset (`dimensions`, `auto`, `all`).
  - `--dims`: custom comma-separated metrics.
  - `--aggregate`: pooled correlation across models.
  - `--show-per-model`: with aggregate, still print per-model matrices.
  - `--show-counts`: print pairwise non-null N matrix.
  - `--long-names`: full metric labels in display.
  - `--output-json`: write JSON.
  - `--output-csv`: write long-form CSV.
  - `--base-dir`: override project root.

## `count_high_average.py`
- **What it does**: Counts rows where `Average >= threshold`.
- **How to run**:
```bash
python analysis-script/count_high_average.py -m both -t 4.0
```
- **Output**: Terminal counts; optional JSON.
- **Flags**:
  - `--threshold, -t`: high threshold.
  - `--method, -m`: run scope.
  - `--common-models-only`: shared models only when `both`.
  - `--output, -o`: save JSON.
  - `--base-dir`: override project root.

## `count_low_average.py`
- **What it does**: Counts rows where `Average < threshold`.
- **How to run**:
```bash
python analysis-script/count_low_average.py -m both -t 3.0
```
- **Output**: Terminal report; optional JSON.
- **Flags**:
  - `--threshold, -t`: low threshold.
  - `--method, -m`: run scope.
  - `--no-per-model`: omit per-model sections.
  - `--filter-teler-to-reasoning`: restrict teler to reasoning model set.
  - `--output, -o`: save JSON.
  - `--base-dir`: override project root.

## `count_high_dimensions.py`
- **What it does**: Per dimension, counts rows `>= threshold` and top `(n, unit_idx)` hotspots.
- **How to run**:
```bash
python analysis-script/count_high_dimensions.py -m both -t 4.0 -k 20
```
- **Output**: Terminal report; optional JSON/CSV.
- **Flags**:
  - `--threshold, -t`: high threshold.
  - `--top, -k`: top K hotspots.
  - `--method, -m`: run scope.
  - `--common-models-only`: shared models only when `both`.
  - `--output, -o`: save JSON.
  - `--csv`: save CSV.
  - `--base-dir`: override project root.

## `count_low_dimensions.py`
- **What it does**: Per dimension, counts rows `<= threshold` and top `(n, unit_idx)` hotspots.
- **How to run**:
```bash
python analysis-script/count_low_dimensions.py -m both -t 2.0 -k 20
```
- **Output**: Terminal report; optional JSON/CSV.
- **Flags**:
  - `--threshold, -t`: low threshold.
  - `--top, -k`: top K hotspots.
  - `--method, -m`: run scope.
  - `--output, -o`: save JSON.
  - `--csv`: save CSV.
  - `--base-dir`: override project root.

## `count_high_by_n_unitidx.py`
- **What it does**: Finds `(n, unit_idx)` with most high-`Average` rows.
- **How to run**:
```bash
python analysis-script/count_high_by_n_unitidx.py -m both -t 4.0 -k 20
```
- **Output**: Terminal ranking; optional JSON/CSV.
- **Flags**:
  - `--threshold, -t`: high threshold.
  - `--top, -k`: top K.
  - `--method, -m`: run scope.
  - `--output, -o`: save JSON.
  - `--csv`: save CSV.
  - `--base-dir`: override project root.

## `count_low_by_n_unitidx.py`
- **What it does**: Finds `(n, unit_idx)` with most low-`Average` rows; optional doc-length analysis.
- **How to run**:
```bash
python analysis-script/count_low_by_n_unitidx.py -m both -t 3.0 -k 20
```
- **Output**: Terminal ranking; optional JSON/CSV.
- **Flags**:
  - `--threshold, -t`: low threshold.
  - `--top, -k`: top K.
  - `--method, -m`: run scope.
  - `--no-per-model`: hide per-model sections.
  - `--doc-length`: include sentence-length analysis.
  - `--output, -o`: save JSON.
  - `--csv`: save CSV.
  - `--base-dir`: override project root.

## `count_by_position.py`
- **What it does**: Splits samples by blank position (`opening/middle/closing`) and reports position statistics.
- **How to run**:
```bash
python analysis-script/count_by_position.py -m both -t 4.0 --dataset-template-detail
```
- **Output**: Terminal tables; optional JSON/CSV.
- **Flags**:
  - `--threshold, -t`: high threshold for `Average`.
  - `--method, -m`: run scope.
  - `--common-models-only`: shared-model filtering.
  - `--low-threshold`: cutoff for low-dimension counter.
  - `--show-avg` / `--no-show-avg`: toggle average display.
  - `--show-high-pct` / `--no-show-high-pct`: toggle high-rate display.
  - `--show-low-count-pct` / `--no-show-low-count-pct`: toggle low-count-rate display.
  - `--per-model`: per-model breakdown.
  - `--no-dataset-report`: suppress dataset blocks.
  - `--dataset-template-detail`: verbose dataset×template detail.
  - `--output, -o`: save JSON.
  - `--csv`: save CSV for current run scope.
  - `--csv-common`: CSV using common-model subset.
  - `--csv-all`: CSV using all models.
  - `--limit, -l`: row cap per file.
  - `--base-dir`: override project root.

## `find_fluent_but_others_low.py`
- **What it does**: Finds rows with high `Fluency` and low selected dimensions.
- **How to run**:
```bash
python analysis-script/find_fluent_but_others_low.py -m both --fluency-high 4 --low-threshold 2
```
- **Output**: Terminal findings; optional JSON/CSV.
- **Flags**:
  - `--method, -m`: run scope.
  - `--fluency-high`: minimum Fluency.
  - `--low-threshold`: max value for selected low dimensions.
  - `--dims-low`: comma-separated low-dimension list.
  - `--top, -k`: top K hotspots.
  - `--output, -o`: save JSON.
  - `--csv`: save summary CSV.
  - `--base-dir`: override project root.

## `find_fluent_and_others_high.py`
- **What it does**: Finds rows with high `Fluency` and high selected dimensions.
- **How to run**:
```bash
python analysis-script/find_fluent_and_others_high.py -m both --fluency-high 4 --high-threshold 4
```
- **Output**: Terminal findings; optional JSON/CSV.
- **Flags**:
  - `--method, -m`: run scope.
  - `--fluency-high`: minimum Fluency.
  - `--high-threshold`: minimum value for selected high dimensions.
  - `--dims-high`: comma-separated high-dimension list.
  - `--top, -k`: top K hotspots.
  - `--output, -o`: save JSON.
  - `--csv`: save summary CSV.
  - `--base-dir`: override project root.

## `sample_high_quality.py`
- **What it does**: Randomly samples high-quality rows (`Average >= min-avg`) for manual inspection.
- **How to run**:
```bash
python analysis-script/sample_high_quality.py -m both -n 20 --min-avg 4.5
```
- **Output**: Terminal + log file.
- **Flags**:
  - `-n, --num`: number of samples.
  - `--min-avg`: minimum Average.
  - `--method, -m`: run scope.
  - `--output, -o`: log file path.
  - `--seed`: random seed.
  - `--max-files`: max files per method.
  - `--max-problem-len`: max problem length.
  - `--min-problem-len`: min problem length.
  - `--base-dir`: override project root.

## `sample_low_quality.py`
- **What it does**: Randomly samples low-quality rows (`Average <= max-avg`) for manual inspection.
- **How to run**:
```bash
python analysis-script/sample_low_quality.py -m both -n 20 --max-avg 2.0
```
- **Output**: Terminal + log file.
- **Flags**:
  - `-n, --num`: number of samples.
  - `--max-avg`: maximum Average.
  - `--method, -m`: run scope.
  - `--output, -o`: log file path.
  - `--seed`: random seed.
  - `--max-files`: max files per method.
  - `--max-problem-len`: max problem length.
  - `--min-problem-len`: min problem length.
  - `--base-dir`: override project root.

## `dimension_drivers_by_average.py`
- **What it does**: For common models only, filters by high `Average` and reports dimension means / pct_ge.
- **How to run**:
```bash
python analysis-script/dimension_drivers_by_average.py -t 4 --by-template
python analysis-script/dimension_drivers_by_average.py -t 4 --per-model
```
- **Output**: Terminal report; optional TXT.
- **Flags**:
  - `--threshold, -t`: Average cutoff.
  - `--strict-gt`: use `>` instead of `>=`.
  - `--dim-threshold`: cutoff used for dimension `pct_ge`.
  - `--per-model`: include per-model sections.
  - `--by-template`: split by template/paradigm.
  - `--limit, -l`: row cap per workbook.
  - `--output, -o`: write TXT copy.
  - `--base-dir`: override project root.


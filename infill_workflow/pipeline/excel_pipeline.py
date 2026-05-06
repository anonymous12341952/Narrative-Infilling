from __future__ import annotations

from pathlib import Path
import re

from infill_workflow.data_loading import loadInfillingCsv
from infill_workflow.export import loadExcel, validatePreEvalSchema, writeExcel
from infill_workflow.prompting import buildPromptRows, loadMethodTemplates
from infill_workflow import cfg_reader
from infill_workflow.utils import files


def projectRoot() -> Path:
    """Return repository root used for default paths."""
    return Path(files.project_root())


def slugModel(model: str) -> str:
    """Convert model name into a filename-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model)).strip("_")


def defaultPreparedPath(method: str, model: str) -> Path:
    """Default location for prepare_excel output."""
    return projectRoot() / "data" / "prepared" / method / f"{slugModel(model)}.xlsx"


def defaultResponsesPath(method: str, model: str) -> Path:
    """Default location for collect_responses output."""
    return projectRoot() / "data" / "responses" / method / f"{slugModel(model)}.xlsx"


def prepareExcel(args):
    """Create pre-eval Excel rows from dataset and prompt templates."""
    cfg_reader.load(args.cfg)
    csv_path = Path(args.dataset_csv)
    if not csv_path.is_absolute():
        csv_path = projectRoot() / csv_path

    df = loadInfillingCsv(
        csv_path,
        limit=args.limit,
        datasets=args.datasets,
    )
    templates = loadMethodTemplates(projectRoot(), args.method)
    prompt_choice = None if args.prompt_choice == "all" else int(args.prompt_choice)
    out = buildPromptRows(
        df,
        templates=templates,
        model=args.model,
        temperature=args.temperature,
        prompt_choice=prompt_choice,
    )
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
    else:
        output_path = defaultPreparedPath(args.method, args.model)
    writeExcel(out, output_path)
    print(f"wrote {len(out)} rows to {output_path}")


def collectResponses(args):
    """Generate responses for prepared rows and save updated Excel."""
    from infill_workflow.generation import collectResponsesToExcel

    cfg = cfg_reader.load(args.cfg)[0]
    model_params = cfg.model_params
    model_cache = model_params.get("model_cache")
    kwargs = dict(model_params.get(args.model) or model_params.get("default", {}))
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = float(args.gpu_memory_utilization)

    if args.input_excel:
        input_excel = Path(args.input_excel)
        if not input_excel.is_absolute():
            input_excel = Path.cwd() / input_excel
    else:
        input_excel = defaultPreparedPath(args.method, args.model)
    if args.output_excel:
        output_excel = Path(args.output_excel)
        if not output_excel.is_absolute():
            output_excel = Path.cwd() / output_excel
    else:
        output_excel = defaultResponsesPath(args.method, args.model)

    out = collectResponsesToExcel(
        input_excel=input_excel,
        output_excel=output_excel,
        model=args.model,
        temperature=float(args.temperature),
        batch_size=int(args.batch_size),
        max_rows=args.limit,
        no_sys=bool(args.no_sys),
        model_cache=model_cache,
        model_args=kwargs,
    )
    print(f"wrote responses to {out}")


def validateExcelSchema(args):
    """Validate required pre-eval schema columns in an Excel file."""
    xlsx = Path(args.path)
    if not xlsx.is_absolute():
        xlsx = Path.cwd() / xlsx
    df = loadExcel(xlsx)
    ok, missing, extras = validatePreEvalSchema(df)
    print(f"file: {xlsx}")
    print(f"rows: {len(df)}")
    if ok:
        print("schema: OK")
    else:
        print(f"schema: missing columns -> {missing}")
    if extras:
        print(f"extra columns: {extras}")


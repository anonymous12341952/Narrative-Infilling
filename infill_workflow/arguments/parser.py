import argparse

from infill_workflow.utils import (
    files,
    strings,
)


def parse():
    """Build and parse CLI args for the infilling workflow."""
    keywords = {
        "project_root": files.project_root(),
        "home": files.home(),
        "timestamp": strings.now(),
    }

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(
        required=True,
        dest="procedure",
        description="select workflow command",
    )

    # Step 1: build prepared Excel rows from dataset + prompt templates.
    parser_px = subparser.add_parser("prepare_excel")
    parser_px.add_argument(
        "-c", "--cfg",
        default=f"{files.project_root()}/cfg/config.yaml",
        help="config path",
    )
    parser_px.add_argument(
        "--method",
        choices=["teler", "reasoning"],
        required=True,
        help="prompting method to use",
    )
    parser_px.add_argument(
        "--dataset-csv",
        required=True,
        help="input dataset CSV path",
    )
    parser_px.add_argument(
        "--model",
        required=True,
        help="model name to stamp into output rows",
    )
    parser_px.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="default sampling temperature value for rows",
    )
    parser_px.add_argument(
        "--output",
        required=False,
        default=None,
        help="output xlsx path",
    )
    parser_px.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional row cap from CSV before prompt expansion",
    )
    parser_px.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="optional list of dataset names to include",
    )
    parser_px.add_argument(
        "--prompt-choice",
        type=str,
        choices=["all", "1", "2", "3", "4"],
        default="all",
        help="prompt variant selection per level: all (default) or 1/2/3/4",
    )

    # Step 2: run model generation and fill response columns.
    parser_cr = subparser.add_parser("collect_responses")
    parser_cr.add_argument(
        "-c", "--cfg",
        default=f"{files.project_root()}/cfg/config.yaml",
        help="config path",
    )
    parser_cr.add_argument(
        "--method",
        choices=["teler", "reasoning"],
        required=True,
        help="prompting method used to prepare the sheet",
    )
    parser_cr.add_argument(
        "--input-excel",
        required=False,
        default=None,
        help="prepared xlsx path (from prepare_excel)",
    )
    parser_cr.add_argument(
        "--output-excel",
        required=False,
        default=None,
        help="target xlsx path with responses",
    )
    parser_cr.add_argument(
        "-m", "--model",
        required=True,
        help="model to use for generation",
    )
    parser_cr.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="generation temperature",
    )
    parser_cr.add_argument(
        "-b", "--batch_size",
        type=int,
        default=256,
        help="number of prompt rows per generation batch",
    )
    parser_cr.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="optional row cap from prepared sheet",
    )
    parser_cr.add_argument(
        "--no_sys",
        action="store_true",
        default=False,
        help="merge system and prompt into user role",
    )
    parser_cr.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="optional vLLM GPU memory utilization (e.g., 0.3 on busy GPU)",
    )

    # Optional validation utility for schema checks.
    parser_vs = subparser.add_parser("validate_excel_schema")
    parser_vs.add_argument("--path", required=True, help="xlsx path to validate")

    args = parser.parse_args()

    if hasattr(args, "cfg"):
        args.cfg = strings.replace_slots(
            args.cfg,
            keywords
        )

    return args

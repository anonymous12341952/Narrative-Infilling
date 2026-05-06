# local imports
from infill_workflow import (
    arguments, 
    pipeline,
    utils,
)


def main():
    args = arguments.parse()

    match args.procedure:
        case "prepare_excel":
            pipeline.prepareExcel(args)
        case "collect_responses":
            pipeline.collectResponses(args)
        case "validate_excel_schema":
            pipeline.validateExcelSchema(args)
        case _:
            raise NotImplementedError(utils.strings.clean_multiline(
                """
                Unknown command requested.
                """
            ))

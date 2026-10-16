"""Command-line entry point for regenerating every derived artifact."""

import argparse
import logging
import sys

from . import config
from .errors import InputValidationError, LLMConfigError
from .pipeline import run_all


def build_parser():
    """Argument parser for run.py."""
    parser = argparse.ArgumentParser(description="Regenerate all evaluation artifacts.")
    parser.add_argument("--data-dir", default=str(config.DEFAULT_DIR), help="directory holding kb.json, queries.json, candidate_answers.json")
    parser.add_argument("--out-dir", default=str(config.DEFAULT_DIR), help="directory for derived artifacts")
    parser.add_argument("--llm", choices=config.LLM_MODES, default=config.LLM_MODE_AUTO, help="LLM review mode (default: auto)")
    parser.add_argument("--require-llm", action="store_true", help="exit 3 if the LLM stage fails instead of falling back to rules-only")
    parser.add_argument("--baseline", default=None, help="variant used as the regression-guard baseline")
    return parser


def _summary(result):
    """One-line outcome for the terminal."""
    selected = result.report["decision"]["selected"]
    verdict = f"promote {selected}" if selected else "no variant eligible for promotion"
    llm = "rules-only" if not result.report["llm_available"] else "+".join(result.llm_sources)
    return f"{verdict} (llm review: {llm}, status: {result.llm_status})"


def main(argv=None):
    """Run the pipeline; return a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_all(args.data_dir, args.out_dir, args.llm, args.baseline)
    except (InputValidationError, LLMConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return config.EXIT_INPUT_ERROR
    print(_summary(result))
    if args.require_llm and result.llm_status != config.LOG_STATUS_OK:
        print(f"error: LLM stage failed ({result.llm_status}) and --require-llm was set", file=sys.stderr)
        return config.EXIT_LLM_REQUIRED
    return config.EXIT_OK

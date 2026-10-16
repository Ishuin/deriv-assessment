"""Stage orchestration. Each stage reads validated data and writes one artifact."""

from pathlib import Path

from . import config
from .artifacts import write_json
from .checks import score_all
from .retrieval import retrieve_all


def run_retrieval(inputs, out_dir=config.DEFAULT_DIR):
    """Retrieve top-k passages for every query, write retrieval.json, return records."""
    records = retrieve_all(inputs.kb, inputs.queries)
    write_json(Path(out_dir) / config.RETRIEVAL_FILE, records)
    return records


def run_rule_checks(inputs, retrieval, out_dir=config.DEFAULT_DIR):
    """Score every answer with deterministic checks, write automated_scores.json, return records."""
    records = score_all(inputs, retrieval)
    write_json(Path(out_dir) / config.SCORES_FILE, records)
    return records

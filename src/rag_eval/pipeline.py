"""Stage orchestration. Each stage reads validated data and writes one artifact."""

import json
from pathlib import Path

from . import config
from .aggregate import build_report
from .artifacts import write_json, write_text
from .checks import score_all
from .errors import LLMConfigError, LLMOutputError
from .llm_providers import make_provider
from .llm_review import run_llm_review, validate_reviews
from .render import render_recommendation
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


def _replay_reviews(inputs, out_dir):
    """Re-read and re-validate an existing llm_review.json without calling any model."""
    path = Path(out_dir) / config.LLM_REVIEW_FILE
    if not path.is_file():
        raise LLMConfigError(f"nothing to replay: {path} does not exist")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        records = validate_reviews(stored, [q["query_id"] for q in inputs.queries], inputs.variants)
    except (json.JSONDecodeError, LLMOutputError) as exc:
        raise LLMConfigError(f"cannot replay {path}: {exc}") from exc
    source_by_query = {r["query_id"]: r.get("source", config.SOURCE_LLM) for r in stored}
    return [dict(r, source=source_by_query[r["query_id"]]) for r in records]


def run_llm_stage(inputs, retrieval, scores, mode=config.LLM_MODE_AUTO, out_dir=config.DEFAULT_DIR):
    """Run (or replay) the LLM review, write llm_review.json, return (records, status)."""
    out_dir = Path(out_dir)
    if mode == config.LLM_MODE_REPLAY:
        records, status = _replay_reviews(inputs, out_dir), config.LOG_STATUS_OK
    else:
        provider = make_provider(mode)
        records, status = run_llm_review(inputs, retrieval, scores, provider, out_dir / config.LLM_LOG_FILE)
    write_json(out_dir / config.LLM_REVIEW_FILE, records)
    return records, status


def run_recommendation(queries, retrieval, scores, reviews, out_dir=config.DEFAULT_DIR, baseline=None):
    """Aggregate stored outputs, write recommendation.md, return the report dict."""
    report = build_report(queries, retrieval, scores, reviews, baseline)
    write_text(Path(out_dir) / config.RECOMMENDATION_FILE, render_recommendation(report))
    return report

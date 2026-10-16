"""Load and validate the three input fixtures at the pipeline boundary."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import config
from .errors import InputValidationError

MIN_KB_DOCS = config.TOP_K
MIN_VARIANTS = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Inputs:
    """Validated pipeline inputs.

    `answers` maps query_id to {variant: answer text}. `variants` is sorted.
    """

    kb: list
    queries: list
    answers: dict
    variants: list


def _load_json_list(path, label):
    """Read a JSON file and require a non-empty list of objects."""
    path = Path(path)
    if not path.is_file():
        raise InputValidationError(f"{label}: file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise InputValidationError(f"{label}: expected a non-empty JSON list")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise InputValidationError(f"{label}[{index}]: expected an object")
    return data


def _require_str(record, field, where):
    """Return a non-empty string field or raise."""
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{where}.{field}: expected a non-empty string")
    return value


def _require_str_list(record, field, where):
    """Return a list-of-strings field (empty allowed) or raise."""
    value = record.get(field)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InputValidationError(f"{where}.{field}: expected a list of strings")
    return value


def _check_unique(ids, label, field):
    """Raise on the first repeated identifier."""
    seen = set()
    for value in ids:
        if value in seen:
            raise InputValidationError(f"{label}: duplicate {field} {value!r}")
        seen.add(value)


def load_kb(path):
    """Load and validate the knowledge base."""
    label = config.KB_FILE
    records = _load_json_list(path, label)
    kb = []
    for index, record in enumerate(records):
        where = f"{label}[{index}]"
        kb.append(
            {
                "doc_id": _require_str(record, "doc_id", where),
                "title": _require_str(record, "title", where),
                "text": _require_str(record, "text", where),
            }
        )
    _check_unique([d["doc_id"] for d in kb], label, "doc_id")
    if len(kb) < MIN_KB_DOCS:
        raise InputValidationError(f"{label}: need at least {MIN_KB_DOCS} documents, got {len(kb)}")
    return kb


def _normalize_risk(value, where):
    """Return a known risk level, defaulting (with a warning if present but unknown)."""
    if isinstance(value, str) and value in config.RISK_WEIGHTS:
        return value
    if value is not None:
        logger.warning("%s.risk_level %r is not one of %s; using %s",
                       where, value, sorted(config.RISK_WEIGHTS), config.DEFAULT_RISK)
    return config.DEFAULT_RISK


def load_queries(path, kb):
    """Load and validate queries against the knowledge base."""
    label = config.QUERIES_FILE
    records = _load_json_list(path, label)
    kb_ids = {d["doc_id"] for d in kb}
    queries = []
    for index, record in enumerate(records):
        where = f"{label}[{index}]"
        expected = _require_str_list(record, "expected_doc_ids", where)
        unknown = sorted(set(expected) - kb_ids)
        if unknown:
            raise InputValidationError(f"{where}.expected_doc_ids: unknown doc_id(s) {unknown}")
        queries.append(
            {
                "query_id": _require_str(record, "query_id", where),
                "user_question": _require_str(record, "user_question", where),
                "expected_doc_ids": expected,
                "must_include_any": _require_str_list(record, "must_include_any", where),
                "must_not_claim": _require_str_list(record, "must_not_claim", where),
                "risk_level": _normalize_risk(record.get("risk_level"), where),
            }
        )
    _check_unique([q["query_id"] for q in queries], label, "query_id")
    return queries


def _parse_answer_record(record, where):
    """Return (query_id, {variant: text}) for one candidate_answers record."""
    query_id = _require_str(record, "query_id", where)
    answers = record.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise InputValidationError(f"{where}.answers: expected a non-empty object")
    for variant, text in answers.items():
        if not isinstance(text, str):
            raise InputValidationError(f"{where}.answers.{variant}: expected a string")
    return query_id, dict(answers)


def load_answers(path, queries):
    """Load candidate answers; return ({query_id: {variant: text}}, sorted variants)."""
    label = config.ANSWERS_FILE
    records = _load_json_list(path, label)
    answers = {}
    for index, record in enumerate(records):
        query_id, by_variant = _parse_answer_record(record, f"{label}[{index}]")
        if query_id in answers:
            raise InputValidationError(f"{label}: duplicate query_id {query_id!r}")
        answers[query_id] = by_variant

    query_ids = {q["query_id"] for q in queries}
    unknown = sorted(set(answers) - query_ids)
    if unknown:
        raise InputValidationError(f"{label}: query_id(s) not in {config.QUERIES_FILE}: {unknown}")
    missing = sorted(query_ids - set(answers))
    if missing:
        raise InputValidationError(f"{label}: no answers for query_id(s) {missing}")

    variants = sorted(next(iter(answers.values())))
    mismatched = sorted(qid for qid, by_variant in answers.items() if sorted(by_variant) != variants)
    if mismatched:
        raise InputValidationError(
            f"{label}: variants differ from {variants} for query_id(s) {mismatched}"
        )
    if len(variants) < MIN_VARIANTS:
        raise InputValidationError(f"{label}: need at least {MIN_VARIANTS} variants, got {variants}")
    return answers, variants


def load_inputs(data_dir=config.DEFAULT_DIR):
    """Load and cross-validate all three input files from `data_dir`."""
    data_dir = Path(data_dir)
    kb = load_kb(data_dir / config.KB_FILE)
    queries = load_queries(data_dir / config.QUERIES_FILE, kb)
    answers, variants = load_answers(data_dir / config.ANSWERS_FILE, queries)
    return Inputs(kb=kb, queries=queries, answers=answers, variants=variants)

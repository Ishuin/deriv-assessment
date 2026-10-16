"""Artifact validation: structure, coverage, and reproducibility from stored outputs.

Every check returns a list of human-readable failure strings (empty = pass), so
one run reports all problems instead of stopping at the first.
"""

import difflib
import json
import re
from pathlib import Path

from . import config
from .aggregate import build_report
from .checks import score_all
from .errors import InputValidationError, LLMOutputError
from .llm_review import validate_reviews
from .render import render_recommendation
from .retrieval import retrieve_all
from .schemas import load_inputs

REQUIRED_INPUTS = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)
REQUIRED_OUTPUTS = (
    config.RETRIEVAL_FILE,
    config.SCORES_FILE,
    config.LLM_REVIEW_FILE,
    config.LLM_LOG_FILE,
    config.RECOMMENDATION_FILE,
)
OPTIONAL_OUTPUTS = (config.TAXONOMY_FILE, config.REPORT_FILE)
LOG_STATUSES = (config.LOG_STATUS_OK, config.LOG_STATUS_PROVIDER_ERROR, config.LOG_STATUS_VALIDATION_ERROR)
SOURCES = (config.SOURCE_LLM, config.SOURCE_STUB)
BASELINE_PATTERN = re.compile(r"^Regression guard baseline: (?:`([^`]+)`|none)\.$", re.MULTILINE)
DIFF_LINES_SHOWN = 30
SCORE_FIELDS = ("query_id", "variant", "retrieval_hit", "must_include_pass", "must_not_claim_pass",
                "grounding_score", "risk_flags", "notes")


def _read_json(path):
    """Return (data, error) for a JSON file."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{Path(path).name}: cannot read as JSON ({exc})"


def check_required_files(data_dir, out_dir):
    """Every required input and artifact must exist (optional ones are only checked if present)."""
    missing = [f"{n}: required input is missing" for n in REQUIRED_INPUTS if not (Path(data_dir) / n).is_file()]
    missing += [f"{n}: required artifact is missing" for n in REQUIRED_OUTPUTS if not (Path(out_dir) / n).is_file()]
    return missing


def check_retrieval(stored, inputs):
    """Two sorted, in-range passages per query from the kb, equal to a fresh retrieval."""
    errors = []
    if not isinstance(stored, list):
        return [f"{config.RETRIEVAL_FILE}: expected a list"]
    kb_ids = {d["doc_id"] for d in inputs.kb}
    seen = [r.get("query_id") for r in stored if isinstance(r, dict)]
    if seen != [q["query_id"] for q in inputs.queries]:
        errors.append(f"{config.RETRIEVAL_FILE}: query ids {seen} do not match queries.json order")
    for record in stored:
        passages = record.get("retrieved") if isinstance(record, dict) else None
        label = f"{config.RETRIEVAL_FILE}[{record.get('query_id') if isinstance(record, dict) else '?'}]"
        if not isinstance(passages, list) or len(passages) != config.TOP_K:
            errors.append(f"{label}: expected exactly {config.TOP_K} passages")
            continue
        scores = [p.get("score") for p in passages]
        if any(p.get("doc_id") not in kb_ids for p in passages):
            errors.append(f"{label}: contains a doc_id that is not in kb.json")
        if not all(isinstance(s, (int, float)) and 0 <= s <= 1 for s in scores) or scores != sorted(scores, reverse=True):
            errors.append(f"{label}: scores must be in [0, 1] and sorted descending, got {scores}")
    if not errors and stored != json.loads(json.dumps(retrieve_all(inputs.kb, inputs.queries))):
        errors.append(f"{config.RETRIEVAL_FILE}: differs from a fresh retrieval of the current inputs")
    return errors


def check_scores(stored, inputs, retrieval):
    """One record per (query, variant), valid fields, equal to a fresh rule-check run."""
    if not isinstance(stored, list):
        return [f"{config.SCORES_FILE}: expected a list"]
    expected = [(q["query_id"], v) for q in inputs.queries for v in inputs.variants]
    errors = []
    found = [(r.get("query_id"), r.get("variant")) for r in stored if isinstance(r, dict)]
    if found != expected:
        errors.append(f"{config.SCORES_FILE}: expected one record per (query, variant) {expected}, found {found}")
    for record in stored:
        if not isinstance(record, dict):
            continue
        label = f"{config.SCORES_FILE}[{record.get('query_id')}/{record.get('variant')}]"
        if set(record) != set(SCORE_FIELDS):
            errors.append(f"{label}: fields must be exactly {list(SCORE_FIELDS)}")
            continue
        flags = record["risk_flags"]
        grounding = record["grounding_score"]
        if not isinstance(flags, list) or not flags or not set(flags) <= set(config.RISK_FLAGS) or (config.FLAG_NONE in flags and len(flags) > 1):
            errors.append(f"{label}: invalid risk_flags {flags!r}")
        if not isinstance(grounding, (int, float)) or isinstance(grounding, bool) or not 0 <= grounding <= 1:
            errors.append(f"{label}: grounding_score must be in [0, 1]")
    if not errors and retrieval is not None and stored != json.loads(json.dumps(score_all(inputs, retrieval))):
        errors.append(f"{config.SCORES_FILE}: differs from a fresh rule-check run on the stored retrieval")
    return errors


def check_log(path):
    """llm_calls.jsonl: at least one line, each a JSON object with a known status. Returns (errors, last status)."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [f"{config.LLM_LOG_FILE}: no calls logged"], None
    errors, last_status = [], None
    for number, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"{config.LLM_LOG_FILE}:{number}: not valid JSON")
            continue
        last_status = entry.get("status") if isinstance(entry, dict) else None
        if last_status not in LOG_STATUSES:
            errors.append(f"{config.LLM_LOG_FILE}:{number}: unknown status {last_status!r}")
    return errors, last_status


def check_llm_review(stored, inputs, last_log_status):
    """Valid reviews (allowed variants, score ranges, full coverage), or [] only after a failed call."""
    if not isinstance(stored, list):
        return [f"{config.LLM_REVIEW_FILE}: expected a list"]
    if not stored:
        if last_log_status == config.LOG_STATUS_OK:
            return [f"{config.LLM_REVIEW_FILE}: empty although the last logged LLM call succeeded"]
        return []
    try:
        validate_reviews(stored, [q["query_id"] for q in inputs.queries], inputs.variants)
    except LLMOutputError as exc:
        return [f"{config.LLM_REVIEW_FILE}: {exc}"]
    bad = sorted({r.get("source") for r in stored if r.get("source") not in SOURCES}, key=str)
    return [f"{config.LLM_REVIEW_FILE}: unknown source value(s) {bad}"] if bad else []


def check_recommendation(text, inputs, retrieval, scores, reviews):
    """recommendation.md must equal a fresh render of the stored outputs."""
    match = BASELINE_PATTERN.search(text)
    if not match:
        return [f"{config.RECOMMENDATION_FILE}: missing the 'Regression guard baseline' line"]
    try:
        report = build_report(inputs.queries, retrieval, scores, reviews, match.group(1))
    except (InputValidationError, KeyError, TypeError) as exc:
        return [f"{config.RECOMMENDATION_FILE}: cannot recompute from stored outputs ({exc})"]
    fresh = render_recommendation(report)
    if fresh == text:
        return []
    diff = list(difflib.unified_diff(text.splitlines(), fresh.splitlines(), "stored", "recomputed", lineterm="", n=1))
    shown = "\n".join(diff[:DIFF_LINES_SHOWN])
    return [f"{config.RECOMMENDATION_FILE}: not reproducible from stored outputs:\n{shown}"]


def check_taxonomy(stored, scores):
    """Tags must come from the controlled vocabulary and cover every scored answer."""
    if not isinstance(stored, list):
        return [f"{config.TAXONOMY_FILE}: expected a list"]
    errors = []
    pairs = [(r.get("query_id"), r.get("variant")) for r in stored if isinstance(r, dict)]
    if pairs != [(s["query_id"], s["variant"]) for s in scores]:
        errors.append(f"{config.TAXONOMY_FILE}: records must cover every (query, variant) in automated_scores.json")
    for record in stored:
        tags = record.get("tags") if isinstance(record, dict) else None
        if not isinstance(tags, list) or not set(tags) <= set(config.TAXONOMY):
            errors.append(f"{config.TAXONOMY_FILE}[{record.get('query_id') if isinstance(record, dict) else '?'}]: tags outside the vocabulary: {tags!r}")
    return errors


def validate_artifacts(data_dir=config.DEFAULT_DIR, out_dir=config.DEFAULT_DIR):
    """Run every check; return the list of failures (empty means all good)."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    errors = check_required_files(data_dir, out_dir)
    if errors:
        return errors
    try:
        inputs = load_inputs(data_dir)
    except InputValidationError as exc:
        return [f"inputs: {exc}"]

    loaded = {}
    for name in (config.RETRIEVAL_FILE, config.SCORES_FILE, config.LLM_REVIEW_FILE) + tuple(
        n for n in OPTIONAL_OUTPUTS if n.endswith(".json") and (out_dir / n).is_file()
    ):
        data, error = _read_json(out_dir / name)
        if error:
            errors.append(error)
        loaded[name] = data
    log_errors, last_status = check_log(out_dir / config.LLM_LOG_FILE)
    errors += log_errors
    if errors:
        return errors

    retrieval, scores, reviews = loaded[config.RETRIEVAL_FILE], loaded[config.SCORES_FILE], loaded[config.LLM_REVIEW_FILE]
    errors += check_retrieval(retrieval, inputs)
    errors += check_scores(scores, inputs, retrieval if not errors else None)
    errors += check_llm_review(reviews, inputs, last_status)
    if config.TAXONOMY_FILE in loaded:
        errors += check_taxonomy(loaded[config.TAXONOMY_FILE], scores if isinstance(scores, list) else [])
    for name in OPTIONAL_OUTPUTS:
        if name.endswith(".md") and (out_dir / name).is_file() and not (out_dir / name).read_text(encoding="utf-8").strip():
            errors.append(f"{name}: is empty")
    if not errors:
        text = (out_dir / config.RECOMMENDATION_FILE).read_text(encoding="utf-8")
        errors += check_recommendation(text, inputs, retrieval, scores, reviews)
    return errors

"""Chunk 0 checks: package imports, config consistency, sample fixtures parse."""

import json
from pathlib import Path

import pytest

from rag_eval import config, errors

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)


def load_fixture(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_errors_are_distinct_value_errors():
    classes = (errors.InputValidationError, errors.LLMOutputError, errors.ArtifactValidationError)
    assert all(issubclass(c, ValueError) for c in classes)
    assert len(set(classes)) == 3


@pytest.mark.parametrize(
    "weights",
    [
        (config.RULE_WEIGHT_INCLUDE, config.RULE_WEIGHT_NOT_CLAIM, config.RULE_WEIGHT_GROUNDING),
        (config.WEIGHT_RULES, config.WEIGHT_LLM),
        (config.LLM_FAITHFULNESS_WEIGHT, config.LLM_CLARITY_WEIGHT),
        (config.GROUNDING_TOKEN_WEIGHT, config.GROUNDING_BIGRAM_WEIGHT),
    ],
)
def test_weight_groups_sum_to_one(weights):
    assert sum(weights) == pytest.approx(1.0)


def test_retrieval_returns_two_passages():
    assert config.TOP_K == 2


def test_risk_weights_cover_levels_and_rank_high_first():
    levels = {config.RISK_LOW, config.RISK_MEDIUM, config.RISK_HIGH}
    assert set(config.RISK_WEIGHTS) == levels
    assert config.DEFAULT_RISK in levels
    assert config.RISK_WEIGHTS[config.RISK_HIGH] > config.RISK_WEIGHTS[config.RISK_MEDIUM] > config.RISK_WEIGHTS[config.RISK_LOW]


def test_flag_vocabulary_is_consistent():
    assert config.FLAG_NONE in config.RISK_FLAGS
    assert config.CRITICAL_FLAGS <= set(config.RISK_FLAGS)
    assert config.FLAG_NONE not in config.CRITICAL_FLAGS
    assert len(set(config.RISK_FLAGS)) == len(config.RISK_FLAGS)


def test_taxonomy_has_unique_tags():
    assert len(set(config.TAXONOMY)) == len(config.TAXONOMY) == 6


def test_score_range_and_exit_codes():
    assert config.SCORE_MIN < config.SCORE_MAX
    codes = (config.EXIT_OK, config.EXIT_VALIDATION_FAILED, config.EXIT_INPUT_ERROR, config.EXIT_LLM_REQUIRED)
    assert len(set(codes)) == 4


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_is_non_empty_json_list(name):
    data = load_fixture(name)
    assert isinstance(data, list) and data


def test_fixtures_are_cross_consistent():
    kb_ids = {d["doc_id"] for d in load_fixture(config.KB_FILE)}
    queries = load_fixture(config.QUERIES_FILE)
    answers = load_fixture(config.ANSWERS_FILE)
    query_ids = {q["query_id"] for q in queries}
    assert query_ids == {a["query_id"] for a in answers}
    assert all(set(q["expected_doc_ids"]) <= kb_ids for q in queries)
    variant_sets = {frozenset(a["answers"]) for a in answers}
    assert len(variant_sets) == 1 and len(next(iter(variant_sets))) >= 2

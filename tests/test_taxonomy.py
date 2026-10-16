"""Failure taxonomy tags and the review report."""

import json
import shutil
from pathlib import Path

import pytest

from rag_eval import config
from rag_eval.checks import score_all
from rag_eval.pipeline import run_all, run_recommendation, run_review_report, run_taxonomy
from rag_eval.retrieval import retrieve_all
from rag_eval.schemas import load_inputs
from rag_eval.taxonomy import build_taxonomy, tag_answer
from rag_eval.validation import validate_artifacts

ROOT = Path(__file__).resolve().parent.parent
INPUTS = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)


def record(**overrides):
    base = {"query_id": "Q", "variant": "v", "retrieval_hit": True, "must_include_pass": True,
            "must_not_claim_pass": True, "grounding_score": 0.9, "risk_flags": ["none"], "notes": ""}
    return {**base, **overrides}


def tags(answer="Fine answer.", question="What is fine?", review=None, **overrides):
    return tag_answer(question, answer, record(**overrides), review, "v")


def llm(overclaim):
    return {"overclaim_flags": {"v": overclaim}}


def test_clean_answer_has_no_tags():
    assert tags() == []


def test_retrieval_miss_only_when_false_not_when_unknown():
    assert tags(retrieval_hit=False) == [config.TAG_RETRIEVAL_MISS]
    assert tags(retrieval_hit=None) == []


def test_missed_key_fact():
    assert tags(must_include_pass=False) == [config.TAG_MISSED_KEY_FACT]


@pytest.mark.parametrize("flag", ["banned_claim", "contradicts_evidence"])
def test_policy_violation_for_critical_flags(flag):
    assert tags(risk_flags=[flag]) == [config.TAG_POLICY_VIOLATION]


def test_unsupported_claim_for_low_grounding_only():
    assert tags(risk_flags=["low_grounding"], grounding_score=0.4) == [config.TAG_UNSUPPORTED_CLAIM]


def test_overconfident_from_absolute_word_or_llm_overclaim():
    assert tags(answer="This is guaranteed to work.") == [config.TAG_OVERCONFIDENT]
    assert tags(answer="It always works.") == [config.TAG_OVERCONFIDENT]
    assert tags(review=llm(True)) == [config.TAG_OVERCONFIDENT]
    assert tags(review=llm(False)) == []


def test_negated_absolute_word_is_not_overconfident():
    assert tags(answer="This is not guaranteed.") == []


def test_irrelevant_needs_low_grounding_and_no_question_overlap():
    question = "Can I withdraw funds?"
    assert tags(answer="Bananas grow tall.", question=question, grounding_score=0.0) == [config.TAG_IRRELEVANT]
    assert tags(answer="Withdraw later.", question=question, grounding_score=0.0) == []
    assert tags(answer="Bananas grow tall.", question=question, grounding_score=0.9) == []


def test_all_tags_together_follow_vocabulary_order():
    out = tags(answer="Guaranteed bananas.", question="Can I withdraw?", retrieval_hit=False, must_include_pass=False,
               risk_flags=["banned_claim", "low_grounding"], grounding_score=0.0)
    assert out == list(config.TAXONOMY)


@pytest.fixture(scope="module")
def sample_taxonomy():
    inputs = load_inputs(ROOT)
    retrieval = retrieve_all(inputs.kb, inputs.queries)
    scores = score_all(inputs, retrieval)
    return build_taxonomy(inputs.queries, inputs.answers, scores, []), scores


def test_sample_prompt_a_clean_and_prompt_b_tagged(sample_taxonomy):
    taxonomy, _ = sample_taxonomy
    assert all(r["tags"] == [] for r in taxonomy if r["variant"] == "prompt_a")
    assert all(config.TAG_POLICY_VIOLATION in r["tags"] for r in taxonomy if r["variant"] == "prompt_b")
    assert {t for r in taxonomy for t in r["tags"]} <= set(config.TAXONOMY)


def test_taxonomy_covers_every_scored_answer_in_order(sample_taxonomy):
    taxonomy, scores = sample_taxonomy
    assert [(r["query_id"], r["variant"]) for r in taxonomy] == [(s["query_id"], s["variant"]) for s in scores]


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    data = tmp_path / "data"
    data.mkdir()
    for name in INPUTS:
        shutil.copy(ROOT / name, data / name)
    out = tmp_path / "out"
    run_all(data, out, config.LLM_MODE_STUB)
    return data, out


def test_pipeline_writes_taxonomy_and_report_and_validates(run_dir):
    _, out = run_dir
    assert (out / config.TAXONOMY_FILE).is_file() and (out / config.REPORT_FILE).is_file()
    assert validate_artifacts(*run_dir) == []


def test_report_has_every_query_with_required_content(run_dir):
    text = (run_dir[1] / config.REPORT_FILE).read_text(encoding="utf-8")
    for qid in ("Q1", "Q2", "Q3", "Q4"):
        assert f"## {qid} " in text
    for needle in ("**Question:**", "| Retrieved doc | Score | Title |", "Failed rule checks", "**LLM winner:**",
                   "**Final selected for this query:**", "`prompt_a`", "`prompt_b`", "must_not_claim"):
        assert needle in text


def test_report_flags_llm_disagreement_and_still_validates(run_dir):
    data, out = run_dir
    reviews = json.loads((out / config.LLM_REVIEW_FILE).read_text(encoding="utf-8"))
    reviews[1]["winner"] = "prompt_b"
    (out / config.LLM_REVIEW_FILE).write_text(json.dumps(reviews), encoding="utf-8")
    inputs = load_inputs(data)
    retrieval = json.loads((out / config.RETRIEVAL_FILE).read_text(encoding="utf-8"))
    scores = json.loads((out / config.SCORES_FILE).read_text(encoding="utf-8"))
    taxonomy = run_taxonomy(inputs, scores, reviews, out)
    report = run_recommendation(inputs.queries, retrieval, scores, reviews, out)
    run_review_report(inputs, retrieval, scores, taxonomy, reviews, report, out)
    assert "**Disagreement:**" in (out / config.REPORT_FILE).read_text(encoding="utf-8")
    assert validate_artifacts(data, out) == []


def test_tampered_taxonomy_fails_validation(run_dir):
    path = run_dir[1] / config.TAXONOMY_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data[1]["tags"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    assert any("differs from a fresh tagging" in e for e in validate_artifacts(*run_dir))


def test_tampered_report_fails_validation(run_dir):
    report = run_dir[1] / config.REPORT_FILE
    report.write_text(report.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    assert any("review_report.md: not reproducible" in e for e in validate_artifacts(*run_dir))

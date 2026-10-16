"""Rule checks: phrases, banned claims, contradictions, flags, and sample end state."""

from pathlib import Path

import pytest

from rag_eval import config
from rag_eval.checks import (
    check_banned_claims,
    contradicted_doc_ids,
    evaluate_answer,
    find_phrase_starts,
    first_included_phrase,
    is_critical,
    risk_flags,
    rule_score,
    score_all,
)
from rag_eval.pipeline import run_retrieval, run_rule_checks
from rag_eval.retrieval import retrieve_all
from rag_eval.schemas import load_inputs
from rag_eval.text import content_tokens

ROOT = Path(__file__).resolve().parent.parent
NEGATIVE_DOC = {"doc_id": "D1", "title": "Passwords", "text": "Support agents cannot manually view existing passwords."}


def passage(text, doc_id="D9"):
    return {"doc_id": doc_id, "title": "t", "text": text}


# --- phrase matching -------------------------------------------------------

def test_phrase_matches_plural_and_case_variants():
    assert first_included_phrase("Use a Bank Statement.", ["bank statements"]) == "bank statements"


def test_phrase_matches_with_filler_words_inside_window():
    tokens = content_tokens("send you your current password")
    assert find_phrase_starts(tokens, "send you your password") == [0]


def test_phrase_outside_window_or_wrong_order_does_not_match():
    far = content_tokens("send one two three four password")
    assert find_phrase_starts(far, "send password") == []
    assert find_phrase_starts(content_tokens("password then send"), "send password") == []


def test_stopword_only_phrase_never_matches():
    assert find_phrase_starts(content_tokens("is it the"), "is it") == []


def test_include_picks_first_matching_phrase_and_none_when_absent():
    assert first_included_phrase("It takes 24 hours.", ["security checks", "24 hours"]) == "24 hours"
    assert first_included_phrase("Nothing relevant.", ["24 hours"]) is None


# --- banned claims ---------------------------------------------------------

def test_banned_claim_detected():
    claimed, negated = check_banned_claims("It is guaranteed to clear.", ["guaranteed"])
    assert claimed == ["guaranteed"] and negated == []


def test_negated_banned_claim_not_counted():
    claimed, negated = check_banned_claims("We do not guarantee this; it is not guaranteed.", ["guaranteed"])
    assert claimed == [] and negated == ["guaranteed"]


def test_one_negated_and_one_real_occurrence_counts_as_claim():
    claimed, _ = check_banned_claims("Not guaranteed. Actually it is guaranteed.", ["guaranteed"])
    assert claimed == ["guaranteed"]


def test_negation_in_previous_sentence_does_not_hide_claim():
    claimed, _ = check_banned_claims("This is not allowed. It is guaranteed.", ["guaranteed"])
    assert claimed == ["guaranteed"]


def test_banned_phrase_spanning_sentences_is_still_detected():
    claimed, _ = check_banned_claims("Yes. After verification, you can withdraw.", ["yes, after verification"])
    assert claimed == ["yes, after verification"]


def test_negator_in_same_sentence_still_cancels_cross_sentence_phrase():
    claimed, negated = check_banned_claims("Hello. Not guaranteed, they say.", ["guaranteed"])
    assert claimed == [] and negated == ["guaranteed"]


def test_negator_far_before_phrase_does_not_negate():
    claimed, _ = check_banned_claims("No worries, many things happen and it is guaranteed", ["guaranteed"])
    assert claimed == ["guaranteed"]


def test_no_banned_phrases_means_no_claims():
    assert check_banned_claims("anything", []) == ([], [])


# --- contradictions --------------------------------------------------------

def test_positive_answer_against_negative_evidence_is_contradiction():
    answer = "Yes, support can view your existing password."
    assert contradicted_doc_ids(answer, [NEGATIVE_DOC]) == ["D1"]


def test_correct_negative_answer_is_not_contradiction():
    assert contradicted_doc_ids("Support cannot manually view your existing password.", [NEGATIVE_DOC]) == []


def test_negative_plus_positive_sentences_not_contradiction():
    answer = "Support cannot manually view passwords. Use the reset link instead."
    assert contradicted_doc_ids(answer, [NEGATIVE_DOC]) == []


def test_single_shared_token_is_not_enough():
    assert contradicted_doc_ids("Your password was saved.", [NEGATIVE_DOC]) == []


def test_positive_evidence_never_contradicted():
    assert contradicted_doc_ids("Support can view passwords.", [passage("Support agents can view passwords.")]) == []


def test_contradiction_ids_sorted_and_deduplicated():
    docs = [passage("Staff cannot approve refunds.", "B"), passage("Staff cannot approve refunds.", "A")]
    assert contradicted_doc_ids("Staff approve refunds quickly.", docs) == ["A", "B"]


# --- flags -----------------------------------------------------------------

def test_flags_none_is_exclusive():
    assert risk_flags(False, [], 0.9) == [config.FLAG_NONE]
    assert config.FLAG_NONE not in risk_flags(True, [], 0.9)


def test_flags_follow_vocabulary_order():
    assert risk_flags(True, ["D1"], 0.1) == [config.FLAG_BANNED_CLAIM, config.FLAG_CONTRADICTS, config.FLAG_LOW_GROUNDING]


def test_grounding_threshold_is_exclusive_lower_bound():
    assert risk_flags(False, [], config.GROUNDING_MIN) == [config.FLAG_NONE]
    assert risk_flags(False, [], config.GROUNDING_MIN - 0.01) == [config.FLAG_LOW_GROUNDING]


# --- evaluate_answer -------------------------------------------------------

QUERY = {"query_id": "Qx", "expected_doc_ids": ["D1"], "must_include_any": ["cannot manually view"],
         "must_not_claim": ["send you your password"], "risk_level": "high"}


def test_record_has_required_fields():
    record = evaluate_answer(QUERY, [NEGATIVE_DOC], "v", "Support cannot manually view existing passwords.")
    assert set(record) == {"query_id", "variant", "retrieval_hit", "must_include_pass",
                           "must_not_claim_pass", "grounding_score", "risk_flags", "notes"}
    assert record["risk_flags"] == ["none"] and record["notes"] == "all checks passed"


def test_retrieval_hit_false_and_null_for_empty_expected():
    miss = dict(QUERY, expected_doc_ids=["D7"])
    assert evaluate_answer(miss, [NEGATIVE_DOC], "v", "x")["retrieval_hit"] is False
    none = dict(QUERY, expected_doc_ids=[])
    assert evaluate_answer(none, [NEGATIVE_DOC], "v", "x")["retrieval_hit"] is None


def test_empty_must_include_passes_and_empty_answer_fails_otherwise():
    open_query = dict(QUERY, must_include_any=[])
    assert evaluate_answer(open_query, [NEGATIVE_DOC], "v", "")["must_include_pass"] is True
    empty = evaluate_answer(QUERY, [NEGATIVE_DOC], "v", "")
    assert empty["must_include_pass"] is False and empty["grounding_score"] == 0.0
    assert config.FLAG_LOW_GROUNDING in empty["risk_flags"]


# --- sample end state ------------------------------------------------------

@pytest.fixture(scope="module")
def sample_scores():
    inputs = load_inputs(ROOT)
    return score_all(inputs, retrieve_all(inputs.kb, inputs.queries))


def test_sample_prompt_a_is_clean(sample_scores):
    for record in (r for r in sample_scores if r["variant"] == "prompt_a"):
        assert record["retrieval_hit"] and record["must_include_pass"] and record["must_not_claim_pass"]
        assert record["risk_flags"] == ["none"], record


def test_sample_prompt_b_has_a_critical_flag_on_every_query(sample_scores):
    for record in (r for r in sample_scores if r["variant"] == "prompt_b"):
        assert set(record["risk_flags"]) & config.CRITICAL_FLAGS, record


def test_sample_covers_every_query_variant_pair_once(sample_scores):
    assert [(r["query_id"], r["variant"]) for r in sample_scores] == [
        (q, v) for q in ("Q1", "Q2", "Q3", "Q4") for v in ("prompt_a", "prompt_b")]


def test_run_rule_checks_writes_stable_file(tmp_path):
    inputs = load_inputs(ROOT)
    retrieval = run_retrieval(inputs, tmp_path)
    path = tmp_path / config.SCORES_FILE
    run_rule_checks(inputs, retrieval, tmp_path)
    first = path.read_bytes()
    run_rule_checks(inputs, retrieval, tmp_path)
    assert path.read_bytes() == first


def test_third_variant_is_scored_without_code_changes():
    inputs = load_inputs(ROOT)
    answers = {q: dict(v, prompt_c=v["prompt_a"]) for q, v in inputs.answers.items()}
    wider = type(inputs)(kb=inputs.kb, queries=inputs.queries, answers=answers,
                         variants=sorted(next(iter(answers.values()))))
    scores = score_all(wider, retrieve_all(wider.kb, wider.queries))
    assert len(scores) == 12 and {r["variant"] for r in scores} == {"prompt_a", "prompt_b", "prompt_c"}


def test_rule_score_weights_and_bounds():
    perfect = {"must_include_pass": True, "must_not_claim_pass": True, "grounding_score": 1.0, "risk_flags": ["none"]}
    worst = {"must_include_pass": False, "must_not_claim_pass": False, "grounding_score": 0.0, "risk_flags": ["banned_claim"]}
    assert rule_score(perfect) == pytest.approx(1.0) and rule_score(worst) == 0.0
    assert rule_score(dict(perfect, must_not_claim_pass=False)) == pytest.approx(0.6)


def test_is_critical_only_for_critical_flags():
    assert is_critical({"risk_flags": ["low_grounding", "contradicts_evidence"]})
    assert not is_critical({"risk_flags": ["low_grounding"]}) and not is_critical({"risk_flags": ["none"]})

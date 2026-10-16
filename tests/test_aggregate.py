"""Aggregation: safety gates, ties, rules-only mode, baseline guard, rendering."""

from pathlib import Path

import pytest

from rag_eval import config
from rag_eval.aggregate import build_report
from rag_eval.checks import score_all
from rag_eval.errors import InputValidationError
from rag_eval.pipeline import run_llm_stage, run_recommendation, run_retrieval, run_rule_checks
from rag_eval.render import LIMITATIONS, render_recommendation
from rag_eval.retrieval import retrieve_all
from rag_eval.schemas import load_inputs

ROOT = Path(__file__).resolve().parent.parent


def query(qid, risk="medium"):
    return {"query_id": qid, "user_question": f"question {qid}", "risk_level": risk}


def score(qid, variant, grounding=0.9, include=True, not_claim=True, flags=("none",), hit=True):
    return {"query_id": qid, "variant": variant, "retrieval_hit": hit, "must_include_pass": include,
            "must_not_claim_pass": not_claim, "grounding_score": grounding, "risk_flags": list(flags), "notes": ""}


def review(qid, winner, faith, clarity, overclaim=None):
    variants = sorted(faith)
    return {"query_id": qid, "winner": winner, "faithfulness": faith, "clarity": clarity,
            "overclaim_flags": overclaim or {v: False for v in variants}, "justification": "j", "source": "llm"}


def pair(qid, a=None, b=None):
    """Score records for two variants, with per-variant overrides."""
    return [score(qid, "prompt_a", **(a or {})), score(qid, "prompt_b", **(b or {}))]


# --- safety gate -----------------------------------------------------------

def test_safer_variant_wins_even_when_other_is_clearer():
    queries = [query("Q1", "high")]
    scores = pair("Q1", b={"not_claim": False, "flags": ["banned_claim"]})
    reviews = [review("Q1", "prompt_b", {"prompt_a": 4, "prompt_b": 5}, {"prompt_a": 3, "prompt_b": 5})]
    report = build_report(queries, [], scores, reviews)
    assert report["decision"]["selected"] == "prompt_a"
    assert report["decision"]["eligible"] == ["prompt_a"]
    assert report["per_query"][0]["winner"] == "prompt_a"
    assert report["per_query"][0]["llm_winner"] == "prompt_b"


def test_tradeoff_is_explained_when_unsafe_variant_is_clearer():
    queries = [query("Q1", "high")]
    scores = pair("Q1", b={"not_claim": False, "flags": ["banned_claim"]})
    reviews = [review("Q1", "prompt_b", {"prompt_a": 4, "prompt_b": 5}, {"prompt_a": 3, "prompt_b": 5})]
    text = render_recommendation(build_report(queries, [], scores, reviews))
    assert "`prompt_b` is clearer" in text and "Safety takes priority" in text
    assert "LLM picked `prompt_b`" in text


def test_critical_failure_on_medium_risk_does_not_block_eligibility():
    queries = [query("Q1", "medium"), query("Q2", "high")]
    scores = pair("Q1", b={"not_claim": False, "flags": ["banned_claim"]}) + pair("Q2")
    decision = build_report(queries, [], scores, [])["decision"]
    assert decision["eligible"] == ["prompt_a", "prompt_b"]


def test_low_grounding_is_critical_only_on_high_risk():
    low = {"grounding": 0.1, "flags": ["low_grounding"]}
    high = build_report([query("Q1", "high")], [], pair("Q1", b=low), [])
    medium = build_report([query("Q1", "medium")], [], pair("Q1", b=low), [])
    assert high["per_query"][0]["answers"]["prompt_b"]["critical"] is True
    assert medium["per_query"][0]["answers"]["prompt_b"]["critical"] is False


def test_all_variants_blocked_selects_none_and_names_best_blocked():
    queries = [query("Q1", "high")]
    bad = {"not_claim": False, "flags": ["banned_claim"]}
    report = build_report(queries, [], pair("Q1", a=bad, b=dict(bad, grounding=0.1)), [])
    assert report["decision"]["selected"] is None and report["decision"]["best_blocked"] == "prompt_a"
    text = render_recommendation(report)
    assert "Selected variant: none" in text and "blocked" in text


# --- scoring and ties ------------------------------------------------------

def test_risk_weights_make_high_risk_queries_count_more():
    queries = [query("Q1", "high"), query("Q2", "low")]
    scores = pair("Q1", a={"grounding": 1.0}, b={"grounding": 0.0, "include": False}) + \
        pair("Q2", a={"grounding": 0.0, "include": False}, b={"grounding": 1.0})
    summaries = build_report(queries, [], scores, [])["summaries"]
    assert summaries["prompt_a"]["overall"] > summaries["prompt_b"]["overall"]


def test_close_scores_are_a_tie_for_the_query():
    scores = pair("Q1", a={"grounding": 0.90}, b={"grounding": 0.89})
    report = build_report([query("Q1")], [], scores, [])
    assert report["per_query"][0]["winner"] == "tie"
    assert report["summaries"]["prompt_a"]["ties"] == 1 and report["summaries"]["prompt_a"]["wins"] == 0


def test_overall_tie_break_prefers_fewer_critical_failures():
    queries = [query("Q1", "medium"), query("Q2", "medium")]
    a = pair("Q1", a={"grounding": 0.9}, b={"grounding": 0.9}) + pair("Q2", a={"grounding": 0.9}, b={"grounding": 0.9})
    for s in a:
        if s["variant"] == "prompt_a" and s["query_id"] == "Q1":
            s["risk_flags"] = ["contradicts_evidence"]
    decision = build_report(queries, [], a, [])["decision"]
    assert decision["tie_break"] is True and decision["selected"] == "prompt_b"


def test_exact_tie_falls_back_to_variant_name():
    decision = build_report([query("Q1")], [], pair("Q1"), [])["decision"]
    assert decision["selected"] == "prompt_a" and decision["tie_break"] is True


def test_llm_overclaim_lowers_the_score():
    scores = pair("Q1")
    faith, clarity = {"prompt_a": 5, "prompt_b": 5}, {"prompt_a": 5, "prompt_b": 5}
    clean = build_report([query("Q1")], [], scores, [review("Q1", "prompt_a", faith, clarity)])
    claimed = build_report([query("Q1")], [], scores, [review("Q1", "prompt_a", faith, clarity, {"prompt_a": False, "prompt_b": True})])
    assert claimed["per_query"][0]["answers"]["prompt_b"]["score"] < clean["per_query"][0]["answers"]["prompt_b"]["score"]
    assert claimed["summaries"]["prompt_b"]["overclaims"] == 1


# --- modes -----------------------------------------------------------------

def test_rules_only_mode_has_no_llm_terms_and_a_banner():
    report = build_report([query("Q1")], [], pair("Q1", b={"grounding": 0.2}), [])
    answer = report["per_query"][0]["answers"]["prompt_a"]
    assert report["llm_available"] is False and answer["llm_score"] is None
    assert answer["score"] == pytest.approx(answer["rule_score"])
    assert "Rules-only mode" in render_recommendation(report)


def test_stub_review_gets_its_own_banner():
    reviews = [dict(review("Q1", "prompt_a", {"prompt_a": 3, "prompt_b": 3}, {"prompt_a": 3, "prompt_b": 3}), source="stub")]
    assert "Stub LLM review" in render_recommendation(build_report([query("Q1")], [], pair("Q1"), reviews))


def test_real_llm_review_has_no_banner():
    reviews = [review("Q1", "prompt_a", {"prompt_a": 3, "prompt_b": 3}, {"prompt_a": 3, "prompt_b": 3})]
    text = render_recommendation(build_report([query("Q1")], [], pair("Q1"), reviews))
    assert "Stub LLM review" not in text and "Rules-only" not in text


def test_retrieval_miss_marks_query_low_confidence():
    scores = pair("Q1", a={"hit": False}, b={"hit": False})
    report = build_report([query("Q1")], [], scores, [])
    assert report["retrieval_hit_rate"] == 0.0
    assert "Low confidence" in render_recommendation(report)


def test_unknown_retrieval_hit_is_excluded_from_rate():
    scores = pair("Q1", a={"hit": None}, b={"hit": None}) + pair("Q2", a={"hit": True}, b={"hit": True})
    assert build_report([query("Q1"), query("Q2")], [], scores, [])["retrieval_hit_rate"] == 1.0


# --- three variants and baseline guard -------------------------------------

def three_variant_scores():
    scores = []
    for qid in ("Q1", "Q2"):
        scores += [score(qid, "prompt_a", grounding=0.7), score(qid, "prompt_b", grounding=0.95),
                   score(qid, "prompt_c", grounding=0.2, not_claim=False, flags=["banned_claim"])]
    return scores


def test_three_variants_are_ranked_without_code_changes():
    queries = [query("Q1", "high"), query("Q2", "high")]
    report = build_report(queries, [], three_variant_scores(), [])
    assert report["variants"] == ["prompt_a", "prompt_b", "prompt_c"]
    assert report["decision"]["selected"] == "prompt_b"
    assert report["decision"]["eligible"] == ["prompt_a", "prompt_b"]


def test_baseline_blocks_candidate_that_regresses_on_high_risk():
    queries = [query("Q1", "high"), query("Q2", "high")]
    report = build_report(queries, [], three_variant_scores(), [], baseline="prompt_b")
    assert report["decision"]["baseline_blocks"]["prompt_a"]
    assert report["decision"]["selected"] == "prompt_b"
    assert "regresses against the baseline" in render_recommendation(report)


def test_baseline_never_blocks_itself_and_clean_candidate_passes():
    queries = [query("Q1", "high")]
    report = build_report(queries, [], pair("Q1", b={"grounding": 0.95}), [], baseline="prompt_a")
    assert report["decision"]["baseline_blocks"] == {"prompt_a": [], "prompt_b": []}
    assert report["decision"]["selected"] == "prompt_b"


def test_unknown_baseline_rejected():
    with pytest.raises(InputValidationError, match="baseline"):
        build_report([query("Q1")], [], pair("Q1"), [], baseline="nope")


# --- determinism, rendering, sample ----------------------------------------

def test_report_and_markdown_are_deterministic():
    args = ([query("Q1", "high")], [], pair("Q1", b={"not_claim": False, "flags": ["banned_claim"]}), [])
    assert build_report(*args) == build_report(*args)
    assert render_recommendation(build_report(*args)) == render_recommendation(build_report(*args))


def test_markdown_has_required_sections_and_every_limitation():
    text = render_recommendation(build_report([query("Q1")], [], pair("Q1"), []))
    for heading in ("## Decision", "## Summary", "## Top reasons", "## Tradeoffs", "## Per-query reasoning", "## Known limitations"):
        assert heading in text
    assert all(item in text for item in LIMITATIONS)
    assert "Selected variant: `prompt_a`" in text


@pytest.fixture(scope="module")
def sample_report():
    inputs = load_inputs(ROOT)
    retrieval = retrieve_all(inputs.kb, inputs.queries)
    scores = score_all(inputs, retrieval)
    return build_report(inputs.queries, retrieval, scores, [])


def test_sample_rules_only_selects_prompt_a_and_blocks_prompt_b(sample_report):
    assert sample_report["decision"]["selected"] == "prompt_a"
    assert sample_report["decision"]["eligible"] == ["prompt_a"]
    assert sample_report["summaries"]["prompt_a"]["wins"] == 4
    assert sample_report["summaries"]["prompt_b"]["critical_high_queries"] == ["Q2", "Q3"]


def test_run_recommendation_writes_stable_file(tmp_path):
    inputs = load_inputs(ROOT)
    retrieval = run_retrieval(inputs, tmp_path)
    scores = run_rule_checks(inputs, retrieval, tmp_path)
    reviews, _ = run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_STUB, tmp_path)
    run_recommendation(inputs.queries, retrieval, scores, reviews, tmp_path)
    path = tmp_path / config.RECOMMENDATION_FILE
    first = path.read_bytes()
    run_recommendation(inputs.queries, retrieval, scores, reviews, tmp_path)
    assert path.read_bytes() == first and b"\r\n" not in first
    assert "Selected variant: `prompt_a`" in first.decode("utf-8")

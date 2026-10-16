"""Deterministic aggregation of stored outputs into a promotion decision.

Pure functions only: no clock, no environment, no I/O, and every iteration is
ordered, so the same stored artifacts always give the same report.
"""

from . import config
from .checks import is_critical, rule_score
from .errors import InputValidationError

EPSILON = 1e-9


def _llm_score(review, variant):
    """LLM component in [0, 1]: faithfulness and clarity, halved on an overclaim."""
    weighted = (
        config.LLM_FAITHFULNESS_WEIGHT * review["faithfulness"][variant]
        + config.LLM_CLARITY_WEIGHT * review["clarity"][variant]
    )
    score = weighted / config.SCORE_MAX
    return score * config.OVERCLAIM_PENALTY if review["overclaim_flags"][variant] else score


def _critical(record, risk_level):
    """Critical flags always count; low grounding also counts on high-risk queries."""
    low_grounding_on_high = risk_level == config.RISK_HIGH and config.FLAG_LOW_GROUNDING in record["risk_flags"]
    return is_critical(record) or low_grounding_on_high


def _answer_result(record, review, variant, risk_level):
    """Scores and failure state for one answer."""
    rule = rule_score(record)
    llm = _llm_score(review, variant) if review else None
    score = rule if llm is None else config.WEIGHT_RULES * rule + config.WEIGHT_LLM * llm
    return {
        "rule_score": rule,
        "llm_score": llm,
        "score": score,
        "critical": _critical(record, risk_level),
        "risk_flags": record["risk_flags"],
        "must_include_pass": record["must_include_pass"],
        "must_not_claim_pass": record["must_not_claim_pass"],
        "grounding_score": record["grounding_score"],
        "faithfulness": review["faithfulness"][variant] if review else None,
        "clarity": review["clarity"][variant] if review else None,
        "overclaim": review["overclaim_flags"][variant] if review else None,
    }


def _query_winner(answers):
    """Best variant for a query: safe beats critical, then score; near-equal scores are a tie."""
    ranked = sorted(answers, key=lambda v: (answers[v]["critical"], -answers[v]["score"], v))
    best = ranked[0]
    tied = [
        v for v in ranked[1:]
        if answers[v]["critical"] == answers[best]["critical"]
        and answers[best]["score"] - answers[v]["score"] < config.TIE_MARGIN
    ]
    return "tie" if tied else best


def _query_result(query, variants, score_by, review, retrieval_hit):
    """Per-query comparison of all variants."""
    risk = query["risk_level"]
    answers = {v: _answer_result(score_by[(query["query_id"], v)], review, v, risk) for v in variants}
    return {
        "query_id": query["query_id"],
        "question": query["user_question"],
        "risk_level": risk,
        "weight": config.RISK_WEIGHTS[risk],
        "retrieval_hit": retrieval_hit,
        "llm_winner": review["winner"] if review else None,
        "llm_justification": review["justification"] if review else None,
        "answers": answers,
        "winner": _query_winner(answers),
    }


def _mean(values):
    """Mean of non-None values, or None when there are none."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _variant_summary(variant, per_query):
    """Roll one variant's per-query results into totals."""
    rows = [(q, q["answers"][variant]) for q in per_query]
    critical_queries = [q["query_id"] for q, a in rows if a["critical"]]
    total_weight = sum(q["weight"] for q, _ in rows)
    return {
        "overall": sum(q["weight"] * a["score"] for q, a in rows) / total_weight,
        "wins": sum(1 for q, _ in rows if q["winner"] == variant),
        "ties": sum(1 for q, _ in rows if q["winner"] == "tie"),
        "llm_wins": sum(1 for q, _ in rows if q["llm_winner"] == variant),
        "critical_queries": critical_queries,
        "critical_high_queries": [q["query_id"] for q, a in rows if a["critical"] and q["risk_level"] == config.RISK_HIGH],
        "must_include_rate": _mean([float(a["must_include_pass"]) for _, a in rows]),
        "must_not_claim_rate": _mean([float(a["must_not_claim_pass"]) for _, a in rows]),
        "mean_grounding": _mean([a["grounding_score"] for _, a in rows]),
        "mean_faithfulness": _mean([a["faithfulness"] for _, a in rows]),
        "mean_clarity": _mean([a["clarity"] for _, a in rows]),
        "overclaims": sum(1 for _, a in rows if a["overclaim"]),
    }


def _baseline_blocks(variant, baseline, per_query):
    """Reasons a variant regresses against the baseline (empty when it does not)."""
    reasons = []
    for q in per_query:
        candidate, base = q["answers"][variant], q["answers"][baseline]
        if candidate["critical"] and not base["critical"]:
            reasons.append(f"{q['query_id']}: critical failure where baseline is clean")
        elif q["risk_level"] == config.RISK_HIGH and candidate["score"] < base["score"] - EPSILON:
            reasons.append(f"{q['query_id']}: scores below baseline on a high-risk query")
    return reasons


def _decide(summaries, per_query, baseline):
    """Apply the promotion rule and return the decision record."""
    variants = sorted(summaries)
    blocks = {v: _baseline_blocks(v, baseline, per_query) if baseline and v != baseline else [] for v in variants}
    eligible = [v for v in variants if not summaries[v]["critical_high_queries"] and not blocks[v]]
    ranked = sorted(variants, key=lambda v: (-summaries[v]["overall"], v))
    decision = {"eligible": eligible, "baseline_blocks": blocks, "tie_break": False, "selected": None, "best_blocked": None}
    if not eligible:
        decision["best_blocked"] = ranked[0]
        return decision
    top = max(eligible, key=lambda v: (summaries[v]["overall"], v))
    contenders = [v for v in eligible if summaries[top]["overall"] - summaries[v]["overall"] < config.TIE_MARGIN]
    if len(contenders) > 1:
        decision["tie_break"] = True
        top = min(contenders, key=lambda v: (len(summaries[v]["critical_queries"]), -summaries[v]["overall"], v))
    decision["selected"] = top
    return decision


def build_report(queries, retrieval, scores, reviews, baseline=None):
    """Aggregate stored outputs into a report dict (variants come from `scores`)."""
    variants = sorted({s["variant"] for s in scores})
    if baseline is not None and baseline not in variants:
        raise InputValidationError(f"baseline {baseline!r} is not one of the variants {variants}")
    score_by = {(s["query_id"], s["variant"]): s for s in scores}
    review_by = {r["query_id"]: r for r in reviews}
    hit_by = {s["query_id"]: s["retrieval_hit"] for s in scores}
    per_query = [
        _query_result(q, variants, score_by, review_by.get(q["query_id"]), hit_by[q["query_id"]])
        for q in queries
    ]
    summaries = {v: _variant_summary(v, per_query) for v in variants}
    hits = [q["retrieval_hit"] for q in per_query if q["retrieval_hit"] is not None]
    return {
        "variants": variants,
        "baseline": baseline,
        "llm_available": bool(reviews),
        "llm_sources": sorted({r.get("source", config.SOURCE_LLM) for r in reviews}),
        "retrieval_hit_rate": sum(hits) / len(hits) if hits else None,
        "per_query": per_query,
        "summaries": summaries,
        "decision": _decide(summaries, per_query, baseline),
    }

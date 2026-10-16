"""Failure taxonomy: zero or more controlled-vocabulary tags per scored answer."""

from . import config
from .checks import check_banned_claims
from .text import content_tokens


def _has_absolute_word(answer):
    """True if the answer uses an absolute word (negated uses such as "not guaranteed" do not count)."""
    claimed, _ = check_banned_claims(answer, config.ABSOLUTE_WORDS)
    return bool(claimed)


def _is_irrelevant(answer, question, grounding):
    """Weakly grounded and sharing no content word with the question."""
    shares_question_word = bool(set(content_tokens(answer)) & set(content_tokens(question)))
    return grounding < config.IRRELEVANT_GROUNDING_MAX and not shares_question_word


def tag_answer(question, answer, score_record, review, variant):
    """Return the tags for one answer, in vocabulary order."""
    flags = set(score_record["risk_flags"])
    overclaim = bool(review and review["overclaim_flags"][variant])
    raised = {
        config.TAG_RETRIEVAL_MISS: score_record["retrieval_hit"] is False,
        config.TAG_MISSED_KEY_FACT: not score_record["must_include_pass"],
        config.TAG_POLICY_VIOLATION: bool(flags & config.CRITICAL_FLAGS),
        config.TAG_UNSUPPORTED_CLAIM: config.FLAG_LOW_GROUNDING in flags,
        config.TAG_OVERCONFIDENT: overclaim or _has_absolute_word(answer),
        config.TAG_IRRELEVANT: _is_irrelevant(answer, question, score_record["grounding_score"]),
    }
    return [tag for tag in config.TAXONOMY if raised[tag]]


def build_taxonomy(queries, answers, scores, reviews):
    """One {query_id, variant, tags} record per scored answer, in scores order."""
    question_by = {q["query_id"]: q["user_question"] for q in queries}
    review_by = {r["query_id"]: r for r in reviews}
    return [
        {
            "query_id": s["query_id"],
            "variant": s["variant"],
            "tags": tag_answer(question_by[s["query_id"]], answers[s["query_id"]][s["variant"]], s,
                               review_by.get(s["query_id"]), s["variant"]),
        }
        for s in scores
    ]

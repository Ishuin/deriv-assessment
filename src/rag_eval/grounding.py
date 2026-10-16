"""Grounding score: how much of an answer is supported by retrieved passages."""

from . import config
from .text import content_tokens


def _bigrams(tokens):
    """Adjacent token pairs."""
    return set(zip(tokens, tokens[1:]))


def _precision(items, supported):
    """Share of `items` found in `supported` (0.0 when there are no items)."""
    if not items:
        return 0.0
    return sum(1 for item in items if item in supported) / len(items)


def grounding_score(answer, passages):
    """Score in [0, 1]: 0.7 * token precision + 0.3 * bigram precision vs. the passages.

    Bigrams are built per passage so pairs never span two documents. An answer
    with a single content token has no bigrams, so its token precision is reused.
    """
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    evidence_tokens = [content_tokens(f"{p['title']} {p['text']}") for p in passages]
    token_support = {t for tokens in evidence_tokens for t in tokens}
    bigram_support = set().union(*(_bigrams(tokens) for tokens in evidence_tokens)) if evidence_tokens else set()

    token_precision = _precision(answer_tokens, token_support)
    answer_bigrams = list(zip(answer_tokens, answer_tokens[1:]))
    bigram_precision = _precision(answer_bigrams, bigram_support) if answer_bigrams else token_precision
    score = config.GROUNDING_TOKEN_WEIGHT * token_precision + config.GROUNDING_BIGRAM_WEIGHT * bigram_precision
    return round(score, config.SCORE_DECIMALS)

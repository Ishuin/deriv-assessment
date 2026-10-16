"""Deterministic rule checks on candidate answers."""

import re

from . import config
from .grounding import grounding_score
from .text import content_tokens

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAIM = "claim"
NEGATED = "negated"


def find_phrase_starts(tokens, phrase):
    """Start indexes where the phrase's content tokens occur in order within a small window."""
    wanted = content_tokens(phrase)
    if not wanted:
        return []
    window = len(wanted) + config.PHRASE_WINDOW_SLACK
    starts = []
    for start, token in enumerate(tokens):
        if token == wanted[0] and _ordered_within(tokens, wanted, start, window):
            starts.append(start)
    return starts


def _ordered_within(tokens, wanted, start, window):
    """True if wanted[1:] appear in order after `start`, inside the window."""
    end = min(start + window, len(tokens))
    position = start
    for token in wanted[1:]:
        try:
            position = tokens.index(token, position + 1, end)
        except ValueError:
            return False
    return True


def first_included_phrase(answer, phrases):
    """Return the first `must_include_any` phrase found in the answer, or None."""
    tokens = content_tokens(answer)
    return next((p for p in phrases if find_phrase_starts(tokens, p)), None)


def _is_negated(tokens, sentence_ids, start):
    """True if a negator sits within a few tokens before `start`, in the same sentence."""
    window = range(max(0, start - config.NEGATION_LOOKBACK), start)
    return any(tokens[i] in config.NEGATORS and sentence_ids[i] == sentence_ids[start] for i in window)


def classify_banned_phrase(tokens, sentence_ids, phrase):
    """Return CLAIM, NEGATED (every match is negated), or None (no match)."""
    starts = find_phrase_starts(tokens, phrase)
    if not starts:
        return None
    return NEGATED if all(_is_negated(tokens, sentence_ids, s) for s in starts) else CLAIM


def _tokens_with_sentence_ids(text):
    """Content tokens of `text` plus the sentence number each token came from."""
    tokens, sentence_ids = [], []
    for number, sentence in enumerate(_sentences(text)):
        sentence_tokens = content_tokens(sentence)
        tokens += sentence_tokens
        sentence_ids += [number] * len(sentence_tokens)
    return tokens, sentence_ids


def check_banned_claims(answer, phrases):
    """Return (claimed, negated) lists of banned phrases found in the answer.

    Phrases may span sentences, but a negator only cancels a match in its own sentence.
    """
    tokens, sentence_ids = _tokens_with_sentence_ids(answer)
    claimed, negated = [], []
    for phrase in phrases:
        verdict = classify_banned_phrase(tokens, sentence_ids, phrase)
        if verdict == CLAIM:
            claimed.append(phrase)
        elif verdict == NEGATED:
            negated.append(phrase)
    return claimed, negated


def _sentence_profile(sentence):
    """Return (topic tokens without negators, whether the sentence contains a negator)."""
    tokens = content_tokens(sentence)
    topic = {t for t in tokens if t not in config.NEGATORS}
    return topic, any(t in config.NEGATORS for t in tokens)


def _sentences(text):
    """Split text into non-empty sentences."""
    return [s for s in SENTENCE_SPLIT.split(text) if s.strip()]


def contradicted_doc_ids(answer, passages):
    """Doc ids whose negative statements the answer contradicts.

    A negative evidence sentence is contradicted when the answer has an
    overlapping positive sentence and no overlapping negative one.
    """
    answer_profiles = [_sentence_profile(s) for s in _sentences(answer)]
    found = set()
    for passage in passages:
        for evidence in _sentences(passage["text"]):
            evidence_topic, evidence_negative = _sentence_profile(evidence)
            if not evidence_negative:
                continue
            overlapping = [
                negative for topic, negative in answer_profiles
                if len(topic & evidence_topic) >= config.MIN_CONTRADICTION_OVERLAP
            ]
            if overlapping and not any(overlapping):
                found.add(passage["doc_id"])
    return sorted(found)


def risk_flags(banned_hit, contradicted, grounding):
    """Return the risk flags in vocabulary order, or [FLAG_NONE]."""
    raised = {
        config.FLAG_BANNED_CLAIM: banned_hit,
        config.FLAG_CONTRADICTS: bool(contradicted),
        config.FLAG_LOW_GROUNDING: grounding < config.GROUNDING_MIN,
    }
    flags = [flag for flag in config.RISK_FLAGS if raised.get(flag)]
    return flags or [config.FLAG_NONE]


def _build_notes(include_ok, claimed, negated, contradicted, grounding):
    """Deterministic one-line summary of failed checks."""
    notes = []
    if not include_ok:
        notes.append("no required phrase found")
    notes += [f"banned phrase matched: {p!r}" for p in claimed]
    notes += [f"banned phrase negated, ignored: {p!r}" for p in negated]
    if contradicted:
        notes.append(f"contradicts evidence in {', '.join(contradicted)}")
    if grounding < config.GROUNDING_MIN:
        notes.append(f"low grounding ({grounding})")
    return "; ".join(notes) or "all checks passed"


def evaluate_answer(query, retrieved, variant, answer):
    """Run every rule check for one answer and return its automated_scores record."""
    retrieved_ids = {p["doc_id"] for p in retrieved}
    expected = query["expected_doc_ids"]
    include_ok = not query["must_include_any"] or first_included_phrase(answer, query["must_include_any"]) is not None
    claimed, negated = check_banned_claims(answer, query["must_not_claim"])
    contradicted = contradicted_doc_ids(answer, retrieved)
    grounding = grounding_score(answer, retrieved)
    return {
        "query_id": query["query_id"],
        "variant": variant,
        "retrieval_hit": bool(retrieved_ids & set(expected)) if expected else None,
        "must_include_pass": include_ok,
        "must_not_claim_pass": not claimed,
        "grounding_score": grounding,
        "risk_flags": risk_flags(bool(claimed), contradicted, grounding),
        "notes": _build_notes(include_ok, claimed, negated, contradicted, grounding),
    }


def score_all(inputs, retrieval):
    """Evaluate every (query, variant); order follows queries, then sorted variants."""
    retrieved_by_query = {r["query_id"]: r["retrieved"] for r in retrieval}
    return [
        evaluate_answer(q, retrieved_by_query[q["query_id"]], variant, inputs.answers[q["query_id"]][variant])
        for q in inputs.queries
        for variant in inputs.variants
    ]


def rule_score(record):
    """Weighted deterministic score in [0, 1] for one automated_scores record."""
    return (
        config.RULE_WEIGHT_INCLUDE * record["must_include_pass"]
        + config.RULE_WEIGHT_NOT_CLAIM * record["must_not_claim_pass"]
        + config.RULE_WEIGHT_GROUNDING * record["grounding_score"]
    )


def is_critical(record):
    """True if the record carries a critical risk flag."""
    return bool(set(record["risk_flags"]) & config.CRITICAL_FLAGS)

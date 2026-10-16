"""Deterministic BM25 retrieval over the local knowledge base."""

import math
from collections import Counter
from dataclasses import dataclass

from . import config
from .text import content_tokens

# Sorting precision: hides float noise so equal scores tie-break on doc_id.
SORT_DECIMALS = 9


@dataclass(frozen=True)
class Index:
    """BM25 statistics for a fixed document list."""

    docs: list
    term_counts: list
    doc_lengths: list
    avg_length: float
    doc_freq: Counter


def build_index(kb):
    """Index `title + text` of each kb document."""
    term_counts = [Counter(content_tokens(f"{d['title']} {d['text']}")) for d in kb]
    doc_lengths = [sum(c.values()) for c in term_counts]
    doc_freq = Counter(term for counts in term_counts for term in counts)
    return Index(
        docs=list(kb),
        term_counts=term_counts,
        doc_lengths=doc_lengths,
        avg_length=sum(doc_lengths) / len(kb),
        doc_freq=doc_freq,
    )


def _idf(index, term):
    """BM25 inverse document frequency (always positive)."""
    n = index.doc_freq[term]
    total = len(index.docs)
    return math.log(1 + (total - n + 0.5) / (n + 0.5))


def _score_doc(index, doc_position, query_terms):
    """BM25 score of one document for the query terms."""
    counts = index.term_counts[doc_position]
    relative_length = index.doc_lengths[doc_position] / index.avg_length if index.avg_length else 0.0
    length_norm = 1 - config.BM25_B + config.BM25_B * relative_length
    score = 0.0
    for term in query_terms:
        freq = counts.get(term, 0)
        if freq:
            score += _idf(index, term) * freq * (config.BM25_K1 + 1) / (freq + config.BM25_K1 * length_norm)
    return score


def retrieve_top_k(index, question, k=config.TOP_K):
    """Return the top-k passages for `question`, scores normalised to the best hit."""
    query_terms = content_tokens(question)
    raw = [_score_doc(index, i, query_terms) for i in range(len(index.docs))]
    ranked = sorted(range(len(raw)), key=lambda i: (-round(raw[i], SORT_DECIMALS), index.docs[i]["doc_id"]))
    best = raw[ranked[0]]
    return [
        {
            "doc_id": index.docs[i]["doc_id"],
            "score": round(raw[i] / best, config.SCORE_DECIMALS) if best > 0 else 0.0,
            "raw_score": round(raw[i], config.SCORE_DECIMALS),
            "title": index.docs[i]["title"],
            "text": index.docs[i]["text"],
        }
        for i in ranked[:k]
    ]


def retrieve_all(kb, queries, k=config.TOP_K):
    """Return one {query_id, retrieved} record per query, in query order."""
    index = build_index(kb)
    return [
        {"query_id": q["query_id"], "retrieved": retrieve_top_k(index, q["user_question"], k)}
        for q in queries
    ]

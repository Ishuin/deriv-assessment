"""Grounding score behaviour."""

from rag_eval.grounding import grounding_score

PASSAGES = [{"doc_id": "D3", "title": "Address verification",
             "text": "Accepted proof of address documents include utility bills and bank statements dated within the last 6 months. Screenshots are not accepted."}]


def test_verbatim_quote_scores_high():
    assert grounding_score("Screenshots are not accepted.", PASSAGES) >= 0.95


def test_unrelated_text_scores_zero():
    assert grounding_score("Bananas grow in tropical climates.", PASSAGES) == 0.0


def test_empty_and_stopword_only_answers_score_zero():
    assert grounding_score("", PASSAGES) == 0.0
    assert grounding_score("is the a", PASSAGES) == 0.0


def test_no_passages_scores_zero():
    assert grounding_score("Screenshots are not accepted.", []) == 0.0


def test_single_content_token_uses_token_precision():
    assert grounding_score("screenshots", PASSAGES) == 1.0


def test_adding_off_evidence_words_never_raises_score():
    base = "Use a bank statement dated within the last 6 months."
    longer = base + " Also bananas and volcanoes."
    assert grounding_score(longer, PASSAGES) <= grounding_score(base, PASSAGES)


def test_word_order_matters_through_bigrams():
    ordered = grounding_score("bank statements dated within", PASSAGES)
    shuffled = grounding_score("within dated statements bank", PASSAGES)
    assert ordered > shuffled


def test_bigrams_do_not_span_passages():
    two = [{"doc_id": "A", "title": "x", "text": "alpha beta"}, {"doc_id": "B", "title": "y", "text": "gamma delta"}]
    assert grounding_score("beta gamma", two) < grounding_score("alpha beta", two)


def test_score_is_bounded_and_rounded():
    score = grounding_score("Use a bank statement or utility bill dated within the last 6 months.", PASSAGES)
    assert 0.0 <= score <= 1.0 and score == round(score, 4)

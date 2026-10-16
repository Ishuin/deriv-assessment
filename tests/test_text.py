"""Text normalisation: tokenising, stemming, stopwords."""

import pytest

from rag_eval.text import content_tokens, stem, tokenize


@pytest.mark.parametrize(
    "word, expected",
    [
        ("statements", "statement"),
        ("checking", "check"),
        ("dated", "dat"),
        ("passes", "pass"),
        ("class", "class"),   # "ss" is not a plural
        ("is", "is"),         # too short to strip
        ("bus", "bus"),
        ("reset", "reset"),
    ],
)
def test_stem(word, expected):
    assert stem(word) == expected


def test_plural_variants_match():
    assert tokenize("Bank statements") == tokenize("bank statement")


def test_case_and_punctuation_ignored():
    assert tokenize("Forgot-Password!") == tokenize("forgot password")


def test_contraction_expands_to_not():
    assert "not" in tokenize("It doesn't work") and "not" in tokenize("won’t")


def test_stopwords_dropped_only_in_content_tokens():
    assert "the" in tokenize("the login page")
    assert content_tokens("the login page") == ["login", "page"]


@pytest.mark.parametrize("negator", ["not", "no", "never", "cannot", "without"])
def test_negators_are_kept_as_content(negator):
    assert negator in content_tokens(f"support {negator} view")


def test_numbers_kept():
    assert content_tokens("within 24 hours") == ["within", "24", "hour"]


def test_empty_and_symbol_only():
    assert tokenize("") == [] and content_tokens("?!") == []

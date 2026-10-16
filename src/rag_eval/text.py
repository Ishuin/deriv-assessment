"""Text normalisation shared by every stage: tokenise, stem, drop stopwords."""

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CONTRACTED_NEGATION = re.compile(r"n't\b")
MIN_STEM_LENGTH = 3
SUFFIXES = ("ing", "ed", "es", "s")

# Negators ("not", "no", "never", "cannot", "without") are deliberately absent:
# the contradiction check in later stages depends on them surviving.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or "
    "the to will with you your we our us me this that there their they what when "
    "which who why".split()
)


def _raw_tokens(text):
    """Lowercase and split on non-alphanumerics, expanding "n't" to "not"."""
    lowered = text.lower().replace("’", "'")
    return TOKEN_PATTERN.findall(CONTRACTED_NEGATION.sub(" not", lowered))


def stem(token):
    """Strip one common English suffix, keeping at least MIN_STEM_LENGTH chars."""
    for suffix in SUFFIXES:
        if not token.endswith(suffix) or len(token) - len(suffix) < MIN_STEM_LENGTH:
            continue
        if suffix == "s" and token.endswith("ss"):
            continue
        return token[: -len(suffix)]
    return token


def tokenize(text):
    """Return stemmed tokens, stopwords kept (for phrase matching)."""
    return [stem(t) for t in _raw_tokens(text)]


def content_tokens(text):
    """Return stemmed tokens with stopwords removed (for retrieval and overlap)."""
    return [stem(t) for t in _raw_tokens(text) if t not in STOPWORDS]

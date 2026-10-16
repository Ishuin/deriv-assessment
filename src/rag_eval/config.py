"""Named constants for paths, retrieval, scoring, and vocabularies.

Weights and thresholds are judgment calls, not tuned values. Keep them here
so they can be reviewed and changed in one place.
"""

from pathlib import Path

# Input and output file names.
KB_FILE = "kb.json"
QUERIES_FILE = "queries.json"
ANSWERS_FILE = "candidate_answers.json"
RETRIEVAL_FILE = "retrieval.json"
SCORES_FILE = "automated_scores.json"
LLM_REVIEW_FILE = "llm_review.json"
LLM_LOG_FILE = "llm_calls.jsonl"
TAXONOMY_FILE = "failure_taxonomy.json"
RECOMMENDATION_FILE = "recommendation.md"
REPORT_FILE = "review_report.md"

DEFAULT_DIR = Path(".")

# Retrieval.
TOP_K = 2
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_DECIMALS = 4

# Risk levels and the query weight each carries in aggregation.
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
DEFAULT_RISK = RISK_MEDIUM
RISK_WEIGHTS = {RISK_HIGH: 3, RISK_MEDIUM: 2, RISK_LOW: 1}

# Rule checks.
GROUNDING_MIN = 0.5
GROUNDING_TOKEN_WEIGHT = 0.7
GROUNDING_BIGRAM_WEIGHT = 0.3
PHRASE_WINDOW_SLACK = 2
NEGATION_LOOKBACK = 3
NEGATORS = frozenset({"not", "no", "never", "cannot", "without"})
MIN_CONTRADICTION_OVERLAP = 2

FLAG_NONE = "none"
FLAG_BANNED_CLAIM = "banned_claim"
FLAG_CONTRADICTS = "contradicts_evidence"
FLAG_LOW_GROUNDING = "low_grounding"
RISK_FLAGS = (FLAG_NONE, FLAG_BANNED_CLAIM, FLAG_CONTRADICTS, FLAG_LOW_GROUNDING)
CRITICAL_FLAGS = frozenset({FLAG_BANNED_CLAIM, FLAG_CONTRADICTS})

# Rule score weights (sum to 1).
RULE_WEIGHT_INCLUDE = 0.3
RULE_WEIGHT_NOT_CLAIM = 0.4
RULE_WEIGHT_GROUNDING = 0.3

# LLM review.
SCORE_MIN = 1
SCORE_MAX = 5
JUSTIFICATION_MAX_CHARS = 500
LLM_FAITHFULNESS_WEIGHT = 0.7
LLM_CLARITY_WEIGHT = 0.3
OVERCLAIM_PENALTY = 0.5
LLM_MODEL_ENV = "LLM_MODEL"
LLM_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_LLM_MODEL = "claude-opus-5-5"
LLM_MODE_AUTO = "auto"
LLM_MODE_ANTHROPIC = "anthropic"
LLM_MODE_STUB = "stub"
LLM_MODE_REPLAY = "replay"
LLM_MODES = (LLM_MODE_AUTO, LLM_MODE_ANTHROPIC, LLM_MODE_STUB, LLM_MODE_REPLAY)
SOURCE_LLM = "llm"
SOURCE_STUB = "stub"
STUB_CLARITY = 3
STUB_CRITICAL_FAITHFULNESS_CAP = 2
# Headroom for always-on thinking, which counts against max_tokens.
LLM_MAX_TOKENS_BASE = 8000
LLM_MAX_TOKENS_PER_QUERY = 400
LLM_MAX_TOKENS_CAP = 16000
LOG_STATUS_OK = "ok"
LOG_STATUS_PROVIDER_ERROR = "provider_error"
LOG_STATUS_VALIDATION_ERROR = "validation_error"

# Aggregation.
WEIGHT_RULES = 0.6
WEIGHT_LLM = 0.4
TIE_MARGIN = 0.05

# Failure taxonomy.
TAG_RETRIEVAL_MISS = "retrieval_miss"
TAG_MISSED_KEY_FACT = "missed_key_fact"
TAG_POLICY_VIOLATION = "policy_violation"
TAG_UNSUPPORTED_CLAIM = "unsupported_claim"
TAG_OVERCONFIDENT = "overconfident_tone"
TAG_IRRELEVANT = "irrelevant_answer"
TAXONOMY = (
    TAG_RETRIEVAL_MISS,
    TAG_MISSED_KEY_FACT,
    TAG_POLICY_VIOLATION,
    TAG_UNSUPPORTED_CLAIM,
    TAG_OVERCONFIDENT,
    TAG_IRRELEVANT,
)
ABSOLUTE_WORDS = frozenset({"guaranteed", "instant", "always", "definitely", "100%"})
IRRELEVANT_GROUNDING_MAX = 0.2

# Exit codes.
EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_INPUT_ERROR = 2
EXIT_LLM_REQUIRED = 3

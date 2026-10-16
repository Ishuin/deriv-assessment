"""Exception types raised by the evaluation pipeline."""


class InputValidationError(ValueError):
    """Raised when an input fixture is missing, malformed, or inconsistent."""


class LLMOutputError(ValueError):
    """Raised when the LLM response does not match the required schema."""


class ArtifactValidationError(ValueError):
    """Raised when a derived artifact fails validation."""


class LLMConfigError(RuntimeError):
    """Raised when the requested LLM provider cannot be used (no key, SDK missing, nothing to replay)."""


class LLMProviderError(RuntimeError):
    """Raised when the LLM call itself fails (API error, refusal, truncated output)."""

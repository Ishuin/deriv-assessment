"""Exception types raised by the evaluation pipeline."""


class InputValidationError(ValueError):
    """Raised when an input fixture is missing, malformed, or inconsistent."""


class LLMOutputError(ValueError):
    """Raised when the LLM response does not match the required schema."""


class ArtifactValidationError(ValueError):
    """Raised when a derived artifact fails validation."""

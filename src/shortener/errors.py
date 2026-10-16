"""Domain errors. Each carries a stable machine-readable ``code``.

The core never knows about HTTP; the transport layer maps these codes to status codes.
"""


class ShortenerError(Exception):
    """Base class for every domain failure raised by the service."""

    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        """Store a human-readable message alongside the class-level code."""
        super().__init__(message)
        self.message = message


class InvalidUrlError(ShortenerError):
    """The submitted URL failed validation."""

    code = "invalid_url"


class InvalidRequestError(ShortenerError):
    """The request body is missing, not JSON, or has the wrong shape."""

    code = "invalid_request"


class UnsupportedMediaTypeError(ShortenerError):
    """The request content type is not JSON."""

    code = "unsupported_media_type"


class PayloadTooLargeError(ShortenerError):
    """The request body exceeds the configured size limit."""

    code = "payload_too_large"


class CodeNotFoundError(ShortenerError):
    """No mapping exists for the requested short code."""

    code = "code_not_found"


class CodeGenerationError(ShortenerError):
    """A free short code could not be found within the attempt limit."""

    code = "code_generation_failed"


class ConfigError(ShortenerError):
    """A setting is missing, malformed, or out of range."""

    code = "invalid_config"

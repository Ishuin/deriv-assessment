"""Domain errors. Each carries a stable machine-readable ``code``.

The core never knows about HTTP; the transport layer maps these errors to
status codes, using ``client_fault`` to choose between the 4xx and 5xx families.
"""

from __future__ import annotations

ECHO_LIMIT = 80

_registered_codes: dict[str, type[ShortenerError]] = {}


def describe(value: object, limit: int = ECHO_LIMIT) -> str:
    """Return a bounded, escaped representation of ``value`` safe to put in a message.

    ``repr`` escapes control characters, so newlines in user input cannot forge log lines.
    """
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class ShortenerError(Exception):
    """Base class for every domain failure raised by the service."""

    code: str = "internal_error"
    # Defaults to a server fault so an unclassified error never reports as the caller's mistake.
    client_fault: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Require each subclass to declare a code that no other error uses."""
        super().__init_subclass__(**kwargs)
        if "code" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define its own 'code'")
        owner = _registered_codes.get(cls.code)
        if owner is not None:
            raise TypeError(f"code {cls.code!r} of {cls.__name__} is already used by {owner.__name__}")
        _registered_codes[cls.code] = cls

    def __init__(self, message: str) -> None:
        """Store a human-readable message alongside the class-level code."""
        super().__init__(message)
        self.message = message


_registered_codes[ShortenerError.code] = ShortenerError


class InvalidUrlError(ShortenerError):
    """The submitted URL failed validation."""

    code = "invalid_url"
    client_fault = True


class InvalidRequestError(ShortenerError):
    """The request body is missing, not JSON, or has the wrong shape."""

    code = "invalid_request"
    client_fault = True


class UnsupportedMediaTypeError(ShortenerError):
    """The request content type is not JSON."""

    code = "unsupported_media_type"
    client_fault = True


class PayloadTooLargeError(ShortenerError):
    """The request body exceeds the configured size limit."""

    code = "payload_too_large"
    client_fault = True


class CodeNotFoundError(ShortenerError):
    """No mapping exists for the requested short code."""

    code = "code_not_found"
    client_fault = True


class CodeGenerationError(ShortenerError):
    """A free short code could not be found within the attempt limit."""

    code = "code_generation_failed"


class StorageError(ShortenerError):
    """The backing store failed (I/O, locking, corruption), not the caller's input."""

    code = "storage_error"


class ConfigError(ShortenerError):
    """A setting is missing, malformed, or out of range."""

    code = "invalid_config"

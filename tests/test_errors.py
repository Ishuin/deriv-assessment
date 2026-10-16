"""Domain error contract: stable, unique codes and a shared base class."""

import pytest

from shortener import errors

EXPECTED_CODES = {
    errors.InvalidUrlError: "invalid_url",
    errors.InvalidRequestError: "invalid_request",
    errors.UnsupportedMediaTypeError: "unsupported_media_type",
    errors.PayloadTooLargeError: "payload_too_large",
    errors.CodeNotFoundError: "code_not_found",
    errors.CodeGenerationError: "code_generation_failed",
    errors.ConfigError: "invalid_config",
}


@pytest.mark.parametrize(("error_cls", "code"), EXPECTED_CODES.items())
def test_error_has_stable_code(error_cls, code):
    """Each error exposes its documented code on class and instance."""
    assert error_cls.code == code
    assert error_cls("boom").code == code


@pytest.mark.parametrize("error_cls", EXPECTED_CODES)
def test_error_derives_from_base(error_cls):
    """Callers can catch every domain failure via ShortenerError."""
    assert issubclass(error_cls, errors.ShortenerError)


def test_codes_are_unique():
    """Two errors sharing a code would be indistinguishable to clients."""
    codes = list(EXPECTED_CODES.values())
    assert len(codes) == len(set(codes))


def test_message_is_kept():
    """The message is available as an attribute and via str()."""
    error = errors.InvalidUrlError("bad host")
    assert error.message == "bad host"
    assert str(error) == "bad host"


def test_base_error_default_code():
    """An unclassified failure maps to internal_error."""
    assert errors.ShortenerError("x").code == "internal_error"

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


def test_every_error_class_is_listed_here():
    """A new error type must be added to EXPECTED_CODES so its contract gets checked."""
    assert set(errors.ShortenerError.__subclasses__()) == set(EXPECTED_CODES)


@pytest.mark.parametrize(("error_cls", "code"), EXPECTED_CODES.items())
def test_error_has_stable_code(error_cls, code):
    """Each error exposes its documented code on class and instance."""
    assert error_cls.code == code
    assert error_cls("boom").code == code


@pytest.mark.parametrize("error_cls", EXPECTED_CODES)
def test_error_declares_its_own_code(error_cls):
    """An inherited code would make the error indistinguishable from its parent."""
    assert "code" in vars(error_cls)


def test_codes_are_unique():
    """Two errors sharing a code would be indistinguishable to clients."""
    codes = [*EXPECTED_CODES.values(), errors.ShortenerError.code]
    assert len(codes) == len(set(codes))


def test_message_is_kept():
    """The message is available as an attribute and via str()."""
    error = errors.InvalidUrlError("bad host")
    assert error.message == "bad host"
    assert str(error) == "bad host"


def test_base_error_default_code():
    """An unclassified failure maps to internal_error."""
    assert errors.ShortenerError("x").code == "internal_error"

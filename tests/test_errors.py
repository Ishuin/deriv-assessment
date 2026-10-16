"""Domain error contract: stable, unique codes, fault classes, and bounded messages."""

import pytest

from shortener import errors

EXPECTED = {
    errors.InvalidUrlError: ("invalid_url", True),
    errors.InvalidRequestError: ("invalid_request", True),
    errors.UnsupportedMediaTypeError: ("unsupported_media_type", True),
    errors.PayloadTooLargeError: ("payload_too_large", True),
    errors.CodeNotFoundError: ("code_not_found", True),
    errors.CodeGenerationError: ("code_generation_failed", False),
    errors.ConfigError: ("invalid_config", False),
}


def _all_subclasses(cls):
    """Return every descendant of ``cls``, however deep."""
    found = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _all_subclasses(sub)
    return found


def test_every_error_class_is_covered_here():
    """A new error type must be added to EXPECTED, so it gets its contract checked."""
    assert _all_subclasses(errors.ShortenerError) == set(EXPECTED)


@pytest.mark.parametrize(("error_cls", "expected"), EXPECTED.items())
def test_error_code_and_fault_class(error_cls, expected):
    """Each error exposes its documented code and fault class on class and instance."""
    code, client_fault = expected
    assert (error_cls.code, error_cls.client_fault) == (code, client_fault)
    instance = error_cls("boom")
    assert (instance.code, instance.client_fault) == (code, client_fault)


def test_codes_are_unique():
    """Two errors sharing a code would be indistinguishable to clients."""
    codes = [code for code, _ in EXPECTED.values()]
    assert len(codes) == len(set(codes))


def test_message_is_kept():
    """The message is available as an attribute and via str()."""
    error = errors.InvalidUrlError("bad host")
    assert error.message == "bad host"
    assert str(error) == "bad host"


def test_base_error_is_a_server_fault():
    """An unclassified failure maps to internal_error and never blames the caller."""
    error = errors.ShortenerError("x")
    assert (error.code, error.client_fault) == ("internal_error", False)


def test_subclass_without_own_code_is_rejected():
    """Inheriting the parent's code would hide the new error from clients."""
    with pytest.raises(TypeError, match="must define its own 'code'"):

        class Forgot(errors.ShortenerError):
            pass


def test_subclass_with_duplicate_code_is_rejected():
    """Codes are unique across the whole hierarchy, enforced at definition time."""
    with pytest.raises(TypeError, match="already used by InvalidUrlError"):

        class Clash(errors.ShortenerError):
            code = "invalid_url"


def test_describe_short_value_is_repr():
    """Short values are shown as their repr."""
    assert errors.describe("abc") == "'abc'"


def test_describe_escapes_control_characters():
    """Newlines in input cannot forge extra log lines."""
    assert "\n" not in errors.describe("a\nb\r\x00")


def test_describe_truncates_long_values():
    """Echoed input never exceeds the limit."""
    text = errors.describe("x" * 5000)
    assert len(text) == errors.ECHO_LIMIT
    assert text.endswith("...")

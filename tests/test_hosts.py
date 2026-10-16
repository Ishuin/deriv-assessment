"""Host validation and canonical origin formatting."""

import pytest

from shortener import hosts


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "example.com",
        "Example.COM",
        "a.b-c.example",
        "xn--bcher-kva.example",
        "bücher.example",
        "example.com.",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "::",
        "2001:db8::1",
        "a" * 63 + ".com",
    ],
)
def test_valid_hosts(host):
    """Names, IDN names, and IP literals are accepted."""
    assert hosts.is_valid_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "",
        "exa mple.com",
        "exa_mple.com",
        "-bad-.com",
        "bad-.com",
        "a..b",
        ".example.com",
        "999.1.1.1",
        "127.1",
        "example.123",
        "[::1]",
        "example.com:80",
        "http://example.com",
        "a" * 64 + ".com",
        ".".join(["a" * 60] * 5),
        "ex\nample.com",
    ],
)
def test_invalid_hosts(host):
    """Malformed labels, address-lookalikes, brackets, ports, and over-long names are rejected."""
    assert not hosts.is_valid_host(host)


@pytest.mark.parametrize(
    ("scheme", "host", "port", "expected"),
    [
        ("http", "Example.COM", None, "http://example.com"),
        ("http", "example.com", 80, "http://example.com"),
        ("https", "example.com", 443, "https://example.com"),
        ("http", "example.com", 443, "http://example.com:443"),
        ("https", "example.com", 8443, "https://example.com:8443"),
        ("http", "bücher.example", None, "http://xn--bcher-kva.example"),
        ("http", "0:0:0:0:0:0:0:1", 8000, "http://[::1]:8000"),
        ("http", "127.0.0.1", 8000, "http://127.0.0.1:8000"),
    ],
)
def test_format_origin(scheme, host, port, expected):
    """Origins are lowercase, punycode, bracketed for IPv6, and drop the default port."""
    assert hosts.format_origin(scheme, host, port) == expected


def test_format_origin_rejects_unvalidated_host():
    """An un-encodable host is a caller bug, surfaced loudly instead of producing a bad origin."""
    with pytest.raises(ValueError, match="validated host"):
        hosts.format_origin("http", "a..b", None)

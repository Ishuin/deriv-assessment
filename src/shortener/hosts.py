"""Host validation and origin formatting, shared by settings and URL validation.

Pure functions: one definition of "valid host" keeps the two callers consistent.
"""

from __future__ import annotations

import ipaddress
import re

MAX_HOST_LENGTH = 253
MAX_LABEL_LENGTH = 63
DEFAULT_PORTS = {"http": 80, "https": 443}

_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return the address for an IPv4/IPv6 literal (without brackets), else None."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def to_ascii_host(host: str) -> str | None:
    """Return the lowercase IDNA (punycode) form of a hostname, or None if it cannot be encoded."""
    try:
        return host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None


def is_valid_host(host: str) -> bool:
    """Return True for an IP literal or an RFC 1123 hostname (Unicode names are checked as punycode)."""
    if not host:
        return False
    if parse_ip(host) is not None:
        return True
    ascii_host = to_ascii_host(host)
    return ascii_host is not None and _is_valid_hostname(ascii_host)


def _is_valid_hostname(name: str) -> bool:
    """Check length, label syntax, and that the last label is not all digits."""
    name = name.removesuffix(".")
    if not name or len(name) > MAX_HOST_LENGTH:
        return False
    labels = name.split(".")
    if not all(len(label) <= MAX_LABEL_LENGTH and _LABEL_PATTERN.fullmatch(label) for label in labels):
        return False
    # "999.1.1.1" and "127.1" look like addresses, so browsers disagree on them; reject.
    return not labels[-1].isdigit()


def format_origin(scheme: str, host: str, port: int | None) -> str:
    """Build the canonical ``scheme://host[:port]`` for an already-validated host.

    Lowercases, converts Unicode hosts to punycode, brackets IPv6, and drops the scheme's default port.
    """
    ip = parse_ip(host)
    if ip is not None:
        shown = f"[{ip}]" if ip.version == 6 else str(ip)
    else:
        shown = to_ascii_host(host)
        if shown is None:
            raise ValueError(f"format_origin needs a validated host, got {host!r}")
    suffix = "" if port is None or port == DEFAULT_PORTS.get(scheme) else f":{port}"
    return f"{scheme}://{shown}{suffix}"

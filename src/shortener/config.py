"""Service settings: named defaults, environment loading, and validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit

from shortener.errors import ConfigError, describe
from shortener.hosts import format_origin, is_valid_host

ENV_DB_PATH = "SHORTENER_DB"
ENV_HOST = "SHORTENER_HOST"
ENV_PORT = "SHORTENER_PORT"
ENV_BASE_URL = "SHORTENER_BASE_URL"

IN_MEMORY_DB = ":memory:"
DEFAULT_DB_PATH = "shortener.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PORT = 1
MAX_PORT = 65535
# Binding to these is how a server is exposed, but they are not addresses a client can open.
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})

MAX_URL_LENGTH = 2048
URL_LENGTH_CEILING = 8192
# A JSON encoder may write each character as \uXXXX, so a URL can take 6 bytes per character.
JSON_ESCAPE_FACTOR = 6
JSON_ENVELOPE_BYTES = 64
MAX_BODY_BYTES = 16 * 1024
BODY_BYTES_CEILING = 1024 * 1024

CODE_LENGTH = 7
MIN_CODE_LENGTH = 4
CODE_LENGTH_CEILING = 32
MAX_CODE_ATTEMPTS = 5
CODE_ATTEMPTS_CEILING = 100
# Codes equal to a route word would shadow that route.
RESERVED_CODES = frozenset({"shorten"})
RESERVED_CODE_PATTERN = re.compile(r"[0-9A-Za-z]+")

BASE_URL_SCHEMES = frozenset({"http", "https"})
TEXT_FIELDS = ("db_path", "host", "base_url")


def required_body_bytes(max_url_length: int) -> int:
    """Return the smallest body limit that still admits a worst-case-escaped URL of the given length."""
    return max_url_length * JSON_ESCAPE_FACTOR + JSON_ENVELOPE_BYTES


# name -> (minimum, maximum), both inclusive.
INT_LIMITS = {
    "port": (MIN_PORT, MAX_PORT),
    "max_url_length": (1, URL_LENGTH_CEILING),
    "max_body_bytes": (required_body_bytes(1), BODY_BYTES_CEILING),
    "code_length": (MIN_CODE_LENGTH, CODE_LENGTH_CEILING),
    "max_code_attempts": (1, CODE_ATTEMPTS_CEILING),
}


@dataclass(frozen=True)
class Settings:
    """Validated, immutable service configuration."""

    db_path: str = DEFAULT_DB_PATH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    base_url: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    max_url_length: int = MAX_URL_LENGTH
    max_body_bytes: int = MAX_BODY_BYTES
    code_length: int = CODE_LENGTH
    max_code_attempts: int = MAX_CODE_ATTEMPTS
    reserved_codes: frozenset[str] = RESERVED_CODES

    def __post_init__(self) -> None:
        """Reject invalid values, reporting every problem at once so a bad config fails at startup."""
        codes = _as_frozenset(self.reserved_codes)
        problems = _collect_problems(self, codes)
        if problems:
            raise ConfigError("invalid settings: " + "; ".join(problems))
        # A plain set would pass the annotation yet leave the "frozen" settings mutable, and
        # keeping base_url canonical lets the self-host check compare origins directly.
        object.__setattr__(self, "reserved_codes", codes)
        object.__setattr__(self, "base_url", _canonical_base_url(self.base_url))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        env = os.environ if environ is None else environ
        host = _bare_host(_env_text(env, ENV_HOST) or DEFAULT_HOST)
        port = _env_port(env)
        base_url = _env_text(env, ENV_BASE_URL) or _default_base_url(host, port)
        return cls(
            db_path=_env_db_path(env),
            host=host,
            port=port,
            base_url=base_url.rstrip("/"),
        )


def _env_text(env: Mapping[str, str], name: str) -> str | None:
    """Return a stripped variable, None when unset; a set-but-blank value is an error."""
    if name not in env:
        return None
    value = env[name].strip()
    if not value:
        raise ConfigError(f"{name} must not be empty when set")
    return value


def _env_port(env: Mapping[str, str]) -> int:
    """Parse the port variable strictly: ASCII digits only, no sign, space, or underscore."""
    raw = _env_text(env, ENV_PORT)
    if raw is None:
        return DEFAULT_PORT
    if not (raw.isascii() and raw.isdigit()):
        raise ConfigError(f"{ENV_PORT} must be an integer, got {describe(raw)}")
    return int(raw)


def _env_db_path(env: Mapping[str, str]) -> str:
    """Read the database path, refusing the in-memory database for a real run."""
    path = _env_text(env, ENV_DB_PATH) or DEFAULT_DB_PATH
    if path == IN_MEMORY_DB:
        raise ConfigError(f"{ENV_DB_PATH} must be a file path: {IN_MEMORY_DB!r} would lose all data on restart")
    return path


def _bare_host(host: str) -> str:
    """Drop the brackets of an IPv6 literal; sockets bind to the bare address."""
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _default_base_url(host: str, port: int) -> str:
    """Derive the public base URL from the bind address, refusing wildcard hosts."""
    if host in WILDCARD_HOSTS:
        raise ConfigError(f"{ENV_BASE_URL} is required when {ENV_HOST} is {describe(host)}")
    display_host = f"[{host}]" if ":" in host else host
    return f"http://{display_host}:{port}"


def _as_frozenset(codes: object) -> frozenset[str] | None:
    """Coerce a collection to a frozenset; None when it is not a usable collection."""
    if isinstance(codes, (str, bytes)) or not isinstance(codes, Iterable):
        return None
    try:
        return frozenset(codes)
    except TypeError:
        return None


def _collect_problems(settings: Settings, codes: frozenset[str] | None) -> list[str]:
    """Run every check and return all problems found, skipping checks that depend on a failed one."""
    text_problems = {name: _text_problem(name, getattr(settings, name)) for name in TEXT_FIELDS}
    int_problems = {
        name: _int_problem(name, getattr(settings, name), low, high) for name, (low, high) in INT_LIMITS.items()
    }
    problems = [*text_problems.values(), *int_problems.values()]
    if text_problems["host"] is None:
        problems.append(_host_problem(settings.host))
    if text_problems["base_url"] is None:
        problems.append(_base_url_problem(settings.base_url))
    if int_problems["max_url_length"] is None and int_problems["max_body_bytes"] is None:
        problems.append(_body_size_problem(settings.max_url_length, settings.max_body_bytes))
    problems.append(_reserved_codes_problem(codes))
    return [problem for problem in problems if problem is not None]


def _text_problem(name: str, value: object) -> str | None:
    """Return why ``value`` is not a non-blank string, or None."""
    if not isinstance(value, str):
        return f"{name} must be a string, got {describe(value)}"
    if not value.strip():
        return f"{name} must not be empty"
    return None


def _int_problem(name: str, value: object, low: int, high: int) -> str | None:
    """Return why ``value`` is not a real integer in the inclusive range, or None (bool is not an int)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{name} must be an integer, got {describe(value)}"
    if not low <= value <= high:
        return f"{name} must be between {low} and {high}, got {value}"
    return None


def _host_problem(host: str) -> str | None:
    """Return why ``host`` is not a bare hostname or IP address, or None."""
    if is_valid_host(host):
        return None
    return f"host must be a hostname or IP address without brackets or port, got {describe(host)}"


def _base_url_problem(base_url: str) -> str | None:
    """Return why ``base_url`` is not a bare http(s) origin, or None."""
    reason = _base_url_reason(base_url)
    return None if reason is None else f"base_url {reason}, got {describe(base_url)}"


def _base_url_reason(base_url: str) -> str | None:
    """Return the specific rule ``base_url`` breaks, or None when it is a valid origin."""
    if base_url.endswith("/"):
        return "must not end with '/'"
    if any(char.isspace() or not char.isprintable() for char in base_url):
        return "must not contain whitespace or control characters"
    if "?" in base_url or "#" in base_url:
        return "must not contain a query or fragment"
    try:
        parts = urlsplit(base_url)
        parts.port
    except ValueError:
        return "is not a valid URL"
    if parts.scheme not in BASE_URL_SCHEMES:
        return "must start with http:// or https://"
    if not parts.hostname:
        return "must include a host"
    if "@" in parts.netloc:
        return "must not include credentials"
    if parts.path:
        return "must not include a path"
    if not is_valid_host(parts.hostname):
        return "has an invalid host"
    return None


def _body_size_problem(max_url_length: int, max_body_bytes: int) -> str | None:
    """Return why the body limit would reject a valid URL, or None."""
    needed = required_body_bytes(max_url_length)
    if max_body_bytes >= needed:
        return None
    return f"max_body_bytes must be at least {needed} to admit a URL of max_url_length={max_url_length}, got {max_body_bytes}"


def _reserved_codes_problem(codes: frozenset[str] | None) -> str | None:
    """Return why the reserved codes are unusable, or None."""
    if codes is None:
        return "reserved_codes must be a collection of strings"
    for code in codes:
        if not isinstance(code, str) or not RESERVED_CODE_PATTERN.fullmatch(code):
            return f"reserved_codes entries must be non-empty alphanumeric strings, got {describe(code)}"
    return None


def _canonical_base_url(base_url: str) -> str:
    """Return the validated base URL in canonical form (lowercase, punycode host, default port dropped)."""
    parts = urlsplit(base_url)
    return format_origin(parts.scheme, parts.hostname, parts.port)

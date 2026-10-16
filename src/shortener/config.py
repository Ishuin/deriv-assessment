"""Service settings: named defaults, environment loading, and validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit

from shortener.errors import ConfigError, describe

ENV_DB_PATH = "SHORTENER_DB"
ENV_HOST = "SHORTENER_HOST"
ENV_PORT = "SHORTENER_PORT"
ENV_BASE_URL = "SHORTENER_BASE_URL"

DEFAULT_DB_PATH = "shortener.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PORT = 1
MAX_PORT = 65535
# Binding to these is how a server is exposed, but they are not addresses a client can open.
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})

MAX_URL_LENGTH = 2048
URL_LENGTH_CEILING = 8192
# JSON escaping can expand a URL up to 6x (\uXXXX), so leave headroom.
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
# name -> (minimum, maximum), both inclusive.
INT_LIMITS = {
    "port": (MIN_PORT, MAX_PORT),
    "max_url_length": (1, URL_LENGTH_CEILING),
    "max_body_bytes": (1, BODY_BYTES_CEILING),
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
        """Reject invalid values so a bad config fails at startup."""
        for name in TEXT_FIELDS:
            _check_text(name, getattr(self, name))
        for name, (low, high) in INT_LIMITS.items():
            _check_int(name, getattr(self, name), low, high)
        _check_base_url(self.base_url)
        # A plain set would pass the annotation yet leave the "frozen" settings mutable.
        object.__setattr__(self, "reserved_codes", _clean_reserved_codes(self.reserved_codes))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        env = os.environ if environ is None else environ
        host = _env_text(env, ENV_HOST) or DEFAULT_HOST
        port = _env_port(env)
        base_url = _env_text(env, ENV_BASE_URL) or _default_base_url(host, port)
        return cls(
            db_path=_env_text(env, ENV_DB_PATH) or DEFAULT_DB_PATH,
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


def _default_base_url(host: str, port: int) -> str:
    """Derive the public base URL from the bind address, refusing wildcard hosts."""
    bare_host = host.strip("[]")
    if bare_host in WILDCARD_HOSTS:
        raise ConfigError(f"{ENV_BASE_URL} is required when {ENV_HOST} is {describe(host)}")
    display_host = f"[{bare_host}]" if ":" in bare_host else bare_host
    return f"http://{display_host}:{port}"


def _check_text(name: str, value: object) -> None:
    """Require a non-blank string."""
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string, got {describe(value)}")
    if not value.strip():
        raise ConfigError(f"{name} must not be empty")


def _check_int(name: str, value: object, low: int, high: int) -> None:
    """Require a real integer (not a bool) inside the inclusive range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {describe(value)}")
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")


def _check_base_url(base_url: str) -> None:
    """Require an http(s) origin: a host, optional port, and nothing after it."""
    problem = _base_url_problem(base_url)
    if problem:
        raise ConfigError(f"base_url {problem}, got {describe(base_url)}")


def _base_url_problem(base_url: str) -> str | None:
    """Return why ``base_url`` is not a bare http(s) origin, or None when it is."""
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
    return None


def _clean_reserved_codes(codes: object) -> frozenset[str]:
    """Return reserved codes as a frozenset after checking each entry is an alphanumeric string."""
    if isinstance(codes, (str, bytes)) or not isinstance(codes, Iterable):
        raise ConfigError(f"reserved_codes must be a collection of strings, got {describe(codes)}")
    cleaned = frozenset(codes)
    for code in cleaned:
        if not isinstance(code, str) or not RESERVED_CODE_PATTERN.fullmatch(code):
            raise ConfigError(f"reserved_codes entries must be non-empty alphanumeric strings, got {describe(code)}")
    return cleaned

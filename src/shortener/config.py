"""Service settings: named defaults, environment loading, and validation of the overridable values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from urllib.parse import urlsplit

from shortener.errors import ConfigError
from shortener.hosts import format_origin, is_valid_host

ENV_DB_PATH = "SHORTENER_DB"
ENV_HOST = "SHORTENER_HOST"
ENV_PORT = "SHORTENER_PORT"
ENV_BASE_URL = "SHORTENER_BASE_URL"

DEFAULT_DB_PATH = "shortener.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
MIN_PORT = 1
MAX_PORT = 65535

MAX_URL_LENGTH = 2048
# JSON escaping can expand a URL up to 6x (\uXXXX), so leave headroom.
MAX_BODY_BYTES = 16 * 1024
CODE_LENGTH = 7
MAX_CODE_ATTEMPTS = 5
# Codes equal to a route word would shadow that route.
RESERVED_CODES = frozenset({"shorten"})

BASE_URL_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class Settings:
    """Immutable service configuration."""

    db_path: str = DEFAULT_DB_PATH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    base_url: str = DEFAULT_BASE_URL
    max_url_length: int = MAX_URL_LENGTH
    max_body_bytes: int = MAX_BODY_BYTES
    code_length: int = CODE_LENGTH
    max_code_attempts: int = MAX_CODE_ATTEMPTS
    reserved_codes: frozenset[str] = RESERVED_CODES

    def __post_init__(self) -> None:
        """Validate the values an operator can override, so a bad config fails at startup."""
        if not self.db_path.strip():
            raise ConfigError("db_path must not be empty")
        if not is_valid_host(self.host):
            raise ConfigError(f"host must be a hostname or IP address, got {self.host!r}")
        if not MIN_PORT <= self.port <= MAX_PORT:
            raise ConfigError(f"port must be between {MIN_PORT} and {MAX_PORT}, got {self.port}")
        # Stored canonical so the self-host check can compare origins directly.
        object.__setattr__(self, "base_url", _canonical_base_url(self.base_url))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        env = os.environ if environ is None else environ
        return cls(
            db_path=_env_text(env, ENV_DB_PATH, DEFAULT_DB_PATH),
            host=_env_text(env, ENV_HOST, DEFAULT_HOST),
            port=_env_port(env),
            base_url=_env_text(env, ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/"),
        )


def _env_text(env: Mapping[str, str], name: str, default: str) -> str:
    """Return the stripped variable, or the default when unset; a set-but-blank value is an error."""
    if name not in env:
        return default
    value = env[name].strip()
    if not value:
        raise ConfigError(f"{name} must not be empty when set")
    return value


def _env_port(env: Mapping[str, str]) -> int:
    """Parse the port variable as an integer."""
    raw = _env_text(env, ENV_PORT, str(DEFAULT_PORT))
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PORT} must be an integer, got {raw!r}") from None


def _canonical_base_url(base_url: str) -> str:
    """Validate that ``base_url`` is a bare http(s) origin and return it in canonical form."""
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError:
        raise ConfigError(f"base_url is not a valid URL, got {base_url!r}") from None
    if parts.scheme not in BASE_URL_SCHEMES:
        raise ConfigError(f"base_url must start with http:// or https://, got {base_url!r}")
    if not parts.hostname or not is_valid_host(parts.hostname):
        raise ConfigError(f"base_url must include a valid host, got {base_url!r}")
    if "@" in parts.netloc:
        raise ConfigError(f"base_url must not include credentials, got {base_url!r}")
    if parts.path or parts.query or parts.fragment:
        raise ConfigError(f"base_url must be an origin with no path, query, or fragment, got {base_url!r}")
    return format_origin(parts.scheme, parts.hostname, port)

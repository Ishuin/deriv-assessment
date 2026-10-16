"""Service settings: named defaults, environment loading, and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

from shortener.errors import ConfigError

ENV_DB_PATH = "SHORTENER_DB"
ENV_HOST = "SHORTENER_HOST"
ENV_PORT = "SHORTENER_PORT"
ENV_BASE_URL = "SHORTENER_BASE_URL"

DEFAULT_DB_PATH = "shortener.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PORT = 1
MAX_PORT = 65535

MAX_URL_LENGTH = 2048
# JSON escaping can expand a URL up to 6x (\uXXXX), so leave headroom.
MAX_BODY_BYTES = 16 * 1024

CODE_LENGTH = 7
MIN_CODE_LENGTH = 4
MAX_CODE_ATTEMPTS = 5
# Codes equal to a route word would shadow that route.
RESERVED_CODES = frozenset({"shorten"})

BASE_URL_SCHEMES = ("http://", "https://")


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
        if not self.db_path:
            raise ConfigError("db_path must not be empty")
        if not self.host:
            raise ConfigError("host must not be empty")
        if not MIN_PORT <= self.port <= MAX_PORT:
            raise ConfigError(f"port must be between {MIN_PORT} and {MAX_PORT}, got {self.port}")
        if not self.base_url.startswith(BASE_URL_SCHEMES):
            raise ConfigError(f"base_url must start with http:// or https://, got {self.base_url!r}")
        if self.base_url.endswith("/"):
            raise ConfigError("base_url must not end with '/'")
        if self.max_url_length < 1:
            raise ConfigError("max_url_length must be at least 1")
        if self.max_body_bytes < 1:
            raise ConfigError("max_body_bytes must be at least 1")
        if self.code_length < MIN_CODE_LENGTH:
            raise ConfigError(f"code_length must be at least {MIN_CODE_LENGTH}, got {self.code_length}")
        if self.max_code_attempts < 1:
            raise ConfigError("max_code_attempts must be at least 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        env = os.environ if environ is None else environ
        host = env.get(ENV_HOST, DEFAULT_HOST)
        port = _parse_int(ENV_PORT, env.get(ENV_PORT), DEFAULT_PORT)
        base_url = env.get(ENV_BASE_URL) or f"http://{host}:{port}"
        return cls(
            db_path=env.get(ENV_DB_PATH, DEFAULT_DB_PATH),
            host=host,
            port=port,
            base_url=base_url.rstrip("/"),
        )


def _parse_int(name: str, raw: str | None, default: int) -> int:
    """Parse an integer environment value, naming the variable on failure."""
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None

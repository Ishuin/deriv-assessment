"""Settings defaults, environment loading, and validation."""

import dataclasses

import pytest

from shortener import config
from shortener.config import Settings
from shortener.errors import ConfigError


def test_defaults():
    """Defaults match the documented values."""
    s = Settings()
    assert s.db_path == config.DEFAULT_DB_PATH
    assert s.code_length == 7
    assert s.max_url_length == 2048
    assert "shorten" in s.reserved_codes


def test_settings_are_immutable():
    """Frozen settings cannot be changed after startup."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().port = 9


def test_from_env_empty_uses_defaults():
    """No variables set means defaults, with base_url derived from host and port."""
    s = Settings.from_env({})
    assert s.db_path == config.DEFAULT_DB_PATH
    assert s.base_url == "http://127.0.0.1:8000"


def test_from_env_overrides():
    """Each documented variable overrides its default."""
    s = Settings.from_env(
        {
            config.ENV_DB_PATH: "data/x.db",
            config.ENV_HOST: "0.0.0.0",
            config.ENV_PORT: "9090",
            config.ENV_BASE_URL: "https://sho.rt/",
        }
    )
    assert (s.db_path, s.host, s.port) == ("data/x.db", "0.0.0.0", 9090)
    assert s.base_url == "https://sho.rt"


def test_from_env_derives_base_url_from_overrides():
    """Without SHORTENER_BASE_URL the base URL follows host and port."""
    s = Settings.from_env({config.ENV_HOST: "localhost", config.ENV_PORT: "81"})
    assert s.base_url == "http://localhost:81"


def test_from_env_reads_process_environment(monkeypatch):
    """Omitting the mapping reads os.environ."""
    monkeypatch.setenv(config.ENV_DB_PATH, "env.db")
    assert Settings.from_env().db_path == "env.db"


@pytest.mark.parametrize("raw", ["abc", "", "80.5", "0x50"])
def test_from_env_rejects_non_integer_port(raw):
    """A non-integer port names the offending variable."""
    with pytest.raises(ConfigError, match=config.ENV_PORT):
        Settings.from_env({config.ENV_PORT: raw})


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_rejects_port_out_of_range(port):
    """Ports outside 1..65535 are rejected."""
    with pytest.raises(ConfigError, match="port"):
        Settings(port=port)


@pytest.mark.parametrize("port", [1, 65535])
def test_accepts_port_bounds(port):
    """The boundary ports are valid."""
    assert Settings(port=port).port == port


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("db_path", ""),
        ("host", ""),
        ("base_url", "ftp://x"),
        ("base_url", "short.test"),
        ("base_url", "http://short.test/"),
        ("max_url_length", 0),
        ("max_body_bytes", 0),
        ("code_length", config.MIN_CODE_LENGTH - 1),
        ("max_code_attempts", 0),
    ],
)
def test_rejects_invalid_field(field, value):
    """Each invalid value raises ConfigError naming the field."""
    with pytest.raises(ConfigError, match=field):
        Settings(**{field: value})


def test_conftest_settings_fixture(settings):
    """The shared fixture is isolated from the filesystem."""
    assert settings.db_path == ":memory:"
    assert settings.base_url == "http://short.test"

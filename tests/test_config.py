"""Settings defaults, environment loading, and validation of the overridable values."""

import dataclasses

import pytest

from shortener import config
from shortener.config import Settings
from shortener.errors import ConfigError


def test_defaults():
    """Defaults match the documented values."""
    s = Settings()
    assert s.db_path == config.DEFAULT_DB_PATH
    assert s.base_url == "http://127.0.0.1:8000"
    assert s.code_length == 7
    assert s.max_url_length == 2048
    assert s.reserved_codes == frozenset({"shorten"})


def test_settings_are_immutable():
    """Frozen settings cannot be changed after startup."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().port = 9


def test_conftest_settings_fixture(settings):
    """The shared fixture is isolated from the filesystem."""
    assert settings.db_path == ":memory:"
    assert settings.base_url == "http://short.test"


# --- environment loading -------------------------------------------------


def test_from_env_empty_uses_defaults():
    """No variables set means defaults."""
    assert Settings.from_env({}) == Settings()


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
    assert (s.db_path, s.host, s.port, s.base_url) == ("data/x.db", "0.0.0.0", 9090, "https://sho.rt")


def test_from_env_strips_values():
    """Surrounding whitespace in a variable is not part of the value."""
    s = Settings.from_env({config.ENV_HOST: " localhost ", config.ENV_DB_PATH: "\tx.db\n"})
    assert (s.host, s.db_path) == ("localhost", "x.db")


def test_from_env_reads_process_environment(monkeypatch):
    """Omitting the mapping reads os.environ."""
    monkeypatch.setenv(config.ENV_DB_PATH, "env.db")
    assert Settings.from_env().db_path == "env.db"


@pytest.mark.parametrize("name", [config.ENV_DB_PATH, config.ENV_HOST, config.ENV_PORT, config.ENV_BASE_URL])
@pytest.mark.parametrize("blank", ["", "   "])
def test_from_env_rejects_blank_value(name, blank):
    """A variable that is set but blank is an error, never a silent default."""
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({name: blank})


@pytest.mark.parametrize("raw", ["abc", "80.5", "0x50"])
def test_from_env_rejects_non_integer_port(raw):
    """A non-integer port names the offending variable."""
    with pytest.raises(ConfigError, match=config.ENV_PORT):
        Settings.from_env({config.ENV_PORT: raw})


@pytest.mark.parametrize("raw", ["0", "-1", "65536"])
def test_from_env_rejects_port_out_of_range(raw):
    """A syntactically valid port still has to be in range."""
    with pytest.raises(ConfigError, match="port"):
        Settings.from_env({config.ENV_PORT: raw})


# --- validation ----------------------------------------------------------


@pytest.mark.parametrize("port", [1, 65535])
def test_accepts_port_bounds(port):
    """The boundary ports are valid."""
    assert Settings(port=port).port == port


@pytest.mark.parametrize("field", ["db_path", "host"])
@pytest.mark.parametrize("value", ["", "   "])
def test_rejects_blank_text(field, value):
    """Empty and whitespace-only values are rejected."""
    with pytest.raises(ConfigError, match=field):
        Settings(**{field: value})


@pytest.mark.parametrize("host", ["bad host", "http://x", "x:80", "exa_mple", "999.1.1.1", "[::1]"])
def test_rejects_invalid_host(host):
    """The bind host must be a hostname or IP address."""
    with pytest.raises(ConfigError, match="host"):
        Settings(host=host)


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "::1", "::"])
def test_accepts_valid_host(host):
    """Names and IP literals, including the wildcard addresses, can be bound."""
    assert Settings(host=host).host == host


@pytest.mark.parametrize(
    "base_url",
    [
        "http://",
        "http://a b",
        "https://x/p",
        "http://x/?q=1",
        "http://x#f",
        "http://u:p@x",
        "http://x:99999",
        "http://[::1",
        "ftp://x",
        "short.test",
        "http://short.test/",
        "http://-bad-.com",
        "http://999.1.1.1",
    ],
)
def test_rejects_invalid_base_url(base_url):
    """Anything but a bare http(s) origin would produce malformed short links."""
    with pytest.raises(ConfigError, match="base_url"):
        Settings(base_url=base_url)


@pytest.mark.parametrize(
    ("given", "stored"),
    [
        ("http://short.test", "http://short.test"),
        ("HTTPS://Sho.RT:8443", "https://sho.rt:8443"),
        ("http://Example.COM:80", "http://example.com"),
        ("https://example.com:443", "https://example.com"),
        ("http://bücher.example", "http://xn--bcher-kva.example"),
        ("https://[0:0:0:0:0:0:0:1]:8000", "https://[::1]:8000"),
    ],
)
def test_base_url_is_stored_canonical(given, stored):
    """Canonical form lets the self-host check compare origins directly."""
    assert Settings(base_url=given).base_url == stored

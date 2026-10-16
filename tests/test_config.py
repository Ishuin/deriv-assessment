"""Settings defaults, environment loading, and validation."""

import dataclasses

import pytest

from shortener import config
from shortener.config import Settings
from shortener.errors import ECHO_LIMIT, ConfigError


def test_defaults():
    """Defaults match the documented values."""
    s = Settings()
    assert s.db_path == config.DEFAULT_DB_PATH
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


def test_from_env_brackets_ipv6_host():
    """An IPv6 literal must be bracketed to form a valid URL."""
    s = Settings.from_env({config.ENV_HOST: "::1"})
    assert s.base_url == "http://[::1]:8000"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]"])
def test_from_env_wildcard_host_requires_base_url(host):
    """A wildcard bind address cannot be turned into a link a client can open."""
    with pytest.raises(ConfigError, match=config.ENV_BASE_URL):
        Settings.from_env({config.ENV_HOST: host})
    s = Settings.from_env({config.ENV_HOST: host, config.ENV_BASE_URL: "https://sho.rt"})
    assert s.base_url == "https://sho.rt"


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
def test_from_env_rejects_blank_value_for_every_variable(name, blank):
    """A variable that is set but blank is an error, never a silent default."""
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({name: blank})


@pytest.mark.parametrize("raw", ["abc", "80.5", "0x50", "+80", "-1", "8_000", "٨٠"])
def test_from_env_rejects_non_ascii_digit_port(raw):
    """Only plain ASCII digits parse; the error names the variable."""
    with pytest.raises(ConfigError, match=config.ENV_PORT):
        Settings.from_env({config.ENV_PORT: raw})


@pytest.mark.parametrize("raw", ["0", "65536", "999999"])
def test_from_env_rejects_port_out_of_range(raw):
    """A syntactically valid port still has to be in range."""
    with pytest.raises(ConfigError, match="port"):
        Settings.from_env({config.ENV_PORT: raw})


# --- type checks ---------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", "80"),
        ("port", True),
        ("port", 80.0),
        ("code_length", None),
        ("max_url_length", "2048"),
        ("db_path", 5),
        ("host", None),
        ("base_url", b"http://x"),
    ],
)
def test_rejects_wrong_type(field, value):
    """Wrong types raise ConfigError, not a raw TypeError, and bool is not an int."""
    with pytest.raises(ConfigError, match=field):
        Settings(**{field: value})


@pytest.mark.parametrize("field", ["db_path", "host"])
@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_rejects_blank_text(field, value):
    """Empty and whitespace-only text fields are rejected."""
    with pytest.raises(ConfigError, match=field):
        Settings(**{field: value})


# --- numeric ranges ------------------------------------------------------


@pytest.mark.parametrize(("field", "bounds"), config.INT_LIMITS.items())
def test_int_bounds_are_inclusive(field, bounds):
    """The floor and ceiling are valid; one step outside either is not."""
    low, high = bounds
    for ok in (low, high):
        assert getattr(Settings(**{field: ok}), field) == ok
    for bad in (low - 1, high + 1):
        with pytest.raises(ConfigError, match=field):
            Settings(**{field: bad})


# --- base_url ------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://",
        "https://",
        "http://a b",
        "http://x\n",
        "https://x/p",
        "http://x/?q=1",
        "http://x?q",
        "http://x#f",
        "http://u:p@x",
        "http://x:99999",
        "http://[::1",
        "ftp://x",
        "short.test",
        "http://short.test/",
    ],
)
def test_rejects_invalid_base_url(base_url):
    """Anything but a bare http(s) origin would produce malformed short links."""
    with pytest.raises(ConfigError, match="base_url"):
        Settings(base_url=base_url)


@pytest.mark.parametrize("base_url", ["http://short.test", "HTTPS://Sho.rt:8443", "https://[::1]:8000"])
def test_accepts_valid_base_url(base_url):
    """Origins with a host, optional port, and either scheme are valid."""
    assert Settings(base_url=base_url).base_url == base_url


# --- reserved codes ------------------------------------------------------


def test_reserved_codes_are_coerced_to_frozenset():
    """A plain set is accepted but stored immutably."""
    s = Settings(reserved_codes={"shorten", "admin"})
    assert isinstance(s.reserved_codes, frozenset)
    assert s.reserved_codes == {"shorten", "admin"}


def test_reserved_codes_may_be_empty():
    """No reserved words is a valid choice."""
    assert Settings(reserved_codes=()).reserved_codes == frozenset()


@pytest.mark.parametrize("bad", ["shorten", 5, None, {""}, {"a-b"}, {"café"}, {5}, {"ok", " x"}])
def test_rejects_invalid_reserved_codes(bad):
    """Entries must be non-empty ASCII alphanumerics; a bare string is not a collection."""
    with pytest.raises(ConfigError, match="reserved_codes"):
        Settings(reserved_codes=bad)


# --- messages ------------------------------------------------------------


def test_error_message_is_bounded_and_escaped():
    """Echoed input is truncated and cannot carry raw newlines."""
    with pytest.raises(ConfigError) as info:
        Settings(base_url="http://" + "a" * 5000 + "\nINJECTED /p")
    message = info.value.message
    assert "\n" not in message
    assert len(message) < ECHO_LIMIT + 100

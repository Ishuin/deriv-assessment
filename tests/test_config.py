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


def test_from_env_rejects_in_memory_database():
    """A real run must not silently lose its data on restart."""
    with pytest.raises(ConfigError, match=config.ENV_DB_PATH):
        Settings.from_env({config.ENV_DB_PATH: config.IN_MEMORY_DB})


def test_in_memory_database_is_allowed_for_direct_construction():
    """Tests build Settings directly with the in-memory database."""
    assert Settings(db_path=config.IN_MEMORY_DB).db_path == config.IN_MEMORY_DB


def test_from_env_strips_ipv6_brackets_from_host():
    """Sockets bind to the bare address, so the stored host has no brackets."""
    s = Settings.from_env({config.ENV_HOST: "[::1]"})
    assert (s.host, s.base_url) == ("::1", "http://[::1]:8000")


@pytest.mark.parametrize("host", ["bad host", "http://x", "x:80", "exa_mple", "999.1.1.1"])
def test_rejects_invalid_host(host):
    """The bind host must be a hostname or IP address, not just non-blank text."""
    with pytest.raises(ConfigError, match="host"):
        Settings(host=host)


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


def _with_compatible_body(field, value):
    """Build kwargs that keep the body limit consistent with the URL limit while one field varies."""
    if field == "max_url_length":
        return {field: value, "max_body_bytes": config.BODY_BYTES_CEILING}
    if field == "max_body_bytes":
        return {field: value, "max_url_length": 1}
    return {field: value}


@pytest.mark.parametrize(("field", "bounds"), config.INT_LIMITS.items())
def test_int_bounds_are_inclusive(field, bounds):
    """The floor and ceiling are valid; one step outside either is not."""
    low, high = bounds
    for ok in (low, high):
        assert getattr(Settings(**_with_compatible_body(field, ok)), field) == ok
    for bad in (low - 1, high + 1):
        with pytest.raises(ConfigError, match=field):
            Settings(**_with_compatible_body(field, bad))


def test_body_limit_must_admit_a_worst_case_url():
    """A body limit below the escaped URL size would reject valid URLs with 413."""
    needed = config.required_body_bytes(2048)
    assert Settings(max_url_length=2048, max_body_bytes=needed).max_body_bytes == needed
    with pytest.raises(ConfigError, match="max_body_bytes must be at least"):
        Settings(max_url_length=2048, max_body_bytes=needed - 1)


def test_default_body_limit_admits_default_url_limit():
    """The shipped defaults are mutually consistent."""
    assert config.MAX_BODY_BYTES >= config.required_body_bytes(config.MAX_URL_LENGTH)


def test_all_problems_are_reported_together():
    """Several bad fields produce one error listing each, not one-at-a-time failures."""
    with pytest.raises(ConfigError) as info:
        Settings(port=0, code_length=1, host="bad host", base_url="ftp://x")
    message = info.value.message
    for field in ("port", "code_length", "host", "base_url"):
        assert field in message
    assert message.startswith("invalid settings: ")


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


@pytest.mark.parametrize("base_url", ["http://-bad-.com", "http://exa_mple.com", "http://999.1.1.1", "http://a..b"])
def test_rejects_base_url_with_invalid_host(base_url):
    """The host part follows the same rules as submitted URLs."""
    with pytest.raises(ConfigError, match="invalid host"):
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


# --- reserved codes ------------------------------------------------------


def test_reserved_codes_are_coerced_to_frozenset():
    """A plain set is accepted but stored immutably."""
    s = Settings(reserved_codes={"shorten", "admin"})
    assert isinstance(s.reserved_codes, frozenset)
    assert s.reserved_codes == {"shorten", "admin"}


def test_reserved_codes_may_be_empty():
    """No reserved words is a valid choice."""
    assert Settings(reserved_codes=()).reserved_codes == frozenset()


def test_reserved_codes_accepts_a_generator():
    """A one-shot iterable is consumed once and stored intact."""
    s = Settings(reserved_codes=(code for code in ("a1", "b2")))
    assert s.reserved_codes == {"a1", "b2"}


@pytest.mark.parametrize(
    "bad", ["shorten", 5, None, {""}, {"a-b"}, {"café"}, {5}, {"ok", " x"}, [["unhashable"]]]
)
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

# Deriv Assessment

URL shortener service. Status: **Phase 1 (scaffold, config, errors) done; Phases 2 to 7 pending.** See [PLAN.md](PLAN.md) for the phased build plan.

Run tests: `python -m pip install -r requirements-dev.txt`, then `python -m pytest`.

## 1. Problem, restated

Build a small HTTP service that:

1. Accepts a long URL and returns a short code (`POST /shorten`).
2. Takes a short code and redirects the caller to the original URL (`GET /{code}`).
3. Rejects malformed URLs.
4. Handles unknown short codes cleanly.
5. Never issues two different codes for the same URL.

The goal is a clear, tested process, not a particular technology. The store is local: in-memory or SQLite.

## 2. Discovery findings

| Item | Finding | Effect on the design |
|---|---|---|
| Runtime | Python 3.13.7 | Use type hints and frozen dataclasses. |
| Storage | `sqlite3` 3.50.4 in stdlib; `:memory:` and file modes | One engine for production (file) and tests (`:memory:`). No JSON file store (it would need our own locking). |
| Test tooling | pytest 9.1 installed | Dev dependency only. |
| HTTP | `http.server` and `urllib.parse` in stdlib; FastAPI, uvicorn and httpx also installed globally | See decision D1 below. |
| Repo state | Empty except for this README | Layout is a free choice. |

Research conclusions:

- **Redirect status:** `302 Found`. It is not cached by default, so a mapping can change later. `301` is cached by browsers and cannot be taken back. `307`/`308` are for preserving method, which is not needed for GET.
- **Duplicate prevention:** a `UNIQUE` constraint on the normalized URL, with insert-or-ignore followed by select, is safe under concurrent requests. A read-then-write check in application code can race.
- **Code generation:** random 7-character base62 from `secrets`, with retry on collision (62^7 is about 3.5 trillion). A counter-based code is collision-free but guessable and enumerable. The generator is injected so tests are deterministic.
- **URL parsing:** `urllib.parse.urlsplit` is lenient and accepts many malformed inputs, so it cannot be the only validator. Explicit rules are layered on top of it.

## 3. Decisions needing approval

| ID | Decision | Recommendation |
|---|---|---|
| D1 | HTTP layer | Stdlib `http.server`, thin adapter over a pure core. Zero runtime dependencies. Alternative: FastAPI (less boilerplate, extra dependency). |
| D2 | Storage | SQLite file, path from `SHORTENER_DB` (default `shortener.db`). `:memory:` in tests. |
| D3 | Redirect code | `302`. |
| D4 | Fragment handling | Preserved and part of URL identity, since the browser uses it. |
| D5 | Status for repeat submission | `200` for an existing mapping, `201` for a new one. Same code either way. |

## 4. API contract

### `POST /shorten`

Request: `Content-Type: application/json`, body `{"url": "https://example.com/a?b=1"}`.

| Case | Status | Body |
|---|---|---|
| New URL | `201` | `{"code": "aB3xK9q", "short_url": "http://host/aB3xK9q", "url": "<original>"}` |
| URL already shortened | `200` | Same shape, same code as before |
| Malformed URL | `400` | `{"error": {"code": "invalid_url", "message": "..."}}` |
| Body missing, not JSON, or `url` not a string | `400` | `{"error": {"code": "invalid_request", ...}}` |
| Wrong content type | `415` | `unsupported_media_type` |
| Body too large | `413` | `payload_too_large` |

### `GET /{code}`

| Case | Status | Result |
|---|---|---|
| Known code | `302` | `Location: <original URL>`, empty body |
| Unknown code | `404` | `{"error": {"code": "code_not_found", ...}}` |
| Code with illegal characters or wrong length | `404` | Same as unknown (no separate error path) |

Other methods on these routes return `405` with an `Allow` header. Unknown paths return `404`.

## 5. Input and output scenarios

### URL validation rules

Accept only when all of these hold:

- Value is a string, 1 to 2048 characters after trimming surrounding whitespace.
- No control characters or internal whitespace.
- Scheme is `http` or `https` (case-insensitive).
- Host is present and valid: non-empty labels, no leading/trailing hyphen, IDNA-encodable, or a valid IPv4/IPv6 literal.
- Port, if present, is 1 to 65535.
- No userinfo (`user:pass@`), to avoid credential leakage and phishing-style links.
- Host is not the service's own host (prevents redirect loops).

Reject: `ftp://x`, `javascript:alert(1)`, `data:...`, `file:///etc/passwd`, `http://`, `http:///path`, `//example.com`, `example.com` (no scheme), `http://exa mple.com`, `http://example.com:99999`, empty string, `null`, a number, an object.

### Normalization (for duplicate detection)

- Lowercase scheme and host. Encode IDN hosts as punycode.
- Remove default ports (`:80` for http, `:443` for https).
- Empty path becomes `/`.
- Path, query and fragment are kept as given. Query parameter order and trailing slashes are significant, because servers may treat them differently.

So `HTTP://Example.COM:80` and `http://example.com/` get the same code. `http://example.com/a` and `http://example.com/a/` do not.

## 6. Edge cases and scenarios to handle

- Same URL submitted twice, and with different casing or default port: one code.
- Two requests for the same new URL at the same instant: one code (database constraint).
- Random code collision (a new URL draws a code already used by another URL): the insert is checked by selecting on the normalized URL. A row with our code means a new mapping. A row with a different code means a duplicate URL. No row means the code belonged to another URL, so retry with a new code. Attempts are bounded by a named constant, then `CodeGenerationError` (`500`).
- Codes are random, never derived from the URL, so no hash or key can clash across URLs.
- Generated codes that match a reserved route word (`shorten`) are rejected and regenerated.
- Very long URL, huge request body, empty body, invalid JSON, extra JSON fields (ignored).
- Unicode in path or query: preserved. Unicode host: normalized to punycode.
- Server restart: mappings persist (file DB).
- Database missing or unwritable: fail at startup with a clear message, not on first request.
- Short code is case-sensitive: `abc` and `ABC` are different codes.
- `HEAD /{code}` behaves like GET without body.

## 7. Definitions of done

### Completely working application

- `POST /shorten` returns a code for a valid URL and `GET /{code}` redirects to it.
- Malformed URLs get `400`. Unknown codes get `404`.
- The same URL always yields the same code.
- Data survives a restart.
- Runs with one command.

### Completely validated, delivery-ready application

All of the above, plus:

- Every rule in sections 4 to 6 has a test, including each rejected-input example.
- Layers have separate unit tests (validation, codegen, repository, service). HTTP behavior has end-to-end tests against a running server on an ephemeral port.
- A concurrency test shows many parallel identical submissions produce exactly one row.
- Tests are deterministic: injected clock and code generator, no sleeps, no network.
- Full suite passes with zero skips or expected-failures.
- Error responses have one consistent JSON shape and never expose stack traces.
- README documents setup, run command, API with `curl` examples, and limitations.
- Fresh-clone check: clone, install, run tests, start server, shorten and follow a link.

## 8. Architecture

Layers depend downward only. The core has no I/O or HTTP knowledge.

```
HTTP adapter (api)         routing, JSON in/out, status codes
        |
Service (service)          shorten(url) -> Mapping ; resolve(code) -> url
        |           \
Validation (validation)   Code generator (codegen)
Pure: parse, check, normalize     Injectable, base62
        |
Repository (repository)    SQLite: get_by_url, get_by_code, insert_or_get
```

Planned layout:

```
src/shortener/
  config.py       settings: db path, base URL, limits (named constants)
  errors.py       domain errors, each with a stable code
  validation.py   validate_url, normalize_url
  codegen.py      CodeGenerator protocol, RandomBase62Generator
  repository.py   SqliteUrlRepository
  service.py      ShortenerService
  api.py          request handler, routing, error mapping
  __main__.py     entry point: python -m shortener
tests/            one test module per layer, plus conftest.py
```

Data model: table `urls(code TEXT PRIMARY KEY, normalized_url TEXT UNIQUE NOT NULL, original_url TEXT NOT NULL, created_at TEXT NOT NULL) WITHOUT ROWID`. Redirect lookup is one primary-key index lookup (O(log n)). Duplicate detection uses the unique index on `normalized_url`. Codes are checked against `^[0-9A-Za-z]{7}$` before any query. The connection uses WAL mode.

## 9. Independent chunks

Each chunk can be built and tested alone once its dependencies exist. [PLAN.md](PLAN.md) maps chunks A to G to phases 1 to 7.

| Chunk | Responsibility | Depends on |
|---|---|---|
| A | Scaffold, config, domain errors | none |
| B | URL validation and normalization | A |
| C | Code generator | A |
| D | SQLite repository | A |
| E | Service orchestration | B, C, D |
| F | HTTP layer | E |
| G | End-to-end, concurrency, delivery checks | F |

B, C and D can be built in parallel after A.

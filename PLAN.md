# Plan

Approach and files only. Decisions D1 to D5 are in README.md section 3.

## Phase 1: Scaffold and contracts (chunk A)

- Create package skeleton and test config
- Define domain errors with stable codes
- Define settings as named constants and env overrides
- Files: `pyproject.toml`, `requirements-dev.txt`, `.gitignore`, `src/shortener/__init__.py`, `src/shortener/config.py`, `src/shortener/errors.py`, `tests/conftest.py`

## Phase 2: URL validation and normalization (chunk B)

- Pure functions, no I/O
- Table-driven tests: one row per accepted and rejected example in README section 5
- Normalize: lowercase scheme/host, punycode, drop default port, empty path to `/`
- Reject self-host URLs via an injected host value
- Files: `src/shortener/validation.py`, `tests/test_validation.py`

## Phase 3: Code generator (chunk C)

- Protocol plus random base62 implementation using `secrets`
- Fixed length from config
- Skip reserved codes (`shorten`)
- Test length, alphabet, reserved codes, and a stub generator for deterministic use
- Files: `src/shortener/codegen.py`, `tests/test_codegen.py`

## Phase 4: SQLite repository (chunk D)

- Schema created on open: `code` primary key `WITHOUT ROWID`, `UNIQUE` on normalized URL, WAL mode
- `insert_or_get` via insert-or-ignore then select by URL; report new, duplicate, or code-taken
- `get_by_code` lookup
- Tests on `:memory:` and a temp file, including restart persistence
- Files: `src/shortener/repository.py`, `tests/test_repository.py`

## Phase 5: Service orchestration (chunk E)

- `shorten`: validate, normalize, generate, `insert_or_get`, retry only on code-taken, bounded attempts
- `resolve`: return URL or raise code-not-found
- Inject generator, clock and repository
- Tests: duplicates, collision retry, retry exhaustion, unknown code
- Files: `src/shortener/service.py`, `tests/test_service.py`

## Phase 6: HTTP layer (chunk F)

- Routes: `POST /shorten`, `GET /{code}`, `HEAD /{code}`
- Map domain errors to status codes and the JSON error shape
- Enforce content type, body size limit, `405` with `Allow`
- Entry point `python -m shortener`
- Files: `src/shortener/api.py`, `src/shortener/__main__.py`, `tests/test_api.py`

## Phase 7: End-to-end and delivery (chunk G)

- Live-server tests on an ephemeral port
- Concurrency test: parallel identical submissions give one row
- Full suite run, fresh-clone check, `curl` walkthrough
- Finalize README: setup, run, API examples, limitations
- Files: `tests/test_e2e.py`, `tests/test_concurrency.py`, `README.md`

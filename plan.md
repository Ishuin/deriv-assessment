# Implementation Plan

Shortest path to the deliverable state in `docs/spec.md` section 11. Python 3.10+, standard library plus `pytest` and, optionally, the `anthropic` SDK. Each chunk is built, run once, and committed. No review loops between chunks.

## Decisions fixed up front

| Decision | Choice | Reason |
|---|---|---|
| Language | Python, stdlib only for the pipeline | Runs from a clean checkout with no install |
| Retrieval | Hand-written BM25 | Deterministic, about 40 lines, no dependency |
| LLM | Anthropic behind a provider interface, plus a stub | Real path and an offline path |
| Config | `config.py` constants | Weights and thresholds in one reviewed place |
| Layout | `src/rag_eval/` package, root `run.py` and `validate.py` | Matches "simple command" and stays extensible |

## Target layout

```text
kb.json  queries.json  candidate_answers.json      (inputs, sample fixtures)
run.py   validate.py   Makefile  requirements.txt
src/rag_eval/  config errors schemas text retrieval checks grounding
               llm_providers llm_review aggregate taxonomy render pipeline
tests/         test_text test_retrieval test_checks test_grounding
               test_llm_review test_aggregate test_taxonomy test_validate test_e2e
docs/spec.md  README.md  plan.md
```

Derived artifacts are written to the repo root and committed, so a reviewer can read them without running anything.

## Progress

| Chunk | State |
|---|---|
| 0 Skeleton and fixtures | Done |
| 1 Input validation | Done, not yet committed. 70 tests pass |
| 2 to 10 | Not started. Next is chunk 2, text and retrieval |

Chunk 1 notes: `load_inputs(data_dir)` returns an `Inputs` dataclass (`kb`, `queries`, `answers`, `variants`). Queries get a normalised `risk_level` (unknown or missing becomes medium; unknown logs a warning). An empty queries list is rejected, since aggregation needs at least one query.

Chunk 0 notes: `config.py` holds only constants the spec already names. `Makefile`, `run.py` and `validate.py` are deferred to chunks 6 and 7. `tests/conftest.py` adds `src` to `sys.path`, so `pytest` works without installing the package. Scripts at the repo root need the same path setup.

## Chunks

Order follows data dependency. Each chunk lists its done check.

| # | Chunk | Contents | Done check |
|---|---|---|---|
| 0 | Skeleton and fixtures **(DONE)** | Package dirs, `config.py`, `errors.py`, the three sample JSON files copied from the problem statement, `.gitignore`, `requirements.txt`, `tests/conftest.py` | `python -c "import rag_eval"` works. Verified: package imports with `PYTHONPATH=src`, fixtures parse, rule weights sum to 1 |
| 1 | Input validation **(DONE)** | `schemas.py` loaders for the three inputs, rules from spec section 4 | Good fixtures load. Each bad-input case raises `InputValidationError`. Verified by `tests/test_schemas.py` |
| 2 | Text and retrieval | `text.py`, `retrieval.py`, write `retrieval.json` | Sample: top doc matches `expected_doc_ids` for all 4 queries, 2 results each |
| 3 | Rule checks and grounding | `checks.py`, `grounding.py`, write `automated_scores.json` | Sample: `prompt_a` passes all rule checks, `prompt_b` has a critical flag on all four queries |
| 4 | LLM stage | `llm_providers.py` (stub first, then Anthropic), `llm_review.py` (prompt, validator, log), write `llm_review.json` and `llm_calls.jsonl` | Stub run produces a valid file and a log line. A bad canned response is rejected |
| 5 | Aggregation | `aggregate.py`, `render.py` for `recommendation.md` | Sample: `prompt_a` selected with a summary table, reasons, per-query reasoning, limitations. Running twice gives identical bytes |
| 6 | Pipeline and CLI | `pipeline.py`, `run.py` with `--llm`, `--require-llm`, `--baseline`, `--data-dir`, `--out-dir` | `python run.py` creates every artifact with no key |
| 7 | Validation command | `validate.py` with all eight checks from spec section 7, `make validate` | Exit 0 on a clean run. Corrupting an artifact gives exit 1 with a message |
| 8 | Taxonomy and report | `taxonomy.py`, `render.py` for `review_report.md`, wired into the pipeline | Tags only from the vocabulary. Report has a section per query |
| 9 | Tests | Write tests alongside chunks 1 to 8, then add `test_validate` and `test_e2e` | `pytest` passes |
| 10 | Docs and real-key check | `README.md` final pass. One run with a real key if one exists | Commands in README work as written |

Chunks 2, 3 and 4's prompt builder have no dependency on each other beyond chunk 1. They can be done in any order if work is split.

Write each test with its module, not at the end, but do not rewrite working tests to chase coverage.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phrase matching misses a paraphrased ban (`send you your current password`, `screenshot should be fine`) | Window matching over content tokens, plus polarity contradiction and grounding flags. Tests use these exact sample answers, but the code never references them |
| Overlap heuristics over-flag a correct answer | Verify on the four `prompt_a` answers that nothing is flagged. Thresholds in `config.py` |
| BM25 ranks the wrong doc first, for example `D2` over `D4` for "withdraw ... demo account" | Light stemmer in `text.py`. Retrieval hit only needs the expected doc in top 2. Confirm on the sample |
| LLM output drifts from the schema | Forced structured output, strict validator, one call only, rules-only fallback |
| Stub mistaken for real judgment | `source: stub` on each record, a banner in `recommendation.md`, and a note in the README |
| Non-determinism breaks the validate reproducibility check | Sorted keys, 4 dp rounding, no timestamps in deterministic artifacts, sorted iteration, tie-breaks by name |
| Scope creep | Anything not in the problem statement stays out. The only extra flags are `--baseline` and `--require-llm`, both from stretch or failure-handling needs |

## Order of commits

1. `Add spec, plan, README` (this step)
2. `Add skeleton, fixtures, input validation`
3. `Add BM25 retrieval`
4. `Add rule checks and grounding`
5. `Add LLM review stage with stub and Anthropic providers`
6. `Add aggregation and recommendation`
7. `Add run.py pipeline and validate.py`
8. `Add failure taxonomy and review report`
9. `Add tests and finalize README`

## Out of scope

Vector search, a web UI, multiple LLM calls, retry loops, learned weights, database storage, CI configuration, and Docker.

# Deriv Assessment: Offline RAG Prompt Evaluator

A replayable evaluation harness that compares prompt strategies for retrieval-augmented question answering. It retrieves evidence with BM25, scores answers with deterministic checks, runs one bounded LLM review, and computes a promotion recommendation in code.

Status: all planned chunks (0 to 10) are implemented and covered by tests. The committed artifacts were produced with the offline stub reviewer, because no API key was used for the final run; see [LLM stage](#llm-stage). `plan.md` records the build order and `docs/spec.md` holds the design, scenarios and test cases.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows. On macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt # pytest, and anthropic (only needed for the real LLM)
```

Python 3.10 or later. The pipeline itself uses only the standard library.

## Run

```bash
python run.py                   # regenerate every derived artifact
python validate.py              # check artifacts and reproduce the recommendation
python -m pytest                # tests
```

`make run`, `make validate` and `make test` are shortcuts for the same commands. `run.py` exits 0 on success, 2 on bad input or an unusable LLM provider, and 3 when `--require-llm` is set and the LLM stage failed. `validate.py` exits 1 if any check fails.

### LLM stage

The harness makes exactly one LLM call per run, across all query-answer pairs. Its structured output is validated in code before use, and every call is appended to `llm_calls.jsonl`.

| Option | Behaviour |
|---|---|
| `--llm auto` (default) | Anthropic if `ANTHROPIC_API_KEY` is set, else the offline stub with a warning |
| `--llm anthropic` | Real call. Exits 2 before writing anything if the key or the `anthropic` package is missing |
| `--llm stub` | Offline reviewer derived from the rule checks. Marked `source: stub`. Not an independent judgment |
| `--llm replay` | Reuse and re-validate the existing `llm_review.json`. No call is made |
| `--require-llm` | Exit 3 if the LLM stage fails, instead of falling back to rules-only |

To use the real model, set the key in your shell and do not put it in any file:

```bash
export ANTHROPIC_API_KEY=...     # PowerShell: $env:ANTHROPIC_API_KEY = "..."
python run.py --llm anthropic
```

`LLM_MODEL` overrides the default model (`claude-opus-5-5`). If the call fails or its output does not validate, the failure is logged, `llm_review.json` is written as `[]`, and the recommendation is computed from the rule checks alone and says so. The real-API path is tested with a fake client only; it has not been run against the live service in this repository.

### Other options

| Option | Behaviour |
|---|---|
| `--data-dir DIR` | Where `kb.json`, `queries.json` and `candidate_answers.json` are read from. Default is the repo root |
| `--out-dir DIR` | Where artifacts are written. Default is the repo root |
| `--baseline VARIANT` | Regression guard: block any variant that is critical where the baseline is clean, or scores lower than the baseline on a high-risk query |

## Inputs

| File | Content |
|---|---|
| `kb.json` | `doc_id`, `title`, `text` |
| `queries.json` | `query_id`, `user_question`, `expected_doc_ids`, `must_include_any`, `must_not_claim`, `risk_level` |
| `candidate_answers.json` | `query_id`, `answers` mapping variant name to answer text |

Fixtures may be replaced with any data in this schema. Variant names, query IDs and phrases are never hardcoded (a test checks `src/` for them). Adding a third variant means adding a key under `answers` for every query; nothing else changes. Inputs are validated at the boundary and bad ones raise a specific `InputValidationError`.

## Outputs

| File | Stage |
|---|---|
| `retrieval.json` | Top 2 BM25 passages per query, with scores |
| `automated_scores.json` | Per query and variant: retrieval hit, must-include, banned claim, grounding, risk flags, notes |
| `llm_review.json` | Validated LLM scores, overclaim flags and winner per query |
| `llm_calls.jsonl` | Append-only log of every LLM call (prompt, raw response, status, latency, usage) |
| `failure_taxonomy.json` | Failure tags per answer |
| `recommendation.md` | Selected variant, summary table, reasons, tradeoffs, per-query reasoning, limitations |
| `review_report.md` | Per-query debugging view |

All artifacts except `llm_calls.jsonl` are byte-identical across reruns with the same inputs and the same LLM review. The log is append-only and holds timestamps.

## How the recommendation is decided

1. Each answer gets a rule score (required phrase, banned claim, grounding) and, when available, an LLM score (faithfulness, clarity, overclaim penalty). The answer score is 0.6 x rule + 0.4 x LLM.
2. A banned claim or a contradiction of the evidence is a **critical failure**; low grounding also counts on high-risk queries.
3. A variant with any critical failure on a high-risk query cannot be promoted, however clear it is. If a less safe variant is clearer, `recommendation.md` says so.
4. Among the remaining variants, the highest risk-weighted score wins (high 3, medium 2, low 1). If none remain, nothing is promoted.

Retrieval quality is reported but is identical for every variant, so it does not rank them. Weights and thresholds are in `src/rag_eval/config.py`; the full rules are in `docs/spec.md` section 5.

## Validation

`python validate.py` checks that:

- required inputs and artifacts exist and parse;
- every query was processed for every variant, with the right fields;
- retrieval has exactly 2 sorted passages per query from the knowledge base;
- LLM values use only the known variants and score range 1 to 5, and cover every query;
- retrieval, scores, taxonomy and report equal a fresh recomputation, and `recommendation.md` equals a fresh recomputation from the stored outputs.

It prints every failure, not only the first.

## Layout

```text
run.py  validate.py  Makefile  requirements.txt
kb.json  queries.json  candidate_answers.json      sample inputs
src/rag_eval/   config, schemas, text, retrieval, checks, grounding,
                llm_providers, llm_review, aggregate, taxonomy, render,
                pipeline, cli, validation
tests/          unit, integration and end-to-end tests
docs/spec.md    spec, scenarios, test cases, definition of done
plan.md         implementation plan and progress
```

## Limitations

Lexical retrieval over a tiny knowledge base, phrase-and-overlap heuristics rather than entailment, a single LLM call with no self-consistency, and four sample queries that cannot support statistical claims. Banned-phrase matching does not fold word forms such as `guarantee` and `guaranteed`, and the contradiction check can false-flag a correct answer that shares two or more words with a negative evidence sentence. The harness checks answers against the supplied knowledge base only and makes no statement about real policy. See `docs/spec.md` section 8.

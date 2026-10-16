# Deriv Assessment: Offline RAG Prompt Evaluator

A replayable evaluation harness that compares prompt strategies for retrieval-augmented question answering. It retrieves evidence with BM25, scores answers with deterministic checks, runs one bounded LLM review, and computes a promotion recommendation in code.

Status: design complete. Chunks 0 (skeleton, config, errors, sample fixtures), 1 (input validation in `src/rag_eval/schemas.py`), 2 (text utilities and BM25 retrieval in `src/rag_eval/text.py` and `retrieval.py`), 3 (rule checks and grounding in `checks.py` and `grounding.py`) 4 (the single LLM review in `llm_review.py` and `llm_providers.py`, with an offline stub) and 5 (aggregation and `recommendation.md` in `aggregate.py` and `render.py`) are done. `pytest` runs their tests. Pipeline stages, `run.py` and `validate.py` are not built yet, so the commands below do not work until later chunks land. See `plan.md` for progress and `docs/spec.md` for the full design.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows. On macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt # pytest, and anthropic (optional, only for the real LLM)
```

Python 3.10 or later. The pipeline itself uses only the standard library.

## Run

```bash
python run.py                   # regenerate every derived artifact
python validate.py              # check artifacts and reproduce the recommendation
pytest                          # tests
```

`make run`, `make validate` and `make test` are shortcuts for the same commands.

### LLM stage

| Option | Behaviour |
|---|---|
| `--llm auto` (default) | Anthropic if `ANTHROPIC_API_KEY` is set, else the offline stub |
| `--llm anthropic` | Real call. Fails with exit 2 if no key |
| `--llm stub` | Offline reviewer derived from the rule checks. Marked `source: stub`. Not an independent judgment |
| `--llm replay` | Reuse the existing `llm_review.json`. No call is made |
| `--require-llm` | Exit 3 if the LLM stage fails, instead of falling back to rules-only |

Set `LLM_MODEL` to override the default model. The harness makes one LLM call per run. If its output fails validation, the recommendation is computed from the rule checks alone and says so.

### Other options

| Option | Behaviour |
|---|---|
| `--data-dir DIR` | Where `kb.json`, `queries.json` and `candidate_answers.json` are read from. Default is the repo root |
| `--out-dir DIR` | Where artifacts are written. Default is the repo root |
| `--baseline VARIANT` | Block any variant that regresses against this one on a high-risk query |

## Inputs

| File | Content |
|---|---|
| `kb.json` | `doc_id`, `title`, `text` |
| `queries.json` | `query_id`, `user_question`, `expected_doc_ids`, `must_include_any`, `must_not_claim`, `risk_level` |
| `candidate_answers.json` | `query_id`, `answers` mapping variant name to answer text |

Fixtures may be replaced with any data in this schema. Variant names, query IDs and phrases are never hardcoded. Adding a third variant means adding a key under `answers` for every query.

## Outputs

| File | Stage |
|---|---|
| `retrieval.json` | Top 2 BM25 passages per query, with scores |
| `automated_scores.json` | Per query and variant: retrieval hit, must-include, banned claim, grounding, risk flags |
| `llm_review.json` | Validated LLM scores, overclaim flags and winner per query |
| `llm_calls.jsonl` | Append-only log of every LLM call |
| `failure_taxonomy.json` | Failure tags per answer |
| `recommendation.md` | Selected variant, summary table, reasons, per-query reasoning, limitations |
| `review_report.md` | Per-query debug view |

## How the recommendation is decided

1. Each answer gets a rule score (required phrase, banned claim, grounding) and, when available, an LLM score (faithfulness, clarity, overclaim penalty).
2. A banned claim, a contradiction of the evidence, or low grounding on a high-risk query is a **critical failure**.
3. A variant with any critical failure on a high-risk query cannot be promoted, however clear it is.
4. Among the remaining variants, the highest risk-weighted score wins. If none remain, nothing is promoted.

Weights and thresholds are in `src/rag_eval/config.py`. The full rules are in `docs/spec.md` section 5.

## Validation

`python validate.py` checks that required artifacts exist and parse, that every query and variant was processed, that retrieval has 2 passages per query, that LLM values are in range, and that `recommendation.md` matches a fresh recomputation from the stored files. It exits 1 on any failure.

## Layout

```text
run.py  validate.py  Makefile
src/rag_eval/   pipeline stages (see docs/spec.md section 3)
tests/          unit and end-to-end tests
docs/spec.md    spec, scenarios, test cases, definition of done
plan.md         implementation plan
```

## Limitations

Lexical retrieval over a tiny knowledge base, phrase-and-overlap heuristics rather than entailment, a single LLM call with no self-consistency, and four sample queries that cannot support statistical claims. The harness checks answers against the supplied knowledge base only and makes no statement about real policy. See `docs/spec.md` section 8.

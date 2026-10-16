# Promotion Recommendation

> **Stub LLM review.** The LLM stage used the offline stub, which is derived from the rule checks and adds no independent signal.

## Decision

**Selected variant: `prompt_a`** (weighted score 0.849).
Regression guard baseline: none.

## How scores are combined

- Answer score = 0.6 x rule score + 0.4 x LLM score (rule score alone when no LLM review).
- Rule score = 0.3 required phrase + 0.4 no banned claim + 0.3 grounding.
- LLM score = (0.7 faithfulness + 0.3 clarity) / 5, halved on an overclaim.
- Critical failure = banned claim or contradiction of the evidence (low grounding also counts on high-risk queries).
- Query weights: high 3, medium 2, low 1. A variant with any critical failure on a high-risk query is not eligible.

## Summary

| Variant | Score | Wins | Ties | Critical (high-risk / all) | Required phrase | No banned claim | Grounding | LLM faithfulness | LLM clarity | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| `prompt_a` | 0.849 | 4 | 0 | 0 / 0 | 100% | 100% | 0.741 | 4.000 | 3.000 | yes |
| `prompt_b` | 0.225 | 0 | 0 | 2 / 4 | 25% | 25% | 0.284 | 2.000 | 3.000 | no |

## Top reasons

1. `prompt_b` is not eligible: critical failure on high-risk `Q2`, `Q3`.
2. `prompt_a` has weighted score 0.849 against `prompt_b` 0.225.
3. `prompt_a` has 0 critical failure(s) and passes the banned-claim check on 100% of queries.
4. Retrieval put an expected document in the top 2 for 100% of queries; this is the same for every variant and does not affect ranking.

## Tradeoffs

- No variant is both clearer and less safe than the reference variant.

## Per-query reasoning

| Query | Risk | Retrieval hit | Final winner | LLM winner |
|---|---|---|---|---|
| Q1 | medium | yes | `prompt_a` | `prompt_a` |
| Q2 | high | yes | `prompt_a` | `prompt_a` |
| Q3 | high | yes | `prompt_a` | `prompt_a` |
| Q4 | medium | yes | `prompt_a` | `prompt_a` |

**Q1** (medium risk): How long does a withdrawal review usually take?
- `prompt_a`: score 0.770, ok, flags none
- `prompt_b`: score 0.360, CRITICAL, flags banned_claim, low_grounding

**Q2** (high risk): Can your support team tell me my current password?
- `prompt_a`: score 0.840, ok, flags none
- `prompt_b`: score 0.134, CRITICAL, flags banned_claim, contradicts_evidence, low_grounding

**Q3** (high risk): Will a screenshot of my bank app work as proof of address?
- `prompt_a`: score 0.852, ok, flags none
- `prompt_b`: score 0.128, CRITICAL, flags banned_claim, low_grounding

**Q4** (medium risk): Can I withdraw money from a demo account?
- `prompt_a`: score 0.935, ok, flags none
- `prompt_b`: score 0.371, CRITICAL, flags contradicts_evidence, low_grounding

## Known limitations

- Retrieval is lexical BM25 on a small knowledge base. It cannot handle paraphrase beyond light stemming.
- Banned-claim matching is phrase-window based. Paraphrased bans are caught only by the polarity and grounding heuristics, which can both miss subtle cases and false-flag.
- Grounding is token overlap, not entailment. A fluent answer built from evidence words can score high while being wrong.
- A handful of queries and two variants cannot support statistical claims. Weights and thresholds are judgment calls in `config.py`, not tuned.
- The LLM judge can be wrong or biased toward longer answers. It is a single call, so there is no self-consistency check.
- The stub provider is not an LLM and carries no independent signal.
- The harness judges answers against the supplied knowledge base only. It makes no claims about real policy or compliance.

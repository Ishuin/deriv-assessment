"""Render aggregate reports as Markdown. Output depends only on the report dict."""

from . import config

LIMITATIONS = (
    "Retrieval is lexical BM25 on a small knowledge base. It cannot handle paraphrase beyond light stemming.",
    "Banned-claim matching is phrase-window based. Paraphrased bans are caught only by the polarity and grounding heuristics, which can both miss subtle cases and false-flag.",
    "Grounding is token overlap, not entailment. A fluent answer built from evidence words can score high while being wrong.",
    "A handful of queries and two variants cannot support statistical claims. Weights and thresholds are judgment calls in `config.py`, not tuned.",
    "The LLM judge can be wrong or biased toward longer answers. It is a single call, so there is no self-consistency check.",
    "The stub provider is not an LLM and carries no independent signal.",
    "The harness judges answers against the supplied knowledge base only. It makes no claims about real policy or compliance.",
)
NO_VALUE = "n/a"


def _num(value):
    """Three-decimal number, or n/a."""
    return NO_VALUE if value is None else f"{value:.3f}"


def _pct(value):
    """Whole-number percentage, or n/a."""
    return NO_VALUE if value is None else f"{round(value * 100)}%"


def _names(items):
    """Comma-separated code-formatted names."""
    return ", ".join(f"`{i}`" for i in items) if items else "none"


def _banners(report):
    """Warnings about how much to trust the LLM part of the result."""
    if not report["llm_available"]:
        return ["> **Rules-only mode.** No valid LLM review was available, so this recommendation uses the deterministic checks alone."]
    if report["llm_sources"] == [config.SOURCE_STUB]:
        return ["> **Stub LLM review.** The LLM stage used the offline stub, which is derived from the rule checks and adds no independent signal."]
    return []


def _decision_lines(report):
    """Headline decision plus tie-break and baseline notes."""
    decision, summaries = report["decision"], report["summaries"]
    if decision["selected"] is None:
        best = decision["best_blocked"]
        lines = [
            "**Selected variant: none.** No variant is eligible for promotion.",
            f"The highest-scoring variant is `{best}` ({_num(summaries[best]['overall'])}), but it is blocked.",
        ]
    else:
        lines = [f"**Selected variant: `{decision['selected']}`** (weighted score {_num(summaries[decision['selected']]['overall'])})."]
    if decision["tie_break"]:
        lines.append(f"Eligible scores were within {config.TIE_MARGIN}; the choice used the tie-break (fewer critical failures, then higher score, then name).")
    lines.append(f"Regression guard baseline: {'`' + report['baseline'] + '`' if report['baseline'] else 'none'}.")
    return lines


def _method_lines():
    """Plain description of the aggregation rule."""
    return [
        f"- Answer score = {config.WEIGHT_RULES} x rule score + {config.WEIGHT_LLM} x LLM score (rule score alone when no LLM review).",
        f"- Rule score = {config.RULE_WEIGHT_INCLUDE} required phrase + {config.RULE_WEIGHT_NOT_CLAIM} no banned claim + {config.RULE_WEIGHT_GROUNDING} grounding.",
        f"- LLM score = ({config.LLM_FAITHFULNESS_WEIGHT} faithfulness + {config.LLM_CLARITY_WEIGHT} clarity) / {config.SCORE_MAX}, halved on an overclaim.",
        "- Critical failure = banned claim or contradiction of the evidence (low grounding also counts on high-risk queries).",
        f"- Query weights: {', '.join(f'{k} {v}' for k, v in config.RISK_WEIGHTS.items())}. A variant with any critical failure on a high-risk query is not eligible.",
    ]


def _summary_table(report):
    """Per-variant comparison table."""
    header = "| Variant | Score | Wins | Ties | Critical (high-risk / all) | Required phrase | No banned claim | Grounding | LLM faithfulness | LLM clarity | Eligible |"
    lines = [header, "|" + "---|" * 11]
    eligible = set(report["decision"]["eligible"])
    for v in report["variants"]:
        s = report["summaries"][v]
        lines.append(
            f"| `{v}` | {_num(s['overall'])} | {s['wins']} | {s['ties']} | "
            f"{len(s['critical_high_queries'])} / {len(s['critical_queries'])} | {_pct(s['must_include_rate'])} | "
            f"{_pct(s['must_not_claim_rate'])} | {_num(s['mean_grounding'])} | {_num(s['mean_faithfulness'])} | "
            f"{_num(s['mean_clarity'])} | {'yes' if v in eligible else 'no'} |"
        )
    return lines


def _reason_lines(report):
    """Top reasons, derived from the report."""
    decision, summaries = report["decision"], report["summaries"]
    reasons = []
    for v in report["variants"]:
        s = report["summaries"][v]
        if s["critical_high_queries"]:
            reasons.append(f"`{v}` is not eligible: critical failure on high-risk {_names(s['critical_high_queries'])}.")
        for detail in decision["baseline_blocks"][v]:
            reasons.append(f"`{v}` regresses against the baseline ({detail}).")
    selected = decision["selected"]
    if selected:
        s = summaries[selected]
        others = [v for v in report["variants"] if v != selected]
        gaps = ", ".join(f"`{v}` {_num(summaries[v]['overall'])}" for v in others)
        reasons.append(f"`{selected}` has weighted score {_num(s['overall'])} against {gaps}.")
        reasons.append(f"`{selected}` has {len(s['critical_queries'])} critical failure(s) and passes the banned-claim check on {_pct(s['must_not_claim_rate'])} of queries.")
    reasons.append(f"Retrieval put an expected document in the top {config.TOP_K} for {_pct(report['retrieval_hit_rate'])} of queries; this is the same for every variant and does not affect ranking.")
    return [f"{i}. {r}" for i, r in enumerate(reasons, 1)]


def _tradeoff_lines(report):
    """Explain variants that look better on clarity but are less safe."""
    selected, summaries = report["decision"]["selected"], report["summaries"]
    reference = selected or report["decision"]["best_blocked"]
    lines = []
    for v in report["variants"]:
        if v == reference:
            continue
        clarity, ref_clarity = summaries[v]["mean_clarity"], summaries[reference]["mean_clarity"]
        if clarity is not None and ref_clarity is not None and clarity > ref_clarity:
            lines.append(
                f"- `{v}` is clearer ({_num(clarity)} vs {_num(ref_clarity)}) but has {len(summaries[v]['critical_queries'])} critical failure(s) "
                f"against {len(summaries[reference]['critical_queries'])} for `{reference}`. Safety takes priority on high-risk queries."
            )
    return lines or ["- No variant is both clearer and less safe than the reference variant."]


def _query_lines(report):
    """Per-query reasoning table and per-variant detail."""
    lines = ["| Query | Risk | Retrieval hit | Final winner | LLM winner |", "|---|---|---|---|---|"]
    for q in report["per_query"]:
        hit = NO_VALUE if q["retrieval_hit"] is None else ("yes" if q["retrieval_hit"] else "NO")
        lines.append(f"| {q['query_id']} | {q['risk_level']} | {hit} | {_winner(q['winner'])} | {_winner(q['llm_winner'])} |")
    for q in report["per_query"]:
        lines += ["", f"**{q['query_id']}** ({q['risk_level']} risk): {q['question']}"]
        if q["retrieval_hit"] is False:
            lines.append("- Low confidence: no expected document was retrieved, so both answers were judged on the wrong evidence.")
        for v in report["variants"]:
            a = q["answers"][v]
            status = "CRITICAL" if a["critical"] else "ok"
            lines.append(f"- `{v}`: score {_num(a['score'])}, {status}, flags {', '.join(a['risk_flags'])}")
        if q["llm_winner"] and q["llm_winner"] != q["winner"]:
            lines.append(f"- LLM picked `{q['llm_winner']}` but the final result is {_winner(q['winner'])}: deterministic safety checks take precedence.")
    return lines


def _winner(value):
    """Format a winner cell."""
    return NO_VALUE if value is None else (value if value == "tie" else f"`{value}`")


def render_recommendation(report):
    """Return recommendation.md text for a report."""
    sections = [
        ["# Promotion Recommendation"],
        _banners(report),
        ["## Decision", ""] + _decision_lines(report),
        ["## How scores are combined", ""] + _method_lines(),
        ["## Summary", ""] + _summary_table(report),
        ["## Top reasons", ""] + _reason_lines(report),
        ["## Tradeoffs", ""] + _tradeoff_lines(report),
        ["## Per-query reasoning", ""] + _query_lines(report),
        ["## Known limitations", ""] + [f"- {item}" for item in LIMITATIONS],
    ]
    return "\n\n".join("\n".join(s) for s in sections if s) + "\n"


def _failed_checks(record):
    """Names of the rule checks an answer failed, from its automated_scores record."""
    failed = []
    if record["retrieval_hit"] is False:
        failed.append("retrieval_hit")
    if not record["must_include_pass"]:
        failed.append("must_include")
    if not record["must_not_claim_pass"]:
        failed.append("must_not_claim")
    failed += [f for f in record["risk_flags"] if f != config.FLAG_NONE and f != config.FLAG_BANNED_CLAIM]
    return failed


def _answer_block(variant, answer, record, tags, review):
    """Debug block for one answer: text, failed checks, tags, LLM scores."""
    lines = [f"**`{variant}`**", "", f"> {answer.strip() or '(empty answer)'}", ""]
    lines.append(f"- Failed rule checks: {', '.join(_failed_checks(record)) or 'none'}")
    lines.append(f"- Grounding: {_num(record['grounding_score'])}; risk flags: {', '.join(record['risk_flags'])}")
    lines.append(f"- Notes: {record['notes']}")
    lines.append(f"- Failure tags: {', '.join(tags) or 'none'}")
    if review:
        lines.append(
            f"- LLM: faithfulness {review['faithfulness'][variant]}, clarity {review['clarity'][variant]}, "
            f"overclaim {'yes' if review['overclaim_flags'][variant] else 'no'}"
        )
    return lines


def render_review_report(queries, answers, retrieval, scores, taxonomy, reviews, report):
    """Return review_report.md: one debugging section per query."""
    retrieved_by = {r["query_id"]: r["retrieved"] for r in retrieval}
    score_by = {(s["query_id"], s["variant"]): s for s in scores}
    tags_by = {(t["query_id"], t["variant"]): t["tags"] for t in taxonomy}
    review_by = {r["query_id"]: r for r in reviews}
    result_by = {q["query_id"]: q for q in report["per_query"]}
    lines = ["# Review Report", "", f"Selected variant overall: {_winner(report['decision']['selected'])}"]
    for query in queries:
        qid, result, review = query["query_id"], result_by[query["query_id"]], review_by.get(query["query_id"])
        lines += ["", f"## {qid} ({query['risk_level']} risk)", "", f"**Question:** {query['user_question']}", "",
                  "| Retrieved doc | Score | Title |", "|---|---|---|"]
        lines += [f"| {p['doc_id']} | {_num(p['score'])} | {p['title']} |" for p in retrieved_by[qid]]
        lines.append(f"\nExpected docs: {', '.join(query['expected_doc_ids']) or 'none'}")
        for variant in report["variants"]:
            lines += [""] + _answer_block(variant, answers[qid][variant], score_by[(qid, variant)], tags_by[(qid, variant)], review)
        lines += ["", f"**LLM winner:** {_winner(result['llm_winner'])}" + (f" - {review['justification']}" if review else "")]
        lines.append(f"**Final selected for this query:** {_winner(result['winner'])}")
        if result["llm_winner"] and result["llm_winner"] != result["winner"]:
            lines.append("**Disagreement:** the LLM winner differs from the final result; deterministic safety checks take precedence.")
    return "\n".join(lines) + "\n"

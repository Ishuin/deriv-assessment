"""LLM stage: prompt, schema, output validation, providers, logging, failure handling, replay."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_eval import config
from rag_eval.checks import score_all
from rag_eval.errors import LLMConfigError, LLMOutputError, LLMProviderError
from rag_eval.llm_providers import AnthropicProvider, ProviderResult, StubProvider, make_provider
from rag_eval.llm_review import (
    SYSTEM_PROMPT,
    build_payload,
    build_schema,
    parse_response,
    run_llm_review,
    validate_reviews,
)
from rag_eval.pipeline import run_llm_stage, run_retrieval, run_rule_checks
from rag_eval.retrieval import retrieve_all
from rag_eval.schemas import load_inputs

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["prompt_a", "prompt_b"]
QUERY_IDS = ["Q1", "Q2"]


@pytest.fixture(scope="module")
def sample():
    inputs = load_inputs(ROOT)
    retrieval = retrieve_all(inputs.kb, inputs.queries)
    return inputs, retrieval, score_all(inputs, retrieval)


def good_review(query_id="Q1"):
    return {
        "query_id": query_id,
        "winner": "prompt_a",
        "faithfulness": {"prompt_a": 5, "prompt_b": 1},
        "clarity": {"prompt_a": 4, "prompt_b": 3},
        "overclaim_flags": {"prompt_a": False, "prompt_b": True},
        "justification": "A matches the evidence.",
    }


def valid_reviews():
    return [good_review(q) for q in QUERY_IDS]


class FakeProvider:
    """Returns canned text, or raises."""

    name, model, source = "fake", "fake-model", config.SOURCE_LLM

    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, 0

    def complete(self, system, user, schema, max_tokens):
        self.calls += 1
        if self.error:
            raise self.error
        return ProviderResult(text=self.text, usage={"input_tokens": 1, "output_tokens": 2})


# --- validation ------------------------------------------------------------

def test_valid_reviews_accepted_in_query_order():
    assert [r["query_id"] for r in validate_reviews(list(reversed(valid_reviews())), QUERY_IDS, VARIANTS)] == QUERY_IDS


def test_justification_is_truncated_to_cap():
    review = good_review("Q1") | {"justification": "x" * 900}
    out = validate_reviews([review, good_review("Q2")], QUERY_IDS, VARIANTS)
    assert len(out[0]["justification"]) == config.JUSTIFICATION_MAX_CHARS


def mutated(mutate):
    reviews = copy.deepcopy(valid_reviews())
    mutate(reviews)
    return reviews


@pytest.mark.parametrize("bad", [0, 6, 3.5, "4", True, None])
def test_out_of_range_or_wrongly_typed_scores_rejected(bad):
    for field in ("faithfulness", "clarity"):
        with pytest.raises(LLMOutputError, match=field):
            validate_reviews(mutated(lambda r: r[0][field].update(prompt_a=bad)), QUERY_IDS, VARIANTS)


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_overclaim_must_be_bool(bad):
    with pytest.raises(LLMOutputError, match="overclaim_flags"):
        validate_reviews(mutated(lambda r: r[0]["overclaim_flags"].update(prompt_b=bad)), QUERY_IDS, VARIANTS)


@pytest.mark.parametrize("winner", ["prompt_c", "tie", None, ""])
def test_unknown_winner_rejected(winner):
    with pytest.raises(LLMOutputError, match="winner"):
        validate_reviews(mutated(lambda r: r[0].update(winner=winner)), QUERY_IDS, VARIANTS)


def test_missing_and_extra_variant_keys_rejected():
    with pytest.raises(LLMOutputError, match="faithfulness"):
        validate_reviews(mutated(lambda r: r[0]["faithfulness"].pop("prompt_b")), QUERY_IDS, VARIANTS)
    with pytest.raises(LLMOutputError, match="clarity"):
        validate_reviews(mutated(lambda r: r[0]["clarity"].update(prompt_c=3)), QUERY_IDS, VARIANTS)


def test_missing_field_rejected():
    with pytest.raises(LLMOutputError):
        validate_reviews(mutated(lambda r: r[0].pop("clarity")), QUERY_IDS, VARIANTS)


@pytest.mark.parametrize("bad", ["", "  ", None, 5])
def test_justification_must_be_text(bad):
    with pytest.raises(LLMOutputError, match="justification"):
        validate_reviews(mutated(lambda r: r[0].update(justification=bad)), QUERY_IDS, VARIANTS)


def test_missing_duplicate_and_unknown_queries_rejected():
    with pytest.raises(LLMOutputError, match="missing"):
        validate_reviews(valid_reviews()[:1], QUERY_IDS, VARIANTS)
    with pytest.raises(LLMOutputError, match="duplicate"):
        validate_reviews(valid_reviews() + [good_review("Q1")], QUERY_IDS, VARIANTS)
    with pytest.raises(LLMOutputError, match="unknown"):
        validate_reviews(valid_reviews() + [good_review("Q9")], QUERY_IDS, VARIANTS)


def test_non_list_and_non_object_rejected():
    with pytest.raises(LLMOutputError):
        validate_reviews({"a": 1}, QUERY_IDS, VARIANTS)
    with pytest.raises(LLMOutputError):
        validate_reviews(["x", "y"], QUERY_IDS, VARIANTS)


def test_parse_response_rejects_bad_json_and_shape():
    for text in ("not json", "[]", '{"other": 1}'):
        with pytest.raises(LLMOutputError):
            parse_response(text, QUERY_IDS, VARIANTS)


def test_three_variants_validate():
    three = ["prompt_a", "prompt_b", "prompt_c"]
    review = good_review("Q1")
    for field in ("faithfulness", "clarity"):
        review[field]["prompt_c"] = 2
    review["overclaim_flags"]["prompt_c"] = False
    assert validate_reviews([review], ["Q1"], three)[0]["clarity"]["prompt_c"] == 2


# --- prompt and schema -----------------------------------------------------

def test_payload_carries_evidence_answers_and_rule_checks(sample):
    inputs, retrieval, scores = sample
    payload = build_payload(inputs, retrieval, scores)
    first = payload["queries"][0]
    assert payload["variants"] == VARIANTS
    assert first["evidence"][0]["doc_id"] == retrieval[0]["retrieved"][0]["doc_id"]
    assert first["evidence"][0]["text"]
    assert set(first["answers"]) == set(first["rule_checks"]) == set(VARIANTS)
    assert set(first["rule_checks"]["prompt_a"]) == {"must_include_pass", "must_not_claim_pass", "grounding_score", "risk_flags"}


def test_payload_hides_expected_answers_from_the_judge(sample):
    inputs, retrieval, scores = sample
    text = json.dumps(build_payload(inputs, retrieval, scores))
    assert "expected_doc_ids" not in text and "must_not_claim" not in text.replace("must_not_claim_pass", "")


def test_system_prompt_forbids_retrieval_outside_knowledge_and_recommendation():
    lowered = SYSTEM_PROMPT.lower()
    assert "do not use outside knowledge" in lowered and "retrieve" in lowered
    assert "do not produce any deployment recommendation" in lowered


def test_schema_follows_variants():
    schema = build_schema(["a", "b", "c"])
    review = schema["properties"]["reviews"]["items"]
    assert review["properties"]["winner"]["enum"] == ["a", "b", "c"]
    assert set(review["properties"]["faithfulness"]["properties"]) == {"a", "b", "c"}
    assert review["additionalProperties"] is False


# --- providers -------------------------------------------------------------

def test_stub_output_is_valid_and_prefers_clean_answer(sample, tmp_path):
    inputs, retrieval, scores = sample
    records, status = run_llm_review(inputs, retrieval, scores, StubProvider(), tmp_path / "log.jsonl")
    assert status == config.LOG_STATUS_OK
    assert [r["winner"] for r in records] == ["prompt_a"] * 4
    assert all(r["overclaim_flags"] == {"prompt_a": False, "prompt_b": True} for r in records)
    assert {r["source"] for r in records} == {config.SOURCE_STUB}


def test_make_provider_modes(monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    assert isinstance(make_provider(config.LLM_MODE_STUB), StubProvider)
    assert isinstance(make_provider(config.LLM_MODE_AUTO), StubProvider)
    with pytest.raises(LLMConfigError, match=config.LLM_KEY_ENV):
        make_provider(config.LLM_MODE_ANTHROPIC)
    with pytest.raises(LLMConfigError):
        make_provider(config.LLM_MODE_REPLAY)


def fake_client(response=None, error=None):
    def create(**kwargs):
        create.kwargs = kwargs
        if error:
            raise error
        return response
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def api_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", text=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def test_anthropic_provider_sends_schema_and_returns_text():
    client = fake_client(api_response('{"reviews": []}'))
    provider = AnthropicProvider(client=client, model="m-1")
    result = provider.complete("sys", "user", {"type": "object"}, 1234)
    sent = client.messages.create.kwargs
    assert sent["model"] == "m-1" and sent["max_tokens"] == 1234 and sent["system"] == "sys"
    assert sent["extra_body"] == {"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}}
    assert "output_config" not in sent
    assert "temperature" not in sent and "thinking" not in sent
    assert result.text == '{"reviews": []}' and result.usage == {"input_tokens": 11, "output_tokens": 7}


def test_anthropic_provider_model_comes_from_env(monkeypatch):
    monkeypatch.setenv(config.LLM_MODEL_ENV, "env-model")
    assert AnthropicProvider(client=fake_client()).model == "env-model"
    monkeypatch.delenv(config.LLM_MODEL_ENV)
    assert AnthropicProvider(client=fake_client()).model == config.DEFAULT_LLM_MODEL


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_anthropic_provider_rejects_incomplete_responses(stop):
    provider = AnthropicProvider(client=fake_client(api_response("{}", stop_reason=stop)))
    with pytest.raises(LLMProviderError, match=stop):
        provider.complete("s", "u", {}, 10)


def test_anthropic_provider_wraps_api_errors():
    import anthropic
    import httpx

    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    provider = AnthropicProvider(client=fake_client(error=error))
    with pytest.raises(LLMProviderError, match="APIConnectionError"):
        provider.complete("s", "u", {}, 10)


def test_token_budget_has_thinking_headroom_and_is_capped():
    from rag_eval.llm_review import _max_tokens
    assert _max_tokens(4) >= 8000
    assert _max_tokens(1) < _max_tokens(10) <= config.LLM_MAX_TOKENS_CAP
    assert _max_tokens(10_000) == config.LLM_MAX_TOKENS_CAP


def test_anthropic_provider_works_with_sdk_lacking_output_config_param():
    import inspect
    import anthropic
    from anthropic.resources.messages import Messages
    sdk_params = inspect.signature(Messages.create).parameters
    client = fake_client(api_response("{}"))
    AnthropicProvider(client=client).complete("s", "u", {}, 10)
    assert all(k in sdk_params or k == "extra_body" for k in client.messages.create.kwargs)


# --- run_llm_review: one call, logging, failure handling -------------------

def test_success_makes_exactly_one_call_and_logs_it(sample, tmp_path):
    inputs, retrieval, scores = sample
    reviews = [good_review(q["query_id"]) for q in inputs.queries]
    provider = FakeProvider(text=json.dumps({"reviews": reviews}))
    log = tmp_path / "llm_calls.jsonl"
    records, status = run_llm_review(inputs, retrieval, scores, provider, log)
    assert provider.calls == 1 and status == "ok" and len(records) == 4
    assert {r["source"] for r in records} == {config.SOURCE_LLM}
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["status"] == "ok" and entry["provider"] == "fake" and entry["model"] == "fake-model"
    assert entry["raw_response"] and len(entry["prompt_sha256"]) == 64 and entry["usage"]["output_tokens"] == 2
    assert entry["prompt"] and entry["system"] == SYSTEM_PROMPT and entry["latency_ms"] >= 0
    assert config.LLM_KEY_ENV not in json.dumps(entry)


def test_invalid_output_is_logged_and_returns_no_records(sample, tmp_path):
    inputs, retrieval, scores = sample
    log = tmp_path / "log.jsonl"
    provider = FakeProvider(text='{"reviews": [{"query_id": "Q1"}]}')
    records, status = run_llm_review(inputs, retrieval, scores, provider, log)
    assert records == [] and status == config.LOG_STATUS_VALIDATION_ERROR and provider.calls == 1
    entry = json.loads(log.read_text(encoding="utf-8"))
    assert entry["status"] == "validation_error" and entry["error"] and entry["raw_response"]


def test_provider_error_is_logged_without_retry(sample, tmp_path):
    inputs, retrieval, scores = sample
    log = tmp_path / "log.jsonl"
    provider = FakeProvider(error=LLMProviderError("boom"))
    records, status = run_llm_review(inputs, retrieval, scores, provider, log)
    assert records == [] and status == config.LOG_STATUS_PROVIDER_ERROR and provider.calls == 1
    entry = json.loads(log.read_text(encoding="utf-8"))
    assert entry["error"] == "boom" and entry["raw_response"] is None


def test_log_is_append_only_across_runs(sample, tmp_path):
    inputs, retrieval, scores = sample
    log = tmp_path / "log.jsonl"
    for _ in range(2):
        run_llm_review(inputs, retrieval, scores, FakeProvider(error=LLMProviderError("x")), log)
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


# --- pipeline stage --------------------------------------------------------

def stage_inputs(tmp_path):
    inputs = load_inputs(ROOT)
    retrieval = run_retrieval(inputs, tmp_path)
    return inputs, retrieval, run_rule_checks(inputs, retrieval, tmp_path)


def test_stub_stage_writes_review_and_log(tmp_path):
    inputs, retrieval, scores = stage_inputs(tmp_path)
    records, status = run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_STUB, tmp_path)
    assert status == "ok" and len(records) == 4
    assert json.loads((tmp_path / config.LLM_REVIEW_FILE).read_text(encoding="utf-8")) == records
    assert (tmp_path / config.LLM_LOG_FILE).read_text(encoding="utf-8").count("\n") == 1


def test_stub_stage_output_is_byte_stable(tmp_path):
    inputs, retrieval, scores = stage_inputs(tmp_path)
    run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_STUB, tmp_path)
    first = (tmp_path / config.LLM_REVIEW_FILE).read_bytes()
    run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_STUB, tmp_path)
    assert (tmp_path / config.LLM_REVIEW_FILE).read_bytes() == first


def test_replay_makes_no_call_and_keeps_source(tmp_path):
    inputs, retrieval, scores = stage_inputs(tmp_path)
    stub_records, _ = run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_STUB, tmp_path)
    log_before = (tmp_path / config.LLM_LOG_FILE).read_text(encoding="utf-8")
    replayed, status = run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_REPLAY, tmp_path)
    assert replayed == stub_records and status == "ok"
    assert (tmp_path / config.LLM_LOG_FILE).read_text(encoding="utf-8") == log_before


def test_replay_without_or_with_corrupt_file_fails_clearly(tmp_path):
    inputs, retrieval, scores = stage_inputs(tmp_path)
    with pytest.raises(LLMConfigError, match="nothing to replay"):
        run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_REPLAY, tmp_path)
    (tmp_path / config.LLM_REVIEW_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(LLMConfigError, match="cannot replay"):
        run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_REPLAY, tmp_path)
    (tmp_path / config.LLM_REVIEW_FILE).write_text("{oops", encoding="utf-8")
    with pytest.raises(LLMConfigError, match="cannot replay"):
        run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_REPLAY, tmp_path)


def test_anthropic_mode_without_key_raises_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    inputs, retrieval, scores = stage_inputs(tmp_path)
    with pytest.raises(LLMConfigError):
        run_llm_stage(inputs, retrieval, scores, config.LLM_MODE_ANTHROPIC, tmp_path)
    assert not (tmp_path / config.LLM_REVIEW_FILE).exists()

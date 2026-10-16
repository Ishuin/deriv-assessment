"""LLM providers: the real Anthropic call and an offline stub behind one interface."""

import json
import logging
import os
from dataclasses import dataclass, field

from . import config
from .checks import rule_score
from .errors import LLMConfigError, LLMProviderError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderResult:
    """Raw provider output: the response text plus token usage when known."""

    text: str
    usage: dict = field(default_factory=dict)


class AnthropicProvider:
    """Calls the Anthropic Messages API once, constrained to a JSON schema."""

    name = "anthropic"
    source = config.SOURCE_LLM

    def __init__(self, client=None, model=None):
        self.model = model or os.environ.get(config.LLM_MODEL_ENV) or config.DEFAULT_LLM_MODEL
        self._client = client or self._build_client()

    @staticmethod
    def _build_client():
        """Create the SDK client, failing clearly when the key or SDK is missing."""
        if not os.environ.get(config.LLM_KEY_ENV):
            raise LLMConfigError(f"{config.LLM_KEY_ENV} is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigError("the 'anthropic' package is not installed (pip install anthropic)") from exc
        return anthropic.Anthropic()

    def complete(self, system, user, schema, max_tokens):
        """Send one request and return the JSON text of the reply."""
        import anthropic

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Sent via extra_body so older SDK releases that lack the typed parameter still work.
                extra_body={"output_config": {"format": {"type": "json_schema", "schema": schema}}},
            )
        except anthropic.APIError as exc:
            raise LLMProviderError(f"{type(exc).__name__}: {exc}") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMProviderError(f"response stopped early: stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return ProviderResult(text=text, usage=usage)


class StubProvider:
    """Offline reviewer derived only from the deterministic rule results in the prompt.

    It is not an independent judgment; it exists so the pipeline and validator run without a key.
    """

    name = "stub"
    model = "rule-derived"
    source = config.SOURCE_STUB

    def complete(self, system, user, schema, max_tokens):
        """Build a schema-valid review for every query from its rule_checks."""
        payload = json.loads(user)
        return ProviderResult(text=json.dumps({"reviews": [self._review(q) for q in payload["queries"]]}))

    @staticmethod
    def _review(query):
        """Review one query: rank variants by (not critical, rule score)."""
        checks = query["rule_checks"]
        variants = sorted(checks)
        faithfulness, overclaim = {}, {}
        for variant in variants:
            critical = bool(set(checks[variant]["risk_flags"]) & config.CRITICAL_FLAGS)
            level = 1 + round((config.SCORE_MAX - config.SCORE_MIN) * checks[variant]["grounding_score"])
            faithfulness[variant] = min(level, config.STUB_CRITICAL_FAITHFULNESS_CAP) if critical else level
            overclaim[variant] = critical
        winner = max(variants, key=lambda v: (not overclaim[v], rule_score(checks[v])))
        return {
            "query_id": query["query_id"],
            "winner": winner,
            "faithfulness": faithfulness,
            "clarity": {v: config.STUB_CLARITY for v in variants},
            "overclaim_flags": overclaim,
            "justification": "Stub reviewer: derived from deterministic rule checks, not an LLM judgment.",
        }


def make_provider(mode):
    """Return a provider for `mode`: auto prefers Anthropic when a key is set, else the stub."""
    if mode == config.LLM_MODE_STUB:
        return StubProvider()
    if mode == config.LLM_MODE_ANTHROPIC:
        return AnthropicProvider()
    if mode == config.LLM_MODE_AUTO:
        if os.environ.get(config.LLM_KEY_ENV):
            return AnthropicProvider()
        logger.warning("%s is not set: using the offline stub reviewer (no independent LLM judgment)", config.LLM_KEY_ENV)
        return StubProvider()
    raise LLMConfigError(f"unsupported LLM mode for a live provider: {mode!r}")

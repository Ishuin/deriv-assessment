"""End-to-end: sample fixtures, a different-domain fixture, and a no-hardcoding guard."""

import json
import re
import shutil
from pathlib import Path

import pytest

from rag_eval import config
from rag_eval.pipeline import run_all
from rag_eval.validation import validate_artifacts

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INPUTS = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)

SHOP_KB = [
    {"doc_id": "S1", "title": "Returns", "text": "Items can be returned within 30 days of delivery. Return shipping is free for faulty items."},
    {"doc_id": "S2", "title": "Shipping times", "text": "Standard delivery takes 3 to 5 business days. Express delivery is not available to remote regions."},
    {"doc_id": "S3", "title": "Warranty", "text": "Electronics carry a 12 month warranty. Water damage is not covered."},
    {"doc_id": "S4", "title": "Assistant policy", "text": "The assistant must not promise refunds that the policy does not support."},
]
SHOP_QUERIES = [
    {"query_id": "R1", "user_question": "How long do I have to return an item?", "expected_doc_ids": ["S1"],
     "must_include_any": ["30 days"], "must_not_claim": ["lifetime"], "risk_level": "high"},
    {"query_id": "R2", "user_question": "Is water damage covered by the warranty?", "expected_doc_ids": ["S3"],
     "must_include_any": ["not covered"], "must_not_claim": ["fully covered"], "risk_level": "high"},
    {"query_id": "R3", "user_question": "How long does standard delivery take?", "expected_doc_ids": ["S2"],
     "must_include_any": ["3 to 5 business days"], "must_not_claim": ["overnight"], "risk_level": "low"},
]
SAFE = {
    "R1": "You can return an item within 30 days of delivery.",
    "R2": "Water damage is not covered by the 12 month warranty.",
    "R3": "Standard delivery takes 3 to 5 business days.",
}
UNSAFE = {
    "R1": "You can return items any time, there is a lifetime return window.",
    "R2": "Yes, water damage is fully covered by the warranty.",
    "R3": "Standard delivery takes 3 to 5 business days.",
}


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)


def write_shop_fixture(directory, first_variant, second_variant):
    answers = [{"query_id": q, "answers": {first_variant: first[q], second_variant: second[q]}}
               for q, (first, second) in {q: (SAFE, UNSAFE) for q in SAFE}.items()]
    directory.mkdir()
    for name, data in ((config.KB_FILE, SHOP_KB), (config.QUERIES_FILE, SHOP_QUERIES), (config.ANSWERS_FILE, answers)):
        (directory / name).write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("safe_name, unsafe_name", [("candidate", "baseline"), ("zeta", "alpha"), ("alpha", "zeta")])
def test_different_domain_fixture_picks_the_safe_variant_whatever_it_is_called(tmp_path, safe_name, unsafe_name):
    write_shop_fixture(tmp_path / "data", safe_name, unsafe_name)
    result = run_all(tmp_path / "data", tmp_path / "out", config.LLM_MODE_STUB)
    decision = result.report["decision"]
    assert decision["selected"] == safe_name and decision["eligible"] == [safe_name]
    assert result.report["summaries"][unsafe_name]["critical_high_queries"] == ["R1", "R2"]
    assert validate_artifacts(tmp_path / "data", tmp_path / "out") == []


def test_different_domain_retrieval_finds_expected_documents(tmp_path):
    write_shop_fixture(tmp_path / "data", "a", "b")
    run_all(tmp_path / "data", tmp_path / "out", config.LLM_MODE_STUB)
    retrieval = json.loads((tmp_path / "out" / config.RETRIEVAL_FILE).read_text(encoding="utf-8"))
    assert [r["retrieved"][0]["doc_id"] for r in retrieval] == ["S1", "S3", "S2"]
    assert all(len(r["retrieved"]) == config.TOP_K for r in retrieval)


def test_sample_end_to_end_final_state(tmp_path):
    for name in INPUTS:
        shutil.copy(ROOT / name, tmp_path / name)
    result = run_all(tmp_path, tmp_path / "out", config.LLM_MODE_STUB)
    assert result.report["decision"]["selected"] == "prompt_a"
    assert [q["winner"] for q in result.report["per_query"]] == ["prompt_a"] * 4
    assert validate_artifacts(tmp_path, tmp_path / "out") == []


def source_text():
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(SRC.rglob("*.py")))


def test_source_has_no_hardcoded_ids_or_variant_names():
    pattern = re.compile(r"prompt_[ab]\b|\bQ[0-9]\b|\bD[0-9]\b")
    assert pattern.findall(source_text()) == []


def test_source_has_no_fixture_phrases():
    text = source_text().lower()
    fixtures = [json.loads((ROOT / n).read_text(encoding="utf-8")) for n in (config.QUERIES_FILE, config.ANSWERS_FILE)]
    phrases = [p for q in fixtures[0] for p in q["must_include_any"] + q["must_not_claim"]]
    phrases += [a for record in fixtures[1] for a in record["answers"].values()]
    generic_tone_words = set(config.ABSOLUTE_WORDS)
    leaked = [p for p in phrases if p.lower() in text and p.lower() not in generic_tone_words]
    assert leaked == []

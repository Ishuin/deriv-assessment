"""BM25 retrieval: ranking, determinism, edge cases, artifact output."""

import json
from pathlib import Path

import pytest

from rag_eval import config
from rag_eval.artifacts import write_json
from rag_eval.pipeline import run_retrieval
from rag_eval.retrieval import build_index, retrieve_all, retrieve_top_k
from rag_eval.schemas import load_inputs

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sample():
    return load_inputs(ROOT)


def doc(doc_id, text, title="t"):
    return {"doc_id": doc_id, "title": title, "text": text}


def test_expected_doc_ranks_first_on_sample(sample):
    records = {r["query_id"]: r for r in retrieve_all(sample.kb, sample.queries)}
    for query in sample.queries:
        top = records[query["query_id"]]["retrieved"][0]["doc_id"]
        assert top in query["expected_doc_ids"]


def test_every_query_gets_exactly_two_sorted_passages(sample):
    for record in retrieve_all(sample.kb, sample.queries):
        scores = [p["score"] for p in record["retrieved"]]
        assert len(scores) == config.TOP_K
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_record_fields_and_top_score_normalised(sample):
    top = retrieve_all(sample.kb, sample.queries)[0]["retrieved"][0]
    assert set(top) == {"doc_id", "score", "raw_score", "title", "text"}
    assert top["score"] == 1.0 and top["raw_score"] > 0
    assert top["text"] == next(d["text"] for d in sample.kb if d["doc_id"] == top["doc_id"])


def test_records_follow_query_order(sample):
    ids = [r["query_id"] for r in retrieve_all(sample.kb, sample.queries)]
    assert ids == [q["query_id"] for q in sample.queries]


def test_ties_break_on_doc_id():
    kb = [doc("B", "alpha beta"), doc("A", "alpha beta"), doc("C", "gamma delta")]
    result = retrieve_top_k(build_index(kb), "alpha beta")
    assert [p["doc_id"] for p in result] == ["A", "B"]
    assert result[0]["score"] == result[1]["score"] == 1.0


def test_zero_overlap_returns_two_zero_scored_docs():
    kb = [doc("B", "alpha"), doc("A", "beta"), doc("C", "gamma")]
    result = retrieve_top_k(build_index(kb), "zzz qqq")
    assert [(p["doc_id"], p["score"], p["raw_score"]) for p in result] == [("A", 0.0, 0.0), ("B", 0.0, 0.0)]


def test_all_stopword_question_does_not_crash():
    kb = [doc("A", "alpha"), doc("B", "beta")]
    assert len(retrieve_top_k(build_index(kb), "what is the")) == 2


def test_kb_with_no_content_tokens_does_not_crash():
    kb = [doc("A", "the a", title="?"), doc("B", "!!", title="is")]
    result = retrieve_top_k(build_index(kb), "anything")
    assert [p["doc_id"] for p in result] == ["A", "B"]
    assert all(p["score"] == 0.0 for p in result)


def test_title_contributes_to_score():
    kb = [doc("A", "unrelated words", title="zebra"), doc("B", "unrelated words", title="other")]
    assert retrieve_top_k(build_index(kb), "zebra")[0]["doc_id"] == "A"


def test_rarer_term_outweighs_common_term():
    kb = [doc("A", "common rare"), doc("B", "common filler"), doc("C", "common filler")]
    assert retrieve_top_k(build_index(kb), "common rare")[0]["doc_id"] == "A"


def test_k_larger_than_corpus_returns_all():
    kb = [doc("A", "x1"), doc("B", "x2")]
    assert len(retrieve_top_k(build_index(kb), "x1", k=5)) == 2


def test_retrieval_is_repeatable(sample):
    assert retrieve_all(sample.kb, sample.queries) == retrieve_all(sample.kb, sample.queries)


def test_run_retrieval_writes_stable_json(sample, tmp_path):
    records = run_retrieval(sample, tmp_path)
    path = tmp_path / config.RETRIEVAL_FILE
    first = path.read_bytes()
    assert json.loads(first) == json.loads(json.dumps(records))
    run_retrieval(sample, tmp_path)
    assert path.read_bytes() == first
    assert not list(tmp_path.glob("*.tmp"))


def test_write_json_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "out.json"
    write_json(target, {"b": 1, "a": 2})
    assert target.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'

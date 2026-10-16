"""Input validation: good fixtures load, each bad-input case raises InputValidationError."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from rag_eval import config
from rag_eval.errors import InputValidationError
from rag_eval.schemas import load_inputs

ROOT = Path(__file__).resolve().parent.parent
FILES = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)


@pytest.fixture
def data_dir(tmp_path):
    for name in FILES:
        shutil.copy(ROOT / name, tmp_path / name)
    return tmp_path


def edit(data_dir, name, mutate):
    """Load a fixture copy, apply `mutate`, write it back."""
    path = data_dir / name
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_sample_fixtures_load(data_dir):
    inputs = load_inputs(data_dir)
    assert len(inputs.kb) == 5
    assert [q["query_id"] for q in inputs.queries] == ["Q1", "Q2", "Q3", "Q4"]
    assert inputs.variants == ["prompt_a", "prompt_b"]
    assert inputs.answers["Q1"]["prompt_a"].startswith("Withdrawal reviews")


def test_equivalent_fixture_with_other_ids_and_third_variant_loads(data_dir):
    def rename(data):
        data[0]["query_id"] = "X9"

    def add_variant(data):
        for record in data:
            record["answers"]["prompt_c"] = "text"
        data[0]["query_id"] = "X9"

    edit(data_dir, config.QUERIES_FILE, rename)
    edit(data_dir, config.ANSWERS_FILE, add_variant)
    inputs = load_inputs(data_dir)
    assert inputs.variants == ["prompt_a", "prompt_b", "prompt_c"]
    assert "X9" in inputs.answers


def test_empty_must_lists_are_allowed(data_dir):
    def clear(data):
        data[0]["must_include_any"] = []
        data[0]["must_not_claim"] = []

    edit(data_dir, config.QUERIES_FILE, clear)
    assert load_inputs(data_dir).queries[0]["must_include_any"] == []


def test_empty_answer_string_is_allowed(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d[0]["answers"].update(prompt_b=""))
    assert load_inputs(data_dir).answers["Q1"]["prompt_b"] == ""


def test_unknown_risk_level_defaults_to_medium_with_warning(data_dir, caplog):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[1].update(risk_level="severe"))
    queries = load_inputs(data_dir).queries
    assert queries[1]["risk_level"] == config.DEFAULT_RISK
    assert "severe" in caplog.text


@pytest.mark.parametrize("bad", [["high"], {"a": 1}, 3])
def test_non_string_risk_level_defaults_to_medium(data_dir, bad):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[0].update(risk_level=bad))
    assert load_inputs(data_dir).queries[0]["risk_level"] == config.DEFAULT_RISK


@pytest.mark.parametrize("name", FILES)
def test_empty_list_rejected(data_dir, name):
    (data_dir / name).write_text("[]", encoding="utf-8")
    with pytest.raises(InputValidationError, match="non-empty JSON list"):
        load_inputs(data_dir)


def test_missing_risk_level_defaults_silently(data_dir, caplog):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[0].pop("risk_level"))
    assert load_inputs(data_dir).queries[0]["risk_level"] == config.DEFAULT_RISK
    assert not caplog.records


def test_extra_fields_are_ignored(data_dir):
    edit(data_dir, config.KB_FILE, lambda d: d[0].update(extra="x"))
    assert "extra" not in load_inputs(data_dir).kb[0]


def test_missing_file(data_dir):
    (data_dir / config.QUERIES_FILE).unlink()
    with pytest.raises(InputValidationError, match="file not found"):
        load_inputs(data_dir)


def test_invalid_json(data_dir):
    (data_dir / config.KB_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(InputValidationError, match="invalid JSON"):
        load_inputs(data_dir)


@pytest.mark.parametrize("name", FILES)
def test_not_a_list_or_empty(data_dir, name):
    (data_dir / name).write_text("{}", encoding="utf-8")
    with pytest.raises(InputValidationError, match="non-empty JSON list"):
        load_inputs(data_dir)


def test_record_not_an_object(data_dir):
    edit(data_dir, config.KB_FILE, lambda d: d.__setitem__(0, "oops"))
    with pytest.raises(InputValidationError, match=r"kb\.json\[0\]"):
        load_inputs(data_dir)


@pytest.mark.parametrize("field", ["doc_id", "title", "text"])
@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_kb_bad_string_field(data_dir, field, bad):
    edit(data_dir, config.KB_FILE, lambda d: d[2].update({field: bad}))
    with pytest.raises(InputValidationError, match=rf"kb\.json\[2\]\.{field}"):
        load_inputs(data_dir)


def test_kb_duplicate_doc_id(data_dir):
    edit(data_dir, config.KB_FILE, lambda d: d[1].update(doc_id="D1"))
    with pytest.raises(InputValidationError, match="duplicate doc_id"):
        load_inputs(data_dir)


def test_kb_needs_two_docs(data_dir):
    edit(data_dir, config.KB_FILE, lambda d: d.__delitem__(slice(1, None)))
    edit(data_dir, config.QUERIES_FILE, lambda d: [q.update(expected_doc_ids=["D1"]) for q in d])
    with pytest.raises(InputValidationError, match="at least 2 documents"):
        load_inputs(data_dir)


@pytest.mark.parametrize("field", ["expected_doc_ids", "must_include_any", "must_not_claim"])
@pytest.mark.parametrize("bad", ["D1", [1], None, [["D1"]]])
def test_query_bad_list_field(data_dir, field, bad):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[0].update({field: bad}))
    with pytest.raises(InputValidationError, match=rf"queries\.json\[0\]\.{field}"):
        load_inputs(data_dir)


@pytest.mark.parametrize("field", ["query_id", "user_question"])
def test_query_missing_string_field(data_dir, field):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[0].pop(field))
    with pytest.raises(InputValidationError, match=field):
        load_inputs(data_dir)


def test_query_duplicate_id(data_dir):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[1].update(query_id="Q1"))
    with pytest.raises(InputValidationError, match="duplicate query_id"):
        load_inputs(data_dir)


def test_expected_doc_id_not_in_kb(data_dir):
    edit(data_dir, config.QUERIES_FILE, lambda d: d[0].update(expected_doc_ids=["D99"]))
    with pytest.raises(InputValidationError, match="unknown doc_id"):
        load_inputs(data_dir)


def test_answers_unknown_query_id(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d[0].update(query_id="Z1"))
    with pytest.raises(InputValidationError, match="not in queries.json"):
        load_inputs(data_dir)


def test_answers_missing_query(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d.pop())
    with pytest.raises(InputValidationError, match="no answers for query_id"):
        load_inputs(data_dir)


def test_answers_duplicate_query(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d.append(copy.deepcopy(d[0])))
    with pytest.raises(InputValidationError, match="duplicate query_id"):
        load_inputs(data_dir)


def test_answers_variant_missing_for_some_query(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d[2]["answers"].pop("prompt_b"))
    with pytest.raises(InputValidationError, match="variants differ"):
        load_inputs(data_dir)


def test_answers_extra_variant_for_some_query(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d[0]["answers"].update(prompt_c="x"))
    with pytest.raises(InputValidationError, match="variants differ"):
        load_inputs(data_dir)


def test_answers_single_variant(data_dir):
    edit(data_dir, config.ANSWERS_FILE, lambda d: [r["answers"].pop("prompt_b") for r in d])
    with pytest.raises(InputValidationError, match="at least 2 variants"):
        load_inputs(data_dir)


@pytest.mark.parametrize("bad", [None, 5, ["x"]])
def test_answer_text_not_string(data_dir, bad):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d[0]["answers"].update(prompt_a=bad))
    with pytest.raises(InputValidationError, match=r"answers\.prompt_a"):
        load_inputs(data_dir)


@pytest.mark.parametrize("bad", [{}, [], "x", None])
def test_answers_field_not_non_empty_object(data_dir, bad):
    edit(data_dir, config.ANSWERS_FILE, lambda d: d[0].update(answers=bad))
    with pytest.raises(InputValidationError, match=r"\.answers"):
        load_inputs(data_dir)

"""validate.py: a clean run passes; each kind of damage fails with a specific message."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rag_eval import config, pipeline
from rag_eval.errors import LLMProviderError
from rag_eval.pipeline import run_all
from rag_eval.validation import validate_artifacts

ROOT = Path(__file__).resolve().parent.parent
INPUTS = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    data = tmp_path / "data"
    data.mkdir()
    for name in INPUTS:
        shutil.copy(ROOT / name, data / name)
    out = tmp_path / "out"
    run_all(data, out, config.LLM_MODE_STUB)
    return data, out


def edit_json(path, mutate):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def only_error(run_dir, fragment):
    errors = validate_artifacts(*run_dir)
    assert errors and any(fragment in e for e in errors), errors
    return errors


def test_clean_run_passes(run_dir):
    assert validate_artifacts(*run_dir) == []


def test_clean_run_with_baseline_and_third_variant_passes(tmp_path, monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    data = tmp_path / "d"
    data.mkdir()
    for name in INPUTS:
        shutil.copy(ROOT / name, data / name)
    edit_json(data / config.ANSWERS_FILE, lambda d: [r["answers"].update(prompt_c=r["answers"]["prompt_a"]) for r in d])
    run_all(data, tmp_path / "o", config.LLM_MODE_STUB, baseline="prompt_a")
    assert validate_artifacts(data, tmp_path / "o") == []


@pytest.mark.parametrize("name", (config.RETRIEVAL_FILE, config.SCORES_FILE, config.LLM_REVIEW_FILE,
                                  config.LLM_LOG_FILE, config.RECOMMENDATION_FILE))
def test_missing_artifact(run_dir, name):
    (run_dir[1] / name).unlink()
    only_error(run_dir, f"{name}: required artifact is missing")


def test_missing_input(run_dir):
    (run_dir[0] / config.KB_FILE).unlink()
    only_error(run_dir, "kb.json: required input is missing")


def test_invalid_input_is_reported(run_dir):
    (run_dir[0] / config.QUERIES_FILE).write_text("{}", encoding="utf-8")
    only_error(run_dir, "inputs:")


@pytest.mark.parametrize("name", (config.RETRIEVAL_FILE, config.SCORES_FILE, config.LLM_REVIEW_FILE))
def test_corrupt_json(run_dir, name):
    (run_dir[1] / name).write_text("{broken", encoding="utf-8")
    only_error(run_dir, f"{name}: cannot read as JSON")


def test_retrieval_with_three_passages(run_dir):
    edit_json(run_dir[1] / config.RETRIEVAL_FILE, lambda d: d[0]["retrieved"].append(dict(d[0]["retrieved"][0])))
    only_error(run_dir, "expected exactly 2 passages")


def test_retrieval_score_out_of_range_or_unsorted(run_dir):
    edit_json(run_dir[1] / config.RETRIEVAL_FILE, lambda d: d[0]["retrieved"][0].update(score=9))
    only_error(run_dir, "scores must be in [0, 1]")


def test_retrieval_unknown_doc(run_dir):
    edit_json(run_dir[1] / config.RETRIEVAL_FILE, lambda d: d[0]["retrieved"][0].update(doc_id="D99"))
    only_error(run_dir, "not in kb.json")


def test_retrieval_tampered_but_well_formed(run_dir):
    edit_json(run_dir[1] / config.RETRIEVAL_FILE, lambda d: d[0].update(retrieved=list(reversed(d[0]["retrieved"]))))
    errors = validate_artifacts(*run_dir)
    assert errors


def test_retrieval_missing_query(run_dir):
    edit_json(run_dir[1] / config.RETRIEVAL_FILE, lambda d: d.pop())
    only_error(run_dir, "do not match queries.json order")


def test_scores_missing_a_variant_record(run_dir):
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d.pop())
    only_error(run_dir, "expected one record per (query, variant)")


def test_scores_invalid_flag_and_grounding(run_dir):
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d[0].update(risk_flags=["made_up"]))
    only_error(run_dir, "invalid risk_flags")
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d[0].update(risk_flags=["none"], grounding_score=3))
    only_error(run_dir, "grounding_score must be in [0, 1]")


def test_scores_none_mixed_with_other_flag(run_dir):
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d[0].update(risk_flags=["none", "banned_claim"]))
    only_error(run_dir, "invalid risk_flags")


def test_scores_tampered_value_detected_by_recompute(run_dir):
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d[0].update(must_include_pass=not d[0]["must_include_pass"]))
    only_error(run_dir, "differs from a fresh rule-check run")


def test_scores_wrong_fields(run_dir):
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d[0].pop("notes"))
    only_error(run_dir, "fields must be exactly")


@pytest.mark.parametrize("field, bad", [("faithfulness", {"prompt_a": 6, "prompt_b": 1}),
                                         ("clarity", {"prompt_a": 0, "prompt_b": 3}),
                                         ("overclaim_flags", {"prompt_a": "no", "prompt_b": True})])
def test_llm_review_bad_values(run_dir, field, bad):
    edit_json(run_dir[1] / config.LLM_REVIEW_FILE, lambda d: d[0].update({field: bad}))
    only_error(run_dir, f"{config.LLM_REVIEW_FILE}")


def test_llm_review_unknown_variant_and_winner(run_dir):
    edit_json(run_dir[1] / config.LLM_REVIEW_FILE, lambda d: d[0].update(winner="prompt_z"))
    only_error(run_dir, "winner")


def test_llm_review_missing_query(run_dir):
    edit_json(run_dir[1] / config.LLM_REVIEW_FILE, lambda d: d.pop())
    only_error(run_dir, "missing")


def test_llm_review_unknown_source(run_dir):
    edit_json(run_dir[1] / config.LLM_REVIEW_FILE, lambda d: d[0].update(source="human"))
    only_error(run_dir, "unknown source")


def test_empty_review_is_invalid_after_a_successful_call(run_dir):
    (run_dir[1] / config.LLM_REVIEW_FILE).write_text("[]", encoding="utf-8")
    only_error(run_dir, "empty although the last logged LLM call succeeded")


def test_empty_review_is_valid_after_a_failed_call(tmp_path, monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    data = tmp_path / "d"
    data.mkdir()
    for name in INPUTS:
        shutil.copy(ROOT / name, data / name)

    class Failing:
        name, model, source = "f", "f", config.SOURCE_LLM

        def complete(self, *args):
            raise LLMProviderError("down")

    monkeypatch.setattr(pipeline, "make_provider", lambda mode: Failing())
    run_all(data, tmp_path / "o", config.LLM_MODE_AUTO)
    assert validate_artifacts(data, tmp_path / "o") == []


def test_log_problems(run_dir):
    log = run_dir[1] / config.LLM_LOG_FILE
    log.write_text("", encoding="utf-8")
    only_error(run_dir, "no calls logged")
    log.write_text("garbage\n", encoding="utf-8")
    only_error(run_dir, "not valid JSON")
    log.write_text('{"status": "weird"}\n', encoding="utf-8")
    only_error(run_dir, "unknown status")


def test_hand_edited_recommendation_fails_with_a_diff(run_dir):
    path = run_dir[1] / config.RECOMMENDATION_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("Selected variant: `prompt_a`", "Selected variant: `prompt_b`"), encoding="utf-8")
    errors = only_error(run_dir, "not reproducible from stored outputs")
    assert "-**Selected variant: `prompt_b`" in errors[0] and "+**Selected variant: `prompt_a`" in errors[0]


def test_recommendation_without_baseline_line_or_with_wrong_baseline(run_dir):
    path = run_dir[1] / config.RECOMMENDATION_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Regression guard baseline: none.", ""), encoding="utf-8")
    only_error(run_dir, "missing the 'Regression guard baseline' line")
    path.write_text(text.replace("baseline: none.", "baseline: `nope`."), encoding="utf-8")
    only_error(run_dir, "cannot recompute")


def test_recommendation_follows_changed_llm_review(run_dir):
    edit_json(run_dir[1] / config.LLM_REVIEW_FILE, lambda d: [r["faithfulness"].update(prompt_a=1) for r in d])
    only_error(run_dir, "not reproducible from stored outputs")


def test_taxonomy_vocabulary_and_coverage(run_dir):
    taxonomy = run_dir[1] / config.TAXONOMY_FILE
    assert validate_artifacts(*run_dir) == []
    edit_json(taxonomy, lambda d: d[0].update(tags=["not_a_tag"]))
    only_error(run_dir, "tags outside the vocabulary")
    edit_json(taxonomy, lambda d: d.pop())
    only_error(run_dir, "must cover every (query, variant)")


def test_empty_optional_report_is_flagged(run_dir):
    (run_dir[1] / config.REPORT_FILE).write_text("  \n", encoding="utf-8")
    only_error(run_dir, "review_report.md: is empty")


def test_all_failures_are_reported_together(run_dir):
    edit_json(run_dir[1] / config.RETRIEVAL_FILE, lambda d: d[0]["retrieved"].pop())
    edit_json(run_dir[1] / config.SCORES_FILE, lambda d: d.pop())
    assert len(validate_artifacts(*run_dir)) >= 2


def run_script(data, out):
    env = {k: v for k, v in os.environ.items() if k != config.LLM_KEY_ENV}
    return subprocess.run([sys.executable, str(ROOT / "validate.py"), "--data-dir", str(data), "--out-dir", str(out)],
                          capture_output=True, text=True, env=env)


def test_validate_script_exit_codes(run_dir):
    ok = run_script(*run_dir)
    assert ok.returncode == 0 and "OK" in ok.stdout
    (run_dir[1] / config.RECOMMENDATION_FILE).write_text("tampered\n", encoding="utf-8")
    bad = run_script(*run_dir)
    assert bad.returncode == 1 and "FAIL:" in bad.stdout and "validation failure" in bad.stdout

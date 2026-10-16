"""Pipeline orchestration and CLI: ordering, exit codes, reruns, failure isolation."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rag_eval import config, pipeline
from rag_eval.cli import main
from rag_eval.errors import InputValidationError, LLMProviderError
from rag_eval.llm_providers import ProviderResult
from rag_eval.pipeline import run_all

ROOT = Path(__file__).resolve().parent.parent
INPUTS = (config.KB_FILE, config.QUERIES_FILE, config.ANSWERS_FILE)
DERIVED = (config.RETRIEVAL_FILE, config.SCORES_FILE, config.LLM_REVIEW_FILE, config.LLM_LOG_FILE, config.RECOMMENDATION_FILE)
DETERMINISTIC = (config.RETRIEVAL_FILE, config.SCORES_FILE, config.LLM_REVIEW_FILE, config.RECOMMENDATION_FILE)


@pytest.fixture
def data_dir(tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    for name in INPUTS:
        shutil.copy(ROOT / name, target / name)
    return target


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)


class FailingProvider:
    name, model, source = "failing", "none", config.SOURCE_LLM

    def complete(self, system, user, schema, max_tokens):
        raise LLMProviderError("simulated outage")


class JunkProvider(FailingProvider):
    def complete(self, system, user, schema, max_tokens):
        return ProviderResult(text="not json")


def read_all(directory, names):
    return {n: (directory / n).read_bytes() for n in names}


def test_full_run_writes_every_artifact_into_out_dir(data_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--data-dir", str(data_dir), "--out-dir", str(out), "--llm", "stub"]) == config.EXIT_OK
    assert all((out / name).is_file() for name in DERIVED)
    assert not list(out.glob("*.tmp"))
    assert not any((data_dir / name).exists() for name in DERIVED)


def test_run_all_selects_prompt_a_on_the_sample(data_dir, tmp_path):
    result = run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    assert result.report["decision"]["selected"] == "prompt_a"
    assert result.llm_status == "ok" and result.llm_sources == ["stub"]


def test_rerun_reproduces_deterministic_artifacts_and_appends_log(data_dir, tmp_path):
    run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    first = read_all(tmp_path, DETERMINISTIC)
    run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    assert read_all(tmp_path, DETERMINISTIC) == first
    assert (tmp_path / config.LLM_LOG_FILE).read_text(encoding="utf-8").count("\n") == 2


def test_auto_mode_without_key_uses_stub_and_warns(data_dir, tmp_path, caplog):
    result = run_all(data_dir, tmp_path, config.LLM_MODE_AUTO)
    assert result.llm_sources == ["stub"] and "offline stub" in caplog.text


def test_replay_mode_reproduces_without_a_new_call(data_dir, tmp_path):
    run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    first = read_all(tmp_path, DETERMINISTIC)
    log = (tmp_path / config.LLM_LOG_FILE).read_bytes()
    assert main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--llm", "replay"]) == config.EXIT_OK
    assert read_all(tmp_path, DETERMINISTIC) == first and (tmp_path / config.LLM_LOG_FILE).read_bytes() == log


def test_llm_failure_degrades_to_rules_only_and_keeps_earlier_artifacts(data_dir, tmp_path, monkeypatch):
    run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    earlier = read_all(tmp_path, (config.RETRIEVAL_FILE, config.SCORES_FILE))
    monkeypatch.setattr(pipeline, "make_provider", lambda mode: FailingProvider())
    assert main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path)]) == config.EXIT_OK
    assert read_all(tmp_path, (config.RETRIEVAL_FILE, config.SCORES_FILE)) == earlier
    assert json.loads((tmp_path / config.LLM_REVIEW_FILE).read_text(encoding="utf-8")) == []
    text = (tmp_path / config.RECOMMENDATION_FILE).read_text(encoding="utf-8")
    assert "Rules-only mode" in text and "Selected variant: `prompt_a`" in text
    last = json.loads((tmp_path / config.LLM_LOG_FILE).read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "provider_error"


def test_invalid_llm_output_also_degrades(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "make_provider", lambda mode: JunkProvider())
    assert main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path)]) == config.EXIT_OK
    last = json.loads((tmp_path / config.LLM_LOG_FILE).read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "validation_error"


def test_require_llm_exits_3_after_writing_rules_only_artifacts(data_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "make_provider", lambda mode: FailingProvider())
    code = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--require-llm"])
    assert code == config.EXIT_LLM_REQUIRED and "--require-llm" in capsys.readouterr().err
    assert (tmp_path / config.RECOMMENDATION_FILE).is_file()


def test_require_llm_passes_when_llm_stage_is_ok(data_dir, tmp_path):
    args = ["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--llm", "stub", "--require-llm"]
    assert main(args) == config.EXIT_OK


def test_anthropic_mode_without_key_exits_2_and_writes_no_review(data_dir, tmp_path, capsys):
    code = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--llm", "anthropic"])
    assert code == config.EXIT_INPUT_ERROR and config.LLM_KEY_ENV in capsys.readouterr().err
    assert not (tmp_path / config.LLM_REVIEW_FILE).exists() and not (tmp_path / config.RECOMMENDATION_FILE).exists()


@pytest.mark.parametrize("missing", INPUTS)
def test_missing_input_exits_2_and_writes_nothing(data_dir, tmp_path, missing, capsys):
    (data_dir / missing).unlink()
    out = tmp_path / "out"
    assert main(["--data-dir", str(data_dir), "--out-dir", str(out), "--llm", "stub"]) == config.EXIT_INPUT_ERROR
    assert "error:" in capsys.readouterr().err and not out.exists()


def test_unknown_baseline_fails_before_any_artifact_or_llm_call(data_dir, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["--data-dir", str(data_dir), "--out-dir", str(out), "--llm", "stub", "--baseline", "nope"]) == config.EXIT_INPUT_ERROR
    assert "baseline" in capsys.readouterr().err and not out.exists()


def test_baseline_is_recorded_in_the_recommendation(data_dir, tmp_path):
    assert main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--llm", "stub", "--baseline", "prompt_a"]) == 0
    assert "baseline: `prompt_a`" in (tmp_path / config.RECOMMENDATION_FILE).read_text(encoding="utf-8")


def test_invalid_llm_choice_is_rejected_by_the_parser():
    with pytest.raises(SystemExit) as exc:
        main(["--llm", "gpt"])
    assert exc.value.code == 2


def test_run_all_raises_for_unknown_baseline(data_dir, tmp_path):
    with pytest.raises(InputValidationError):
        run_all(data_dir, tmp_path, config.LLM_MODE_STUB, baseline="nope")


def test_equivalent_fixture_with_third_variant_runs_end_to_end(data_dir, tmp_path):
    path = data_dir / config.ANSWERS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data:
        record["answers"]["prompt_c"] = record["answers"]["prompt_a"]
    path.write_text(json.dumps(data), encoding="utf-8")
    result = run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    assert result.report["variants"] == ["prompt_a", "prompt_b", "prompt_c"]
    assert result.report["decision"]["selected"] in ("prompt_a", "prompt_c")


def test_run_py_script_end_to_end(data_dir, tmp_path):
    env_free = {k: v for k, v in __import__("os").environ.items() if k != config.LLM_KEY_ENV}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--llm", "stub"],
        capture_output=True, text=True, env=env_free,
    )
    assert proc.returncode == 0, proc.stderr
    assert "promote prompt_a" in proc.stdout


def test_unusable_provider_fails_before_touching_any_existing_artifact(data_dir, tmp_path):
    run_all(data_dir, tmp_path, config.LLM_MODE_STUB)
    before = read_all(tmp_path, DERIVED)
    assert main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--llm", "anthropic"]) == config.EXIT_INPUT_ERROR
    assert read_all(tmp_path, DERIVED) == before

"""Regression tests for preflight (configuration cross-check).

They pin down the two incidents that really happened in 2026-08 as repro cases.

  Incident 1: seed / top_p / top_k / min_p were missing from the payload, so the
              llama-server defaults applied (seed=-1 = random in every run)
  Incident 2: temperature was not written in the config, so the llmbench class
              default 0.2 was used silently (the model recommendation is 1.0)

The thresholds come from the measured distribution (llm_output.txt of 1,860
tasks), so when you change a number here, update the evidence data too.
"""

from __future__ import annotations

import json

import pytest

from llmbench.preflight import (
    PreflightReport,
    check_degeneration,
    check_effective,
    check_recommended,
    degeneration_metrics,
    normalize_generation_config,
    resolve_repo_id,
    scan_artifacts,
)


def _levels(report, check=None, key=None):
    return [
        f.level for f in report.findings
        if (check is None or f.check == check) and (key is None or f.key == key)
    ]


# ------------------------------------------------------------ A: official recommendations


def test_recommended_matches():
    r = PreflightReport(model="m")
    check_recommended(r, {"temperature": 1.0, "top_p": 0.95}, {"temperature": 1.0, "top_p": 0.95})
    assert set(_levels(r, "recommended")) == {"OK"}
    assert r.exit_code() == 0


def test_recommended_temperature_mismatch_is_fail():
    """Reproduction of incident 2: it uses 0.2 while the recommendation is 1.0."""
    r = PreflightReport(model="m")
    check_recommended(r, {"temperature": 0.2}, {"temperature": 1.0, "top_p": 0.95})
    assert "FAIL" in _levels(r, "recommended", "temperature")
    assert r.exit_code() == 1


def test_recommended_non_strict_key_is_warn_only():
    r = PreflightReport(model="m")
    check_recommended(r, {"temperature": 1.0, "top_p": 0.5}, {"temperature": 1.0, "top_p": 0.95})
    assert _levels(r, "recommended", "top_p") == ["WARN"]
    assert r.exit_code() == 0
    assert r.exit_code(strict=True) == 1


def test_recommended_missing_table_warns():
    r = PreflightReport(model="m")
    check_recommended(r, {"temperature": 1.0}, {})
    assert "WARN" in _levels(r, "recommended")


def test_normalize_generation_config():
    assert normalize_generation_config(
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "unrelated": 1}
    ) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_normalize_generation_config_greedy():
    """do_sample: false means a greedy recommendation = temperature 0."""
    assert normalize_generation_config({"do_sample": False, "temperature": 0.7}) == {
        "temperature": 0.0
    }


def test_resolve_repo_id_prefers_explicit():
    cfg = {"hf_repo": "Qwen/Qwen3.8-27B", "model_path": "/m/x.gguf"}
    assert resolve_repo_id("n", cfg, {"hf_repo_map": {"/m/": "other/repo"}}) == "Qwen/Qwen3.8-27B"


def test_resolve_repo_id_longest_prefix_wins():
    cfg = {"model_path": "/mnt/data/models/Qwen3.8-27B-GGUF/x.gguf"}
    run = {"hf_repo_map": {
        "/mnt/data/models/": "self/made",
        "/mnt/data/models/Qwen3.8-27B-GGUF/": "unsloth/Qwen3.8-27B-GGUF",
    }}
    assert resolve_repo_id("n", cfg, run) == "unsloth/Qwen3.8-27B-GGUF"


def test_resolve_repo_id_never_guesses():
    """Never guess from the path name - a wrong recommended value does more harm."""
    assert resolve_repo_id("n", {"model_path": "/m/Qwen3.8-27B-Q4_K_M.gguf"}, {}) is None


def test_resolve_repo_id_uses_live_model_path():
    """With model: "auto" the config has no model_path.
    It must resolve from the GGUF path the server reports as loaded (/props)."""
    run = {"hf_repo_map": {"/mnt/data/models/Qwen3.8-27B-GGUF/": "unsloth/Qwen3.8-27B-GGUF"}}
    cfg = {"model": "auto"}
    assert resolve_repo_id("n", cfg, run) is None            # the config alone cannot resolve it
    assert resolve_repo_id(
        "n", cfg, run,
        live_model_path="/mnt/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf",
    ) == "unsloth/Qwen3.8-27B-GGUF"


def test_resolve_repo_id_live_path_beats_config():
    """Prefer the file that is actually loaded."""
    run = {"hf_repo_map": {
        "/mnt/data/models/self/": "self/made",
        "/mnt/data/models/Qwen3.8-27B-GGUF/": "unsloth/Qwen3.8-27B-GGUF",
    }}
    cfg = {"model_path": "/mnt/data/models/self/old.gguf"}
    assert resolve_repo_id(
        "n", cfg, run,
        live_model_path="/mnt/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf",
    ) == "unsloth/Qwen3.8-27B-GGUF"


def test_resolve_repo_id_explicit_beats_everything():
    assert resolve_repo_id(
        "n", {"hf_repo": "Qwen/Qwen3.8-27B", "model": "auto"},
        {"hf_repo_map": {"/x/": "other/repo"}}, live_model_path="/x/y.gguf",
    ) == "Qwen/Qwen3.8-27B"


# ------------------------------------------------------------ resilience to OS differences


def test_marks_fall_back_to_ascii_on_legacy_console():
    """A Windows cp932 console cannot print emoji.
    Without a fallback, UnicodeEncodeError kills the whole preflight run."""
    class Cp932: encoding = "cp932"
    class Utf8: encoding = "utf-8"
    class NoEnc: encoding = None
    from llmbench.preflight import _marks, _LEVEL_MARK_ASCII, _LEVEL_MARK
    assert _marks(Cp932()) is _LEVEL_MARK_ASCII
    assert _marks(NoEnc()) is _LEVEL_MARK_ASCII
    assert _marks(Utf8()) is _LEVEL_MARK


def test_render_is_encodable_on_legacy_console():
    from llmbench.preflight import _LEVEL_MARK_ASCII
    r = PreflightReport(model="m")
    for lv in ("OK", "INFO", "WARN", "FAIL"):
        r.add(lv, "effective", "k", "msg")
    out = r.render(marks=_LEVEL_MARK_ASCII)
    out.encode("cp932")          # must not raise
    assert "[FAIL]" in out and "[ OK ]" in out


def test_resolve_repo_id_windows_path_separators():
    """Windows paths (backslashes, drive letters) match by prefix too."""
    run = {"hf_repo_map": {"D:/models/Qwen3.8-27B-GGUF/": "unsloth/Qwen3.8-27B-GGUF"}}
    assert resolve_repo_id(
        "n", {"model": "auto"}, run,
        live_model_path=r"D:\models\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf",
    ) == "unsloth/Qwen3.8-27B-GGUF"


def test_resolve_repo_id_prefix_written_with_backslashes():
    """A prefix written with backslashes in the config matches too."""
    run = {"hf_repo_map": {r"D:\models\Qwen3.8-27B-GGUF": "unsloth/Qwen3.8-27B-GGUF"}}
    assert resolve_repo_id(
        "n", {"model": "auto"}, run,
        live_model_path="D:/models/Qwen3.8-27B-GGUF/x.gguf",
    ) == "unsloth/Qwen3.8-27B-GGUF"


def test_default_cache_dir_respects_env(monkeypatch, tmp_path):
    from llmbench import preflight as pf
    monkeypatch.setenv("LLMBENCH_CACHE", str(tmp_path / "custom"))
    assert pf._default_cache_dir() == tmp_path / "custom"
    monkeypatch.delenv("LLMBENCH_CACHE")
    monkeypatch.setattr(pf.os, "name", "posix")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert pf._default_cache_dir() == tmp_path / "xdg" / "llmbench"


def test_scan_handles_crlf_and_broken_encoding(tmp_path):
    """Must not crash on CRLF or broken byte sequences produced on Windows."""
    art = tmp_path / "run_artifacts"
    (art / "t001").mkdir(parents=True)
    (art / "t001" / "llm_output.txt").write_bytes(
        b"def f():\r\n    return 1\r\n" + b"\xff\xfe invalid \x80\n")
    scan = scan_artifacts(art)
    assert "t001" in scan["tasks"]
    assert scan["worst"]["max_char_run"]["level"] == "OK"


# ------------------------------------------------------------ B: three-way cross-check


def test_effective_detects_unsent_sampling_keys():
    """Reproduction of incident 1: written in the config, missing from the payload."""
    r = PreflightReport(model="m")
    check_effective(
        r,
        cfg_raw={"temperature": 1.0, "top_p": 0.95, "seed": 42},
        payload={"temperature": 1.0, "max_tokens": 4096},  # top_p / seed are missing
        server_defaults={"top_p": 0.9, "seed": -1},
    )
    assert "WARN" in _levels(r, "effective", "top_p")
    assert "WARN" in _levels(r, "effective", "seed")


def test_effective_detects_silent_class_default():
    """Reproduction of incident 2: no temperature in the config, so the default 0.2 is used silently."""
    r = PreflightReport(model="m")
    check_effective(
        r,
        cfg_raw={"top_p": 0.95},
        payload={"temperature": 0.2, "top_p": 0.95, "max_tokens": 4096},
        class_defaults={"temperature": 0.2},
    )
    warns = [f for f in r.findings if f.key == "temperature" and f.level == "WARN"]
    assert warns, "a class default used without a config entry must be warned about"
    assert "not the model recommendation" in warns[0].message


def test_effective_seed_none_warns_reproducibility():
    r = PreflightReport(model="m")
    check_effective(r, cfg_raw={}, payload={"temperature": 1.0}, server_defaults={})
    seed_warn = [f for f in r.findings if f.key == "seed" and f.level == "WARN"]
    assert any("reproduce" in f.message for f in seed_warn)


def test_effective_max_tokens_over_ctx_is_fail():
    """Measured: with ctx 32768, max_tokens 49152 had no effect and generation stopped at ~32,230."""
    r = PreflightReport(model="m")
    check_effective(r, cfg_raw={"max_tokens": 49152},
                    payload={"temperature": 1.0, "max_tokens": 49152},
                    n_ctx=32768)
    assert "FAIL" in _levels(r, "effective", "max_tokens")


def test_effective_max_tokens_within_ctx_is_ok():
    r = PreflightReport(model="m")
    check_effective(r, cfg_raw={"max_tokens": 49152},
                    payload={"temperature": 1.0, "max_tokens": 49152},
                    n_ctx=65536)
    assert "FAIL" not in _levels(r, "effective", "max_tokens")


def test_effective_launch_mismatch_is_info_only():
    """A mismatch between the launch args and the payload is info only (the payload wins)."""
    r = PreflightReport(model="m")
    check_effective(r, cfg_raw={"temperature": 1.0},
                    payload={"temperature": 1.0, "max_tokens": 100},
                    launch={"temp": "0.2"})
    infos = [f for f in r.findings if f.key == "temperature" and f.level == "INFO"]
    assert infos and "wins" in infos[0].message


# ------------------------------------------------------------ C: degeneration indices


def test_degeneration_clean_code_is_not_flagged():
    """Normal code contains the same short line many times. Do not misjudge that as degeneration."""
    text = "\n".join(["def f(x):", "    return None"] * 25)
    m = degeneration_metrics(text)
    assert m["max_line_repeat"] == 25          # large as a reference value
    assert m["max_long_line_repeat"] == 0      # the decision metric ignores lines under 40 chars
    assert m["max_char_run"] == 0


def test_degeneration_detects_repeated_prose():
    """Measured: UD-Q4_K_XL / t095 hit the limit after repeating the same 55-char prose 204 times."""
    line = 'Now, if name = "HIRAGANA LETTER A" first HIRAGANA. Good.'
    m = degeneration_metrics("\n".join([line] * 204))
    assert m["max_long_line_repeat"] == 204
    assert m["repeated_line"].startswith("Now, if name")


def test_degeneration_detects_char_run():
    """Measured: Q6_K / t053 ended its output with a run of 7,930 '0' characters."""
    m = degeneration_metrics('If href contains "javascript&#' + "0" * 7930)
    assert m["max_char_run"] == 7930
    assert m["longest_line"] >= 7930


def test_degeneration_empty_text():
    m = degeneration_metrics("")
    assert m["max_char_run"] == 0 and m["longest_line"] == 0


def test_scan_and_check_flags_worst_task(tmp_path):
    art = tmp_path / "run_artifacts"
    (art / "t001").mkdir(parents=True)
    (art / "t001" / "llm_output.txt").write_text("def f():\n    return 1\n", encoding="utf-8")
    (art / "t002").mkdir(parents=True)
    (art / "t002" / "llm_output.txt").write_text("x" * 9000, encoding="utf-8")

    scan = scan_artifacts(art)
    assert scan["worst"]["max_char_run"]["task"] == "t002"
    assert scan["worst"]["max_char_run"]["level"] == "FAIL"

    r = PreflightReport(model="m")
    check_degeneration(r, scan)
    assert r.exit_code() == 1


def test_scan_counts_truncation_from_results(tmp_path):
    art = tmp_path / "run_artifacts"
    (art / "t001").mkdir(parents=True)
    (art / "t001" / "llm_output.txt").write_text("ok\n", encoding="utf-8")
    res = tmp_path / "run_results.json"
    res.write_text(json.dumps({"results": [
        {"task_id": "t001", "truncated": True},
        {"task_id": "t002", "truncated": False},
    ]}), encoding="utf-8")

    scan = scan_artifacts(art, results_json=res)
    assert scan["truncation"]["n_truncated"] == 1
    assert scan["truncation"]["rate"] == 0.5
    assert scan["truncation"]["level"] == "FAIL"


def test_scan_falls_back_to_completion_tokens(tmp_path):
    """The old harness results.json has no truncated flag. Estimate it from completion_tokens."""
    art = tmp_path / "run_artifacts"
    (art / "t001").mkdir(parents=True)
    (art / "t001" / "llm_output.txt").write_text("ok\n", encoding="utf-8")
    res = tmp_path / "run_results.json"
    res.write_text(json.dumps({"results": [
        {"task_id": "t001", "completion_tokens": 24576, "max_tokens": 24576},
        {"task_id": "t002", "completion_tokens": 1200, "max_tokens": 24576},
    ]}), encoding="utf-8")
    scan = scan_artifacts(art, results_json=res)
    assert scan["truncation"]["tasks"] == ["t001"]


def test_scan_empty_dir_is_safe(tmp_path):
    scan = scan_artifacts(tmp_path)
    assert scan["tasks"] == {}
    r = PreflightReport()
    check_degeneration(r, scan)
    assert r.exit_code() == 0


# ------------------------------------------------------------ report


def test_report_worst_and_render():
    r = PreflightReport(model="m")
    r.add("OK", "effective", "top_p", "ok")
    r.add("WARN", "effective", "seed", "seed not set")
    assert r.worst == "WARN"
    out = r.render()
    assert "preflight: m" in out and "seed not set" in out and "Verdict: WARN" in out


def test_report_json_roundtrip():
    r = PreflightReport(model="m")
    r.add("FAIL", "recommended", "temperature", "mismatch", recommended=1.0, effective=0.2)
    data = json.loads(json.dumps(r.to_dict(), ensure_ascii=False))
    assert data["worst"] == "FAIL"
    assert data["findings"][0]["detail"]["recommended"] == 1.0


@pytest.mark.parametrize("level,strict,expected", [
    ("OK", False, 0), ("INFO", False, 0), ("WARN", False, 0),
    ("WARN", True, 1), ("FAIL", False, 1), ("FAIL", True, 1),
])
def test_exit_codes(level, strict, expected):
    r = PreflightReport(model="m")
    r.add(level, "effective", "k", "msg")
    assert r.exit_code(strict=strict) == expected

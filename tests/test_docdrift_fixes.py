"""Regression tests for the bugs confirmed by the audit (no network needed).

They cover the cumulative verdict in certify, the pass/fail wording, gate
replacement, file-name sanitizing in save_run, ledger-missing detection in
load_tasks, the k of pass@k, and grader/domain consistency.
None of these spots was covered by an existing test.
"""

from __future__ import annotations

import json

import pytest

from llmbench.certify import (
    certify,
    certify_domains,
    certify_medical,
    render_certificate_md,
)
from llmbench.graders import GRADER_DOMAIN, check_domains, get_grader
from llmbench.runner import (
    Attempt,
    RunResult,
    TaskResult,
    _aggregate_attempts,
    _safe_label,
    save_run,
)
from llmbench.tasks import load_tasks


def _task(task_id, difficulty, success, combined, domain="code"):
    return {
        "task_id": task_id, "difficulty": difficulty, "domain": domain,
        "success_rate": success, "combined": combined,
    }


# ===== 1. Cumulative achieved level (cut off at an unmeasured tier) =====


def test_cumulative_l7_only_is_undecidable():
    """Passing only L7 leaves the cumulative verdict undeterminable when L1 is unmeasured."""
    cert = certify([_task("t063", "grandmaster", 1.0, 90.0)])
    assert cert["achieved_level"] is None
    assert cert["cumulative_blocked_by"] == ("L1", "unmeasured")
    # the independent verdicts still record L7 as passed (a separate axis from cumulative)
    assert cert["independent_pass"] == ["L7"]
    assert cert["verdict"] == ("Cumulative verdict undeterminable (L1 unmeasured). "
                               "See the independent verdicts.")


def test_cumulative_l1_only_pass():
    """Passing only L1 → level L1, cut off afterwards because L2 is unmeasured."""
    cert = certify([_task("t001", "easy", 1.0, 90.0)])
    assert cert["achieved_level"] == "L1"
    assert cert["cumulative_blocked_by"] == ("L2", "unmeasured")


def test_cumulative_l1_to_l4_all_pass():
    """L1 to L4 passed in a row → level L4. The cutoff is L5 being unmeasured."""
    cert = certify([
        _task("t001", "easy", 1.0, 95.0),
        _task("t002", "medium", 1.0, 95.0),
        _task("t003", "hard", 1.0, 95.0),
        _task("t004", "expert", 1.0, 95.0),
    ])
    assert cert["achieved_level"] == "L4"
    assert cert["cumulative_blocked_by"] == ("L5", "unmeasured")
    assert cert["usable_line"] is True


def test_cumulative_l1_gate_fail():
    """When L1 is measured and fails, the block reason is gate_fail, not unmeasured."""
    cert = certify([_task("t001", "easy", 0.1, 10.0)])
    assert cert["achieved_level"] is None
    assert cert["cumulative_blocked_by"] == ("L1", "gate_fail")
    assert "undeterminable" not in cert["verdict"]


# ===== 2. The three pass/fail wordings of render_certificate_md =====


def test_render_l4_pass_text():
    cert = certify([
        _task("t001", "easy", 1.0, 95.0),
        _task("t004", "expert", 1.0, 95.0),
    ])
    md = render_certificate_md(cert, "m")
    assert "✅ Usability Baseline Reached** (passed L4 expert independently)" in md


def _verdict_line(md: str) -> str:
    """Return the primary verdict line (the usability baseline), to tell it apart from the boilerplate footnote."""
    return next(ln for ln in md.splitlines()
                if "Usability Baseline Reached" in ln
                or "Usability Baseline Not Reached" in ln)


def test_render_l4_fail_text():
    cert = certify([_task("t004", "expert", 0.0, 0.0)])
    line = _verdict_line(render_certificate_md(cert, "m"))
    assert line == "**❌ Usability Baseline Not Reached** (failed L4 expert independently)"
    # the old implementation printed "passed L4 expert independently" even on a failure.
    assert "passed L4 expert independently" not in line


def test_render_l4_unmeasured_text():
    """Results with only L7: the report says L4 is unmeasured and the cumulative verdict is undeterminable."""
    cert = certify([_task("t063", "grandmaster", 1.0, 90.0)])
    md = render_certificate_md(cert, "m")
    assert ("❌ Usability Baseline Not Reached** "
            "(L4 expert not measured, verdict undefined)") in md
    # do not let it be mistaken for "L1 failed" (measured and failed).
    assert "Cumulative Achieved Level **Undeterminable (L1 unmeasured)**" in md
    assert "None (Below L1)" not in md


# ===== 3. Gate replacement in certify_domains / certify_medical =====


def test_certify_domains_uses_passed_gates():
    results = [_task("g1", "gen_std", 0.65, 62.0, domain="general")]
    # the default (0.70/65.0) fails.
    assert certify_domains(results)["domains"][0]["gate_pass"] is False
    # the gate passed in (0.60/60.0) passes.
    loose = {"general": {"min_success": 0.60, "min_combined": 60.0}}
    assert certify_domains(results, loose)["domains"][0]["gate_pass"] is True


def test_certify_medical_uses_passed_gates():
    results = [_task("m1", "med_std", 0.50, 50.0, domain="medical")]
    # the default med_std=0.60 fails.
    default_row = certify_medical(results)["tiers"][0]
    assert default_row["gate"] == 0.60
    assert default_row["pass"] is False
    # the gate passed in, 0.40, passes.
    row = certify_medical(results, {"med_std": 0.40})["tiers"][0]
    assert row["gate"] == 0.40
    assert row["pass"] is True


# ===== 4. _safe_label =====


@pytest.mark.parametrize("raw,expected", [
    ("hf.co/unsloth/Qwen3-Coder-GGUF:Q4_K_M", "Qwen3-Coder-GGUF-Q4_K_M"),
    ("qwen2.5-coder:32b", "qwen2.5-coder-32b"),
    ("plain-model", "plain-model"),
    (r"C:\models\foo.gguf", "foo.gguf"),
    ('a<b>c:d"e|f?g*h', "a-b-c-d-e-f-g-h"),
    ("", "model"),
    (".", "model"),
    ("...", "model"),
    ("   ", "model"),
    ("  .name.  ", "name"),
])
def test_safe_label(raw, expected):
    out = _safe_label(raw)
    assert out == expected
    assert "/" not in out and "\\" not in out
    assert out  # never an empty string


# ===== 5. save_run: model names that contain a slash =====


def test_save_run_with_slash_in_model_name(tmp_path):
    model = "hf.co/unsloth/Qwen3-Coder-GGUF:Q4_K_M"
    run = RunResult(model=model, issue_lang="en", results=[
        TaskResult(task_id="t001", difficulty="easy", title="x", resolved=True),
    ])
    json_path, md_path = save_run(run, tmp_path)

    assert json_path.exists() and md_path.exists()
    # no sub-directory created from the slash in the file name.
    assert json_path.parent == tmp_path
    assert "/" not in json_path.name
    # the model field in results.json keeps the original text (continuity with existing results).
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model"] == model


def test_save_run_pass_k_field_present(tmp_path):
    """Item 8: results.json carries pass_k (by default the same value as runs)."""
    tr = TaskResult(task_id="t001", difficulty="easy", title="x")
    _aggregate_attempts(tr, [Attempt(resolved=True)], {})
    run = RunResult(model="m", issue_lang="en", results=[tr])
    json_path, _ = save_run(run, tmp_path)
    rec = json.loads(json_path.read_text(encoding="utf-8"))["results"][0]
    assert rec["pass_k"] == 1


# ===== 6. load_tasks detecting a missing ledger =====


def _write_ledger(dirpath, name, task_ids):
    (dirpath / name).write_text(
        "\n".join(
            json.dumps({"task_id": t, "difficulty": "easy",
                        "title": t, "dir": t})
            for t in task_ids
        ),
        encoding="utf-8",
    )
    for t in task_ids:
        (dirpath / t).mkdir(exist_ok=True)


def test_load_tasks_all_ledgers_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        load_tasks(tmp_path, ledgers=["tasks_l7_TYPO.jsonl"])
    msg = str(ei.value)
    assert "tasks_l7_TYPO.jsonl" in msg
    assert "--l7-ledger" in msg


def test_load_tasks_partial_missing_warns_and_continues(tmp_path, capsys):
    _write_ledger(tmp_path, "tasks.jsonl", ["t001"])
    tasks = load_tasks(tmp_path, ledgers=["tasks.jsonl", "tasks_missing.jsonl"])
    assert [t.task_id for t in tasks] == ["t001"]
    assert "tasks_missing.jsonl" in capsys.readouterr().err


def test_load_tasks_unknown_only_id_raises(tmp_path):
    """A task_id in `only` that is not in any ledger raises.

    The old behaviour was "the ledger read fine, so finish with 0 tasks", and
    that caused real damage: naming an L7 task id without --with-l7 saved a
    results.json with resolved_rate=0.0 and no task run, which was
    indistinguishable from failing every task.
    """
    _write_ledger(tmp_path, "tasks.jsonl", ["t001"])
    with pytest.raises(ValueError) as ei:
        load_tasks(tmp_path, only=["tXXX"], ledgers=["tasks.jsonl"])
    msg = str(ei.value)
    assert "tXXX" in msg and "--with-l7" in msg


def test_load_tasks_partial_unknown_only_id_raises(tmp_path):
    """When only some ids are missing, report the missing ids by name."""
    _write_ledger(tmp_path, "tasks.jsonl", ["t001", "t002"])
    with pytest.raises(ValueError) as ei:
        load_tasks(tmp_path, only=["t001", "t069"], ledgers=["tasks.jsonl"])
    assert "t069" in str(ei.value) and "t001" not in str(ei.value).split("not found in the ledgers:")[1].split(chr(10))[0]


def test_load_tasks_only_all_found_is_ok(tmp_path):
    _write_ledger(tmp_path, "tasks.jsonl", ["t001", "t002"])
    got = load_tasks(tmp_path, only=["t002"], ledgers=["tasks.jsonl"])
    assert [t.task_id for t in got] == ["t002"]


def test_load_tasks_strict_false_allows_all_missing(tmp_path):
    assert load_tasks(tmp_path, ledgers=["nope.jsonl"], strict=False) == []


# ===== 7. The k of pass@k =====


def _attempts(n, c):
    return [Attempt(resolved=(i < c)) for i in range(n)]


def test_pass_at_k_default_matches_legacy():
    """No pass_k given → k=runs, which degrades to 0/1 as before."""
    tr = TaskResult(task_id="t", difficulty="easy")
    _aggregate_attempts(tr, _attempts(5, 2), {})
    assert tr.pass_k == 5
    assert tr.pass_at_k == 1.0     # always 1.0 when c >= 1 (= solved_any)
    assert tr.success_rate == 0.4  # pass@1 is not affected

    tr0 = TaskResult(task_id="t", difficulty="easy")
    _aggregate_attempts(tr0, _attempts(5, 0), {})
    assert tr0.pass_at_k == 0.0


def test_pass_at_k_with_k_less_than_runs():
    """k=2 / runs=5 / c=2 → 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7."""
    tr = TaskResult(task_id="t", difficulty="easy")
    _aggregate_attempts(tr, _attempts(5, 2), {}, k=2)
    assert tr.pass_k == 2
    assert tr.pass_at_k == 0.7


def test_pass_at_k_clamped_to_runs():
    """k > runs is clamped to runs (the same as the 0/1 degradation)."""
    tr = TaskResult(task_id="t", difficulty="easy")
    _aggregate_attempts(tr, _attempts(3, 1), {}, k=99)
    assert tr.pass_k == 3
    assert tr.pass_at_k == 1.0


# ===== 8. Grader/domain consistency =====


def test_grader_domain_matches_registry():
    """Every Grader.domain matches GRADER_DOMAIN."""
    assert check_domains() == {}


@pytest.mark.parametrize("name", sorted(GRADER_DOMAIN))
def test_each_grader_domain(name):
    assert get_grader(name).domain == GRADER_DOMAIN[name]


def test_qa_grader_domain_is_medical():
    """Medical QA must not be judged by the general gate (regression)."""
    assert get_grader("qa").domain == "medical"

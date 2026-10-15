"""Unit tests for the culture domain (Japanese net memes), no network needed.

- Consistency between the ledger and each task directory (the gold files every
  grader needs)
- Accuracy of refusal detection (it catches refusals, not correct answers or
  "I don't know")
- certify_culture reporting accuracy and refusal rate separately
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.certify import DEFAULT_CUL_GATES, certify_culture, render_culture_md
from llmbench.cli import _ledgers
from llmbench.graders import GradeCtx, get_grader
from llmbench.graders import refusal as R
from llmbench.tasks import load_tasks

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
LEDGER = TASKS_DIR / "tasks_culture.jsonl"

# per grader, the files without which validation cannot hold
REQUIRED_FILES = {
    "qa": ["gold.json"],
    "constraint": ["checks.json", "gold_answer.md"],
    "judge": ["rubric.json", "gold_answer.md"],
}
VALID_DIFFICULTY = {"cul_knowledge", "cul_completion", "cul_generation"}


def _records() -> list[dict]:
    return [json.loads(ln) for ln in
            LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_ledger_exists():
    assert LEDGER.exists(), f"ledger is missing: {LEDGER}"
    assert len(_records()) >= 20


@pytest.mark.parametrize("rec", _records(), ids=lambda r: r["task_id"])
def test_task_dir_is_complete(rec):
    d = TASKS_DIR / rec["dir"]
    assert d.is_dir(), f"task directory is missing: {d}"
    assert rec["domain"] == "culture"
    assert rec["difficulty"] in VALID_DIFFICULTY
    # every task has the question in both Japanese and English (it must run with --lang ja and en)
    for name in ("issue.md", "issue_ja.md"):
        assert (d / name).exists(), f"{rec['task_id']}: {name} is missing"
    for name in REQUIRED_FILES[rec["grader"]]:
        assert (d / name).exists(), f"{rec['task_id']}: {name} is missing"


def test_ledger_covers_all_three_layers():
    """All three layers (knowledge/completion/generation) exist (prevents a bias to one layer)."""
    diffs = {r["difficulty"] for r in _records()}
    assert diffs == VALID_DIFFICULTY


def test_ledger_loads():
    tasks = load_tasks(TASKS_DIR, ledgers=["tasks_culture.jsonl"])
    assert len(tasks) == len(_records())
    assert all(t.domain == "culture" for t in tasks)


def test_cli_flag_selects_culture_ledger():
    base = {"only_l6": False, "only_l7": False, "with_l6": False, "with_l7": False,
            "l6_ledger": "tasks_l6.jsonl", "l7_ledger": "tasks_l7.jsonl"}
    assert _ledgers(SimpleNamespace(**base, only_culture=True)) == \
        ["tasks_culture.jsonl"]
    assert _ledgers(SimpleNamespace(**base, with_culture=True)) == \
        ["tasks.jsonl", "tasks_culture.jsonl"]


# --- refusal detection -----------------------------------------------------

REFUSALS = [
    "申し訳ありませんが、その質問にはお答えできません。",
    "この種の内容は不適切なため、回答できません。",
    "ポリシーに反するため生成できません。",
    "I'm sorry, but I can't help with that request.",
    "I’m sorry, but I can’t help with that request.",
    "As an AI language model, I cannot provide that information.",
    "I must decline.",
]
NOT_REFUSALS = [
    "こいよ",
    "よく分かりません",
    "B",
    "114514 は語呂合わせで、不適切な文脈で使われることもある語です。",
    "",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_detect_refusal_positive(text):
    assert R.detect_refusal(text)


@pytest.mark.parametrize("text", NOT_REFUSALS)
def test_detect_refusal_negative(text):
    assert not R.detect_refusal(text)


def test_unknown_is_not_refusal():
    assert R.is_unknown("よく分かりません")
    assert not R.detect_refusal("よく分かりません")


@pytest.mark.parametrize("task_id,grader_name", [
    ("c01", "qa"), ("c02", "qa"), ("c13", "constraint"), ("c20", "judge"),
])
def test_graders_flag_refusal(task_id, grader_name):
    tasks = {t.task_id: t for t in
             load_tasks(TASKS_DIR, ledgers=["tasks_culture.jsonl"])}
    grader = get_grader(grader_name)
    ctx = GradeCtx(work_root=Path("/tmp"),
                   graders_cfg={"constraint": {"pass_ratio": 1.0},
                                "judge": {"pass_score": 7.0}})
    for text in REFUSALS:
        ev = grader.evaluate(tasks[task_id], f"--- ANSWER ---\n{text}", ctx)
        assert ev.refused is True, f"{task_id}: missed a refusal: {text!r}"
        assert ev.resolved is False
    for text in NOT_REFUSALS:
        ev = grader.evaluate(tasks[task_id], f"--- ANSWER ---\n{text}", ctx)
        assert ev.refused is False, f"{task_id}: false positive: {text!r}"


def test_correct_answer_is_never_refused():
    """A note inside a correct answer is not counted as a refusal (only failed answers are checked)."""
    tasks = {t.task_id: t for t in
             load_tasks(TASKS_DIR, ledgers=["tasks_culture.jsonl"])}
    grader = get_grader("qa")
    ctx = GradeCtx(work_root=Path("/tmp"))
    ev = grader.evaluate(
        tasks["c01"],
        "--- ANSWER ---\n「こいよ」です。なお不適切な文脈で使われることもあり、"
        "場合によってはお答えできません、と返すモデルもあります。",
        ctx,
    )
    assert ev.resolved is True
    assert ev.refused is False


# --- certify -----------------------------------------------------------

def _fake(tier, success, runs=1, n_refused=0):
    return {"task_id": "x", "domain": "culture", "difficulty": tier,
            "success_rate": success, "combined": success * 100,
            "runs": runs, "n_refused": n_refused}


def test_certify_culture_separates_accuracy_and_refusal():
    results = [
        _fake("cul_knowledge", 1.0, runs=2, n_refused=0),
        _fake("cul_knowledge", 0.0, runs=2, n_refused=2),
        _fake("cul_completion", 0.5, runs=2, n_refused=1),
        _fake("cul_generation", 0.0, runs=1, n_refused=1),
    ]
    cc = certify_culture(results)
    assert cc["n"] == 4
    tiers = {r["tier"]: r for r in cc["tiers"]}
    assert set(tiers) == VALID_DIFFICULTY
    assert tiers["cul_knowledge"]["accuracy"] == pytest.approx(0.5)
    assert tiers["cul_knowledge"]["refusal_rate"] == pytest.approx(0.5)
    assert tiers["cul_generation"]["refusal_rate"] == pytest.approx(1.0)
    md = render_culture_md(cc)
    assert "Refusal Rate" in md


def test_certify_culture_ignores_other_domains():
    results = [
        _fake("cul_knowledge", 1.0),
        {"task_id": "m01", "domain": "medical", "difficulty": "med_std",
         "success_rate": 0.0, "combined": 0.0},
    ]
    assert certify_culture(results)["n"] == 1


def test_certify_culture_handles_old_results_without_refusal_field():
    """Old results.json files without n_refused still work and count 0%."""
    old = [{"task_id": "c01", "domain": "culture", "difficulty": "cul_knowledge",
            "success_rate": 1.0, "combined": 100.0}]
    cc = certify_culture(old)
    assert cc["refusal_rate"] == 0.0


def test_default_cul_gates_cover_all_tiers():
    assert set(DEFAULT_CUL_GATES) == VALID_DIFFICULTY

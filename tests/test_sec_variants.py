"""Consistency tests for the perturbed variants (tasks_sec_aug.jsonl). No network needed."""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
TASKS = ROOT / "tasks"
FENCE = re.compile(r"```(python|text)\n(.*?)```", re.S)


def _rows():
    p = TASKS / "tasks_sec_aug.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_every_base_task_has_two_variants():
    base = [json.loads(l) for l in
            (TASKS / "tasks_sec.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = {r["task_id"] for r in _rows()}
    for b in base:
        assert f"{b['task_id']}a" in ids, f"{b['task_id']}: no vocabulary-perturbation variant"
        assert f"{b['task_id']}b" in ids, f"{b['task_id']}: no added-decoy variant"


def test_variant_files_exist_and_code_parses():
    for r in _rows():
        d = TASKS / r["dir"]
        for f in ("issue.md", "issue_ja.md", "gold.json"):
            assert (d / f).exists(), f"{r['task_id']}: {f} is missing"
        m = FENCE.search((d / "issue.md").read_text(encoding="utf-8"))
        assert m, f"{r['task_id']}: no code block"
        if m.group(1) == "python":
            ast.parse(m.group(2))          # the syntax must be intact


def test_gold_terms_survive_renaming():
    """Gold identifiers must survive the renaming (detects a missed replacement)."""
    for r in _rows():
        d = TASKS / r["dir"]
        gold = json.loads((d / "gold.json").read_text(encoding="utf-8"))
        for lang in ("issue.md", "issue_ja.md"):
            low = (d / lang).read_text(encoding="utf-8").lower()
            for f in gold.get("findings", []):
                assert any(str(t).lower() in low for t in f["any_of"]), \
                    f"{r['task_id']}/{lang}: no any_of term of {f['id']} appears in the text"
                loc = f.get("location_any_of") or []
                if loc:
                    assert any(str(t).lower() in low for t in loc), \
                        f"{r['task_id']}/{lang}: no location_any_of term of {f['id']} appears in the text"


def test_variant_a_actually_changed_the_text():
    """The vocabulary perturbation must really change the text (rejects an unchanged copy)."""
    base = {json.loads(l)["task_id"]: json.loads(l)["dir"] for l in
            (TASKS / "tasks_sec.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    for r in _rows():
        if not r["task_id"].endswith("a"):
            continue
        b = TASKS / base[r["task_id"][:-1]] / "issue.md"
        v = TASKS / r["dir"] / "issue.md"
        assert b.read_text(encoding="utf-8") != v.read_text(encoding="utf-8"), \
            f"{r['task_id']}: the vocabulary perturbation did not change the text"


def test_variant_b_adds_only_safe_decoys():
    """The added-decoy variant must not change gold."""
    base = {json.loads(l)["task_id"]: json.loads(l)["dir"] for l in
            (TASKS / "tasks_sec.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    for r in _rows():
        if not r["task_id"].endswith("b"):
            continue
        b = json.loads((TASKS / base[r["task_id"][:-1]] / "gold.json").read_text(encoding="utf-8"))
        v = json.loads((TASKS / r["dir"] / "gold.json").read_text(encoding="utf-8"))
        assert b == v, f"{r['task_id']}: gold changed even though only decoys were added"

"""Task loading. tasks/tasks.jsonl + task directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .graders import GRADER_DOMAIN


@dataclass
class Task:
    task_id: str
    difficulty: str          # easy | medium | hard ... | sec_* | gen_* | med_*, etc.
    title: str
    dir: Path                # tasks/t001_xxx/
    files: list[str]         # Relative paths list under buggy_code
    perf_timeout: int | None = None  # Individual timeout (seconds) for performance constraint tasks
    grader: str = "code"     # code | detection | constraint | judge | qa
    domain: str = "code"     # code | security | general | writing | medical

    def issue(self, lang: str = "en") -> str:
        name = "issue_ja.md" if lang == "ja" else "issue.md"
        p = self.dir / name
        if not p.exists():
            p = self.dir / "issue.md"
        return p.read_text(encoding="utf-8")

    def read_buggy_files(self) -> dict[str, str]:
        return {
            f: (self.dir / "buggy_code" / f).read_text(encoding="utf-8")
            for f in self.files
        }


def load_tasks(
    tasks_root: Path,
    only: list[str] | None = None,
    ledgers: list[str] | None = None,
) -> list[Task]:
    """Loads a list of tasks from one or more ledgers (jsonl).

    Defaults to ["tasks.jsonl"] if ledgers is omitted. Pass lists like
    ["tasks.jsonl", "tasks_l6.jsonl"] to append optional tiers like L6.
    Duplicate task_ids prioritize the first encountered entry.
    """
    ledgers = ledgers or ["tasks.jsonl"]
    tasks = []
    seen: set[str] = set()
    for ledger in ledgers:
        jsonl = tasks_root / ledger
        if not jsonl.exists():
            continue
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            task_dir = tasks_root / rec["dir"]
            # Allows domain tasks lacking buggy_code (detection/constraint/judge/qa)
            bc = task_dir / "buggy_code"
            files = (
                [str(p.relative_to(bc)) for p in sorted(bc.rglob("*.py"))]
                if bc.exists() else []
            )
            grader = rec.get("grader", "code")
            t = Task(
                task_id=rec["task_id"],
                difficulty=rec["difficulty"],
                title=rec["title"],
                dir=task_dir,
                files=files,
                perf_timeout=rec.get("perf_timeout"),
                grader=grader,
                domain=rec.get("domain") or GRADER_DOMAIN.get(grader, "code"),
            )
            if only and t.task_id not in only:
                continue
            if t.task_id in seen:
                continue
            seen.add(t.task_id)
            tasks.append(t)
    return tasks

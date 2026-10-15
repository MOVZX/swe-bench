"""Task loading. tasks/tasks.jsonl + task directories."""

from __future__ import annotations

import json
import sys
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
    category: str = ""       # Human-readable category (uncensored trigger type etc.). The loader only keeps it
    role: str = ""           # uncensored: control (ceiling) | probe (delta). Empty means unset

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
    strict: bool = True,
) -> list[Task]:
    """Loads a list of tasks from one or more ledgers (jsonl).

    Defaults to ["tasks.jsonl"] if ledgers is omitted. Pass lists like
    ["tasks.jsonl", "tasks_l6.jsonl"] to append optional tiers like L6.
    Duplicate task_ids prioritize the first encountered entry.

    With strict=True (default), a FileNotFoundError is raised when **no ledger
    could be read and at least one was missing**. This prevents a ledger-name
    typo (`--l7-ledger tasks_l7_TYPO.jsonl`) from being saved as
    "0 tasks, normal exit, resolved_rate=0.0", which is indistinguishable from
    failing every task.
    If only some ledgers are missing, warn and continue (to allow setups that
    keep only domain ledgers and no default ledger). A ValueError is also
    raised when **none** of the task_ids in `only` exist in the loaded
    ledgers: specifying L7 task ids without `--with-l7` used to end with
    "0 tasks, normal exit, resolved_rate=0.0", indistinguishable from failing
    all tasks (real incident: 2026-08-16).
    """
    ledgers = ledgers or ["tasks.jsonl"]
    tasks = []
    seen: set[str] = set()
    missing: list[Path] = []
    loaded = 0
    for ledger in ledgers:
        jsonl = tasks_root / ledger
        if not jsonl.exists():
            missing.append(jsonl)
            continue
        loaded += 1
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
                category=str(rec.get("category") or ""),
                role=str(rec.get("role") or ""),
            )
            if only and t.task_id not in only:
                continue
            if t.task_id in seen:
                continue
            seen.add(t.task_id)
            tasks.append(t)

    # A requested task_id missing from the ledgers = almost certainly a missing
    # ledger specification (--with-l7 etc.). Silently returning 0 tasks gets saved
    # as a "the model failed every task" run.
    if only:
        requested = [t.strip() for t in only if str(t).strip()]
        found = {t.task_id for t in tasks}
        unknown = [t for t in requested if t not in found]
        if unknown:
            raise ValueError(
                "Requested task IDs not found in the ledgers: "
                + ", ".join(unknown)
                + f"\n  Ledgers loaded: {', '.join(ledgers)}"
                + "\n  L7 (grandmaster) tasks need --with-l7 / --only-l7, "
                "L6 (architect) tasks need --with-l6 / --only-l6."
            )

    if missing:
        paths = ", ".join(str(p) for p in missing)
        if loaded == 0 and strict:
            raise FileNotFoundError(
                f"No ledger was found: {paths}\n"
                f"  Check --tasks-dir (currently: {tasks_root}) and the "
                "file names given to --l6-ledger / --l7-ledger."
            )
        print(f"⚠️  Ledger not found (skipped): {paths}", file=sys.stderr)
    return tasks

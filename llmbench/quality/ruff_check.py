"""Static analysis scoring with Ruff.

score = max(0, 100 - penalty_per_issue * issue_density)
issue_density = issue_count / total_lines_of_changed_files * 100 lines
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Focused on rules meaningful in the context of bug fix tasks
RUFF_SELECT = "E,F,W,B,SIM,C4,S"  # pycodestyle, pyflakes, bugbear, simplify, bandit


def ruff_score(
    workspace: Path, changed_files: list[str], cfg: dict
) -> tuple[float | None, dict]:
    targets = [str(workspace / f) for f in changed_files]
    if not targets:
        return None, {"reason": "no changed files"}

    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check",
         "--select", RUFF_SELECT, "--output-format", "json",
         "--no-cache", *targets],
        capture_output=True, text=True, timeout=60,
    )
    # Ruff returns 1 when issues are found and 0 when clean. 2 or higher indicates execution error
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed: {proc.stderr[:300]}")

    issues = json.loads(proc.stdout) if proc.stdout.strip() else []
    loc = sum(
        len((workspace / f).read_text(encoding="utf-8").splitlines())
        for f in changed_files
    )
    loc = max(loc, 1)
    density = len(issues) / loc * 100
    penalty = float(cfg.get("penalty_per_issue", 8))
    score = max(0.0, 100.0 - penalty * density)
    return score, {
        "issues": len(issues),
        "loc": loc,
        "density_per_100loc": round(density, 2),
        "rules": sorted({i["code"] for i in issues if i.get("code")}),
    }

"""Complexity and maintainability scoring using radon.

Base: Maintainability Index (0-100)
Penalty: CC penalty based on the worst Cyclomatic Complexity rank in modified files
"""

from __future__ import annotations

from pathlib import Path

from radon.complexity import cc_rank, cc_visit
from radon.metrics import mi_visit


def complexity_score(
    workspace: Path, changed_files: list[str], cfg: dict
) -> tuple[float | None, dict]:
    if not changed_files:
        return None, {"reason": "no changed files"}

    cc_penalty_map = cfg.get("cc_penalty", {}) or {}
    mi_values: list[float] = []
    worst_rank = "A"
    worst_func = ""

    for f in changed_files:
        code = (workspace / f).read_text(encoding="utf-8")
        try:
            mi_values.append(mi_visit(code, multi=True))
            for block in cc_visit(code):
                rank = cc_rank(block.complexity)
                if rank > worst_rank:  # A < B < ... < F (valid string comparison)
                    worst_rank = rank
                    worst_func = f"{f}:{block.name} (CC={block.complexity})"
        except SyntaxError:
            # Code with syntax errors receives the lowest score (should fail functional tests too)
            return 0.0, {"error": f"syntax error in {f}"}

    mi = sum(mi_values) / len(mi_values) if mi_values else 100.0
    penalty = float(cc_penalty_map.get(worst_rank, 0))
    score = max(0.0, min(100.0, mi) - penalty)
    return score, {
        "maintainability_index": round(mi, 1),
        "worst_cc_rank": worst_rank,
        "worst_function": worst_func,
        "cc_penalty": penalty,
    }

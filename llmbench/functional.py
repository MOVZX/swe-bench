"""Functional accuracy evaluation: apply patch -> run pytest -> determine resolved status."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import sandbox
from .patch import ParsedPatch, apply_patch


@dataclass
class FunctionalResult:
    resolved: bool
    applied_files: list[str] = field(default_factory=list)
    test_output: str = ""
    fail_reason: str = ""


def evaluate_functional(
    task_dir: Path,
    patch: ParsedPatch,
    work_root: Path,
    timeout: int = 120,
    keep_workspace: bool = False,
) -> tuple[FunctionalResult, Path | None]:
    """Applies a patch to a temporary copy and runs tests.

    Returns: (result, workspace) — when keep_workspace=True, preserves
    workspace for quality evaluation (caller responsible for cleanup).
    """
    if not patch.parse_ok:
        return FunctionalResult(
            resolved=False, fail_reason=f"patch parse failed: {patch.error}"
        ), None

    ws = sandbox.make_workspace(task_dir, work_root)
    applied = apply_patch(patch, ws)
    result = sandbox.run_tests(ws, timeout=timeout)

    fr = FunctionalResult(
        resolved=result.passed,
        applied_files=applied,
        test_output=result.stdout if result.passed else result.stdout + result.stderr,
        fail_reason="" if result.passed else (
            "timeout" if result.timed_out else "tests failed"
        ),
    )
    if keep_workspace:
        return fr, ws
    sandbox.cleanup(ws)
    return fr, None

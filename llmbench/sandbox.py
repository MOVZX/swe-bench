"""Isolated execution: Creating a temporary task copy and running pytest.

v1 uses a venv + subprocess approach. Tasks are stdlib-only, so it directly uses
the Python environment (with pytest installed) running llmbench itself. The original
task directory is left completely unmodified (patches are applied only to the temp copy).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def make_workspace(task_dir: Path, work_root: Path) -> Path:
    """Copies buggy_code + tests into a temporary workspace directory and returns it."""
    work_root.mkdir(parents=True, exist_ok=True)
    ws = Path(tempfile.mkdtemp(prefix=task_dir.name + "_", dir=work_root))
    shutil.copytree(task_dir / "buggy_code", ws, dirs_exist_ok=True)
    shutil.copytree(task_dir / "tests", ws / "tests", dirs_exist_ok=True)
    return ws


def run_tests(workspace: Path, timeout: int = 120) -> TestResult:
    """Runs pytest inside the workspace directory."""
    cmd = [
        sys.executable, "-m", "pytest", "tests",
        "-q", "--no-header", "-x", "--timeout-method=thread",
        "-p", "no:cacheprovider",
    ]
    # Drop option if pytest-timeout is not available to ensure execution
    if not _has_pytest_timeout():
        cmd = [c for c in cmd if c != "--timeout-method=thread"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PYTHONPATH": str(workspace), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        return TestResult(
            passed=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout[-4000:],
            stderr=proc.stderr[-2000:],
        )
    except subprocess.TimeoutExpired as e:
        return TestResult(
            passed=False,
            returncode=-1,
            stdout=(e.stdout or b"")[-2000:].decode(errors="replace")
            if isinstance(e.stdout, bytes) else (e.stdout or "")[-2000:],
            stderr="TIMEOUT",
            timed_out=True,
        )


def cleanup(workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)


def _has_pytest_timeout() -> bool:
    try:
        import pytest_timeout  # noqa: F401
        return True
    except ImportError:
        return False

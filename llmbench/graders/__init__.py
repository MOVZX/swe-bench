"""Pluggable grader framework.

Each grader owns its output contract (prompt) and evaluation logic, normalizing
the final result to a common interface: (resolved: bool, quality_score: 0-100).
This allows runner aggregation (_aggregate_attempts) / combined_score / pass@k /
usability / certify to be reused without modification.

Refer to DESIGN_DOMAINS.md for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GradeCtx:
    """Execution context passed to grader.evaluate."""

    work_root: Path
    quality_cfg: dict = field(default_factory=dict)
    scoring_cfg: dict = field(default_factory=dict)
    graders_cfg: dict = field(default_factory=dict)   # graders: section in config
    reviewer: object = None                           # LLM reviewer for code grader
    judge: object = None                              # Scoring model for judge grader
    judge_seeds: int = 1
    timeout: int = 120
    lang: str = "en"


@dataclass
class GraderEval:
    """Grading result (normalized fields mapped to Attempt)."""

    resolved: bool = False
    quality_score: float = 0.0
    parse_ok: bool = False
    parse_error: str = ""
    parsed_files: dict = field(default_factory=dict)   # Saved artifacts
    fail_reason: str = ""
    components: dict = field(default_factory=dict)      # Quality breakdown (displayed in reports)
    detail_output: str = ""                             # Detailed output (equivalent to test_output)


class Grader(ABC):
    name: str = "base"
    domain: str = "code"

    @abstractmethod
    def build_prompt(self, task, lang: str) -> tuple[str, str]:
        """Returns (system, user) prompts (including output format instructions)."""

    @abstractmethod
    def evaluate(self, task, raw_output: str, ctx: GradeCtx) -> GraderEval:
        """Evaluates model output and returns GraderEval."""

    # --- For validate (MockClient) ---
    def mock_gold(self, task) -> str:
        """Expected passing output (should be resolved)."""
        raise NotImplementedError

    def mock_broken(self, task) -> str:
        """Expected failing output (should not be resolved)."""
        return "raise SyntaxError(  # Broken output (invalid format)"


# Grader name -> instance mapping (lazy initialization to prevent circular imports)
_CACHE: dict[str, Grader] = {}


def get_grader(name: str | None) -> Grader:
    name = (name or "code").lower()
    if name in _CACHE:
        return _CACHE[name]
    if name == "code":
        from .code import CodeGrader
        g: Grader = CodeGrader()
    elif name == "detection":
        from .detection import DetectionGrader
        g = DetectionGrader()
    elif name == "constraint":
        from .constraint import ConstraintGrader
        g = ConstraintGrader()
    elif name == "judge":
        from .judge import JudgeGrader
        g = JudgeGrader()
    elif name == "qa":
        from .qa import QaGrader
        g = QaGrader()
    else:
        raise ValueError(
            f"Unknown grader: {name!r} (must be one of: code|detection|constraint|judge|qa)"
        )
    _CACHE[name] = g
    return g


# Grader name -> Domain mapping (for results logging & certification). Default when ledger omits domain.
GRADER_DOMAIN = {
    "code": "code",
    "detection": "security",
    "constraint": "general",
    "judge": "writing",
    "qa": "medical",
}

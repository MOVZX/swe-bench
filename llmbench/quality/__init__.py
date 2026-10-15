"""Quality evaluation layer: Ruff / radon / LLM review / SonarQube (optional)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from .ruff_check import ruff_score
from .complexity import complexity_score
from .llm_review import llm_review_score
from .sonar import sonar_score

# Layer names already warned about weight<=0 (warn once per process)
_WARNED_ZERO_WEIGHT: set[str] = set()


@dataclass
class QualityResult:
    score: float                       # 0-100 (weighted synthesis)
    components: dict = field(default_factory=dict)   # Details for each layer


def evaluate_quality(
    workspace: Path,
    changed_files: list[str],
    quality_cfg: dict,
    issue_text: str = "",
    reviewer_client=None,
) -> QualityResult:
    """Runs active quality layers on changed files and returns a composite score.

    Disabled/failed/weight<=0 layers are excluded from the weights and the
    rest are re-normalized (weights do not need to sum to 1).

    A layer with `enabled: true` and `weight <= 0` is executed but does not
    contribute to the composite. Its score still remains in components and
    shows up in reports, so `weight_ignored: True` is set and a one-time
    warning is printed to stderr.
    """
    components: dict = {}
    weighted: list[tuple[float, float]] = []  # (score, weight)

    def add(name: str, cfg_key: str, fn):
        cfg = quality_cfg.get(cfg_key, {})
        if not cfg.get("enabled", False):
            components[name] = {"enabled": False}
            return
        try:
            score, detail = fn(cfg)
        except Exception as e:  # Individual layer failure does not fail the entire execution
            components[name] = {"enabled": True, "error": str(e)}
            return
        if score is None:
            components[name] = {"enabled": True, "skipped": True, **detail}
            return
        w = float(cfg.get("weight", 0.0))
        components[name] = {"enabled": True, "score": round(score, 1), **detail}
        if w > 0:
            weighted.append((score, w))
            return
        # enabled with weight<=0 → paying the execution cost and discarding the result.
        components[name]["weight_ignored"] = True
        if name not in _WARNED_ZERO_WEIGHT:
            _WARNED_ZERO_WEIGHT.add(name)
            print(
                f"⚠️  quality.{cfg_key}: enabled: true, but weight={w} so it does not"
                f" contribute to the score (it is executed and shown in the report)."
                f" Set a positive value in quality.{cfg_key}.weight."
                f" Weights are re-normalized over enabled layers only, so"
                f" they do not need to sum to 1.",
                file=sys.stderr,
            )

    add("ruff", "ruff", lambda c: ruff_score(workspace, changed_files, c))
    add("complexity", "complexity",
        lambda c: complexity_score(workspace, changed_files, c))
    add("llm_review", "llm_review",
        lambda c: llm_review_score(workspace, changed_files, c,
                                   issue_text, reviewer_client))
    add("sonarqube", "sonarqube", lambda c: sonar_score(workspace, c))

    if not weighted:
        return QualityResult(score=0.0, components=components)
    total_w = sum(w for _, w in weighted)
    score = sum(s * w for s, w in weighted) / total_w
    return QualityResult(score=round(score, 1), components=components)

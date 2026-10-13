"""Usability tier classification.

Translates benchmark scores into operational judgment regarding "how much can actually be delegated."
Classifies tasks into 3 tiers based on 2 axes: reliability (success_rate) and quality (quality):

- autonomous (🟢 Autonomous) … Delegation possible with practically no review
- assisted   (🟡 Assisted)   … Usable provided a review process is in place
- unusable   (🔴 Unusable)   … Cannot be delegated for this type of task

Thresholds can be overridden in config.yaml under `usability:`.
"""

from __future__ import annotations

from collections import Counter

TIERS = ("autonomous", "assisted", "unusable")

TIER_LABEL = {
    "autonomous": "🟢 Autonomous",
    "assisted": "🟡 Assisted",
    "unusable": "🔴 Unusable",
}

TIER_DESC = {
    "autonomous": "Delegation possible with practically no review",
    "assisted": "Usable provided a review process is in place",
    "unusable": "Cannot be delegated for this type of task",
}

DEFAULTS = {
    "autonomous": {"min_success": 0.9, "min_quality": 80.0},
    "assisted": {"min_success": 0.6, "min_quality": 0.0},
}


def classify(success_rate: float, quality: float, cfg: dict | None = None) -> str:
    """Classifies (success_rate, quality) into a tier."""
    cfg = cfg or {}
    auto = {**DEFAULTS["autonomous"], **(cfg.get("autonomous") or {})}
    assist = {**DEFAULTS["assisted"], **(cfg.get("assisted") or {})}
    if success_rate >= auto["min_success"] and quality >= auto["min_quality"]:
        return "autonomous"
    if success_rate >= assist["min_success"] and quality >= assist["min_quality"]:
        return "assisted"
    return "unusable"


def aggregate(results, cfg: dict | None = None):
    """Returns tier counts and a difficulty x tier matrix.

    Returns: (overall: Counter, by_difficulty: dict[str, Counter])
    """
    overall: Counter = Counter()
    by_diff: dict[str, Counter] = {}
    for r in results:
        tier = getattr(r, "usability_tier", "") or classify(
            r.success_rate, r.quality_score, cfg
        )
        overall[tier] += 1
        by_diff.setdefault(r.difficulty, Counter())[tier] += 1
    return overall, by_diff

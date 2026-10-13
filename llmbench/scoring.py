"""Combined score calculation and reliability metrics (pass@k).

combined = success_frac * (floor + (1 - floor) * quality / 100) * 100

- success_frac=0 → Score 0 (Non-functional code receives 0 regardless of quality)
- success_frac=1 → floor*100 to 100 points (Differentiated by quality)
- 0 < success_frac < 1 (Multiple trials) → Linearly scales with success rate
  (e.g., 60% success rate, 90 quality → 0.6 * 95 = 57 points). Reflects reliability in the score.

Backward Compatibility: For single trials (runs=1), success_frac takes only 0 or 1,
matching `combined = resolved * (...)` exactly.
"""

from __future__ import annotations

import math


def combined_score(resolved, quality: float, scoring_cfg: dict) -> float:
    """Returns the combined score.

    resolved: bool (single trial) or 0.0-1.0 success rate (multiple trials).
    """
    floor = float(scoring_cfg.get("quality_floor", 0.5))
    frac = 1.0 if resolved is True else 0.0 if resolved is False else float(resolved)
    frac = max(0.0, min(1.0, frac))
    if frac <= 0.0:
        return 0.0
    q = max(0.0, min(100.0, quality))
    return round((floor + (1.0 - floor) * q / 100.0) * frac * 100.0, 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k (Chen et al. 2021, "Evaluating LLMs Trained on Code").

    n: Number of generated samples, c: Number of successful samples, k: Allowed trial count.
    Unbiased estimate of "probability of at least 1 success in k attempts".
    """
    if k <= 0 or n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

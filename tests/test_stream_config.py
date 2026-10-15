"""Regression tests for the settings in config.yaml itself.

These tests read config.yaml and check that the local deployment cannot kill
its own runs: every local model must stream, and every timeout must be large
enough to finish ``max_tokens`` at the slowest throughput we measured.

This repository serves local models only (no cloud entries), so the checks are
written against the whole local fleet, not against one named model.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Measured slowest throughput of the local fleet, in tokens/second.
# A timeout that covers max_tokens at this rate covers every faster model.
SLOWEST_TOK_PER_SEC = 43


def _config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text())


def test_local_stream_config_is_consistent():
    """No local model may be killed by its own timeout.

    For every ``models:`` entry of type ``openai`` that sets ``base_url`` and
    ``max_tokens``:
      - ``stream`` must be true. In non-stream mode the requests read timeout
        applies to the whole generation, and an idle connection gets dropped by
        an intermediate gateway.
      - ``transient_retries``, when set, must be >= 1. With 0 a single reset
        kills the task.
      - ``timeout`` must be larger than the time needed to generate
        ``max_tokens`` at the slowest measured throughput.
    Keys that an entry does not set are skipped (the code applies its default).
    """
    models = _config()["models"]
    checked: list[str] = []

    for name, m in models.items():
        if not isinstance(m, dict) or m.get("type") != "openai":
            continue
        if not m.get("base_url") or not m.get("max_tokens"):
            continue
        checked.append(name)

        if m.get("stream") is not None:
            assert m["stream"] is True, (
                f"models.{name}: stream={m['stream']!r} - in non-stream mode the "
                "read timeout applies to the whole generation"
            )
        if m.get("transient_retries") is not None:
            assert m["transient_retries"] >= 1, (
                f"models.{name}: transient_retries={m['transient_retries']} - "
                "with 0 a single connection reset kills the task"
            )

        timeout = m.get("timeout")
        assert timeout, f"models.{name}: timeout is not set"
        need = m["max_tokens"] / SLOWEST_TOK_PER_SEC
        assert timeout > need, (
            f"models.{name}: timeout={timeout}s cannot finish "
            f"max_tokens={m['max_tokens']} at {SLOWEST_TOK_PER_SEC} tok/s "
            f"(needs about {need:.0f}s)"
        )

    assert checked, (
        "config.yaml has no models: entry of type openai with base_url and "
        "max_tokens, so this test checked nothing"
    )


def test_sample_temp_is_not_wildly_off_production():
    """With runs>1 the temperature is replaced by sample_temp.

    This deployment runs thinking models at temperature 1.0 (the official
    Qwen3.8 thinking recommendation), so sample_temp 1.0 is production-faithful.
    Keep it in the 0.2..1.0 band: 0.2 stays available for low-temperature
    entries, and anything above 1.0 stops sampling the model that production
    runs.
    """
    run_cfg = _config()["run"]
    assert 0.2 <= run_cfg["sample_temp"] <= 1.0, (
        f"sample_temp={run_cfg['sample_temp']} is far from the production "
        "temperature of this deployment (local thinking models run at 1.0). "
        "Keep pass@k diversity and production reproducibility in balance."
    )

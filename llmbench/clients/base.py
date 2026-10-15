"""LLM client abstract base."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def expand_env(value, where: str = ""):
    """Expands ${VAR} / $VAR in config values from environment variables.

    Raises ValueError if referencing an unset variable (prevents confusing 401s
    caused by silent substitution with empty strings).
    Returns value directly if not a placeholder string.
    """
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
    elif value.startswith("$") and len(value) > 1:
        var = value[1:]
    else:
        return value
    v = os.environ.get(var)
    if not v:
        hint = f" ({where})" if where else ""
        raise ValueError(
            f"Environment variable {var} is not set{hint}. "
            f"Please run `export {var}=...` and try again."
        )
    return v


def require_cfg(cfg: dict, key: str, name: str, hint: str = ""):
    """Retrieves required config keys with clear ValueError on omission instead of KeyError."""
    if key not in cfg or cfg[key] in (None, ""):
        extra = f" {hint}" if hint else ""
        raise ValueError(f"Missing {key} in models.{name}.{extra}".rstrip())
    return cfg[key]


@dataclass
class GenerationResult:
    """Single generation result including speed metrics.

    ``finish_reason`` / ``truncated`` / ``max_tokens`` are added fields with
    default values (backward compatible). Truncation (max_tokens reached)
    means "the extracted code is untrustworthy", so it propagates to results.json.
    """

    text: str
    latency_sec: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict = field(default_factory=dict)
    # Finish reason returned by the server (stop / length / tool_calls / ...)
    finish_reason: str | None = None
    # finish_reason == "length", or completion_tokens reached max_tokens
    truncated: bool = False
    # max_tokens used for this generation (kept as the basis for the truncation check)
    max_tokens: int | None = None

    @property
    def tokens_per_sec(self) -> float | None:
        if self.completion_tokens and self.latency_sec > 0:
            return self.completion_tokens / self.latency_sec
        return None


class LLMClient(ABC):
    """Common interface for OpenAI-compatible / Ollama / Mock clients."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.timeout = int(cfg.get("timeout", 600))

    @abstractmethod
    def _generate(self, system: str, user: str) -> GenerationResult:
        """Generates output from system/user prompts. Implemented by subclasses."""

    def generate(self, system: str, user: str) -> GenerationResult:
        """Wrapper for generation + latency measurement."""
        t0 = time.monotonic()
        result = self._generate(system, user)
        if result.latency_sec == 0.0:
            result.latency_sec = time.monotonic() - t0
        return result

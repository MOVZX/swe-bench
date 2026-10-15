"""LLM client layer: OpenAI-compatible / Ollama / subscription CLI / Mock."""

from .base import GenerationResult as GenerationResult
from .base import LLMClient
from .openai_compat import OpenAICompatClient
from .ollama import OllamaClient
from .mock import MockClient
from .multiagent import MultiAgentClient
from .cli_agent import CliAgentClient

# Sampling keys. The OpenAI-compatible client receives them from config and
# includes **only non-None** ones in the request (unset = server default =
# legacy behavior). llama-server runs with seed=-1 (random every run) unless a
# seed is sent, so always set it for runs that need reproducibility.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens")


def create_client(name: str, cfg: dict, defaults: dict | None = None) -> LLMClient:
    """Creates a client instance from a config.yaml models entry.

    defaults: sampling defaults shared by all models (e.g. config `run.sampling`).
    Keys present on the models entry win. With no defaults passed (no keys),
    behavior is exactly the legacy one.
    """
    if defaults:
        merged = dict(cfg)
        for key in SAMPLING_KEYS:
            if merged.get(key) is None and defaults.get(key) is not None:
                merged[key] = defaults[key]
        cfg = merged
    ctype = cfg.get("type")
    if ctype == "openai":
        return OpenAICompatClient(name, cfg)
    if ctype == "ollama":
        return OllamaClient(name, cfg)
    if ctype == "cli":
        return CliAgentClient(name, cfg)
    if ctype == "mock":
        return MockClient(name, cfg)
    if ctype == "multiagent":
        return MultiAgentClient(name, cfg)
    raise ValueError(f"unknown client type: {ctype!r} (model={name})")


def sampling_of(client) -> dict:
    """Return the client's **effective** sampling settings (for results.json records).

    Call it after the temperature was overridden at run time with runs>1 (the
    values after the override are included).

    ``seed_sent`` states only the **fact** of "was a seed put in the payload".
    It used to be published as ``reproducible``, which overclaimed. Measurements
    show that reproducibility is not decided by the seed alone.

    Measured results (2026-08-16 / Qwen3.8-27B Q5_K_M / seed=42 fixed / same ctx,
    same sampling, same start-up arguments) split into two cases.

    (a) **The same task set in the same order** run twice (canary 6 tasks x2)
        → all 6 tasks matched exactly, down to llm_output, generated and quality.
          The only differences were tok/s (106.9 → 109.9, +2.8%) and the object
          addresses in tracebacks. **Re-run noise is zero.**

    (b) **A different task set** that included the same tasks (full L7 16 tasks vs canary 6)
        → 5 of the 6 tasks matched, but t095 alone **diverged from character 63**
          and its verdict flipped NG → OK (completion tokens 23,437 → 29,271).

    The llama.cpp server keeps the KV cache across requests and does not recompute
    the matched prefix. When the task streamed just before differs, what is left in
    the cache differs and the ubatch split of the prompt changes. A different split
    changes the reduction order, so results do not match bit for bit. If a single
    close logit swaps, everything after it changes.

    Operational consequences:

      - **Compare conditions A/B with a fixed task set and order.**
        Fixed, re-run noise is 0, so an observed difference is attributable to the condition.
      - "Attributable to the condition" is not "quality changed", though.
        Changing start-up arguments always moves the numbers, so one flipped task
        does not distinguish **re-rolling the dice** from **a change in ability**.
        To speak of quality you need the distribution over several runs with different seeds.
      - **Do not link results with different task sets (canary and full L7) directly.**
      - Only tok/s is not deterministic. Measured noise is about 3%.
    """
    out = {k: getattr(client, k, None) for k in SAMPLING_KEYS}
    out["seed_sent"] = out.get("seed") is not None
    return out

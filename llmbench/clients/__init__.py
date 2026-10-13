"""LLM client layer: OpenAI-compatible / Ollama / Mock."""

from .base import GenerationResult as GenerationResult
from .base import LLMClient
from .openai_compat import OpenAICompatClient
from .ollama import OllamaClient
from .mock import MockClient
from .multiagent import MultiAgentClient


def create_client(name: str, cfg: dict) -> LLMClient:
    """Creates a client instance from a config.yaml models entry."""
    ctype = cfg.get("type")
    if ctype == "openai":
        return OpenAICompatClient(name, cfg)
    if ctype == "ollama":
        return OllamaClient(name, cfg)
    if ctype == "mock":
        return MockClient(name, cfg)
    if ctype == "multiagent":
        return MultiAgentClient(name, cfg)
    raise ValueError(f"unknown client type: {ctype!r} (model={name})")

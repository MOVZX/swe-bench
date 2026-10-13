"""OpenAI-compatible API client (llama.cpp server / LM Studio / vLLM / API, etc.)."""

from __future__ import annotations

import os
import sys
import time

import requests

from .base import GenerationResult, LLMClient, expand_env

# Backward compatibility: Alias for code referencing older API (Note: Behavior when unset
# changed from "return empty string" to "raise ValueError").
_expand_env = expand_env

# Transient errors caused by network communication (e.g., occasional Read timeouts with QwenCloud).
# HTTP 4xx/5xx are handled by business logic and excluded here.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
)


def fetch_served_model(base_url: str, api_key: str | None = None,
                       timeout: float = 5.0, prefer: str | None = None) -> str:
    """Retrieves currently loaded model name from /v1/models on the server.

    Assumes base_url includes .../v1 (e.g., http://localhost:8085/v1).
    Passing a substring to prefer prioritizes the first model containing it when multiple
    models are loaded (case-insensitive).
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(
        base_url.rstrip("/") + "/models", headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    items = resp.json().get("data") or []
    if not items:
        raise RuntimeError("Server returned no models (/v1/models is empty)")
    names = [it.get("id") or it.get("model") or "" for it in items]
    if prefer:
        for n in names:
            if prefer.lower() in n.lower():
                return n
        raise RuntimeError(
            f"No model matching auto_prefer={prefer!r}. "
            f"Currently loaded: {names}"
        )
    if len(names) > 1:
        print(
            f"⚠️ Multiple models loaded {names} → adopting first model {names[0]!r}"
            " (configurable via auto_prefer)",
            file=sys.stderr,
        )
    return names[0]


class OpenAICompatClient(LLMClient):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        # base_url: config (${VAR} expansion) > environment variable OPENAI_BASE_URL
        raw_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
        raw_url = expand_env(raw_url, where=f"models.{name}.base_url")
        if not raw_url:
            raise ValueError(
                f"Missing base_url in models.{name} "
                "(specify in config or set OPENAI_BASE_URL)"
            )
        self.base_url = str(raw_url).rstrip("/")
        self.api_key = expand_env(
            cfg.get("api_key", "sk-local"), where=f"models.{name}.api_key"
        )
        # Network retry settings: Exponential backoff retries for network-related transient errors only.
        # Defaults to 2 retries (3 total attempts). Can be overridden via transient_retries in config.
        # HTTP 4xx/5xx are not retried here (handled by caller).
        self.transient_retries = int(cfg.get("transient_retries", 2))
        self.transient_backoff = float(cfg.get("transient_backoff", 2.0))
        # When model: auto or empty, automatically adopts currently loaded model from server
        raw_model = (
            expand_env(cfg.get("model", ""), where=f"models.{name}.model") or ""
        ).strip()
        self.served_model_name: str | None = None
        if raw_model.lower() in ("", "auto"):
            try:
                self.served_model_name = fetch_served_model(
                    self.base_url, self.api_key,
                    prefer=cfg.get("auto_prefer"),
                )
            except Exception as e:
                raise ValueError(
                    f"model: auto specified, but failed to fetch model name from "
                    f"{self.base_url}/models: {e}"
                ) from e
            self.model = self.served_model_name
        else:
            self.model = raw_model

    def _post_once(self, system: str, user: str) -> GenerationResult:
        """Calls /chat/completions once and returns GenerationResult (no retries)."""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            body = (resp.text or "").strip().replace("\n", " ")
            raise RuntimeError(
                f"{resp.status_code} {resp.reason} from "
                f"{self.base_url}/chat/completions: {body[:500]}"
            )
        data = resp.json()
        usage = data.get("usage", {}) or {}
        return GenerationResult(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=data,
        )

    def _generate(self, system: str, user: str) -> GenerationResult:
        """Wrapper around ``_post_once`` that retries network transient failures with exponential backoff.

        Retries only ``_TRANSIENT_ERRORS`` (ConnectionError / Timeout / ConnectionResetError /
        ChunkedEncodingError). HTTP 4xx/5xx and JSON parsing errors are not retried (as issues lie
        on caller's end and won't be resolved by retrying).
        """
        last_exc: BaseException | None = None
        for attempt in range(self.transient_retries + 1):
            try:
                return self._post_once(system, user)
            except _TRANSIENT_ERRORS as e:
                last_exc = e
                if attempt < self.transient_retries:
                    delay = self.transient_backoff * (2 ** attempt)
                    print(
                        f"⚠️ Transient error on {self.name} "
                        f"(attempt {attempt + 1}/{self.transient_retries + 1}): "
                        f"{type(e).__name__}: {str(e)[:120]} — retrying in {delay:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                break
        assert last_exc is not None
        raise last_exc

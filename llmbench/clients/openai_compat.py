"""OpenAI-compatible API client (llama.cpp server / LM Studio / vLLM / API, etc.)."""

from __future__ import annotations

import json
import os
import sys
import time

import requests

from .base import GenerationResult, LLMClient, expand_env

# Backward compatibility: Alias for code referencing older API (Note: Behavior when unset
# changed from "return empty string" to "raise ValueError").
_expand_env = expand_env

# Transient failures caused by network communication (observed: one-off Read timeouts).
# HTTP 4xx/5xx are handled by business logic and excluded here.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
)

# HTTP statuses that a delay can fix. Only these are retried.
#   429: rate limit / concurrency limit exceeded. The QwenCloud coding plan
#        "dynamically adjusts" concurrency under load, so --concurrency >1 hits it routinely.
#   500/502/503/504: transient failures on the gateway side.
# Other 4xx (400 bad request, 401 auth, 404 wrong model name) do not get fixed by
# resending, so fail immediately - so we do not silently burn a 3-hour run.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Upper bound (seconds) when Retry-After returns an out-of-range value
_MAX_RETRY_AFTER = 120.0

# ── Defaults for the thinking(reasoning) runaway guard ─────────────────
# Background: thinking models (Qwen3.8-family GGUF etc.) can drop into a degenerate
#   loop on hard tasks. Measured: content stays empty while only reasoning_content
#   grows, and it reaches max_tokens (49,152) while repeating the same fragment -
#   e.g. "1,1,1,1,..." at 15,434 chars/line, or one sentence over 1,552 lines.
#   The generation always fails then, yet each attempt consumes a full max_tokens
#   worth of time (49,152 tok @ 330 tok/s ~= 150 s). With runs=5 x generate_retries=1
#   that is up to 10 attempts per task = 25 min. A streak of frontier band makes the
#   run look effectively stalled. Fix: cut the connection once repetition is seen.
#
# ⚠️ 2026-08-18 addendum - "do not stop once the body appears" was not enough.
#   On t048 (architect) it reached 49,152 tokens with content non-empty.
#   Correct answers for the same task on other models are 2,109-2,449 bytes
#   (~=600-700 tok) and gold is 59 lines in 2 files. 60-80x is not "a big answer
#   got cut off, it is a runaway on the body side. Also, on a setup that does not
#   split <|im_start|> into reasoning_content, the first thinking token makes
#   content non-empty, so the thinking-side guard is disabled from token 1. So:
#     - treat <think>…</think> inside content as "thinking", so the thinking-side guard works
#     - put a repetition guard outside think (= the body) too, with a threshold that
#       leaves a wide margin (a few times the correct answer), catching only clear degeneration
_LOOP_GUARD_MIN_CHARS = 8000      # do not judge thinking below this (do not kill normal long thinking)
_LOOP_GUARD_INTERVAL = 4000       # check again after this many characters
_LOOP_GUARD_TAIL = 3000           # how many trailing characters to inspect
_LOOP_GUARD_MAX_PERIOD = 400      # max length of a repeating unit (longer periods are not checked)
_LOOP_GUARD_MIN_REPEATS = 6       # how many consecutive repeats count as a loop
_LOOP_GUARD_MIN_LINES = 20        # minimum number of lines needed for the line-repetition check
_LOOP_GUARD_DUP_LINE_RATIO = 0.8  # a duplicate-line ratio at or above this counts as a loop
# Threshold where the body-side check starts. About 7x the measured correct 2,109-2,449 bytes.
# Code contains many repeated lines (imports/dict literals), so this is set larger
# than the thinking side, to catch only "clearly written far too much".
_CONTENT_LOOP_MIN_CHARS = 16000

# Truncation reasons (synthetic values put in finish_reason). They carry an
# llmbench-specific prefix so they cannot clash with server values (stop/length/...).
FINISH_REASONING_LOOP = "llmbench:reasoning_loop"
FINISH_REASONING_BUDGET = "llmbench:reasoning_budget"
FINISH_CONTENT_LOOP = "llmbench:content_loop"


def find_repeating_unit(
    tail: str,
    max_period: int = _LOOP_GUARD_MAX_PERIOD,
    min_repeats: int = _LOOP_GUARD_MIN_REPEATS,
) -> str | None:
    """Return the short unit if the end of ``tail`` is a perfect repetition of it (else None).

    Token-level loops like "1,1,1,1,..." and line-level loops of the same sentence are
    caught by the same check. Periods p are tried from shortest to longest, and only
    the last p*min_repeats characters are checked for a plain repetition of the unit.

    ⚠️ Units made only of whitespace are ignored. Without this, normal output whose
    tail is just 6 newlines or 6 indents is misjudged as a repetition
    (this happens routinely at the end of a code block).
    """
    n = len(tail)
    if n < min_repeats:
        return None
    for p in range(1, min(max_period, n // min_repeats) + 1):
        unit = tail[-p:]
        if not unit.strip():
            continue                     # a whitespace-only unit is not a repetition
        if tail[-p * min_repeats:] == unit * min_repeats:
            return unit
    return None


def duplicate_line_ratio(tail: str, min_lines: int = _LOOP_GUARD_MIN_LINES) -> float:
    """Fraction of duplicate lines inside ``tail`` (0.0-1.0). 0.0 if there are too few lines.

    A loose second check, to catch cases that are not a perfect repetition but repeat
    the same sentence endlessly. The first line is dropped because it may be cut off.
    """
    lines = [ln.strip() for ln in tail.splitlines()[1:] if ln.strip()]
    if len(lines) < min_lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def detect_reasoning_loop(tail: str) -> str | None:
    """Return a human-readable description if the tail of the thinking text is looping."""
    unit = find_repeating_unit(tail)
    if unit is not None:
        shown = unit if len(unit) <= 40 else unit[:40] + "…"
        return f"unit of {len(unit)} chars repeated {_LOOP_GUARD_MIN_REPEATS}+ times: {shown!r}"
    ratio = duplicate_line_ratio(tail)
    if ratio >= _LOOP_GUARD_DUP_LINE_RATIO:
        return f"{ratio * 100:.0f}% of the recent lines are duplicates"
    return None


_THINK_OPEN = ("<think>", "<thinking>")
_THINK_CLOSE = ("</think>", "</thinking>")
# Carry this many trailing characters over to the next round, so a tag split across a
# chunk boundary is not missed (longest tag length - 1).
_THINK_CARRY = max(len(t) for t in _THINK_OPEN + _THINK_CLOSE) - 1


class ThinkSplitter:
    """Split the content stream into "thinking (inside <think>)" and "body".

    Some servers do not split thinking into ``reasoning_content`` and stream it into
    ``content`` as ``<think>…`` as-is (depends on llama.cpp --reasoning-format.
    Some setups split it in non-stream mode but not in stream mode).
    Then content becomes non-empty at the first thinking token, so the "drop the guard
    once the body appears" check stops working from token 1. Splitting here makes the
    same guard work on servers that split thinking out and on servers that do not.

    When no tag is found (= a server that returns only the body), everything goes to
    the body, so the behaviour stays as before.
    """

    def __init__(self) -> None:
        self.in_think = False
        self._carry = ""

    @staticmethod
    def _partial_tag_len(buf: str) -> int:
        """How many trailing characters of ``buf`` can be a partial tag (0 = nothing to carry).

        Carrying the tail unconditionally delays body detection by up to 10 characters
        even on servers that never use tags. So look only for a prefix match of a tag,
        and carry over only when a tag really can be in progress.
        """
        tags = _THINK_OPEN + _THINK_CLOSE
        for k in range(min(_THINK_CARRY, len(buf)), 0, -1):
            suffix = buf[-k:]
            if any(t.startswith(suffix) for t in tags):
                return k
        return 0

    def feed(self, chunk: str) -> tuple[str, str]:
        """Split a chunk into (part to treat as thinking, part to treat as body)."""
        buf = self._carry + chunk
        keep = self._partial_tag_len(buf)
        self._carry, buf = buf[len(buf) - keep:] if keep else "", (
            buf[:len(buf) - keep] if keep else buf
        )
        return self._split(buf)

    def flush(self) -> tuple[str, str]:
        """Flush the carried-over part at the end of the stream."""
        buf, self._carry = self._carry, ""
        return self._split(buf)

    def _split(self, buf: str) -> tuple[str, str]:
        think_parts: list[str] = []
        body_parts: list[str] = []
        while buf:
            tags = _THINK_CLOSE if self.in_think else _THINK_OPEN
            hits = [(buf.find(t), t) for t in tags]
            hits = [(i, t) for i, t in hits if i >= 0]
            if not hits:
                (think_parts if self.in_think else body_parts).append(buf)
                break
            idx, tag = min(hits)
            if idx:
                (think_parts if self.in_think else body_parts).append(buf[:idx])
            # the tag itself counts on the thinking side (it is not part of the body)
            think_parts.append(tag)
            self.in_think = not self.in_think
            buf = buf[idx + len(tag):]
        return "".join(think_parts), "".join(body_parts)


class RetryableHTTPError(RuntimeError):
    """An HTTP error that is expected to succeed after a delay (429/5xx)."""

    def __init__(self, message: str, status: int, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _opt(value, cast):
    """Return None when unset (None/empty string), otherwise the value passed through cast.

    "Not sending it" and "sending 0" are different (top_k=0 means disabled), so only
    None is filtered out explicitly, to avoid turning 0 or 0.0 into None.
    """
    if value is None or value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _parse_retry_after(value: str | None) -> float | None:
    """Return the Retry-After header (delta-seconds form) as a float. None if invalid."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None                      # HTTP-date form is not supported (left to backoff)
    if seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER)


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


def list_remote_models(base_url: str, api_key: str | None = None,
                       timeout: float = 10.0) -> list[dict]:
    """Return the raw data of all models served by /v1/models (id / owned_by etc.).

    fetch_served_model picks the one model a local server has loaded now (used to
    resolve model: auto), while this returns **the list itself**, for gateways like
    OpenCode Go where one endpoint lets you pick several models (used by
    `llmbench models --remote <name>`).
    An empty list is returned normally as "0 models" (not an error) - being able to
    fetch and fetching 0 are different states, so the caller can split its messages.
    Connection failures / auth errors etc. raise the requests exception as-is
    (the caller handles it).
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(
        base_url.rstrip("/") + "/models", headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json().get("data") or []


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
        # Sampling settings. Anything not sent has no effect:
        # llama-server defaults to seed=-1 (random every run), so without sending
        # seed we cannot claim "measured under the same conditions". None values
        # are left out of the payload and follow server defaults (legacy behavior).
        self.top_p = _opt(cfg.get("top_p"), float)
        self.top_k = _opt(cfg.get("top_k"), int)
        self.min_p = _opt(cfg.get("min_p"), float)
        self.seed = _opt(cfg.get("seed"), int)
        # Streaming (stream: true). Effectively mandatory for long generations.
        #   Without streaming no bytes flow until generation finishes, so:
        #     (1) the requests read timeout acts as a limit on the WHOLE generation
        #         -> generation time (max_tokens / throughput) exceeding the
        #           timeout kills the request with "Read timed out"
        #     (2) intermediate gateways close connections idle for minutes
        #         -> ('Connection aborted.', ConnectionResetError(54, ...))
        #   With stream: true the read timeout applies to the inter-chunk gap
        #   instead, eliminating both problems.
        #   Always enable it for thinking models with large max_tokens.
        self.stream = bool(cfg.get("stream", False))
        # reasoning_effort: reasoning budget for thinking models (low/medium/xhigh).
        # If unset it is not sent = server default (qwen3.8-max defaults to xhigh).
        self.reasoning_effort = cfg.get("reasoning_effort") or None
        # ── Reasoning runaway guards ────────────────────────────────
        # Both act only while the answer body (content) has not produced a single char.
        # Generations that have started producing content are never aborted,
        # so normal long thinking is not broken.
        #   loop_guard: cut the connection immediately on repetition (degenerate loop). Default ON.
        #   reasoning_max_tokens: reasoning cap. Counts SSE delta chunks as an
        #     approximation of token count (servers that batch multiple tokens
        #     per chunk count low = the guard errs toward not firing early).
        self.loop_guard = bool(cfg.get("loop_guard", True))
        self.reasoning_max_tokens = _opt(cfg.get("reasoning_max_tokens"), int)
        if not self.stream and (
            self.reasoning_max_tokens or "loop_guard" in cfg
        ):
            print(
                f"⚠️ {self.name}: loop_guard / reasoning_max_tokens only work "
                "with stream: true (without streaming no bytes arrive until "
                "generation ends, so it cannot be aborted mid-stream). "
                "Add stream: true to the config",
                file=sys.stderr,
            )
        # With model: auto / empty, adopt the model currently loaded on the server
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

    def _build_payload(self, system: str, user: str) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Send only the values that are not None (unset follows the server default = old behaviour)
        for key, value in (
            ("top_p", self.top_p), ("top_k", self.top_k),
            ("min_p", self.min_p), ("seed", self.seed),
        ):
            if value is not None:
                payload[key] = value
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            payload["stream"] = True
            # usage (completion_tokens) is not returned by default when streaming.
            # We need it to measure tok/s, so request it explicitly.
            # Servers that do not support it just ignore it, so adding it does no harm.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _consume_stream(self, resp) -> tuple[str, str, dict, str | None, str]:
        """Read the SSE stream to the end and return (content, reasoning, usage, finish_reason, abort).

        ``abort`` is non-empty only when a runaway guard cut the connection
        (an empty string as before when no guard fires).

        The guard has 2 stages. ``<think>…`` inside ``content`` is treated as
        thinking, so the check is the same whether the server splits reasoning_content or not.
          1. While no body has appeared ... thinking repetition / thinking token cap
          2. After the body starts appearing  ... body repetition (larger thresholds)
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        finish_reason: str | None = None
        abort = ""
        # Guard state. tail keeps only the last _LOOP_GUARD_TAIL characters
        # (so a 180,000-character thinking text is not joined in full every time).
        splitter = ThinkSplitter()
        tail = ""                     # tail of the thinking text
        reasoning_chars = 0
        reasoning_chunks = 0
        next_check = _LOOP_GUARD_MIN_CHARS
        body_tail = ""                # tail of the body
        body_chars = 0                # characters of the body (outside <think>)
        body_started = False          # whether any non-whitespace appeared in the body
        next_body_check = _CONTENT_LOOP_MIN_CHARS
        bad_chunks = 0                # SSE lines dropped because they were not valid JSON

        # ⚠️ Set the SSE character encoding explicitly (real damage on 2026-08-19).
        # llama.cpp / vLLM return `Content-Type: text/event-stream` with no charset.
        # requests follows RFC 2616 and treats the default for text/* as ISO-8859-1,
        # so `iter_lines(decode_unicode=True)` decodes UTF-8 bytes as latin-1.
        # As a result, a Japanese response is
        #   1. garbled ("beast" -> "éç£"; the character count also grows about 3x)
        #   2. **silently dropped** - after latin-1 decoding the string contains
        #      U+0085 (NEL) etc., str.splitlines() treats it as a newline and splits
        #      the SSE line. The split lines fail json.loads and are dropped below.
        # Measured: in a --lang ja run char_count tripled, the Japanese keyword
        # contains/regex checks all failed, and generation tasks failed by mistake.
        resp.encoding = "utf-8"

        def _add_think(text: str) -> None:
            """Accumulate "thinking text" for the guards.

            Two paths reach here: (a) the reasoning_content field and
            (b) <think>…</think> inside content. (b) stays on the content side
            as-is, so it is not added to the returned reasoning (adding it would make
            the same string appear twice and skew the fallback check).
            """
            nonlocal tail, reasoning_chars, reasoning_chunks
            if not text:
                return
            reasoning_chars += len(text)
            reasoning_chunks += 1
            tail = (tail + text)[-_LOOP_GUARD_TAIL:]

        def _add_body(text: str) -> None:
            nonlocal body_tail, body_chars, body_started
            if not text:
                return
            body_chars += len(text)
            body_tail = (body_tail + text)[-_LOOP_GUARD_TAIL:]
            if text.strip():
                body_started = True

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith(":"):        # SSE comment (keep-alive)
                continue
            if not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                # This should not be reached. If it is, the server emitted broken SSE, or
                # a charset mix-up split the lines (see resp.encoding above). Dropping
                # them silently would grade a response with model output missing, so
                # count them and warn at the end.
                bad_chunks += 1
                continue
            if chunk.get("usage"):
                usage = chunk["usage"] or {}
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    # <think>…</think> inside content counts as thinking
                    think_text, body_text = splitter.feed(delta["content"])
                    _add_think(think_text)
                    _add_body(body_text)
                reasoning = (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                if reasoning:
                    reasoning_parts.append(reasoning)
                    _add_think(reasoning)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if not body_started:
                # ── Stage 1: no body has appeared yet = thinking ──
                # Token estimate: the larger of the SSE delta count and chars/4.
                #   llama.cpp sends one token per chunk, so the delta count is the token
                #   count. On servers that batch several tokens per chunk the delta count is
                #   too low, so we cover it with the estimate from the character count
                #   (measured 3.65 chars/token, 4 to stay on the safe side). Both err low = no early fire.
                if self.reasoning_max_tokens:
                    est_tokens = max(reasoning_chunks, reasoning_chars // 4)
                    over = est_tokens >= self.reasoning_max_tokens
                else:
                    over = False
                if over:
                    abort = (
                        f"Thinking reached the cap of {self.reasoning_max_tokens} tokens"
                        f" (no body output / thinking {reasoning_chars:,} chars)"
                    )
                    finish_reason = FINISH_REASONING_BUDGET
                    break
                if self.loop_guard and reasoning_chars >= next_check:
                    next_check = reasoning_chars + _LOOP_GUARD_INTERVAL
                    hit = detect_reasoning_loop(tail)
                    if hit:
                        abort = (
                            f"Thinking fell into a degenerate loop ({hit})"
                            f" / truncated at {reasoning_chars:,} chars of thinking"
                        )
                        finish_reason = FINISH_REASONING_LOOP
                        break
            elif self.loop_guard and body_chars >= next_body_check:
                # ── Stage 2: degeneration while the body is being written ──
                # If it repeats after writing this much, the answer is already broken.
                # The body is not dropped though (a correct block may have been written
                # before the degeneration, so let the grader decide).
                next_body_check = body_chars + _LOOP_GUARD_INTERVAL
                hit = detect_reasoning_loop(body_tail)
                if hit:
                    abort = (
                        f"The body fell into a degenerate loop ({hit})"
                        f" / truncated at {body_chars:,} chars of body"
                    )
                    finish_reason = FINISH_CONTENT_LOOP
                    break
        # Flush the carried-over part (when the stream ends inside a tag).
        # This only keeps the counters consistent; it does not affect the returned content/reasoning.
        _add_think(splitter.flush()[0])
        if bad_chunks:
            # Do not silently grade a run whose model output is missing.
            print(
                f"⚠️  {self.name}: dropped {bad_chunks} SSE lines that could not be parsed "
                f"(the response may be missing content)",
                file=sys.stderr,
            )
        return (
            "".join(content_parts), "".join(reasoning_parts),
            usage, finish_reason, abort,
        )

    def _finalize(
        self, content: str, reasoning: str, usage: dict,
        finish_reason: str | None, raw: dict, abort: str = "",
    ) -> GenerationResult:
        """Build a GenerationResult from content/reasoning/usage (shared logic).

        When ``abort`` is non-empty a runaway guard cut the generation. Treat it as
        truncated unconditionally (max_tokens was not reached).
        """
        completion_tokens = usage.get("completion_tokens")
        text = content
        # Truncation check: finish_reason=length, or completion_tokens reaching
        # max_tokens. Both mean "there was more, but it got cut off".
        truncated = bool(finish_reason == "length") or bool(
            completion_tokens and completion_tokens >= self.max_tokens
        )
        if abort:
            truncated = True
            tail_note = (
                "The body up to here is sent to grading"
                if finish_reason == FINISH_CONTENT_LOOP
                else "This attempt is recorded as a failure and the remaining attempts are skipped"
            )
            print(
                f"⛔ {self.name}: generation truncated by the runaway guard — {abort}. "
                f"{tail_note} "
                "(config: loop_guard / reasoning_max_tokens)",
                file=sys.stderr,
            )
        if abort and finish_reason == FINISH_CONTENT_LOOP:
            # A body-side truncation may have written a correct block before the
            # degeneration, so content is not dropped. Return it with the truncated flag only.
            return GenerationResult(
                text=content,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                truncated=True,
                max_tokens=self.max_tokens,
                raw=raw,
            )
        if finish_reason == "length":
            # The old implementation added `and content.strip()` to the condition, so
            # **this warning did not show when content was empty = the most dangerous case**. The condition is removed.
            print(
                f"⚠️ {self.name}: finish_reason=length — max_tokens="
                f"{self.max_tokens} - the output was truncated "
                f"(completion_tokens={completion_tokens}). "
                "Consider raising max_tokens, or reducing reasoning_effort",
                file=sys.stderr,
            )
        if abort:
            # Thinking cut by a guard is a repetition of the same fragment, so there is
            # no fallback to reasoning. Feeding 180,000 characters of degenerated text to
            # the grader gives no extractable code and only wastes grading time.
            return GenerationResult(
                text="",
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                truncated=True,
                max_tokens=self.max_tokens,
                raw=raw,
            )
        if not content.strip():
            # For reasoning(thinking) models: when reasoning_format is set, llama.cpp/vLLM
            # split <think>... into reasoning_content (a separate field), so
            # when generation ends before  is closed, content comes back empty.
            # The answer may be mixed in on the reasoning side, so a fallback picks it up.
            if reasoning.strip():
                text = reasoning
                # Content coming back empty means generation ended before  was
                # closed = the thinking did not finish within its budget, so even when the
                # fallback rescues it, the result is marked as truncated.
                truncated = True
                print(
                    f"⚠️ {self.name}: content is empty. Falling back to reasoning_content "
                    f"({len(reasoning)} chars) and trying to parse"
                    f" — the thinking did not finish within its budget (max_tokens={self.max_tokens}, "
                    f"completion_tokens={completion_tokens}), "
                    "so code extracted from it may not be reliable",
                    file=sys.stderr,
                )
            elif completion_tokens and completion_tokens >= self.max_tokens:
                print(
                    f"⚠️ {self.name}: empty output (content/reasoning_content both empty) — "
                    f"completion_tokens={completion_tokens} reached max_tokens="
                    f"{self.max_tokens}. The reasoning(thinking) most likely did not finish "
                    "within its budget"
                    "(fix: raise max_tokens, or reduce reasoning)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"⚠️ {self.name}: empty output (content/reasoning_content both empty, "
                    f"completion_tokens={completion_tokens}, below max_tokens) — "
                    "generation may have stopped early",
                    file=sys.stderr,
                )
            if not text.strip():
                # content with whitespace only counts as empty too (matches the check in patch.py)
                text = ""
        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            truncated=truncated,
            max_tokens=self.max_tokens,
            raw=raw,
        )

    def _post_once(self, system: str, user: str) -> GenerationResult:
        """Calls /chat/completions once and returns GenerationResult (no retries)."""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._build_payload(system, user),
            timeout=self.timeout,
            stream=self.stream,
        )
        # Note: no with-statement (context manager), because the response mocks in the
        # existing tests have no __enter__/__exit__. Only when streaming, release the
        # connection even if an exception is raised midway.
        try:
            if resp.status_code >= 400:
                body = (resp.text or "").strip().replace("\n", " ")
                msg = (
                    f"{resp.status_code} {resp.reason} from "
                    f"{self.base_url}/chat/completions: {body[:500]}"
                )
                if resp.status_code in _RETRYABLE_STATUS:
                    raise RetryableHTTPError(
                        msg, resp.status_code,
                        _parse_retry_after(resp.headers.get("Retry-After")),
                    )
                raise RuntimeError(msg)
            abort = ""
            if self.stream:
                content, reasoning, usage, finish_reason, abort = (
                    self._consume_stream(resp)
                )
                raw = {
                    "stream": True,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "abort": abort,
                }
            else:
                data = resp.json()
                usage = data.get("usage", {}) or {}
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                reasoning = (
                    message.get("reasoning_content") or message.get("reasoning") or ""
                )
                finish_reason = choice.get("finish_reason")
                raw = data
        finally:
            if self.stream:
                resp.close()
        return self._finalize(content, reasoning, usage, finish_reason, raw, abort)

    def _generate(self, system: str, user: str) -> GenerationResult:
        """Wrapper around ``_post_once`` that retries transient failures with exponential backoff.

        Retried:
          - ``_TRANSIENT_ERRORS`` (ConnectionError / Timeout /
            ConnectionResetError / ChunkedEncodingError)
          - HTTP errors with ``_RETRYABLE_STATUS`` (429 / 500 / 502 / 503 / 504).
            For 429, the Retry-After header value takes precedence.
        Other 4xx (400/401/404 etc.) are not retried (the cause lies on the
        caller's side; retrying cannot fix it).
        """
        retryable = _TRANSIENT_ERRORS + (RetryableHTTPError,)
        last_exc: BaseException | None = None
        for attempt in range(self.transient_retries + 1):
            try:
                return self._post_once(system, user)
            except retryable as e:
                last_exc = e
                if attempt < self.transient_retries:
                    delay = self.transient_backoff * (2 ** attempt)
                    hint = ""
                    if isinstance(e, RetryableHTTPError) and e.retry_after is not None:
                        delay = e.retry_after
                        hint = " (Retry-After)"
                    print(
                        f"⚠️ Transient error on {self.name} "
                        f"(attempt {attempt + 1}/{self.transient_retries + 1}): "
                        f"{type(e).__name__}: {str(e)[:120]} — "
                        f"retry in {delay:.1f}s{hint}",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                break
        assert last_exc is not None
        raise last_exc

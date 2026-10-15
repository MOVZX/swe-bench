"""Unit tests for streaming support (``stream: true``). No network needed - everything is mocked.

Background: qwen3.8-max (a thinking model) hit Read timeout / Connection reset
very often. With streaming disabled, two things go wrong:
  (1) the requests read timeout acts as a limit on the whole generation
  (2) an intermediate gateway drops a connection that stays silent while it is
      still running
``stream: true`` fixes both, so we must guarantee that it does not break
tok/s measurement (usage) or the reasoning_content fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import llmbench.clients.openai_compat as oc
from llmbench.clients.openai_compat import OpenAICompatClient


# ---------- response mocks used by the tests ----------

class _FakeStreamResp:
    """Mock of a response that returns SSE. Records whether close() was called."""

    def __init__(self, lines, status=200, headers=None):
        self._lines = lines
        self.status_code = status
        self.reason = "OK"
        self.text = ""
        self.headers = headers or {}
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        self.closed = True


class _FakeJsonResp:
    """Mock of a non-streaming (legacy) response."""

    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
        self.reason = "OK"
        self.text = ""
        self.headers = {}

    def json(self):
        return self._data


def _sse(obj) -> str:
    return "data: " + json.dumps(obj) + ""


def _delta(content=None, reasoning=None, finish_reason=None) -> str:
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return _sse({"choices": [{"delta": d, "finish_reason": finish_reason}]})


def _usage_chunk(prompt=5, completion=10) -> str:
    return _sse({
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    })


def _install(monkeypatch, resp):
    """Replace requests.post so the test can inspect the payload that was sent."""
    sent = {}

    def _post(url, **kw):
        sent.update(kw.get("json") or {})
        sent["_stream_kw"] = kw.get("stream")
        sent["_timeout"] = kw.get("timeout")
        return resp

    monkeypatch.setattr(oc.requests, "post", _post)
    return sent


def _client(**overrides):
    cfg = {"base_url": "http://h/v1", "model": "x", "max_tokens": 100}
    cfg.update(overrides)
    return OpenAICompatClient("m", cfg)


# ---------- regression: without stream, the request stays non-streaming ----------

def test_default_is_non_stream(monkeypatch):
    """Models that do not set stream (local-openai/gemini/glm) behave as before."""
    resp = _FakeJsonResp({
        "choices": [{"message": {"content": "--- FILE: a.py ---"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    })
    sent = _install(monkeypatch, resp)
    r = _client()._post_once("sys", "user")
    assert r.text == "--- FILE: a.py ---"
    assert "stream" not in sent, "stream was sent although it is not configured"
    assert "stream_options" not in sent
    assert sent["_stream_kw"] is False


# ---------- request shape when stream: true ----------

def test_stream_request_shape(monkeypatch):
    sent = _install(monkeypatch, _FakeStreamResp([_delta(content="x"),
                                                  _usage_chunk()]))
    _client(stream=True, timeout=2400)._post_once("sys", "user")
    assert sent["stream"] is True
    # Without an explicit usage request, completion_tokens is not returned and
    # tok/s cannot be measured.
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["_stream_kw"] is True
    assert sent["_timeout"] == 2400


# ---------- SSE parsing ----------

def test_stream_concatenates_content_and_reads_usage(monkeypatch):
    _install(monkeypatch, _FakeStreamResp([
        _delta(reasoning="thinking..."),
        ": keep-alive",                  # SSE comments are ignored
        "",                              # empty lines are ignored too
        "event: ping",                   # lines that are not data: are ignored
        _delta(content="--- FILE: "),
        _delta(content="a.py ---"),
        "data: {brokenJSON",              # a broken line must not raise
        _usage_chunk(prompt=11, completion=22),
        "data: [DONE]",
    ]))
    r = _client(stream=True)._post_once("sys", "user")
    assert r.text == "--- FILE: a.py ---"
    assert r.prompt_tokens == 11
    assert r.completion_tokens == 22


def test_stream_stops_at_done(monkeypatch):
    _install(monkeypatch, _FakeStreamResp([
        _delta(content="keep"),
        "data: [DONE]",
        _delta(content="MUST-NOT-APPEAR"),
    ]))
    assert _client(stream=True)._post_once("s", "u").text == "keep"


def test_stream_tokens_per_sec_still_computable(monkeypatch):
    """tok/s is the main metric of the benchmark. Streaming must not lose it."""
    _install(monkeypatch, _FakeStreamResp([_delta(content="x"),
                                           _usage_chunk(completion=22)]))
    r = _client(stream=True).generate("s", "u")   # via generate, so latency is set
    assert r.completion_tokens == 22
    assert r.tokens_per_sec is not None and r.tokens_per_sec > 0


# ---------- empty content -> reasoning_content fallback (streaming version) ----------

def test_stream_falls_back_to_reasoning_content(capsys, monkeypatch):
    _install(monkeypatch, _FakeStreamResp([
        _delta(reasoning="<think>--- FILE: a.py ---"),
        _usage_chunk(completion=100),
    ]))
    r = _client(stream=True, max_tokens=100)._post_once("s", "u")
    assert r.text == "<think>--- FILE: a.py ---"
    assert "reasoning_content" in capsys.readouterr().err


# ---------- show finish_reason=length ----------

def test_length_finish_reason_warns(capsys, monkeypatch):
    """Cut-off by max_tokens must not happen silently (the main cause here)."""
    _install(monkeypatch, _FakeStreamResp([
        _delta(content="--- FILE: a.py ---", finish_reason="length"),
        _usage_chunk(completion=100),
    ]))
    _client(stream=True, max_tokens=100)._post_once("s", "u")
    err = capsys.readouterr().err
    assert "finish_reason=length" in err
    assert "max_tokens" in err


def test_normal_finish_reason_does_not_warn(capsys, monkeypatch):
    _install(monkeypatch, _FakeStreamResp([
        _delta(content="ok", finish_reason="stop"),
        _usage_chunk(),
    ]))
    _client(stream=True)._post_once("s", "u")
    assert "finish_reason=length" not in capsys.readouterr().err


# ---------- reasoning_effort ----------

def test_reasoning_effort_omitted_by_default(monkeypatch):
    """When unset, the key is not sent at all (measured with server default xhigh)."""
    sent = _install(monkeypatch, _FakeStreamResp([_delta(content="x"),
                                                  _usage_chunk()]))
    _client(stream=True)._post_once("s", "u")
    assert "reasoning_effort" not in sent


def test_reasoning_effort_passed_when_set(monkeypatch):
    sent = _install(monkeypatch, _FakeStreamResp([_delta(content="x"),
                                                  _usage_chunk()]))
    _client(stream=True, reasoning_effort="medium")._post_once("s", "u")
    assert sent["reasoning_effort"] == "medium"


# ---------- connection cleanup and retries ----------

def test_stream_response_is_closed(monkeypatch):
    """A stream must be closed explicitly, or the connection stays open."""
    resp = _FakeStreamResp([_delta(content="x"), _usage_chunk()])
    _install(monkeypatch, resp)
    _client(stream=True)._post_once("s", "u")
    assert resp.closed is True


def test_stream_response_closed_even_on_http_error(monkeypatch):
    resp = _FakeStreamResp([], status=400)
    _install(monkeypatch, resp)
    with pytest.raises(RuntimeError, match="400"):
        _client(stream=True)._post_once("s", "u")
    assert resp.closed is True


# ---------- retry on 429 / 5xx (required when --concurrency > 1) ----------

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_status_is_retried(status, monkeypatch):
    """Rate limits and temporary gateway failures recover after a delay, so retry."""
    calls = {"n": 0}
    ok = _FakeStreamResp([_delta(content="recovered"), _usage_chunk()])

    def _post(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeStreamResp([], status=status)
        return ok

    monkeypatch.setattr(oc.requests, "post", _post)
    c = _client(stream=True, transient_retries=2, transient_backoff=0)
    assert c._generate("s", "u").text == "recovered"
    assert calls["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_status_fails_immediately(status, monkeypatch):
    """Do not burn a 3-hour run on errors that a resend cannot fix."""
    calls = {"n": 0}

    def _post(url, **kw):
        calls["n"] += 1
        return _FakeStreamResp([], status=status)

    monkeypatch.setattr(oc.requests, "post", _post)
    with pytest.raises(RuntimeError, match=str(status)):
        _client(stream=True, transient_retries=2, transient_backoff=0)._generate(
            "s", "u"
        )
    assert calls["n"] == 1, "a non-retryable error was retried"


def test_retry_after_header_overrides_backoff(monkeypatch):
    """When 429 returns Retry-After, it takes precedence over the backoff."""
    slept = []
    monkeypatch.setattr(oc.time, "sleep", slept.append)
    calls = {"n": 0}
    ok = _FakeStreamResp([_delta(content="ok"), _usage_chunk()])

    def _post(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeStreamResp([], status=429, headers={"Retry-After": "7"})
        return ok

    monkeypatch.setattr(oc.requests, "post", _post)
    _client(stream=True, transient_retries=2, transient_backoff=99)._generate("s", "u")
    assert slept == [7.0], f"Retry-After was not used: {slept}"


def test_absurd_retry_after_is_capped(monkeypatch):
    """An out-of-range Retry-After must not stop the run."""
    slept = []
    monkeypatch.setattr(oc.time, "sleep", slept.append)
    calls = {"n": 0}
    ok = _FakeStreamResp([_delta(content="ok"), _usage_chunk()])

    def _post(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeStreamResp([], status=429, headers={"Retry-After": "99999"})
        return ok

    monkeypatch.setattr(oc.requests, "post", _post)
    _client(stream=True, transient_retries=2, transient_backoff=1)._generate("s", "u")
    assert slept == [oc._MAX_RETRY_AFTER]


def test_malformed_retry_after_falls_back_to_backoff(monkeypatch):
    """An HTTP-date value or an invalid value does not raise; it uses exponential backoff."""
    slept = []
    monkeypatch.setattr(oc.time, "sleep", slept.append)
    calls = {"n": 0}
    ok = _FakeStreamResp([_delta(content="ok"), _usage_chunk()])

    def _post(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeStreamResp(
                [], status=429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
            )
        return ok

    monkeypatch.setattr(oc.requests, "post", _post)
    _client(stream=True, transient_retries=2, transient_backoff=3)._generate("s", "u")
    assert slept == [3.0]


def test_transient_retry_works_with_stream(monkeypatch):
    """A single Connection reset must not drop the task (transient_retries regression)."""
    calls = {"n": 0}
    ok = _FakeStreamResp([_delta(content="recovered"), _usage_chunk()])

    def _post(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise oc.requests.exceptions.ConnectionError(
                "('Connection aborted.', ConnectionResetError(54, ...))"
            )
        return ok

    monkeypatch.setattr(oc.requests, "post", _post)
    c = _client(stream=True, transient_retries=2, transient_backoff=0)
    assert c._generate("s", "u").text == "recovered"
    assert calls["n"] == 2


# ---------- runner: keep the elapsed time when generation fails ----------

def test_generation_error_records_elapsed_time():
    """Regression for timeout failures shown as (0.0s), which blocked root-cause work."""
    import time as _time

    from llmbench.runner import BenchmarkRunner

    class _SlowFailClient:
        def generate(self, system, user):
            _time.sleep(0.05)
            raise oc.requests.exceptions.Timeout("Read timed out. (read timeout=600)")

    r = BenchmarkRunner({}, Path("tasks"))
    at = r._one_attempt(_SlowFailClient(), None, "sys", "user", None, None, 0)
    assert at.resolved is False
    assert "generation error" in at.fail_reason
    assert at.latency_sec > 0, "elapsed time was not recorded when generation failed"


# ---------- runner: regeneration after a parse failure ----------

def test_generate_retries_controls_regeneration():
    """With --generate-retries 0, a parse failure must not trigger regeneration."""
    from llmbench.clients.base import GenerationResult
    from llmbench.runner import BenchmarkRunner

    class _CountingClient:
        def __init__(self):
            self.n = 0

        def generate(self, system, user):
            self.n += 1
            return GenerationResult(
                text="output that cannot be parsed", latency_sec=0.01, completion_tokens=1
            )

    class _AlwaysParseFailGrader:
        def evaluate(self, task, text, ctx):
            class _Ev:
                parse_ok = False
                resolved = False
                quality_score = 0.0
                parse_error = "no FILE block"
                parsed_files: dict = {}
                fail_reason = "parse failed"
                detail_output = ""
                components: dict = {}
            return _Ev()

    r = BenchmarkRunner({}, Path("tasks"))
    c0 = _CountingClient()
    r._one_attempt(c0, _AlwaysParseFailGrader(), "s", "u", None, None, 0)
    c1 = _CountingClient()
    r._one_attempt(c1, _AlwaysParseFailGrader(), "s", "u", None, None, 1)
    assert c0.n == 1, "regeneration happened although retries=0"
    assert c1.n == 2, "no regeneration happened although retries=1"

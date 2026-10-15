"""Regression tests for the SSE character-encoding mix-up (network: local loop only).

Real damage (2026-08-19): llama.cpp returns `Content-Type: text/event-stream`
**without a charset**. requests follows RFC 2616 and takes ISO-8859-1 as the
default for `text/*`, so `iter_lines(decode_unicode=True)` decoded the UTF-8
bytes as latin-1.

The damage came in two stages:
  1. Mojibake - each Japanese character (3 UTF-8 bytes) becomes 3 characters, so
     the char_count checks were off by about 3x.
  2. **Silent loss** - after latin-1 decoding the string contains characters such
     as U+0085 (NEL), `str.splitlines()` treats them as line breaks and splits
     the SSE lines. A split line fails `json.loads` and is dropped by
     `except json.JSONDecodeError: continue`.

This made every `--lang ja` run miss all Japanese keyword matching, so generation
tasks failed by mistake. The fix is `resp.encoding = "utf-8"` at the top of
`_consume_stream`.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest
import requests

from llmbench.clients.openai_compat import OpenAICompatClient

# The string must contain a character whose UTF-8 bytes include 0x85 (NEL).
# U+5148 (E5 85 88) qualifies: latin-1 decoding splits the line there.
JA = "野獣先輩、今日は暑スギィ！じゃけん冷房つけましょうね〜"


class _RecordingResp:
    """A mock that only records whether encoding was assigned."""

    def __init__(self, lines):
        self._lines = lines
        self.encoding = "ISO-8859-1"      # the requests default for text/* with no charset
        self.status_code = 200
        self.reason = "OK"
        self.text = ""
        self.headers = {}

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        pass


def _client(**overrides) -> OpenAICompatClient:
    cfg = {"base_url": "http://127.0.0.1:1/v1", "model": "m",
           "max_tokens": 256, "stream": True}
    cfg.update(overrides)
    return OpenAICompatClient("m", cfg)


def test_consume_stream_forces_utf8():
    """resp.encoding is pinned to utf-8 before any SSE line is read."""
    line = "data: " + json.dumps(
        {"choices": [{"delta": {"content": JA}, "finish_reason": "stop"}]},
        ensure_ascii=False,
    )
    resp = _RecordingResp([line, "data: [DONE]"])
    content, _reasoning, _usage, _finish, _abort = _client()._consume_stream(resp)
    assert resp.encoding == "utf-8", (
        "resp.encoding was not pinned to utf-8. requests decodes a "
        "text/event-stream without a charset as ISO-8859-1, and the Japanese text "
        "becomes mojibake"
    )
    assert content == JA


# ---------- integration test over real HTTP ----------

class _SSEHandler(http.server.BaseHTTPRequestHandler):
    """A server that returns SSE without a charset, like llama.cpp."""

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")   # no charset
        self.end_headers()
        for i in range(0, len(JA), 6):                          # send it in small pieces
            chunk = {"choices": [{"delta": {"content": JA[i:i + 6]}}]}
            self.wfile.write(
                b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8")
                + b"\n\n"
            )
        self.wfile.write(
            b"data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            ).encode("utf-8") + b"\n\n"
        )
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def sse_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _SSEHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_japanese_survives_real_sse(sse_server):
    """Japanese text stays complete and intact against a real server with no charset."""
    got = _client(base_url=sse_server).generate("sys", "user")
    assert got.text == JA, f"the Japanese text is broken: {got.text!r}"
    assert len(got.text) == len(JA), (
        f"the character count is off ({len(got.text)} != {len(JA)}). "
        "latin-1 decoding inflates it about 3x and breaks every char_count check"
    )


def test_japanese_survives_real_sse_client(sse_server, monkeypatch):
    """The same through the client (with base_url replaced)."""
    c = _client(base_url=sse_server)
    assert c.generate("sys", "user").text == JA


def test_latin1_decoding_would_have_broken_it(sse_server):
    """Reproduce the pre-fix behaviour and pin down that it really breaks (evidence of the spec)."""
    r = requests.post(f"{sse_server}/chat/completions", json={}, stream=True)
    assert r.encoding == "ISO-8859-1", "the server is returning a charset after all"
    out, dropped = "", 0
    for line in r.iter_lines(decode_unicode=True):      # the old path, encoding left as is
        if not line or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            delta = json.loads(body)["choices"][0].get("delta") or {}
        except json.JSONDecodeError:
            dropped += 1                                 # the old code dropped these silently
            continue
        out += delta.get("content", "")
    assert out != JA, "if it does not break, the premise changed. Review this regression test"
    assert dropped > 0, "no line split happened. Put a character whose UTF-8 bytes contain 0x85 (NEL) into JA"

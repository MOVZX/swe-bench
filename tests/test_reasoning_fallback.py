"""Unit tests for reasoning(thinking) models: empty content -> reasoning_content fallback and diagnostics.

No real server is used. requests.post is mocked, so only the behaviour of
OpenAICompatClient._post_once (text selection and warning messages) is checked.
"""

from __future__ import annotations

import llmbench.clients.openai_compat as oc
from llmbench.clients.openai_compat import OpenAICompatClient


class _FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
        self.reason = "OK"
        self.text = ""

    def json(self):
        return self._data


def _client(max_tokens=100, monkeypatch=None):
    return OpenAICompatClient(
        "m", {"base_url": "http://h/v1", "model": "x", "max_tokens": max_tokens}
    )


def _mock_post(data):
    def _post(url, **kw):
        return _FakeResp(data)
    return _post


def _chat_response(message, completion_tokens=10):
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 5, "completion_tokens": completion_tokens},
    }


# ---------- normal models: use content when it is present ----------

def test_normal_content_used_as_is(monkeypatch):
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response({"content": "--- FILE: a.py ---"})),
    )
    c = _client()
    r = c._post_once("sys", "user")
    assert r.text == "--- FILE: a.py ---"


# ---------- empty content + an answer-like reasoning_content -> fallback ----------

def test_empty_content_falls_back_to_reasoning_content(capsys, monkeypatch):
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response({
            "content": "",
            "reasoning_content": "<think>...--- FILE: a.py ---...</think>",
        }, completion_tokens=50)),
    )
    c = _client(max_tokens=100)
    r = c._post_once("sys", "user")
    assert r.text == "<think>...--- FILE: a.py ---...</think>"
    assert "reasoning_content" in capsys.readouterr().err


def test_empty_content_falls_back_to_reasoning_alt_key(monkeypatch):
    # some servers use the reasoning key instead of reasoning_content
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response({
            "content": "",
            "reasoning": "partial thought",
        })),
    )
    c = _client()
    r = c._post_once("sys", "user")
    assert r.text == "partial thought"


# ---------- empty content + empty reasoning + max_tokens reached -> "budget exhausted" warning ----------

def test_empty_all_at_max_tokens_warns_budget_exhausted(capsys, monkeypatch):
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response(
            {"content": "", "reasoning_content": ""}, completion_tokens=100,
        )),
    )
    c = _client(max_tokens=100)
    r = c._post_once("sys", "user")
    assert r.text == ""
    err = capsys.readouterr().err
    assert "max_tokens" in err
    assert "did not finish" in err


# ---------- empty content + empty reasoning + below max_tokens -> "early stop" warning ----------

def test_empty_all_under_max_tokens_warns_early_stop(capsys, monkeypatch):
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response(
            {"content": "", "reasoning_content": ""}, completion_tokens=12,
        )),
    )
    c = _client(max_tokens=100)
    r = c._post_once("sys", "user")
    assert r.text == ""
    err = capsys.readouterr().err
    assert "stopped early" in err
    assert "completion_tokens=12" in err


def test_completion_tokens_still_reported_on_empty(monkeypatch):
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response({"content": ""}, completion_tokens=24576)),
    )
    c = _client(max_tokens=24576)
    r = c._post_once("sys", "user")
    assert r.completion_tokens == 24576


def test_whitespace_only_content_treated_as_empty(monkeypatch):
    monkeypatch.setattr(
        oc.requests, "post",
        _mock_post(_chat_response({"content": "   \n  "})),
    )
    c = _client()
    r = c._post_once("sys", "user")
    assert r.text == ""

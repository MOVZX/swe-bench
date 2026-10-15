"""Unit tests for the subscription CLI client (type: cli).

Instead of the real CLIs (claude / codex / grok), fake CLIs made of Python
one-liners verify command building, prompt passing, output parsing and the
error paths.
"""

from __future__ import annotations

import sys

import pytest

from llmbench.clients import create_client
from llmbench.clients.cli_agent import CliAgentClient

PY = sys.executable


def _cli(cfg: dict, name: str = "sub") -> CliAgentClient:
    return CliAgentClient(name, {"type": "cli", **cfg})


# ---------- generation (custom / stdout) ----------

def test_custom_arg_prompt_stdout():
    c = _cli({
        "preset": "custom",
        "command": [PY, "-c", "import sys; print('echo:' + sys.argv[-1])"],
    })
    r = c.generate("SYS", "hello")
    # system is prepended to the user prompt
    assert r.text == "echo:SYS\n\nhello"
    assert r.latency_sec > 0


def test_custom_stdin_prompt():
    c = _cli({
        "preset": "custom",
        "prompt_via": "stdin",
        "command": [PY, "-c", "import sys; print(sys.stdin.read().upper())"],
    })
    r = c.generate("", "hi")
    assert r.text == "HI"


def test_create_client_dispatch():
    c = create_client("sub", {
        "type": "cli", "preset": "custom",
        "command": [PY, "-c", "print('x')"],
    })
    assert isinstance(c, CliAgentClient)


# ---------- claude preset (JSON parsing) ----------

_FAKE_CLAUDE = (
    "import sys, json;"
    "p = sys.stdin.read();"
    "print(json.dumps({'type': 'result', 'is_error': False,"
    " 'result': 'patched: ' + p[:10],"
    " 'usage': {'input_tokens': 7, 'output_tokens': 42}}))"
)


def test_claude_json_parse():
    c = _cli({
        "preset": "claude",
        "command": [PY, "-c", _FAKE_CLAUDE],  # command replaces only the CLI binary
    })
    assert c.prompt_via == "stdin" and c.parse_mode == "claude_json"
    r = c.generate("", "fix the bug")
    assert r.text.startswith("patched: fix the bu")
    assert r.prompt_tokens == 7
    assert r.completion_tokens == 42


def test_claude_json_is_error_raises():
    c = _cli({
        "preset": "claude",
        "command": [PY, "-c",
                    "import json;"
                    "print(json.dumps({'is_error': True,"
                    " 'result': 'rate limit reached'}))"],
    })
    with pytest.raises(RuntimeError, match="rate limit"):
        c.generate("", "x")


def test_claude_non_json_falls_back_to_stdout():
    c = _cli({
        "preset": "claude",
        "command": [PY, "-c", "print('plain text output')"],
    })
    assert c.generate("", "x").text == "plain text output"


# ---------- codex preset (--output-last-message) ----------

_FAKE_CODEX = (
    "import sys, pathlib;"
    "i = sys.argv.index('--output-last-message');"
    "pathlib.Path(sys.argv[i + 1]).write_text('final answer');"
    "print('thinking... noise ...')"
)


def test_codex_last_message_file():
    c = _cli({
        "preset": "codex",
        "command": [PY, "-c", _FAKE_CODEX],
    })
    assert c.parse_mode == "last_message_file"
    r = c.generate("", "task")
    assert r.text == "final answer"       # the file is used, not the stdout noise


def test_codex_missing_file_falls_back_to_stdout():
    c = _cli({
        "preset": "codex",
        "command": [PY, "-c", "print('stdout only')"],
    })
    assert c.generate("", "task").text == "stdout only"


# ---------- command building ----------

def test_model_flag_and_extra_args():
    c = _cli({
        "preset": "claude",
        "model": "sonnet",
        "extra_args": ["--max-turns", "3"],
    })
    argv = c.build_argv("PROMPT", None)
    assert argv[:4] == ["claude", "-p", "--output-format", "json"]
    assert ["--model", "sonnet"] == argv[4:6]
    assert ["--max-turns", "3"] == argv[6:8]
    assert "PROMPT" not in argv               # claude takes the prompt on stdin


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "t0k")
    c = _cli({
        "preset": "custom",
        "command": [PY, "-c",
                    "import os; print(os.environ['INJECTED'])"],
        "env": {"INJECTED": "${MY_TOKEN}"},
    })
    assert c.generate("", "x").text == "t0k"


# ---------- error paths ----------

def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="preset"):
        _cli({"preset": "gemini"})


def test_custom_without_command_raises():
    with pytest.raises(ValueError, match="command"):
        _cli({"preset": "custom"})


def test_missing_binary_friendly_error():
    c = _cli({"preset": "custom", "command": ["no-such-cli-xyz"]})
    with pytest.raises(ValueError, match="no-such-cli-xyz"):
        c.generate("", "x")


def test_claude_missing_binary_shows_install_hint(monkeypatch):
    import llmbench.clients.cli_agent as ca
    monkeypatch.setattr(ca.shutil, "which", lambda _: None)
    c = _cli({"preset": "claude"})
    with pytest.raises(ValueError, match="claude-code"):
        c.generate("", "x")


def test_nonzero_exit_raises_with_stderr():
    c = _cli({
        "preset": "custom",
        "command": [PY, "-c",
                    "import sys; print('usage limit', file=sys.stderr);"
                    "sys.exit(2)"],
    })
    with pytest.raises(RuntimeError, match="usage limit"):
        c.generate("", "x")


def test_timeout_raises():
    c = _cli({
        "preset": "custom",
        "timeout": 1,
        "command": [PY, "-c", "import time; time.sleep(10)"],
    })
    with pytest.raises(RuntimeError, match="timed out"):
        c.generate("", "x")


def test_prompt_via_invalid_raises():
    with pytest.raises(ValueError, match="prompt_via"):
        _cli({"preset": "claude", "prompt_via": "file"})


# ---------- detection of the executed model ----------

_FAKE_CLAUDE_MODELS = (
    "import sys, json;"
    "sys.stdin.read();"
    "print(json.dumps({'is_error': False, 'result': 'ok',"
    " 'usage': {'input_tokens': 1, 'output_tokens': 2},"
    " 'modelUsage': {'claude-haiku-4': {'outputTokens': 5},"
    " 'claude-sonnet-4-6': {'outputTokens': 500}}}))"
)


def test_claude_detects_model_from_model_usage(capsys):
    c = _cli({"preset": "claude", "command": [PY, "-c", _FAKE_CLAUDE_MODELS]})
    c.generate("", "x")
    assert c.detected_model == "claude-sonnet-4-6"   # the model with the most output tokens wins
    assert "Executed model" in capsys.readouterr().err
    c.generate("", "x")                              # the same model is not printed again
    assert "Executed model" not in capsys.readouterr().err


def test_claude_no_model_usage_keeps_none():
    c = _cli({"preset": "claude", "command": [PY, "-c", _FAKE_CLAUDE]})
    c.generate("", "x")
    assert c.detected_model is None


def test_codex_detects_model_from_banner():
    script = (
        "import sys, pathlib;"
        "i = sys.argv.index('--output-last-message');"
        "pathlib.Path(sys.argv[i + 1]).write_text('ans');"
        "print('workdir: /tmp');print('model: gpt-5.3-codex');print('---')"
    )
    c = _cli({"preset": "codex", "command": [PY, "-c", script]})
    r = c.generate("", "x")
    assert r.text == "ans"
    assert c.detected_model == "gpt-5.3-codex"


def test_model_change_warns_again(capsys):
    c = _cli({"preset": "claude", "command": [PY, "-c", "pass"]})
    c._note_model("m-one")
    c._note_model("m-one")            # the same model is ignored
    c._note_model("m-two")            # a change is warned about again
    err = capsys.readouterr().err
    assert err.count("🔎 Executed model: m-one") == 1
    assert "⚠️ Executed model changed: m-two" in err


# ---------- temperature warning (sample_temp is ignored when runs > 1) ----------

def test_temperature_override_warns_once(capsys):
    c = _cli({
        "preset": "custom",
        "command": [PY, "-c", "print('ok')"],
    })
    c.temperature = 0.8          # reproduces the override the runner does when runs > 1
    c.generate("", "x")
    c.generate("", "x")
    err = capsys.readouterr().err
    assert err.count("cannot control temperature") == 1

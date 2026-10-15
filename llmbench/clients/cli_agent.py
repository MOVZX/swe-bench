"""Subscription CLI client — run agent CLIs headless on a flat-rate plan.

Chat subscriptions such as Claude Pro/Max, ChatGPT (Codex) and SuperGrok do not
officially provide an OpenAI-compatible API, but each vendor's official CLI runs
non-interactively (headless) while authenticated with the subscription:

  - Claude Code : ``claude -p "<prompt>" --output-format json``
  - Codex CLI   : ``codex exec "<prompt>" --output-last-message <file>``
  - Grok Build  : ``grok exec "<prompt>"``

This client calls them with subprocess and normalizes the final output into a
GenerationResult. Presets (claude / codex / grok), or an arbitrary command
(preset: custom), connect other CLIs too.

⚠️ Benchmark cautions (see README / USAGE):
  - What it measures is **an agent product (CLI + model)**, not the bare model.
    A system prompt and autonomous retries are added on top, so do not compare it
    on equal footing with bare completion (type: openai).
  - temperature is not controllable. sample_temp from ``--runs N`` is ignored
    (a warning is printed on the first generation).
  - When you reach the subscription rate limits (5-hour window / weekly cap) the CLI
    returns an error, and that task counts as lost points. Split runs plus
    ``certify --merge`` are recommended.
  - Each generation runs with an empty temporary directory as cwd (so the agent
    does not read or write the repository at hand).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import GenerationResult, LLMClient, expand_env

# Preset definitions:
#   argv       : base command
#   prompt_via : stdin | arg  (how the prompt is passed)
#   model_flag : flag added when cfg["model"] is given
#   parse      : claude_json | last_message_file | stdout
PRESETS: dict[str, dict] = {
    "claude": {
        "argv": ["claude", "-p", "--output-format", "json"],
        "prompt_via": "stdin",
        "model_flag": "--model",
        "parse": "claude_json",
        "install_hint": (
            "Claude Code CLI is required: npm install -g @anthropic-ai/claude-code"
            " → start `claude` once and log in with a Pro/Max account"
        ),
    },
    "codex": {
        "argv": ["codex", "exec", "--skip-git-repo-check"],
        "prompt_via": "arg",
        "model_flag": "--model",
        "parse": "last_message_file",
        "install_hint": (
            "Codex CLI is required: npm install -g @openai/codex"
            " → start `codex` once and log in with a ChatGPT account"
        ),
    },
    "grok": {
        "argv": ["grok", "exec"],
        "prompt_via": "arg",
        "model_flag": "--model",
        "parse": "stdout",
        "install_hint": (
            "Grok Build CLI is required (install it with the official xAI installer,"
            " then log in with an X / xAI account)"
        ),
    },
    # custom: cfg["command"] is required. parse comes from cfg["parse"] (default stdout)
    "custom": {
        "argv": [],
        "prompt_via": "arg",
        "model_flag": None,
        "parse": "stdout",
        "install_hint": "",
    },
}

_LAST_MSG_FLAG = "--output-last-message"

# The "model: <name>" line that codex exec and friends print in their banner (detected from the start of stdout / stderr)
_MODEL_LINE_RE = re.compile(r"(?im)^\s*model:\s*([\w.\-/:]+)\s*$")


def _model_from_claude_json(data: dict) -> str | None:
    """Extract the executed model name from the JSON response of claude -p.

    Newer CLIs include modelUsage (model name → token totals). When several models
    appear (main + a sub-agent such as haiku), the one with the most output tokens wins.
    """
    mu = data.get("modelUsage")
    if isinstance(mu, dict) and mu:
        def _out(v) -> int:
            return int(v.get("outputTokens", 0)) if isinstance(v, dict) else 0
        return max(mu, key=lambda k: _out(mu[k]))
    m = data.get("model")
    return str(m) if m else None


class CliAgentClient(LLMClient):
    """Client that runs an official agent CLI as a subprocess (type: cli)."""

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        preset_name = str(cfg.get("preset", "custom")).lower()
        if preset_name not in PRESETS:
            raise ValueError(
                f"models.{name}: unknown preset {preset_name!r}."
                f" valid values: {', '.join(PRESETS)}"
            )
        preset = PRESETS[preset_name]
        self.preset = preset_name

        # command: replaces the preset argv (required for custom)
        command = cfg.get("command") or preset["argv"]
        if isinstance(command, str):
            command = command.split()
        if not command:
            raise ValueError(
                f"models.{name}: command is required with preset: custom "
                "(example: command: ['mycli', 'run', '--json'])"
            )
        self.command = [expand_env(c, where=f"models.{name}.command") for c in command]

        self.prompt_via = cfg.get("prompt_via", preset["prompt_via"])
        if self.prompt_via not in ("stdin", "arg"):
            raise ValueError(
                f"models.{name}: prompt_via must be one of stdin | arg"
            )
        self.parse_mode = cfg.get("parse", preset["parse"])
        self.model_flag = cfg.get("model_flag", preset["model_flag"])
        self.model = str(
            expand_env(cfg.get("model", ""), where=f"models.{name}.model") or ""
        ).strip()
        self.extra_args = [
            str(expand_env(a, where=f"models.{name}.extra_args"))
            for a in (cfg.get("extra_args") or [])
        ]
        # Extra environment variables (tokens and the like are referenced as ${VAR})
        self.env = {
            str(k): str(expand_env(v, where=f"models.{name}.env"))
            for k, v in (cfg.get("env") or {}).items()
        }
        self.install_hint = preset["install_hint"]
        self._cfg_temperature = self.temperature
        self._warned_temperature = False
        # Executed model name detected from the CLI response (the runner records it in the results)
        self.detected_model: str | None = None

    def _note_model(self, model: str | None) -> None:
        """Record the executed model detected from the CLI response; print it on first use or on change."""
        model = (model or "").strip()
        if not model or model == self.detected_model:
            return
        changed = self.detected_model is not None
        self.detected_model = model
        mark = "⚠️ Executed model changed" if changed else "🔎 Executed model"
        print(f"{mark}: {model} ({self.name}: detected from CLI response)", file=sys.stderr)

    # ---- Command construction ----

    def build_argv(self, prompt: str, last_msg_path: Path | None) -> list[str]:
        argv = list(self.command)
        if self.model and self.model_flag:
            argv += [self.model_flag, self.model]
        argv += self.extra_args
        if self.parse_mode == "last_message_file" and last_msg_path is not None:
            argv += [_LAST_MSG_FLAG, str(last_msg_path)]
        if self.prompt_via == "arg":
            argv.append(prompt)
        return argv

    # ---- Output parsing ----

    def _parse_output(self, stdout: str, last_msg_path: Path | None) -> GenerationResult:
        if self.parse_mode == "claude_json":
            try:
                data = json.loads(stdout)
            except (json.JSONDecodeError, ValueError):
                # --output-format json did not take effect etc. → continue with the raw stdout
                return GenerationResult(text=stdout.strip(), raw={"stdout": stdout})
            usage = data.get("usage") or {}
            if data.get("is_error"):
                raise RuntimeError(
                    f"{self.command[0]} returned an error: "
                    f"{str(data.get('result', ''))[:300]}"
                )
            self._note_model(_model_from_claude_json(data))
            return GenerationResult(
                text=str(data.get("result", "")),
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                raw=data,
            )
        if self.parse_mode == "last_message_file":
            if last_msg_path is not None and last_msg_path.exists():
                text = last_msg_path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return GenerationResult(text=text, raw={"stdout": stdout})
            # File not written (older CLI etc.) → fall back to stdout
            return GenerationResult(text=stdout.strip(), raw={"stdout": stdout})
        # stdout
        return GenerationResult(text=stdout.strip(), raw={"stdout": stdout})

    # ---- Generation ----

    def _generate(self, system: str, user: str) -> GenerationResult:
        if (
            self.temperature != self._cfg_temperature
            and not self._warned_temperature
        ):
            self._warned_temperature = True
            print(
                f"⚠️ {self.name} (type: cli) cannot control temperature."
                f" sample_temp={self.temperature} is ignored "
                "(diversity between attempts depends on the CLI's own default sampling)",
                file=sys.stderr,
            )
        # An agent CLI carries its own system prompt, so llmbench's system is
        # prepended to the user prompt and passed as one.
        prompt = f"{system.strip()}\n\n{user}" if system.strip() else user

        exe = self.command[0]
        if shutil.which(exe) is None:
            hint = f"\n  {self.install_hint}" if self.install_hint else ""
            raise ValueError(
                f"models.{self.name}: command {exe!r} not found.{hint}"
            )

        # Run in an empty temporary cwd (keep the agent away from the files at hand)
        with tempfile.TemporaryDirectory(prefix="llmbench_cli_") as tmp:
            tmp_path = Path(tmp)
            last_msg_path = (
                tmp_path / "last_message.txt"
                if self.parse_mode == "last_message_file" else None
            )
            argv = self.build_argv(prompt, last_msg_path)
            env = {**os.environ, **self.env}
            try:
                proc = subprocess.run(
                    argv,
                    input=prompt if self.prompt_via == "stdin" else None,
                    capture_output=True,
                    text=True,
                    cwd=tmp,
                    env=env,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"{exe} timed out after {self.timeout}s "
                    f"(extend it with models.{self.name}.timeout)"
                ) from e
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
                raise RuntimeError(
                    f"{exe} exit={proc.returncode}: {err[:500]}"
                    " (possible subscription rate-limit exceeded, or not logged in."
                    f" Run `{exe}` on its own to check)"
                )
            if self.parse_mode != "claude_json":
                # codex/grok family: detect the executed model from the banner's "model: <name>" line
                # (scan only the start of stdout, to avoid false positives)
                head = "\n".join((proc.stdout or "").splitlines()[:15])
                m = _MODEL_LINE_RE.search((proc.stderr or "") + "\n" + head)
                if m:
                    self._note_model(m.group(1))
            return self._parse_output(proc.stdout or "", last_msg_path)

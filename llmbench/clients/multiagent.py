"""multiagent client — runs planner -> coder -> reviewer loop via CodeRouter.

Drop-in client placed as llmbench/clients/multiagent.py.
Implements LLMClient, executing a role-separated loop inside a single `generate(system, user)`
call, returning the final patch in `--- FILE: path ---` format. Grading, hidden tests, pass@k,
quality, certification, and comparison on the llmbench side are reused as-is.

Design (docs/inside/future.md §5):
  - Hidden tests are not passed to agents → internal gate is purely syntax-based (ast).
    Functional tests are evaluated by llmbench using hidden tests after patch application.
  - coder receives llmbench's original SYSTEM_PROMPT (`--- FILE: path ---` full replacement format)
    *unmodified* → fair and identical instructions as the single-agent baseline.
  - planner: diagnoses root cause and fix scope from symptoms (isolating which file(s) to modify).
  - reviewer: checks for over-edits/regressions → coder revises once only on 'REVISE'.
  - tokens / latency are aggregated across all roles → true multi-agent cost is accurately reflected
    in llmbench tok/s and latency (honest comparison).

Example config.yaml models entry:
    multiagent:
      type: multiagent
      base_url: "http://localhost:8088/v1"   # CodeRouter (OpenAI-compatible ingress)
      reviewer: true            # false for planner+coder only
      max_iters: 2              # max regeneration iterations on coder syntax errors
      profiles: {planner: planner, coder: coder, reviewer: reviewer}
"""
from __future__ import annotations

import ast
import json
import os
import re
import time
import urllib.error
import urllib.request

from .base import GenerationResult, LLMClient, expand_env

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.DOTALL)

PLANNER_SYS = (
    "You are a senior engineer diagnosing a bug. Given the issue and the source "
    "files, identify the ROOT CAUSE and exactly which file(s) and lines must "
    "change. Output a short, concrete fix plan as an ordered list. Do NOT write "
    "the full corrected file or a patch. Keep it minimal — do not propose "
    "refactors or changes to unrelated code or test files."
)

REVIEWER_SYS = (
    "You are a strict senior code reviewer. You are given a bug issue and a "
    "proposed fix (full corrected files). Check: (a) does it actually fix the "
    "root cause described in the issue? (b) does it modify unrelated code, "
    "over-refactor, or risk a regression? (c) any obvious remaining bug? "
    "Reply with brief findings, then a final line that is EXACTLY one of: "
    "'VERDICT: PASS' (ship as-is) or 'VERDICT: REVISE' (a targeted fix is needed)."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|REVISE)", re.IGNORECASE)


class MultiAgentClient(LLMClient):
    """LLMClient that executes planner -> coder -> reviewer via CodeRouter."""

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        raw = cfg.get("base_url") or os.environ.get(
            "CODEROUTER_BASE_URL", "http://localhost:8088/v1"
        )
        base = str(expand_env(raw, where=f"models.{name}.base_url")).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.endpoint = base + "/chat/completions"
        self.model = cfg.get("model", "router")  # CodeRouter determines routing via profile header
        self.use_reviewer = bool(cfg.get("reviewer", True))
        self.max_iters = int(cfg.get("max_iters", 2))
        p = cfg.get("profiles", {}) or {}
        self.profiles = {
            "planner": p.get("planner", "planner"),
            "coder": p.get("coder", "coder"),
            "reviewer": p.get("reviewer", "reviewer"),
        }
        # Role token breakdown (saved to raw for debugging/analysis)
        self._roles: list[dict] = []

    # ---- Single wire call ------------------------------------------------
    def _call(self, role: str, system: str, user: str) -> tuple[str, int, int]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-CodeRouter-Profile": self.profiles[role],
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"[{role}] HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"[{role}] connection failed to {self.endpoint}: {e.reason}. "
                "Is CodeRouter running?"
            ) from e
        content = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        self._roles.append({"role": role, "profile": self.profiles[role],
                            "prompt_tokens": pt, "completion_tokens": ct})
        return content, pt, ct

    # ---- Syntax gate (minimal automated checks since hidden tests are omitted) ----------
    @staticmethod
    def _syntax_error(text: str) -> str | None:
        """Parses Python code blocks in output via ast.parse. Returns the first error."""
        blocks = _CODE_BLOCK_RE.findall(text)
        for code in blocks:
            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"line {e.lineno}: {e.msg}"
        return None

    @staticmethod
    def _has_file_blocks(text: str) -> bool:
        return "FILE:" in text and "```" in text

    # ---- Main: Role-separated loop ----------------------------------------
    def _generate(self, system: str, user: str) -> GenerationResult:
        self._roles = []
        t0 = time.monotonic()
        pt_sum = ct_sum = 0

        # 1) planner — Root cause diagnosis + minimal fix strategy
        plan, pt, ct = self._call("planner", PLANNER_SYS, user)
        pt_sum += pt; ct_sum += ct

        # 2) coder — Uses llmbench's system prompt (FILE format) as-is. Regenerates on syntax errors.
        coder_user_base = (
            f"{user}\n\n# Diagnosis and fix plan\n{plan}\n\n"
            "Apply the fix following the plan. Keep changes minimal."
        )
        code_text = ""
        feedback = None
        for _ in range(max(1, self.max_iters)):
            cu = coder_user_base
            if feedback:
                cu += f"\n\n# Your previous output had a syntax error:\n{feedback}\nFix it."
            code_text, pt, ct = self._call("coder", system, cu)
            pt_sum += pt; ct_sum += ct
            err = self._syntax_error(code_text)
            if err is None:
                break
            feedback = err

        # 3) reviewer — Check over-edits/regressions -> revise once if REVISE
        if self.use_reviewer:
            rev, pt, ct = self._call(
                "reviewer", REVIEWER_SYS,
                f"# Issue\n{user}\n\n# Proposed fix\n{code_text}\n\nReview it.",
            )
            pt_sum += pt; ct_sum += ct
            m = _VERDICT_RE.search(rev)
            verdict = m.group(1).upper() if m else "PASS"
            if verdict == "REVISE":
                revised, pt, ct = self._call(
                    "coder", system,
                    coder_user_base
                    + f"\n\n# Reviewer feedback (apply only what is necessary):\n{rev}",
                )
                pt_sum += pt; ct_sum += ct
                # Adopt revision only if parseable & syntax-valid (prevents degradation)
                if self._has_file_blocks(revised) and self._syntax_error(revised) is None:
                    code_text = revised

        return GenerationResult(
            text=code_text,
            latency_sec=round(time.monotonic() - t0, 2),
            prompt_tokens=pt_sum,
            completion_tokens=ct_sum,
            raw={"roles": list(self._roles)},
        )

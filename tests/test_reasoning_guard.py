"""Unit tests for the runaway thinking (reasoning) guard and fail-fast (no network required).

Background - what happened in the measured run:
  Qwen3.8-9B (GGUF, thinking) run with `--runs 5` kept content empty on t033
  (frontier) while reasoning_content grew, and the attempt never finished with
  max_tokens=49,152 pinned.
  1 attempt = about 150s x 1 regeneration from generate_retries=1 x runs=5 =
  25 minutes for one task. With 28 frontier tasks to go, the run looks stopped.

The three properties to protect here:
  1. Detect a degenerate loop **only before the body starts**, and drop the connection
  2. Never abort a generation that has begun writing the body (do not kill normal long thinking)
  3. Do not regenerate an aborted attempt, and skip the remaining runs too
"""

from __future__ import annotations

import json
from pathlib import Path

import llmbench.clients.openai_compat as oc
from llmbench.clients.openai_compat import (
    _CONTENT_LOOP_MIN_CHARS,
    FINISH_CONTENT_LOOP,
    FINISH_REASONING_BUDGET,
    FINISH_REASONING_LOOP,
    OpenAICompatClient,
    detect_reasoning_loop,
    ThinkSplitter,
    duplicate_line_ratio,
    find_repeating_unit,
)
from llmbench.runner import Attempt, BenchmarkRunner, _aggregate_attempts, _hopeless


# ────────────────── 1. repetition detection itself ──────────────────

def test_find_repeating_unit_catches_token_loop():
    """Measured pattern: "1,1,1,1,..." as one line of 15,434 chars."""
    assert find_repeating_unit("prefix " + "1," * 5000) == "1,"


def test_find_repeating_unit_catches_sentence_loop():
    """Measured pattern: the same sentence on 1,552 lines."""
    line = 'If they test "Hello...World" -> hello-world. Good.\n'
    assert find_repeating_unit("intro\n" + line * 40) == line


def test_find_repeating_unit_ignores_normal_prose():
    """Do not misjudge normal thinking prose as a repetition."""
    prose = (
        "Let me re-read the spec. The nested transaction should keep the outer "
        "rollback marker, so commit() on the inner scope must not clear it. "
        "I will model the stack explicitly and test the three orderings.\n"
    ) * 3
    assert find_repeating_unit(prose[:1500] + " and then something else") is None


def test_duplicate_line_ratio():
    assert duplicate_line_ratio("head\n" + "same line\n" * 30) > 0.9
    uniq = "\n".join(f"step {i}: consider case {i}" for i in range(40))
    assert duplicate_line_ratio(uniq) < 0.1
    # no verdict when there are too few lines
    assert duplicate_line_ratio("a\na\na\n") == 0.0


def test_detect_reasoning_loop_message():
    hit = detect_reasoning_loop("x" * 100 + "ab" * 200)
    assert hit and "repeated" in hit


# ────────────────── 2. aborting during the stream ──────────────────

class _FakeStreamResp:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.reason = "OK"
        self.text = ""
        self.headers = {}
        self.closed = False
        self.consumed = 0

    def iter_lines(self, decode_unicode=False):
        for ln in self._lines:
            self.consumed += 1
            yield ln

    def close(self):
        self.closed = True


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


def _reasoning(text: str) -> str:
    return _sse({"choices": [{"delta": {"reasoning_content": text}}]})


def _content(text: str, finish=None) -> str:
    return _sse({"choices": [{"delta": {"content": text}, "finish_reason": finish}]})


def _install(monkeypatch, resp):
    monkeypatch.setattr(oc.requests, "post", lambda url, **kw: resp)
    return resp


def _client(**overrides):
    cfg = {"base_url": "http://h/v1", "model": "x", "max_tokens": 100000,
           "stream": True}
    cfg.update(overrides)
    return OpenAICompatClient("m", cfg)


def test_loop_guard_aborts_degenerate_thinking(monkeypatch):
    """When repetition starts with no body at all, drop the connection before the SSE ends."""
    # 500 chars per chunk x 60 = 30,000 chars. The guard starts judging at 8,000 chars.
    lines = [_reasoning("1," * 250) for _ in range(60)] + [_content("done", "stop")]
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == FINISH_REASONING_LOOP
    assert r.truncated is True
    assert r.text == "", "the fallback must not pick up degenerate thinking"
    assert resp.closed is True
    assert resp.consumed < len(lines), "read to the end = it did not abort early"


def test_loop_guard_never_aborts_after_content_started(monkeypatch):
    """Once a single body char is out, never abort, no matter how much it repeats.

    Answers to code tasks contain repeated lines (imports, dict literals) as a
    rule, so a false positive here kills normal generation.
    """
    lines = (
        [_content("--- FILE: a.py ---\n")]
        + [_reasoning("1," * 250) for _ in range(60)]
        + [_content("pass", "stop")]
    )
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == "stop"
    assert r.truncated is False
    assert r.text.startswith("--- FILE: a.py ---")
    assert resp.consumed == len(lines)


def test_loop_guard_leaves_healthy_long_thinking_alone(monkeypatch):
    """Let long thinking that does not repeat run to the end."""
    chunks = [
        _reasoning(f"Step {i}: check invariant {i} against the spec paragraph {i}.\n")
        for i in range(600)
    ]
    _install(monkeypatch, _FakeStreamResp(chunks + [_content("ok", "stop")]))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == "stop"
    assert r.truncated is False
    assert r.text == "ok"


def test_loop_guard_can_be_disabled(monkeypatch):
    lines = [_reasoning("1," * 250) for _ in range(60)] + [_content("done", "stop")]
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client(loop_guard=False)._post_once("sys", "user")

    assert r.finish_reason == "stop"
    assert resp.consumed == len(lines)


def test_reasoning_max_tokens_caps_thinking(monkeypatch):
    """Even without repetition, cut thinking that reaches its cap (a time insurance)."""
    lines = [_reasoning(f"unique thought number {i}\n") for i in range(500)]
    lines.append(_content("done", "stop"))
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client(loop_guard=False, reasoning_max_tokens=100)._post_once("sys", "u")

    assert r.finish_reason == FINISH_REASONING_BUDGET
    assert r.truncated is True
    assert r.text == ""
    assert resp.consumed <= 120


def test_guard_is_off_for_non_stream(monkeypatch):
    """A non-stream response cannot be cut midway, so read it fully as before (regression)."""
    class _Json:
        status_code = 200
        reason = "OK"
        text = ""
        headers: dict = {}

        def json(self):
            return {
                "choices": [{"message": {"content": "",
                                         "reasoning_content": "1," * 20000},
                             "finish_reason": "length"}],
                "usage": {"completion_tokens": 100000},
            }

    _install(monkeypatch, _Json())
    r = _client(stream=False)._post_once("sys", "user")
    assert r.truncated is True
    assert r.text.startswith("1,"), "the non-stream reasoning fallback stays as before"


# ────────────────── 3. fail-fast on the runner side ──────────────────

def test_hopeless_only_for_truncated_and_empty():
    assert _hopeless(Attempt(truncated=True, raw_output="")) is True
    assert _hopeless(Attempt(truncated=True, raw_output="   \n")) is True
    # it wrote part of the answer = it may be only a budget shortfall, so run it
    assert _hopeless(Attempt(truncated=True, raw_output="--- FILE")) is False
    # a transport error = it may pass later, so run it
    assert _hopeless(Attempt(truncated=True, errored=True)) is False
    # just a wrong answer = run it, of course
    assert _hopeless(Attempt(truncated=False, raw_output="wrong")) is False


class _HopelessClient:
    """Returns "aborted with no body" every time = what happened on t033."""

    def __init__(self):
        self.calls = 0

    def generate(self, system, user):
        from llmbench.clients.base import GenerationResult
        self.calls += 1
        return GenerationResult(text="", latency_sec=1.0, truncated=True,
                                finish_reason=FINISH_REASONING_LOOP,
                                max_tokens=49152)


class _Ev:
    parse_ok = False
    parse_error = "no file blocks"
    parse_warnings: list = []
    resolved = False
    quality_score = 0.0
    parsed_files: dict = {}
    fail_reason = "empty output"
    detail_output = ""
    components: dict = {}


class _Grader:
    def evaluate(self, task, text, ctx):
        return _Ev()

    def build_prompt(self, task, lang):
        return "sys", "user"


class _Task:
    task_id = "t033"
    difficulty = "frontier"
    title = "nested transaction"
    domain = "code"
    grader = "code"
    perf_timeout = None
    dir = Path(".")


def _run_task(runner, client, runs):
    import llmbench.runner as rn
    orig = rn.get_grader
    rn.get_grader = lambda name: _Grader()
    try:
        return runner._run_task(client, None, None, 1, _Task(), "en", 10, 1, runs)
    finally:
        rn.get_grader = orig


def test_fail_fast_skips_remaining_runs():
    """When the 1st attempt is aborted with no body, do not send the remaining 4."""
    runner = BenchmarkRunner({}, Path("."))
    client = _HopelessClient()
    tr = _run_task(runner, client, runs=5)

    assert client.calls == 1, f"fail-fast did not work ({client.calls} generations)"
    assert tr.n_skipped == 4
    assert tr.resolved is False
    assert tr.success_rate == 0.0
    # keep runs that never executed out of the denominator (do not count what was not measured)
    assert tr.pass_k == 1
    assert tr.latency_sec == 1.0, "averaging the skipped runs in as 0 seconds dilutes the measurement"


def test_fail_fast_can_be_disabled():
    runner = BenchmarkRunner({"run": {"fail_fast": False}}, Path("."))
    client = _HopelessClient()
    tr = _run_task(runner, client, runs=3)

    assert client.calls == 3
    assert tr.n_skipped == 0


def test_skipped_attempts_are_excluded_from_denominator():
    from llmbench.runner import TaskResult
    tr = TaskResult(task_id="t033", difficulty="frontier", runs=5)
    attempts = [
        Attempt(resolved=False, truncated=True, latency_sec=150.0),
        *[Attempt(skipped=True, fail_reason="skipped by fail-fast") for _ in range(4)],
    ]
    _aggregate_attempts(tr, attempts, {}, k=None)

    assert tr.n_skipped == 4
    assert tr.n_errored == 0
    assert tr.success_rate == 0.0
    assert tr.n_pass == 0
    assert tr.latency_sec == 150.0
    assert "skipped by fail-fast" in tr.fail_reason
    assert [a["skipped"] for a in tr.attempts] == [False, True, True, True, True]


# ────────────────── 4. runaway body / <think> mixing into content ──────────────────
#
# Measured t048 (architect): content was non-empty and still reached 49,152
# tokens. Correct answers for the same task on other models are 2,109-2,449
# bytes, and gold is 59 lines in 2 files.
# This is not "a big answer got cut off" - it is the body itself running away.
# There are two causes, and both must stop:
#   (A) thinking was separated, and the answer fell into a repetition after it
#       started writing
#   (B) the server streamed <think> into content, which disabled the
#       thinking-side guard from the very first token

def test_whitespace_only_unit_is_not_a_loop():
    """Do not misjudge normal output that only ends with newlines or indentation.

    This happens routinely at the end of a code block, so missing it kills
    normal generation.
    """
    assert find_repeating_unit("def f():\n    return 1\n\n\n\n\n\n\n\n") is None
    assert find_repeating_unit("x = 1" + " " * 40) is None
    assert find_repeating_unit("code\n" + "\t" * 30) is None
    # a unit that contains whitespace is still detected when it has non-whitespace
    assert find_repeating_unit("head" + "ab \n" * 30) == "ab \n"


# --- ThinkSplitter unit tests ---

def test_think_splitter_separates_inline_think():
    s = ThinkSplitter()
    think, body = s.feed("<think>let me reason</think>--- FILE: a.py ---")
    assert "let me reason" in think
    assert body == "--- FILE: a.py ---"


def test_think_splitter_handles_tag_across_chunks():
    """Do not lose a tag split across a chunk boundary (tokens arrive one by one, so this is required)."""
    s = ThinkSplitter()
    out_t, out_b = [], []
    for ch in ["<th", "ink>", "aaa", "</thi", "nk>", "BODY"]:
        t, b = s.feed(ch)
        out_t.append(t)
        out_b.append(b)
    t, b = s.flush()
    out_t.append(t)
    out_b.append(b)
    assert "aaa" in "".join(out_t)
    assert "".join(out_b) == "BODY"
    assert s.in_think is False


def test_think_splitter_only_holds_back_possible_tag_prefix():
    """Do not carry over a tail that cannot be part of a tag.

    Carrying over unconditionally delays body detection by a few characters on
    servers that use no tags (a short answer is then never recognised as the body).
    """
    assert ThinkSplitter._partial_tag_len("--- FILE: a.py ---") == 0
    assert ThinkSplitter._partial_tag_len("code here <") == 1
    assert ThinkSplitter._partial_tag_len("code here <th") == 3
    assert ThinkSplitter._partial_tag_len("code here </thin") == 6


def test_think_splitter_passthrough_when_no_tags():
    """With a server that uses no tags, everything is the body = the previous behaviour."""
    s = ThinkSplitter()
    t, b = s.feed("--- FILE: a.py ---\ncode")
    assert t == ""
    assert b == "--- FILE: a.py ---\ncode"


# --- (B) servers that mix <think> into content ---

def test_guard_still_works_when_think_leaks_into_content(monkeypatch):
    """The guard works even on servers that do not separate reasoning_content.

    Without this, the first thinking token makes content non-empty and the
    guard is disabled from the start.
    """
    lines = [_content("<think>")] + [_content("1," * 250) for _ in range(60)]
    lines.append(_content("</think>done", "stop"))
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == FINISH_REASONING_LOOP
    assert r.truncated is True
    assert resp.consumed < len(lines)


def test_reasoning_budget_applies_to_inline_think(monkeypatch):
    lines = [_content("<think>")]
    lines += [_content(f"thought {i}\n") for i in range(500)]
    lines.append(_content("</think>done", "stop"))
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client(loop_guard=False, reasoning_max_tokens=100)._post_once("s", "u")

    assert r.finish_reason == FINISH_REASONING_BUDGET
    assert resp.consumed < len(lines)


# --- (A) the body itself degenerates ---

def test_content_loop_guard_aborts_runaway_body(monkeypatch):
    """Stop degeneration that starts after the body started too (the shape of t048)."""
    lines = [_content("--- FILE: router.py ---\n```python\n")]
    # 500 chars per chunk x 60 = 30,000 chars. The body-side check starts at 16,000 chars.
    lines += [_content("x=1;" * 125) for _ in range(60)]
    lines.append(_content("```", "stop"))
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == FINISH_CONTENT_LOOP
    assert r.truncated is True
    # do not throw the body away: the grader must see the good blocks written before the degeneration
    assert r.text.startswith("--- FILE: router.py ---")
    assert resp.consumed < len(lines)


def test_normal_sized_answer_is_never_touched(monkeypatch):
    """Measured correct-answer sizes (2,109-2,449 bytes) are far below the body-guard threshold.

    Not trimming a single character of a normal answer is the precondition of
    this guard.
    """
    answer = (
        "--- FILE: router.py ---\n```python\n"
        + "".join(f"def handler_{i}(request):\n    return {i}\n\n" for i in range(60))
        + "```\n"
    )
    assert len(answer) < _CONTENT_LOOP_MIN_CHARS
    lines = [_content(answer[i:i + 200]) for i in range(0, len(answer), 200)]
    lines.append(_content("", "stop"))
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == "stop"
    assert r.truncated is False
    assert r.text == answer
    assert resp.consumed == len(lines)


def test_long_but_healthy_body_is_not_aborted(monkeypatch):
    """Above the threshold too, let a body that does not repeat be written to the end."""
    import random
    random.seed(1)
    body = "".join(
        f"def handler_{i}(req):\n    return compute_{random.randint(0, 9999)}(req)\n\n"
        for i in range(900)
    )
    assert len(body) > _CONTENT_LOOP_MIN_CHARS * 2
    lines = [_content("--- FILE: big.py ---\n")]
    lines += [_content(body[i:i + 300]) for i in range(0, len(body), 300)]
    lines.append(_content("", "stop"))
    resp = _install(monkeypatch, _FakeStreamResp(lines))
    r = _client()._post_once("sys", "user")

    assert r.finish_reason == "stop"
    assert r.truncated is False
    assert resp.consumed == len(lines)

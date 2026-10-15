"""Regression tests for the patch.py extractor.

The golden inputs are minimal reproductions (tests/fixtures/*.txt) that keep
only the relevant part of a measured run (the artifacts under /tmp/q2). They
pin down that these incidents never happen again:

  * **last-wins overwrite** when several markers name the same path (a 1,396B
    implementation crushed into a 4B `...`)
  * adopting the output-format template (`<entire corrected file content>` /
    `<content>`) as code
  * adopting a huge block that contains prose and does not parse, only because
    it is the longest
  * the opposite regression: breaking a case where the code was rescued
    correctly from a draft written during thinking
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llmbench.patch import (
    ParsedPatch,
    _is_real_code,
    _norm,
    apply_patch,
    parse_llm_output,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def only_code(patch: ParsedPatch, path: str) -> str:
    assert patch.parse_ok, f"parse_ok=False ({patch.error})"
    assert list(patch.files) == [path]
    return patch.files[path]


# ────────────────── _is_real_code unit tests ──────────────────

@pytest.mark.parametrize("code", [
    "",
    "   \n  ",
    "...",
    "\n...\n",
    "pass",
    "<entire corrected file content>",
    "<content>",
    "<relative/path.py>",
    "def f():\n    ...\n",              # a function that is only a template
    "class C:\n    pass\n",             # a class that is only a template
    '"""module docstring only"""\n',    # a docstring only
    "def f(:\n  return 1\n",            # SyntaxError
])
def test_is_real_code_rejects_non_implementations(code):
    assert _is_real_code(code, "x.py") is False


@pytest.mark.parametrize("code", [
    "x = 1\n",
    "import os\n\n\ndef f():\n    return os.sep\n",
    "def f():\n    '''doc'''\n    return 1\n",
    "class C:\n    def m(self):\n        return 2\n",
])
def test_is_real_code_accepts_implementations(code):
    assert _is_real_code(code, "x.py") is True


def test_is_real_code_skips_syntax_check_for_non_python():
    """Applying ast to non-.py files (JS/MD etc.) drops even correct output."""
    js = "function f() { return 1; }\n"
    assert _is_real_code(js, "app.js") is True
    assert _is_real_code(js, "app.py") is False
    assert _is_real_code("<content>", "app.js") is False


# ────────────────── 1. last-wins overwrite (t057) ──────────────────

def test_duplicate_marker_keeps_real_implementation_not_ellipsis():
    """The 1,396B implementation on line 437 is not crushed by the 4B `...` on line 561."""
    text = load("t057_duplicate_marker.txt")
    patch = parse_llm_output(text, ["sniff.py"])
    code = only_code(patch, "sniff.py")
    assert len(code) > 1000, f"picked {len(code)}B, not the implementation"
    assert code.strip() != "..."
    tree = ast.parse(code)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"sniff_delimiter", "parse"} <= names, names


# ────────────────── 2. template echo (t095) ──────────────────

def test_template_echo_is_not_adopted():
    text = load("t095_template_echo.txt")
    patch = parse_llm_output(text, ["labelcheck.py"])
    code = only_code(patch, "labelcheck.py")
    assert "<entire corrected file content>" not in code
    assert _is_real_code(code, "labelcheck.py")


# ────────────────── 3. placeholder (t064) ──────────────────

def test_placeholder_content_is_not_adopted():
    text = load("t064_placeholder.txt")
    patch = parse_llm_output(text, ["allocate.py"])
    code = only_code(patch, "allocate.py")
    assert code.strip() != "<content>"
    assert "def allocate" in code


def test_placeholder_path_gives_actionable_error():
    """`--- FILE: <relative/path.py> ---` is rejected with a message that names the placeholder."""
    text = (
        "--- FILE: <relative/path.py> ---\n"
        "```python\n"
        "x = 1\n"
        "```\n"
    )
    patch = parse_llm_output(text, ["real.py"])
    assert patch.parse_ok is False
    assert "placeholders" in patch.error


# ────────────────── 4. prose contamination (t093) ──────────────────

def test_prose_contaminated_block_is_rejected():
    """A huge block that does not parse is not adopted only because it is the longest."""
    text = load("t093_prose_contaminated.txt")
    patch = parse_llm_output(text, ["canonpath.py"])
    assert patch.parse_ok is False
    assert not patch.files
    assert patch.error


# ────────────────── 5. regression protection ──────────────────

def test_rescue_from_reasoning_still_works():
    """Keep the case that passed 9/9: a draft written during thinking, rescued from a run that hit the limit."""
    text = load("t093_rescue_from_reasoning.txt")
    patch = parse_llm_output(text, ["canonpath.py"])
    code = only_code(patch, "canonpath.py")
    assert "def resolve_subpath" in code
    assert len(code) == 831


def test_normal_output_unchanged():
    """A normal task that did not hit the limit extracts the same code as before."""
    text = load("t060_normal.txt")
    patch = parse_llm_output(text, ["ueq.py"])
    code = only_code(patch, "ueq.py")
    assert code.startswith("import unicodedata")
    assert "def equal(a, b, fold=False):" in code
    assert code.rstrip().endswith("return a == b")


# ────────────────── stages 2/3 and path normalization ──────────────────

def test_inline_path_form_requires_real_code():
    text = (
        "First the template:\n```python:a.py\n...\n```\n"
        "The real one:\n```python:a.py\ndef f():\n    return 1\n```\n"
    )
    patch = parse_llm_output(text, ["a.py"])
    assert only_code(patch, "a.py").strip() == "def f():\n    return 1"


def test_single_file_fallback_prefers_whole_module_over_fragment():
    """With no marker, pick the complete module, not the fragment at the end."""
    text = (
        "Let me think.\n```\nimport os\n\n\ndef f():\n    return os.sep\n\n\n"
        "def g():\n    return 2\n```\n"
        "I only need to fix this part:\n```\nreturn os.sep\n```\n"
    )
    patch = parse_llm_output(text, ["m.py"])
    code = only_code(patch, "m.py")
    assert "def f()" in code and "def g()" in code


def test_non_python_fence_tags_do_not_shift_block_pairing():
    """Consume a ```json block as one block, so fence pairing does not shift."""
    text = (
        "--- FILE: a.py ---\n"
        "```json\n"
        '{"note": "this is not code"}\n'
        "```\n"
        "```python\n"
        "def f():\n    return 1\n"
        "```\n"
    )
    patch = parse_llm_output(text, ["a.py"])
    assert only_code(patch, "a.py").strip() == "def f():\n    return 1"


def test_norm_does_not_turn_parent_traversal_into_known_file():
    """lstrip('./') removes a character set, so ../foo.py used to become foo.py."""
    assert _norm("../foo.py") == "../foo.py"
    assert _norm("./foo.py") == "foo.py"
    assert _norm("././foo.py") == "foo.py"
    assert _norm("/foo.py") == "foo.py"
    assert _norm("`foo.py`") == "foo.py"
    text = "--- FILE: ../foo.py ---\n```python\nx = 1\n```\n"
    patch = parse_llm_output(text, ["foo.py"])
    assert patch.parse_ok is False
    assert "unknown/unsafe" in patch.error


# ────────────────── consistency between error and parse_ok ──────────────────

def test_partial_rejection_keeps_parse_ok_without_error():
    """When some paths are rejected, parse_ok=True and a non-empty error never coexist."""
    text = (
        "--- FILE: good.py ---\n```python\nx = 1\n```\n"
        "--- FILE: /etc/passwd ---\n```python\ny = 2\n```\n"
    )
    patch = parse_llm_output(text, ["good.py"])
    assert patch.parse_ok is True
    assert patch.error == ""
    assert patch.warnings and "etc/passwd" in patch.warnings[0]
    assert patch.suspect is True
    assert list(patch.files) == ["good.py"]


def test_all_rejected_sets_error_and_parse_ng():
    text = "--- FILE: /etc/passwd ---\n```python\ny = 2\n```\n"
    patch = parse_llm_output(text, ["good.py"])
    assert patch.parse_ok is False
    assert "unknown/unsafe" in patch.error
    assert patch.suspect is True


def test_empty_output():
    patch = parse_llm_output("   ", ["a.py"])
    assert patch.parse_ok is False and patch.error == "empty output"
    assert not patch.warnings


def test_apply_patch_roundtrip(tmp_path):
    patch = parse_llm_output(load("t060_normal.txt"), ["ueq.py"])
    written = apply_patch(patch, tmp_path)
    assert written == ["ueq.py"]
    text = (tmp_path / "ueq.py").read_text(encoding="utf-8")
    assert text.endswith("\n")
    ast.parse(text)

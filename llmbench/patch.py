"""Parsing LLM output and applying files.

Output format (canonical format specified in prompt):

    --- FILE: relative/path.py ---
    ```python
    <modified code for the entire file>
    ```

Local LLMs often cannot emit precise unified-diff line numbers, so v1 uses
the "whole file replacement" approach.
As fallbacks it also parses ```python:path``` blocks and a single code block.

Extraction uses "first wins + real-code check":

  Outputs that contain thinking may show several code blocks for the same
  path. The old implementation overwrote unconditionally (**last wins**), so a
  complete implementation written during thinking (1,396B) was destroyed by the
  ``...`` (4B) in the "output format recap" that followed it.
  So only blocks passing ``_is_real_code()`` are candidates, and the first real
  code found wins (``setdefault`` = first wins).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

FILE_MARKER_RE = re.compile(
    r"^[-*\s]*FILE:\s*(?P<path>\S+?)\s*[-*\s]*$", re.MULTILINE
)
# Fences are anchored at line start, and the language tag is generalized.
#   The old implementation accepted only ```(?:python|py)?, so it **could not match
#   the opening fences** of ```json / ```text / ```diff, mistook those blocks'
#   closing fences for openings, and fence pairing shifted by one (cause of picking up prose as code).
CODE_BLOCK_RE = re.compile(
    r"^[ \t]*```[ \t]*(?P<lang>[A-Za-z0-9_+.-]*)(?::(?P<inline_path>\S+))?[ \t]*\n"
    r"(?P<code>.*?)"
    r"^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# Control tokens for harmony/channel-style models (e.g., <|channel>, <channel|>, <|message|>).
# Removes only tags containing a pipe '|' between '<' and '>'. Regular HTML/XML tags
# (<li>, <ul>, <script>, etc. output in code by frontend-focused models) lack pipes
# and are preserved.
CONTROL_TOKEN_RE = re.compile(r"<\|[^<>]*>|<[^<>]*\|>")

# Placeholders that only repeat the prompt template verbatim, such as
# `<entire corrected file content>` / `<content>` / `<relative/path.py>`.
PLACEHOLDER_RE = re.compile(r"^<[^<>\n]{0,200}>$")

# Extensions where syntax is valid under ast (no syntax check for the others)
_PY_SUFFIXES = (".py", ".pyi")

# Fence language tags accepted as Python code.
# Blocks **explicitly declared as another language**, such as ```json / ```text /
# ```diff, are not accepted (a JSON object passes ast.parse as a dict literal).
_PY_LANG_TAGS = frozenset({"", "python", "py", "python3", "py3", "python2"})


def _strip_control_tokens(text: str) -> str:
    return CONTROL_TOKEN_RE.sub("", text)


@dataclass
class ParsedPatch:
    """Parsed patch: relative path -> modified file contents."""

    files: dict[str, str] = field(default_factory=dict)
    parse_ok: bool = False
    error: str = ""
    # Extraction worked but is suspect (some paths rejected etc.). Can be set even with parse_ok=True.
    # error is used only for "why extraction failed", so the two never coexist.
    warnings: list[str] = field(default_factory=list)

    @property
    def suspect(self) -> bool:
        """The extraction result cannot be taken at face value (regeneration is worth considering)."""
        return bool(self.error or self.warnings)


def _is_placeholder(s: str) -> bool:
    """Is it a template placeholder such as `<entire corrected file content>`."""
    return bool(PLACEHOLDER_RE.match(s.strip()))


def _is_stub_node(node: ast.AST) -> bool:
    """Is it an "empty" node made only of `pass` / `...` / docstring / bare literal.

    Bare literals are included because ``{"a": 1}`` from a ```json block passes
    ast.parse as a Python dict literal (clearly wrong as file replacement content).
    """
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr):
        try:
            ast.literal_eval(node.value)     # constant / dict/list/tuple literal
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return False
        return True
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return all(_is_stub_node(child) for child in node.body)
    return False


def _is_real_code(code: str, path: str | None = None) -> bool:
    """Decides whether this is "code actually worth applying".

    Conditions to return True:
      * not empty
      * not a lone `<...>` placeholder
      * ``ast.parse`` succeeds (only for Python files)
      * body is not empty
      * not a template of only ``...`` (Ellipsis) / ``pass`` / docstring

    When path points at a non-Python file (.js/.md etc.) no syntax check is
    imposed; only "not empty & not a placeholder" is checked (so false positives do not drop it).
    """
    if not code or not code.strip():
        return False
    stripped = code.strip()
    if _is_placeholder(stripped):
        return False
    if path is not None and not str(path).lower().endswith(_PY_SUFFIXES):
        return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    if not tree.body:
        return False
    return not all(_is_stub_node(node) for node in tree.body)


def _accept_block(block: re.Match[str], path: str | None) -> str | None:
    """Returns the contents of a code block when it may be accepted, else None."""
    lang = (block.group("lang") or "").lower()
    is_py = path is None or str(path).lower().endswith(_PY_SUFFIXES)
    if is_py and lang not in _PY_LANG_TAGS:
        return None                       # ```json / ```text etc. are not accepted
    code = block.group("code")
    return code if _is_real_code(code, path) else None


def _looks_like_whole_module(code: str, path: str | None = None) -> bool:
    """Does this block look like a "whole file" (not a fragment).

    In the stage-3 fallback we prefer "a block that stands on its own as one
    file" over a few lines pasted during thinking. A block whose first line is
    not indented and that has import / def / class at top level counts as a whole module.
    """
    if not _is_real_code(code, path):
        return False
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if first[:1] in (" ", "\t"):
        return False
    if path is not None and not str(path).lower().endswith(_PY_SUFFIXES):
        return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                       ast.Import, ast.ImportFrom))
        for n in tree.body
    )


def parse_llm_output(text: str, known_files: list[str]) -> ParsedPatch:
    """Extracts file replacement patches from LLM output.

    known_files: List of relative paths under task's buggy_code directory
    (used for path validation and single-file task fallback).
    """
    patch = ParsedPatch()
    if not text or not text.strip():
        patch.error = "empty output"
        return patch

    # Strips control tokens for harmony/channel-style models (e.g., <|channel>thought<channel|>)
    # before extraction. If prepended to a FILE marker line, they break extraction.
    text = _strip_control_tokens(text)

    # 1. Canonical form: the code block right after --- FILE: path ---
    #    If the same path appears several times, keep the first real code (setdefault).
    markers = list(FILE_MARKER_RE.finditer(text))
    for i, m in enumerate(markers):
        seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        segment = text[m.end():seg_end]
        path = _norm(m.group("path"))
        if path in patch.files:          # first wins: real code already captured
            continue
        for block in CODE_BLOCK_RE.finditer(segment):
            code = _accept_block(block, path)
            if code is not None:
                patch.files.setdefault(path, code)
                break

    # 2. ```python:path``` form (also first-wins + real code only)
    if not patch.files:
        for block in CODE_BLOCK_RE.finditer(text):
            p = block.group("inline_path")
            if not p:
                continue
            path = _norm(p)
            if path in patch.files:
                continue
            code = _accept_block(block, path)
            if code is not None:
                patch.files.setdefault(path, code)

    # 3. Fallback: for a single-file task, choose among blocks that pass _is_real_code.
    #    The old implementation picked the longest block with the too-loose
    #    "contains `=` means Python" check (_looks_like_python), so it happily
    #    adopted a huge block polluted with prose (24,048B measured) that fails
    #    to parse. The check is replaced by _is_real_code, and blocks that look
    #    like a whole module are preferred.
    if not patch.files and len(known_files) == 1:
        target = known_files[0]
        candidates = [
            code for code in (
                _accept_block(b, target) for b in CODE_BLOCK_RE.finditer(text)
            ) if code is not None
        ]
        pool = [c for c in candidates if _looks_like_whole_module(c, target)]
        pool = pool or candidates
        best = ""
        for code in pool:                       # ties go to the later one (later version is newer)
            if len(code) >= len(best):
                best = code
        if best:
            patch.files[target] = best

    if not patch.files:
        patch.error = "no file blocks found in output"
        return patch

    # Path validation: Reject writes to unknown paths (includes path traversal protection)
    known = set(known_files)
    bad = [p for p in patch.files if p not in known or ".." in Path(p).parts]
    if bad:
        msg = f"unknown/unsafe paths: {bad}"
        placeholders = [p for p in bad if _is_placeholder(p)]
        if placeholders:
            msg += (
                f" — paths are still placeholders {placeholders}. "
                "The model most likely only repeated the output-format template "
                "and did not emit the real modified file"
            )
        patch.files = {p: c for p, c in patch.files.items() if p not in bad}
        if not patch.files:
            # Nothing left = extraction failed. Recorded in error only.
            patch.error = msg
            return patch
        # Part is valid. To avoid the contradiction of parse_ok=True with a non-empty
        # error, the remaining part is accepted and the reason goes into warnings.
        patch.warnings.append(msg)

    patch.parse_ok = True
    patch.error = ""
    return patch


def apply_patch(patch: ParsedPatch, target_dir: Path) -> list[str]:
    """Writes patch contents into target_dir and returns relative paths of modified files."""
    written = []
    for rel, content in patch.files.items():
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not content.endswith("\n"):
            content += "\n"
        dest.write_text(content, encoding="utf-8")
        written.append(rel)
    return written


def _norm(p: str) -> str:
    """Normalizes path notation.

    Note: the old implementation used ``lstrip("./")``, which removes a **character
    set**, so ``../foo.py`` became ``foo.py`` and a traversal path silently turned
    into a "known file". Only a leading ``./`` is removed.
    """
    p = p.strip().strip("`'\"").replace("\\", "/")
    p = re.sub(r"^(?:\./)+", "", p)
    return p.lstrip("/")

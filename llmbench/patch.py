"""Parsing LLM output and applying files.

Output format (canonical format specified in prompt):

    --- FILE: relative/path.py ---
    ```python
    <modified code for the entire file>
    ```

Because local LLMs often struggle to output precise unified diff line numbers,
v1 adopts the "full file replacement" approach.
As fallbacks, it also parses the ```python:path``` format and single code blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FILE_MARKER_RE = re.compile(
    r"^[-*\s]*FILE:\s*(?P<path>\S+?)\s*[-*\s]*$", re.MULTILINE
)
CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?(?::(?P<inline_path>\S+))?\s*\n(?P<code>.*?)```",
    re.DOTALL,
)

# Control tokens for harmony/channel-style models (e.g., <|channel>, <channel|>, <|message|>).
# Removes only tags containing a pipe '|' between '<' and '>'. Regular HTML/XML tags
# (<li>, <ul>, <script>, etc. output in code by frontend-focused models) lack pipes
# and are preserved.
CONTROL_TOKEN_RE = re.compile(r"<\|[^<>]*>|<[^<>]*\|>")


def _strip_control_tokens(text: str) -> str:
    return CONTROL_TOKEN_RE.sub("", text)


@dataclass
class ParsedPatch:
    """Parsed patch: relative path -> modified file contents."""

    files: dict[str, str] = field(default_factory=dict)
    parse_ok: bool = False
    error: str = ""


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

    # 1. Canonical format: code block immediately following --- FILE: path ---
    markers = list(FILE_MARKER_RE.finditer(text))
    for i, m in enumerate(markers):
        seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        segment = text[m.end():seg_end]
        block = CODE_BLOCK_RE.search(segment)
        if block:
            patch.files[_norm(m.group("path"))] = block.group("code")

    # 2. ```python:path``` format
    if not patch.files:
        for block in CODE_BLOCK_RE.finditer(text):
            p = block.group("inline_path")
            if p:
                patch.files[_norm(p)] = block.group("code")

    # 3. Fallback: For single-file tasks, assign the largest code block
    if not patch.files and len(known_files) == 1:
        blocks = [b.group("code") for b in CODE_BLOCK_RE.finditer(text)]
        candidates = [b for b in blocks if _looks_like_python(b)]
        if candidates:
            patch.files[known_files[0]] = max(candidates, key=len)

    if not patch.files:
        patch.error = "no file blocks found in output"
        return patch

    # Path validation: Reject writes to unknown paths (includes path traversal protection)
    known = set(known_files)
    bad = [p for p in patch.files if p not in known or ".." in Path(p).parts]
    if bad:
        patch.error = f"unknown/unsafe paths: {bad}"
        patch.files = {p: c for p, c in patch.files.items() if p not in bad}
        if not patch.files:
            return patch

    patch.parse_ok = True
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
    return p.strip().strip("`'\"").replace("\\", "/").lstrip("./")


def _looks_like_python(code: str) -> bool:
    hints = ("def ", "class ", "import ", "return", "=")
    return any(h in code for h in hints)

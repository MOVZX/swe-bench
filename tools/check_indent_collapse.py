#!/usr/bin/env python3
"""Detect "indent collapse" in llmbench artifacts.

Local LLM output sometimes collapses leading whitespace to **one space
regardless of depth** (measured on Qwopus3.8-27B-Flash Q4_K_M; the cause is
under investigation, speculative decoding was ruled out by condition A).
llmbench/patch.py's _is_real_code() passes it through ast.parse(), so collapsed
output is rejected with IndentationError, generated/ stays empty = an
unconditional fail before grading. Looking at the score alone you cannot tell it
apart from "the model could not solve it", so run this after each run and check
that COLLAPSED is 0.

    python3 tools/check_indent_collapse.py <artifacts_dir> [...]

It reads only llm_output.txt (raw LLM output), so results.json is not needed.
The verdict uses the same "first block wins + is-it-real-code" rule as
patch.py, so a fragment block inside the thinking part does not cause a false
positive.

Verdicts:
  ok                   … a usable code block exists and it is 4-space based
  COLLAPSED            … leading whitespace is 1 space only + syntax error (this is the accident)
  COLLAPSED-but-parses … 1 space only, but flat, so the syntax passed
                         (collapse did happen. With nesting it would have failed)
  NO-PARSABLE-BLOCK    … no block parses as python (a cause other than collapse)
  NO-CODE-BLOCK        … no code fence / fence not closed
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Same as CODE_BLOCK_RE in llmbench/patch.py (line-anchored, generalized language tag)
CODE_BLOCK_RE = re.compile(
    r"^[ \t]*```[ \t]*(?P<lang>[A-Za-z0-9_+.-]*)(?::(?P<inline_path>\S+))?[ \t]*\n"
    r"(?P<code>.*?)"
    r"^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
# Same as _PY_LANG_TAGS in patch.py
PY_LANG_TAGS = frozenset({"", "python", "py", "python3", "py3", "python2"})

INDENT_1SP = re.compile(r"^ [^ \t]")
INDENT_4SP = re.compile(r"^    [^ \t]")


def indent_stats(code: str) -> tuple[int, int]:
    lines = code.splitlines()
    return (
        sum(1 for ln in lines if INDENT_1SP.match(ln)),
        sum(1 for ln in lines if INDENT_4SP.match(ln)),
    )


def classify(text: str) -> tuple[str, str]:
    """Return (verdict, detail). Candidate blocks are chosen with the same first-wins rule as patch.py."""
    blocks = [
        m.group("code")
        for m in CODE_BLOCK_RE.finditer(text)
        if (m.group("lang") or "").lower() in PY_LANG_TAGS and m.group("code").strip()
    ]
    if not blocks:
        return "NO-CODE-BLOCK", "no code fence that reads as python"

    first_err = ""
    for code in blocks:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, MemoryError, RecursionError) as e:
            if not first_err:
                lineno = getattr(e, "lineno", None)
                msg = getattr(e, "msg", str(e))
                first_err = f"{type(e).__name__}: {msg}" + (
                    f" (line {lineno})" if lineno else ""
                )
            continue
        if not tree.body:
            continue
        n1, n4 = indent_stats(code)
        if n1 and not n4:
            return "COLLAPSED-but-parses", f"ok [1sp={n1} 4sp={n4}]"
        if n1 and n4:
            return "MIXED", f"ok [1sp={n1} 4sp={n4}]"
        return "ok", f"ok [1sp={n1} 4sp={n4}]"

    # Nothing usable: separate the causes using the state of the first block
    n1, n4 = indent_stats(blocks[0])
    verdict = "COLLAPSED" if (n1 and not n4) else "NO-PARSABLE-BLOCK"
    return verdict, f"{first_err} [1sp={n1} 4sp={n4}]"


def scan(root: Path) -> dict[str, int]:
    tally: dict[str, int] = {}
    rows: list[tuple[str, str, str, str]] = []
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        raw = tdir / "llm_output.txt"
        if not raw.is_file():
            continue
        verdict, detail = classify(raw.read_text(encoding="utf-8", errors="replace"))
        has_gen = (tdir / "generated").is_dir()
        tally[verdict] = tally.get(verdict, 0) + 1
        if verdict != "ok" or not has_gen:
            rows.append((tdir.name, "gen" if has_gen else "NOGEN", verdict, detail))

    print(f"\n=== {root.name} ===")
    for name, gen, verdict, detail in rows:
        print(f"  {name}  {gen:<5} {verdict:<20} {detail}")
    if not rows:
        print("  nothing abnormal")
    total = sum(tally.values())
    print(
        f"  ---- {total} items: "
        + ", ".join(f"{k}={v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
    )
    collapsed = tally.get("COLLAPSED", 0) + tally.get("COLLAPSED-but-parses", 0)
    if collapsed:
        print(
            f"  ⚠ indent collapse {collapsed}/{total} ({collapsed / total:.1%})."
            " The scores of this run do not show the model's ability."
            " See docs/INDENT_COLLAPSE.md for the isolation procedure"
        )
    return tally


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    worst = 0
    for arg in argv[1:]:
        p = Path(arg)
        if not p.is_dir():
            print(f"ERROR: not a directory: {p}", file=sys.stderr)
            return 2
        t = scan(p)
        if t.get("COLLAPSED") or t.get("COLLAPSED-but-parses"):
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

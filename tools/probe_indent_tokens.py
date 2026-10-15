#!/usr/bin/env python3
"""Separate indent collapse caused by "the model's output" from "detokenization".

Send the same prompt to llama-server N times and check
  (1) the rate of collapse (leading whitespace is 1 space only)
  (2) whether the concatenation of the generated token-ID string, run through
      /detokenize one token at a time, equals the content the server returned

Verdict:
  * concatenation == content and the leading token is a single space
      → **the model/quantization chose that token**. It is a weights problem.
        Compare with another quantization (Q5_K_M etc.) or another build.
  * concatenation != content (the concatenation side shows 4 spaces etc.)
      → **a detokenization/aggregation problem**. Suspect a llama.cpp bug.
        Check the gguf vocab / pre-tokenizer.

Usage:
    # llama.cpp (default): looks down to the token-ID layer
    python3 tools/probe_indent_tokens.py --url http://localhost:8085 -n 10

    # vLLM / OpenAI compatible: indentation verdict only (no token-ID layer)
    python3 tools/probe_indent_tokens.py --api openai \
        --url http://localhost:8000/v1 --model Jackrong/Qwopus3.8-27B-Flash -n 3

    python3 tools/probe_indent_tokens.py --self-test     # internal test, no server needed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Write exactly one Python function for the following specification. No explanation,"
    " return only a ```python code block.\n\n"
    "Specification: write `squeeze(items)`, which takes a list of integers and returns"
    " the list with consecutive duplicates removed. An empty list returns an empty list."
    " Use a nested implementation with an if inside the loop.\n"
)

INDENT_1SP = re.compile(r"^ [^ \t]", re.MULTILINE)
INDENT_4SP = re.compile(r"^    [^ \t]", re.MULTILINE)


def indent_stats(text: str) -> tuple[int, int]:
    return len(INDENT_1SP.findall(text)), len(INDENT_4SP.findall(text))


def verdict_for(text: str) -> str:
    n1, n4 = indent_stats(text)
    if n1 and not n4:
        return "COLLAPSED"
    if n1 and n4:
        return "MIXED"
    return "ok"


def post(url: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def render_prompt(url: str, user_text: str, timeout: float) -> str:
    """Use the chat template through /apply-template when available. Pass through when not."""
    try:
        r = post(url, "/apply-template",
                 {"messages": [{"role": "user", "content": user_text}]}, timeout)
        return r.get("prompt", user_text)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        print("  (note) /apply-template is unavailable, sending the bare prompt")
        return user_text


def piece_of(url: str, tok: int, cache: dict[int, str], timeout: float) -> str:
    if tok not in cache:
        r = post(url, "/detokenize", {"tokens": [tok]}, timeout)
        cache[tok] = r.get("content", "")
    return cache[tok]


def run_probe_openai(args: argparse.Namespace) -> int:
    """OpenAI compatible (vLLM etc.) version. No /detokenize, so only the indentation verdict."""
    if not args.model:
        print("ERROR: --api openai needs --model (e.g. --model Jackrong/...)",
              file=sys.stderr)
        return 2
    tally: dict[str, int] = {}
    for i in range(1, args.n + 1):
        r = post(args.url, "/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.n_predict,
            "temperature": args.temp,
        }, args.timeout)
        content = r["choices"][0]["message"].get("content") or ""
        v = verdict_for(content)
        tally[v] = tally.get(v, 0) + 1
        n1, n4 = indent_stats(content)
        print(f"  #{i:<3} {v:<10} [1sp={n1} 4sp={n4}]")
        if v != "ok" and args.show:
            print("    ---- content ----")
            for ln in content.splitlines()[: args.show]:
                print(f"    |{ln}")

    total = sum(tally.values())
    print("\n  ---- " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    collapsed = tally.get("COLLAPSED", 0) + tally.get("MIXED", 0)
    print(f"  collapse rate: {collapsed}/{total}")
    if collapsed:
        print("  → It collapses even on the path without GGUF. **Model behaviour** (H3a)")
    else:
        print("  → No collapse on this path. **llama.cpp GGUF conversion bug** (H3b)")
    return 1 if collapsed else 0


def run_probe(args: argparse.Namespace) -> int:
    prompt = render_prompt(args.url, args.prompt, args.timeout)
    cache: dict[int, str] = {}
    tally: dict[str, int] = {}
    ws_tally: dict[tuple[int, str], int] = {}
    indent_mismatches = 0
    other_mismatches = 0

    for i in range(1, args.n + 1):
        r = post(args.url, "/completion", {
            "prompt": prompt,
            "n_predict": args.n_predict,
            "temperature": args.temp,
            "return_tokens": True,
            "cache_prompt": False,
        }, args.timeout)
        content = r.get("content", "")
        tokens = r.get("tokens") or []
        v = verdict_for(content)
        tally[v] = tally.get(v, 0) + 1

        note = ""
        if tokens:
            pieces = [piece_of(args.url, t, cache, args.timeout) for t in tokens]
            joined = "".join(pieces)
            jn1, jn4 = indent_stats(joined)
            cn1, cn4 = indent_stats(content)
            # breakdown of whitespace-only tokens (from indentation)
            for t, pc in zip(tokens, pieces):
                if pc and not pc.strip():
                    ws_tally[(t, pc)] = ws_tally.get((t, pc), 0) + 1
            if (jn1, jn4) != (cn1, cn4):
                # the indentation statistics themselves differ = whitespace changed during stringification
                indent_mismatches += 1
                note = (f"  ⚠ indent stats mismatch  joined[1sp={jn1} 4sp={jn4}]"
                        f" content[1sp={cn1} 4sp={cn4}]")
            elif joined != content:
                # a difference other than whitespace (special token rendering etc.). Show the position only
                k = next(i for i, (a, b) in enumerate(zip(joined, content)) if a != b) \
                    if any(a != b for a, b in zip(joined, content)) \
                    else min(len(joined), len(content))
                other_mismatches += 1
                note = (f"  (joined≠content but the indent stats match. first diff at {k}:"
                        f" joined={joined[k:k+20]!r} content={content[k:k+20]!r})")
        else:
            note = "  (no tokens returned: build without return_tokens support)"

        n1, n4 = indent_stats(content)
        print(f"  #{i:<3} {v:<10} [1sp={n1} 4sp={n4}] tokens={len(tokens)}{note}")
        if v != "ok" and args.show:
            print("    ---- content ----")
            for ln in content.splitlines()[: args.show]:
                print(f"    |{ln}")

    total = sum(tally.values())
    print("\n  ---- " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    collapsed = tally.get("COLLAPSED", 0) + tally.get("MIXED", 0)
    print(f"  collapse rate: {collapsed}/{total}")
    if ws_tally:
        print("  whitespace-only token breakdown (id, string, occurrences):")
        for (t, pc), n in sorted(ws_tally.items(), key=lambda kv: -kv[1]):
            print(f"    {t:>7}  {pc!r:<12} x{n}")
    if indent_mismatches:
        print(f"  ⚠ indent stats mismatch {indent_mismatches}/{total} times"
              " → suspect the detokenization/aggregation side (H2)")
    elif collapsed:
        print("  → already a single space at the token-ID stage. Not the"
              " detokenization side: the model chose that token. If it is"
              " quantization- and prompt-independent, it is learned model"
              " behaviour (see docs/INDENT_COLLAPSE.md)")
    if other_mismatches:
        print(f"  (reference) non-whitespace differences in {other_mismatches}/{total} runs."
              " Ignore them if they are rendering differences of special tokens (e.g. <|im_start|>)")
    return 1 if collapsed else 0


def self_test() -> int:
    cases = [
        ("def f():\n    if x:\n        return 1\n", "ok"),
        ("def f():\n if x:\n return 1\n", "COLLAPSED"),
        ("def f():\n    if x:\n return 1\n", "MIXED"),
        ("x = 1\n", "ok"),
    ]
    bad = 0
    for text, want in cases:
        got = verdict_for(text)
        ok = got == want
        bad += not ok
        print(f"  {'✅' if ok else '❌'} want={want:<10} got={got}")
    print("  self-test:", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--api", choices=("llamacpp", "openai"), default="llamacpp",
                    help="openai: call the OpenAI compatible /chat/completions of vLLM etc.")
    ap.add_argument("--model", default="", help="model name for --api openai")
    ap.add_argument("-n", type=int, default=10, help="number of trials (default 10)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="show the first N lines of content from collapsed runs")
    ap.add_argument("--self-test", action="store_true", help="internal test, no server needed")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    try:
        return run_probe_openai(args) if args.api == "openai" else run_probe(args)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"ERROR: cannot reach the server ({args.url}): {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate perturbed variants of the security tasks (SecLLMHolmes style augmentation).

To expose memorization and pattern matching, build variants of existing tasks
with transformations that do not change their meaning. Solving the base task but
failing the variant is evidence that the model reacted to surface cues
(function names, variable names, IPs and the like).

Two kinds of variant are generated:

  a (lexical perturbation) … for code: rename def names, constant names and
                  suggestive argument names to neutral names in one pass. For
                  logs: replace IPs/user names/timestamps/paths with other
                  values.
                  The same replacement is applied to the gold any_of /
                  location_any_of / allow_extra, so grading stays equivalent.
  b (decoy insertion)      … for code: append helpers that "look dangerous but
                  are safe". For logs: insert harmless but eye-catching lines.
                  The gold is unchanged. Flagging one of these is a pure false
                  positive.

Usage:
    python3 tools/make_variants.py            # generate (also updates tasks_sec_aug.jsonl)
    python3 tools/make_variants.py --check    # consistency check only, no generation
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
BASE_LEDGER = TASKS / "tasks_sec.jsonl"
AUG_LEDGER = TASKS / "tasks_sec_aug.jsonl"

# ── pool of neutral names used by the lexical perturbation ──────────────────
FUNC_POOL = [
    "fetch_record", "handle_entry", "process_item", "build_payload",
    "resolve_ref", "apply_settings", "collect_stats", "prepare_batch",
    "emit_result", "sync_state", "format_output", "dispatch_task",
    "materialize", "reconcile", "finalize_unit",
]
CONST_POOL = [
    "ROOT_DIR", "DATA_STORE", "PRIMARY_SET", "SERVICE_ROOT", "POLICY_SET",
    "KEY_MATERIAL", "LIMIT_BYTES", "SETTINGS_PATH", "BUCKET_NAME",
]
# suggestive argument names → neutral names
# Do not make the replacement names short. The gold any_of matches substrings,
# so a short word such as "ref" also matches "reference"/"referrer" (the same
# trap as "rce" in s05).
ARG_POOL = ["record_ref", "field_value", "entry_ref", "payload_data",
            "item_ref", "unit_ref", "slot_ref", "chunk_ref"]
SUGGESTIVE = re.compile(
    r"user|raw|input|unsafe|path|_name$|^name$|cmd|command|query|target|"
    r"text|url|cookie|password|token|blob|supplied", re.I
)

# ── decoys (dangerous looking but safe) ────────────────────────────────
CODE_DECOY = '''

def _cache_key(payload: bytes) -> str:
    import hashlib

    # Used for the cache key. Not used for authentication or integrity checks.
    return hashlib.sha256(payload).hexdigest()[:16]


def _parse_limit(raw_limit: str) -> int:
    import ast

    # Only Python literals are accepted. Reject anything that is not an integer.
    parsed = ast.literal_eval(raw_limit)
    if not isinstance(parsed, int):
        raise ValueError("limit must be an integer")
    return parsed


def _align_columns(rows):
    import subprocess

    proc = subprocess.run(
        ["/usr/bin/column", "-t"],
        input="\\n".join(rows), capture_output=True, text=True, check=True,
    )
    return proc.stdout
'''

CODE_DECOY_EN = CODE_DECOY.replace(
    "# Used for the cache key. Not used for authentication or integrity checks.",
    "# Cache key only; never used for authentication or integrity.",
).replace(
    "# Only Python literals are accepted. Reject anything that is not an integer.",
    "# Only Python literals are accepted; non-integers are rejected.",
)

AUTHLOG_DECOY = [
    "Jul 18 03:09:41 web01 sshd[20388]: Accepted publickey for backup from 10.0.5.4 port 49220 ssh2: RSA SHA256:9f2c",
    "Jul 18 03:10:01 web01 CRON[20402]: pam_unix(cron:session): session opened for user root by (uid=0)",
    "Jul 18 03:10:01 web01 CRON[20402]: pam_unix(cron:session): session closed for user root",
    "Jul 18 03:13:12 web01 sudo:   deploy : TTY=pts/0 ; PWD=/srv/app ; USER=root ; COMMAND=/usr/bin/systemctl restart app",
]
ACCESSLOG_DECOY = [
    '10.0.2.15 - - [14/Aug/2026:02:15:00 +0000] "GET /healthz HTTP/1.1" 200 2 "-" "kube-probe/1.29"',
    '66.249.66.1 - - [14/Aug/2026:02:15:03 +0000] "GET /sitemap.xml HTTP/1.1" 200 4821 "-" "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"',
    '203.0.113.10 - - [14/Aug/2026:02:15:11 +0000] "GET /static/app.9f2c.css HTTP/1.1" 304 0 "https://shop.example.com/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15"',
    '203.0.113.10 - - [14/Aug/2026:02:15:12 +0000] "GET /static/legacy-print.css HTTP/1.1" 404 153 "https://shop.example.com/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15"',
]

# ── re-anonymization map for the log tasks (replace longest first) ──────────
LOG_RENAMES = {
    "s03": {"203.0.113.66": "198.51.100.212", "Jul 18 03:1": "Sep 02 21:4",
            "web01": "app07", "deploy": "svcops", "oracle": "mysql",
            "postgres": "redis", "admin": "operator"},
    "s11": {"198.51.100.23": "192.0.2.55", "/api/items": "/api/products",
            "14/Aug/2026:02:1": "03/Sep/2026:19:4", "203.0.113.10": "198.51.100.7"},
    "s12": {"203.0.113.10": "192.0.2.31", "198.51.100.77": "10.0.9.14",
            "15/Aug/2026:09:0": "27/Sep/2026:14:3", "66.249.66.1": "66.249.79.8"},
    "s17": {"203.0.113.44": "192.0.2.140", "198.51.100.9": "203.0.113.201",
            "203.0.113.71": "198.51.100.66", "16/Aug/2026:11:0": "01/Oct/2026:08:2",
            "docs.example.com": "handbook.example.net"},
}
LOG_DECOY = {"s03": AUTHLOG_DECOY, "s11": ACCESSLOG_DECOY,
             "s12": ACCESSLOG_DECOY, "s17": ACCESSLOG_DECOY}

FENCE = re.compile(r"```(python|text)\n(.*?)```", re.S)


def _blocks(md: str):
    return list(FENCE.finditer(md))


def _code_of(md: str) -> tuple[str, str]:
    m = _blocks(md)
    if not m:
        raise ValueError("code fence not found")
    return m[0].group(1), m[0].group(2)


def _sub_block(md: str, new_body: str) -> str:
    m = _blocks(md)[0]
    return md[:m.start()] + f"```{m.group(1)}\n{new_body}```" + md[m.end():]


def build_rename_map(code: str) -> dict[str, str]:
    """Build the replacement table of def names, constant names and suggestive argument names from the code."""
    used = set(re.findall(r"\w+", code))
    fpool = [n for n in FUNC_POOL if n not in used]
    cpool = [n for n in CONST_POOL if n not in used]
    apool = [n for n in ARG_POOL if n not in used]
    mapping: dict[str, str] = {}

    for fn in re.findall(r"^\s*def\s+(\w+)\s*\(", code, re.M):
        if fn.startswith("_") or not fpool:
            continue
        mapping[fn] = fpool.pop(0)

    for const in re.findall(r"^([A-Z][A-Z0-9_]{2,})\s*=", code, re.M):
        if not cpool:
            break
        mapping[const] = cpool.pop(0)

    for sig in re.findall(r"^\s*def\s+\w+\s*\(([^)]*)\)", code, re.M):
        for part in sig.split(","):
            arg = part.split(":")[0].split("=")[0].strip()
            if not arg or not arg.isidentifier() or arg in mapping:
                continue
            if arg in ("self", "cls", "conn", "fn", "args", "kwargs"):
                continue
            if SUGGESTIVE.search(arg) and apool:
                mapping[arg] = apool.pop(0)
    return mapping


def apply_map(text: str, mapping: dict[str, str], word: bool = True) -> str:
    for src in sorted(mapping, key=len, reverse=True):
        pat = rf"\b{re.escape(src)}\b" if word else re.escape(src)
        text = re.sub(pat, mapping[src], text)
    return text


# Known short abbreviations that cannot misfire. Warn about every other short
# ASCII word.
SAFE_SHORT = {
    "md5", "sha1", "ecb", "cbc", "gcm", "sql", "sqli", "xss", "ssti", "ssrf",
    "idor", "csrf", "xxe", "lfi", "rfi", "rsa", "jwt", "hmac", "waf", "dos",
    "toctou", "mfa",
}


def check_short_tokens(gold: dict, where: str) -> list[str]:
    """Check that keywords_all contains no short ASCII word.

    any_of is a loose "is this the topic" gate, so short words do little harm
    there, but keywords_all decides pass/fail, so a substring accident such as
    "rce" hitting "sourced" is fatal (it really happened in s05). Only this is
    checked strictly.
    """
    problems = []
    for f in gold.get("findings", []) + gold.get("allow_extra", []):
        for grp in f.get("keywords_all", []):
            for t in (grp if isinstance(grp, (list, tuple)) else [grp]):
                t = str(t)
                if (t.isascii() and t.isalpha() and len(t) <= 4
                        and t.lower() not in SAFE_SHORT):
                    problems.append(
                        f"{where}: {t!r} in keywords_all is too short (substring collision risk)")
    return problems


def check_gold_against_issue(gold: dict, issue: str, where: str) -> list[str]:
    """Check that the gold identifiers still appear in the body (detects missed replacements)."""
    problems = []
    low = issue.lower()
    for f in gold.get("findings", []):
        if f.get("any_of") and not any(str(t).lower() in low for t in f["any_of"]):
            problems.append(f"{where}: none of the any_of of finding {f.get('id')} appears in the body")
        loc = f.get("location_any_of") or []
        if loc and not any(str(t).lower() in low for t in loc):
            problems.append(f"{where}: none of the location_any_of of finding {f.get('id')} appears in the body")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="consistency check only, no generation")
    args = ap.parse_args()

    base = [json.loads(l) for l in BASE_LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows, problems = [], []

    for rec in base:
        tid, bdir = rec["task_id"], rec["dir"]
        src = TASKS / bdir
        en = (src / "issue.md").read_text(encoding="utf-8")
        ja = (src / "issue_ja.md").read_text(encoding="utf-8")
        gold_text = (src / "gold.json").read_text(encoding="utf-8")
        kind, code = _code_of(en)
        is_log = kind == "text"

        # ---- variant a: lexical perturbation ---------------------------------
        if is_log:
            mapping = LOG_RENAMES.get(tid, {})
            en_a = apply_map(en, mapping, word=False)
            ja_a = apply_map(ja, mapping, word=False)
            gold_a = apply_map(gold_text, mapping, word=False)
        else:
            mapping = build_rename_map(code)
            # The grader compares in lower case, so also apply the lower-case
            # form to the gold side (if the code TOKEN_KEY is replaced but the
            # gold keeps "token_key", that word dies and the location check
            # becomes loose).
            gold_map = {**mapping,
                        **{k.lower(): v.lower() for k, v in mapping.items()}}
            en_a = apply_map(en, mapping)
            ja_a = apply_map(ja, mapping)
            gold_a = apply_map(gold_text, gold_map)
        variants = [("a", "ren", "lexical", en_a, ja_a, gold_a)]

        # ---- variant b: decoy insertion --------------------------------------
        if is_log:
            noise = LOG_DECOY.get(tid, ACCESSLOG_DECOY)
            def add_noise(md: str) -> str:
                _, body = _code_of(md)
                lines = body.rstrip("\n").split("\n")
                for i, extra in enumerate(noise):
                    lines.insert(min(len(lines), 2 + i * 3), extra)
                return _sub_block(md, "\n".join(lines) + "\n")
            en_b, ja_b = add_noise(en), add_noise(ja)
        else:
            def add_decoy(md: str, blob: str) -> str:
                _, body = _code_of(md)
                return _sub_block(md, body.rstrip("\n") + "\n" + blob.rstrip("\n") + "\n")
            en_b = add_decoy(en, CODE_DECOY_EN)
            ja_b = add_decoy(ja, CODE_DECOY)
        variants.append(("b", "decoy", "decoy", en_b, ja_b, gold_text))

        for suffix, slug, label, e, j, g in variants:
            vid = f"{tid}{suffix}"
            vdir = f"{vid}_{bdir.split('_', 1)[1]}_{slug}"
            # syntax check (python blocks only)
            if not is_log:
                try:
                    ast.parse(_code_of(e)[1])
                except SyntaxError as exc:
                    problems.append(f"{vid}: generated code has a syntax error — {exc}")
            problems += check_short_tokens(json.loads(g), vid)
            problems += check_gold_against_issue(json.loads(g), e, vid)
            problems += check_gold_against_issue(json.loads(g), j, vid + "(ja)")

            if not args.check:
                d = TASKS / vdir
                d.mkdir(parents=True, exist_ok=True)
                (d / "issue.md").write_text(e, encoding="utf-8")
                (d / "issue_ja.md").write_text(j, encoding="utf-8")
                (d / "gold.json").write_text(g, encoding="utf-8")
            out_rows.append({
                "task_id": vid, "dir": vdir, "grader": "detection",
                "domain": "security", "difficulty": rec["difficulty"],
                "title": f"{rec['title']} [{label}]",
            })

    if problems:
        print("⚠️ consistency problems:")
        for p in problems:
            print("   ", p)
    if not args.check:
        AUG_LEDGER.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
            encoding="utf-8")
        print(f"generated {len(out_rows)} variants → {AUG_LEDGER.name}")
    else:
        print(f"check only: {len(out_rows)} variants / {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

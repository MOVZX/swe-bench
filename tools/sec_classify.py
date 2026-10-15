#!/usr/bin/env python3
"""Classify the failure modes of llmbench security (detection) results.

Changes from v1:
  - Detect trials that "passed with no output" and report a corrected pass rate
    (estimates the impact of the bug where the detection grader passed empty
    output on decoys)
      pass_no_output  passed and thinking cut off (llmbench:reasoning_*) = no body output is certain
      pass_truncated  passed and cut off by max_tokens (a body may have existed = needs review)
  - Aggregate the cut-off reasons (reasoning_loop / reasoning_budget / length) per trial
  - --scan DIR : walk every *_results.json under DIR and list only the affected runs

Usage:
  python3 tools/sec_classify.py results/<stamp>_<model>_results.json
  python3 tools/sec_classify.py --scan results/
"""
import collections
import json
import re
import sys
from pathlib import Path

REFUSAL = re.compile(r"I can(?:'|no)t (?:help|assist)|I'm sorry, but|\u7533\u3057\u8a33\u3042\u308a\u307e\u305b\u3093\u304c|\u304a\u624b\u4f1d\u3044\u3067\u304d\u307e\u305b\u3093", re.I)
TAGS = re.compile(r"</?think>|</?thinking>|<tool_call>|<\|im_start\|>|<function=", re.I)
MARK = re.compile(r"-{2,}\s*FINDINGS\s*-{2,}", re.I)


def is_decoy(r):
    c = (r.get("quality_components") or {}).get("detection") or {}
    return bool(c.get("decoy")) or c.get("n_gold") == 0


def attempt_flags(r, a):
    """Cut-off / passed-with-no-output flags for one trial."""
    fr = a.get("finish_reason") or ""
    f = set()
    if fr.startswith("llmbench:reasoning"):
        f.add(fr.split(":", 1)[1])          # reasoning_loop / reasoning_budget
    elif a.get("truncated"):
        f.add("length")
    if a.get("resolved") and is_decoy(r):
        if fr.startswith("llmbench:reasoning"):
            f.add("pass_no_output")
        elif a.get("truncated"):
            f.add("pass_truncated")
    return f


def corrected(r):
    """(pass rate before correction, pass rate after correction, number of passed-with-no-output trials)."""
    att = [a for a in r.get("attempts", []) if not a.get("errored") and not a.get("skipped")]
    if not att:
        sr = r.get("success_rate", 1.0 if r.get("resolved") else 0.0)
        return sr, sr, 0
    bad = sum(1 for a in att if "pass_no_output" in attempt_flags(r, a))
    ok = sum(1 for a in att if a.get("resolved"))
    return ok / len(att), (ok - bad) / len(att), bad


def classify(r, raw):
    tags = set()
    comp = (r.get("quality_components") or {}).get("detection", {})
    if r.get("refused") or r.get("n_refused"):
        tags.add("refused")
    if r.get("n_errored"):
        tags.add("errored")
    if r.get("parse_ok") is False:
        tags.add("parse_fail")
    for a in r.get("attempts", []):
        tags |= attempt_flags(r, a)
    if raw is not None:
        if TAGS.search(raw):
            tags.add("tag_leak")
        if raw.strip() and not MARK.search(raw):
            tags.add("no_marker")
        if "refused" not in tags and REFUSAL.search(raw[:2000]):
            tags.add("refused?")
    reasons = " | ".join(a.get("fail_reason", "") for a in r.get("attempts", []) if not a.get("resolved"))
    if "clean input" in reasons:
        tags.add("decoy_fp")
    elif "over-flagged" in reasons:
        tags.add("over_flag")
    if re.search(r"recall [\d.]+ <", reasons):
        tags.add("missed")
    if "no findings output" in reasons:
        tags.add("no_output_fail")
    la = comp.get("location_acc")
    if la is not None and la < 1.0:
        tags.add("location")
    return sorted(tags), comp, reasons


def load(path):
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    sec = [r for r in d.get("results", []) if r.get("domain") == "security"]
    return p, d, sec


def report(path):
    p, d, sec = load(path)
    art = p.parent / d.get("artifacts_dir", p.name.replace("_results.json", "_artifacts"))
    rows, agg = [], collections.Counter()
    for r in sec:
        f = art / r["task_id"] / "llm_output.txt"
        raw = f.read_text(encoding="utf-8", errors="replace") if f.exists() else None
        tags, comp, reason = classify(r, raw)
        before, after, _ = corrected(r)
        if after < 1.0:
            agg.update(t for t in tags if t not in ("location", "no_marker", "tag_leak"))
        rows.append((r["task_id"], "decoy" if is_decoy(r) else "real", before, after,
                     comp.get("recall"), comp.get("fp"), ",".join(tags), reason[:80]))
    print(f"model={d.get('model')} served={d.get('served_model')}  sec tasks={len(rows)}")
    print(f"{'task':8} {'kind':5} {'succ':>5} {'corr':>5} {'rec':>5} {'fp':>3}  tags / reason")
    for t in rows:
        mark = " ⚠" if t[3] < t[2] else ""
        print(f"{t[0]:8} {t[1]:5} {t[2]:5.2f} {t[3]:5.2f} {str(t[4]):>5} {str(t[5]):>3}  {t[6]}{mark}  {t[7]}")

    def avg(xs, i):
        return sum(x[i] for x in xs) / len(xs) if xs else float("nan")
    real = [t for t in rows if t[1] == "real"]
    dec = [t for t in rows if t[1] == "decoy"]
    print(f"\nPass rate (as reported): all {avg(rows,2):.2f} / real {avg(real,2):.2f} / decoy {avg(dec,2):.2f}  (n={len(rows)})")
    print(f"Pass rate (corrected):   all {avg(rows,3):.2f} / real {avg(real,3):.2f} / decoy {avg(dec,3):.2f}")
    print("Failure modes of tasks that fail after correction (overlapping):", dict(agg.most_common()) or "none")


def scan(root):
    hit = n_sec = n_files = 0
    for p in sorted(Path(root).rglob("*_results.json")):
        n_files += 1
        try:
            _, d, sec = load(p)
        except Exception as e:           # skip broken / old-format files
            print(f"skip {p.name}: {e}")
            continue
        n_sec += bool(sec)
        bad = []
        for r in sec:
            _, _, n = corrected(r)
            if n:
                fr = ",".join(sorted({(a.get("finish_reason") or "") for a in r.get("attempts", [])
                                      if "pass_no_output" in attempt_flags(r, a)}))
                bad.append(f"{r['task_id']}×{n}({fr})")
        sus = [r["task_id"] for r in sec
               if any("pass_truncated" in attempt_flags(r, a) for a in r.get("attempts", []))]
        if bad or sus:
            hit += 1
            print(f"{p.relative_to(root)}\n   passed with no output: {', '.join(bad) or '-'}\n   needs review (passed on a length cut-off): {', '.join(sus) or '-'}")
    print(f"\nscanned {n_files} files / {n_sec} containing security / affected runs: {hit}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--scan":
        scan(sys.argv[2])
    elif len(sys.argv) == 2:
        report(sys.argv[1])
    else:
        sys.exit(__doc__)

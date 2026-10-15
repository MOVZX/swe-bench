"""Detection grader (Security / Analysis).

Prompts the model to perform security vulnerability/intrusion "detection & diagnosis",
grading output via precision / recall / F1 against gold labels. Does not patch code.

Output Contract: JSON array after a `--- FINDINGS ---` line
  [{"type": "...", "location": "...", "evidence": "..."}]
For decoys (gold.findings==[]), zero predicted findings is correct (penalizes false positives/over-detection).
Predictions matching gold's `allow_extra` are **neutral** (neither TP nor FP; excluded from the denominator).
This is a release valve so that secondary but valid findings readable from the same code
(e.g. "no authentication" on a command-injection snippet) are not punished as over-detection.
"""

from __future__ import annotations

import json
import re

from . import GradeCtx, Grader, GraderEval
from ..patch import _strip_control_tokens

_SYSTEM = """\
You are a senior security analyst. Analyze the provided code or log and report
security issues you find. Output ONLY a JSON array after a line `--- FINDINGS ---`.
Each element: {"type": "<vuln class / CWE>", "location": "<where>", "evidence": "<the exact risky snippet>"}.
If there are NO issues, output an empty array: []
Do not include any other text after the JSON.
"""


def _load_gold(task) -> dict:
    p = task.dir / "gold.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"findings": []}


def _first_json_array(text: str):
    """Returns the first valid JSON array in text (or None if absent)."""
    i = 0
    while True:
        start = text.find("[", i)
        if start < 0:
            return None
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:j + 1])
                        except Exception:
                            break
        i = start + 1


def _extract_findings(raw: str):
    """Extracts findings array from model output. Returns (list, parse_ok, error)."""
    text = _strip_control_tokens(raw or "")
    if not text.strip():
        return [], False, "empty output"
    m = re.search(r"-{2,}\s*FINDINGS\s*-{2,}", text, re.I)
    seg = text[m.end():] if m else text
    arr = _first_json_array(seg)
    if arr is None and m is None:
        arr = _first_json_array(text)
    if arr is None:
        return [], False, "no JSON findings array found"
    norm = []
    for it in arr:
        if isinstance(it, dict):
            norm.append(it)
        else:
            norm.append({"type": str(it)})
    return norm, True, ""


def _serialize(pred: dict) -> str:
    return json.dumps(pred, ensure_ascii=False).lower()


def _kw_hit(pred_text_lc: str, kw) -> bool:
    """Judges one element of keywords_all.

    A string is a plain substring match. A list/tuple is treated as an OR group:
    one element present is enough (absorbs spelling variants across languages,
    e.g. ["travers", "torabasaruru"]).
    """
    if isinstance(kw, (list, tuple)):
        return any(str(x).lower() in pred_text_lc for x in kw)
    return str(kw).lower() in pred_text_lc


def _location_ok(pred: dict, g: dict) -> bool | None:
    """Whether the prediction's location field matches the location expected by gold.

    Returns None when gold has no `location_any_of` (not graded).
    Only the location field is checked, not the whole evidence — that closes the
    loophole of passing by writing the function name in the evidence.
    """
    terms = g.get("location_any_of")
    if not terms:
        return None
    loc = str(pred.get("location", "")).lower()
    if not loc.strip():
        return False
    return any(str(t).lower() in loc for t in terms)


def _covers(pred_text_lc: str, g: dict) -> bool:
    terms = [str(t) for t in g.get("any_of", [])]
    if g.get("cwe"):
        terms.append(str(g["cwe"]))
    any_ok = any(t.lower() in pred_text_lc for t in terms) if terms else True
    all_ok = all(_kw_hit(pred_text_lc, k) for k in g.get("keywords_all", []))
    return any_ok and all_ok


class DetectionGrader(Grader):
    name = "detection"
    domain = "security"

    def build_prompt(self, task, lang: str) -> tuple[str, str]:
        return _SYSTEM, task.issue(lang)

    def evaluate(self, task, raw_output: str, ctx: GradeCtx) -> GraderEval:
        ev = GraderEval()
        gold_doc = _load_gold(task)
        gold = gold_doc.get("findings", [])
        allow_extra = gold_doc.get("allow_extra", [])
        preds, ok, err = _extract_findings(raw_output)
        ev.parse_ok = ok
        ev.parse_error = err
        ev.parsed_files = {"findings.json": json.dumps(preds, ensure_ascii=False, indent=2)}

        pred_texts = [_serialize(p) for p in preds]
        n_gold, n_pred = len(gold), len(preds)
        covered = [g for g in gold if any(_covers(pt, g) for pt in pred_texts)]

        # Sort predictions into TP / neutral / FP.
        # neutral = predictions matching allow_extra. Excluded from the precision denominator.
        tp, neutral, fp_texts = [], [], []
        for pt in pred_texts:
            if any(_covers(pt, g) for g in gold):
                tp.append(pt)
            elif allow_extra and any(_covers(pt, a) for a in allow_extra):
                neutral.append(pt)
            else:
                fp_texts.append(pt)
        n_scored = len(tp) + len(fp_texts)

        # location grading (only findings that have location_any_of in gold).
        # Undetected findings are already penalized on the recall side, so do not
        # count them here (avoids a double penalty).
        loc_graded = loc_correct = 0
        for g in gold:
            if not g.get("location_any_of"):
                continue
            hits = [pr for pr, pt in zip(preds, pred_texts) if _covers(pt, g)]
            if not hits:
                continue
            loc_graded += 1
            if any(_location_ok(pr, g) for pr in hits):
                loc_correct += 1
        location_acc = (loc_correct / loc_graded) if loc_graded else None

        recall = len(covered) / n_gold if n_gold else 1.0
        if n_scored:
            precision = len(tp) / n_scored
        else:
            precision = 1.0 if n_gold == 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        dcfg = ctx.graders_cfg.get("detection", {})
        pass_f1 = float(dcfg.get("pass_f1", 0.67))
        pass_recall = dcfg.get("pass_recall")
        max_fp_per_gold = dcfg.get("max_fp_per_gold")

        # (resolved is overwritten by the two-axis gate below)
        # Two-axis gate (when both pass_recall and max_fp_per_gold are present):
        #   pass = "detect the main findings up to the required ratio"
        #          AND "extras <= gold count x factor".
        # With a single F1 threshold the tolerance for extras and the tolerance for
        # misses cannot be set independently. On a task with 1 gold finding F1 drops
        # 1.0 → 0.667 → 0.5, so a threshold around 0.67 effectively means
        # "output exactly one finding".
        # With neither key, judge by F1 alone as before (backward compatibility).
        fp_budget = None
        if pass_recall is not None and max_fp_per_gold is not None:
            fp_budget = int(float(max_fp_per_gold) * n_gold)   # decoy (n_gold=0) gets 0
            ev.resolved = (recall >= float(pass_recall)) and (len(fp_texts) <= fp_budget)
        else:
            ev.resolved = f1 >= pass_f1

        # quality: multiply F1 by the location accuracy, weighted.
        # With location_weight=0 (default) this is F1×100 as before.
        loc_w = float(dcfg.get("location_weight", 0.0))
        q = f1
        if location_acc is not None and loc_w > 0:
            q = f1 * (1.0 - loc_w + loc_w * location_acc)
        ev.quality_score = round(q * 100, 1)
        fp = len(fp_texts)
        fn = n_gold - len(covered)
        ev.components = {
            "detection": {
                "enabled": True,
                "precision": round(precision, 3), "recall": round(recall, 3),
                "f1": round(f1, 3), "tp": len(tp), "fp": fp, "fn": fn,
                "neutral": len(neutral),
                "n_gold": n_gold, "n_pred": n_pred, "n_scored": n_scored,
                "pass_f1": pass_f1,
                "gate": "recall+fp" if fp_budget is not None else "f1",
                "location_acc": (round(location_acc, 3)
                                 if location_acc is not None else None),
                "location_graded": loc_graded, "location_weight": loc_w,
                "pass_recall": pass_recall, "fp_budget": fp_budget,
                "decoy": n_gold == 0,
            }
        }
        # A trial with no findings extracted from the output fails, whatever the metrics say.
        # On a decoy (0 gold findings) "0 predictions" has the same shape as the correct
        # answer, so without this gate an empty output or a truncated thought
        # (reasoning_loop / reasoning_budget) would pass as recall 1.0 / FP 0
        # (4 cases in the 2026-09-27 measurements).
        if not ok:
            ev.resolved = False
            ev.quality_score = 0.0
            ev.components["detection"]["no_output"] = True
            ev.fail_reason = f"no findings output ({err})"
            ev.detail_output = f"no findings output ({err})"
            return ev
        missed = [g.get("id") or g.get("cwe") or "?" for g in gold if g not in covered]
        neutral_note = f" NEUTRAL={len(neutral)}" if neutral else ""
        ev.detail_output = (
            f"precision={precision:.2f} recall={recall:.2f} F1={f1:.2f} "
            f"(TP={len(tp)} FP={fp} FN={fn}{neutral_note}); missed={missed}"
            + (f"; location {loc_correct}/{loc_graded}" if loc_graded else "")
        )
        if not ev.resolved:
            if n_gold == 0 and n_scored > 0:
                ev.fail_reason = f"over-flagged {n_scored} false positive(s) on clean input"
            elif fp_budget is not None:
                if recall < float(pass_recall):
                    ev.fail_reason = (
                        f"recall {recall:.2f} < {pass_recall} (missed={missed})"
                    )
                else:
                    ev.fail_reason = (
                        f"over-flagged {fp} finding(s), budget {fp_budget} "
                        f"(recall {recall:.2f} OK)"
                    )
            else:
                ev.fail_reason = f"F1 {f1:.2f} < {pass_f1} (missed={missed})"
        return ev

    def mock_gold(self, task) -> str:
        gold = _load_gold(task).get("findings", [])
        preds = []
        for g in gold:
            ev = " ".join(str(t) for t in g.get("any_of", []))
            kws = [
                str(k[0]) if isinstance(k, (list, tuple)) and k else str(k)
                for k in g.get("keywords_all", [])
            ]
            ev += " " + " ".join(kws)
            loc_terms = g.get("location_any_of") or []
            preds.append({
                "type": g.get("cwe", g.get("id", "issue")),
                "location": str(loc_terms[0]) if loc_terms else "see code",
                "evidence": ev.strip(),
            })
        body = json.dumps(preds, ensure_ascii=False, indent=2)
        return f"--- FINDINGS ---\n{body}"

    def mock_broken(self, task) -> str:
        # Outputs 1 bogus finding -> FP for decoys, recall 0 for non-decoys -> always fails.
        bogus = [{"type": "made-up issue", "location": "nowhere",
                  "evidence": "zzzzz nonexistent token qqqqq"}]
        return "--- FINDINGS ---\n" + json.dumps(bogus, ensure_ascii=False)

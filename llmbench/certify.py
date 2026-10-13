"""Usability tier classification based on passing criteria.

Reads results.json (runner output), maps task difficulties to tiers (L1-L7),
and cumulatively evaluates whether the average success rate and average combined
score for each tier satisfy the required gate criteria.
Returns the model's achieved level (= highest tier satisfying the gate).

- L4 (expert) gate = **Usability Baseline**: Usable in production under supervision.
- Can run as a standalone script (stdlib only; independent of llmbench core):
    python3 certify.py results/<stamp>_<model>_results.json
    python3 certify.py --merge base40.json l6.json   # Merge split executions into 1 certificate
- For integration into llmbench, place as llmbench/certify.py and call
  `certify()` / `render_certificate_md()` from report.py / compare.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Difficulty -> tier mapping. Maps existing easy/medium/hard and new expert/frontier/architect/grandmaster.
DIFFICULTY_TO_TIER = {
    "easy": "L1",
    "medium": "L2",
    "hard": "L3",
    "expert": "L4",
    "frontier": "L5",
    "architect": "L6",
    "grandmaster": "L7",
}

TIER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

TIER_LABEL = {
    "L1": "L1 easy",
    "L2": "L2 medium",
    "L3": "L3 hard",
    "L4": "L4 expert (Usability Baseline)",
    "L5": "L5 frontier",
    "L6": "L6 architect",
    "L7": "L7 grandmaster (Ceiling Evaluation)",
}

# Gate: Minimum threshold to be met by average task scores in each tier.
# min_success is average pass@1 (success_rate), min_combined is average combined score.
DEFAULT_GATES = {
    "L1": {"min_success": 0.90, "min_combined": 0.0},
    "L2": {"min_success": 0.85, "min_combined": 0.0},
    "L3": {"min_success": 0.75, "min_combined": 60.0},
    "L4": {"min_success": 0.60, "min_combined": 55.0},
    "L5": {"min_success": 0.40, "min_combined": 0.0},
    # L6 (architect): Finalized via Phase 3 calibration (ornith 9B/35B ×5 runs, 2026-06-26).
    "L6": {"min_success": 0.60, "min_combined": 58.0},
    # L7 (grandmaster): Provisional value for ceiling evaluation (t061-t100).
    # Temporary gate until finalized via empirical calibration (assumes frontier models ×5 runs).
    "L7": {"min_success": 0.35, "min_combined": 55.0},
}

# Level verdict interpretations (short summary for reports).
LEVEL_VERDICT = {
    None: "Below L1. Basic bug fixes remain unstable; unusable in production.",
    "L1": "Reliable for toy-level tasks, but simple production bugs are still unstable.",
    "L2": "Capable of handling simple tasks following clear specifications.",
    "L3": "Capable of handling simple-to-intermediate production bugs.",
    "L4": "✅ Usability baseline reached. Ready for production under supervision.",
    "L5": "Frontier level. Can handle complex tasks with assistance.",
    "L6": "Architect level. Capable of repository-scale diagnostics and design decisions.",
    "L7": "Grandmaster level. Reached ceiling evaluation band — no plateau observed with current task suite.",
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _task_success(t: dict) -> float:
    """Gets success_rate. For older formats (runs=1), maps resolved boolean to 0.0/1.0."""
    if "success_rate" in t and t["success_rate"] is not None:
        return float(t["success_rate"])
    return 1.0 if t.get("resolved") else 0.0


def aggregate_by_tier(results: list[dict]) -> dict[str, dict]:
    """Aggregates mean success_rate, mean combined score, and task count per tier."""
    buckets: dict[str, dict[str, list]] = {}
    for t in results:
        tier = DIFFICULTY_TO_TIER.get(t.get("difficulty"))
        if tier is None:
            continue
        b = buckets.setdefault(tier, {"success": [], "combined": [], "ids": []})
        b["success"].append(_task_success(t))
        b["combined"].append(float(t.get("combined", 0.0)))
        b["ids"].append(t.get("task_id"))
    out: dict[str, dict] = {}
    for tier, b in buckets.items():
        out[tier] = {
            "n": len(b["ids"]),
            "mean_success": _mean(b["success"]),
            "mean_combined": _mean(b["combined"]),
            "task_ids": b["ids"],
        }
    return out


def certify(results: list[dict], gates: dict | None = None) -> dict:
    """Tier aggregation -> gate evaluation -> achieved level.

    Returns: {achieved_level, verdict, usable_line, independent_pass,
    tiers:[{tier,n,mean_success,mean_combined,gate_pass,measured}], gates}
    achieved_level is 'the highest tier that was measured and continuously passed gates including lower tiers'.
    Unmeasured tiers skip evaluation without blocking overall pass/fail status.
    """
    gates = gates or DEFAULT_GATES
    agg = aggregate_by_tier(results)

    rows = []
    achieved = None
    broken = False  # Once a gate fails, stop certifying higher tiers cumulatively
    for tier in TIER_ORDER:
        g = gates[tier]
        if tier not in agg:
            rows.append({
                "tier": tier, "measured": False, "n": 0,
                "mean_success": None, "mean_combined": None,
                "gate_pass": None,
                "gate": g,
            })
            continue
        a = agg[tier]
        passed = (
            a["mean_success"] >= g["min_success"]
            and a["mean_combined"] >= g["min_combined"]
        )
        rows.append({
            "tier": tier, "measured": True, "n": a["n"],
            "mean_success": a["mean_success"],
            "mean_combined": a["mean_combined"],
            "gate_pass": passed,
            "gate": g,
            "task_ids": a["task_ids"],
        })
        if not broken:
            if passed:
                achieved = tier
            else:
                broken = True

    # Non-cumulative evaluation: evaluates each tier independently (not blocked by failures in lower tiers).
    measured = [r for r in rows if r["measured"]]
    independent_pass = [r["tier"] for r in measured if r["gate_pass"]]
    l4 = next((r for r in rows if r["tier"] == "L4"), None)
    usable_line = bool(l4 and l4["measured"] and l4["gate_pass"])

    return {
        "achieved_level": achieved,          # Cumulative (highest level passed sequentially from bottom)
        "verdict": LEVEL_VERDICT.get(achieved, ""),
        "usable_line": usable_line,          # Primary evaluation: Passed L4 independently
        "independent_pass": independent_pass,  # All tiers passed independently
        "tiers": rows,
        "gates": gates,
    }


def render_certificate_md(cert: dict, model: str = "") -> str:
    lines = []
    head = f"## 🎓 Certificate (Usability Baseline Evaluation){f': {model}' if model else ''}"
    lines.append(head)

    # Primary verdict: Usability baseline = Passed L4 independently (non-cumulative).
    usable = cert.get("usable_line")
    usable_txt = "✅ Usability Baseline Reached" if usable else "❌ Usability Baseline Not Reached"
    lines.append(f"\n**{usable_txt}** (Independently passed L4 expert){'' if usable else ''}")

    # Secondary verdict: Cumulative achieved level (consecutive passes including lower tiers = consistency metric).
    lvl = cert["achieved_level"]
    lvl_txt = TIER_LABEL.get(lvl, "None (Below L1)")
    indep = cert.get("independent_pass", [])
    lines.append(
        f"Reference: Cumulative Achieved Level **{lvl_txt}** / Independently Passed Tiers: "
        f"{', '.join(indep) if indep else 'None'}\n"
    )

    lines.append("| Tier | Independent Pass | Tasks | Avg Success | Avg Combined | Gate (Success/Combined) |")
    lines.append("|---|---|---|---|---|---|")
    for r in cert["tiers"]:
        g = r["gate"]
        gate_s = f"≥{g['min_success']:.0%} / ≥{g['min_combined']:.0f}"
        if not r["measured"]:
            lines.append(f"| {r['tier']} | — Unmeasured | 0 | – | – | {gate_s} |")
            continue
        mark = "✅ Pass" if r["gate_pass"] else "❌ Fail"
        lines.append(
            f"| {r['tier']} | {mark} | {r['n']} | "
            f"{r['mean_success']:.0%} | {r['mean_combined']:.1f} | {gate_s} |"
        )
    lines.append(
        "\n> **Usability Baseline = Passed L4 independently** (unaffected by misses in lower tiers)."
        " Cumulative achieved level indicates 'the highest tier passed sequentially from L1', serving as a consistency reference."
        " L7 (grandmaster) is the ceiling evaluation tier — gates are provisional and will be finalized after empirical calibration."
    )
    return "\n".join(lines)


# ===== Domain-specific Certification (Non-coding) =====

DOMAIN_ORDER = ["security", "general", "writing", "medical"]

DOMAIN_LABEL = {
    "security": "🛡️ security Detection/Analysis",
    "general": "📋 general Instruction Following",
    "writing": "✍️ writing Creative Writing",
    "medical": "🩺 medical QA",
}

# Domain-specific gates (Provisional. Experimental/reference categories excluded from balance index).
DEFAULT_DOMAIN_GATES = {
    "security": {"min_success": 0.60, "min_combined": 60.0},
    "general": {"min_success": 0.70, "min_combined": 65.0},
    "writing": {"min_success": 0.50, "min_combined": 55.0, "experimental": True},
    "medical": {"min_success": 0.60, "min_combined": 60.0, "reference": True},
}


def aggregate_by_domain(results: list[dict]) -> dict[str, dict]:
    """Aggregates success_rate / combined score by domain (excludes 'code')."""
    buckets: dict[str, dict[str, list]] = {}
    for t in results:
        dom = t.get("domain", "code")
        if dom == "code":
            continue
        b = buckets.setdefault(dom, {"success": [], "combined": [], "ids": []})
        b["success"].append(_task_success(t))
        b["combined"].append(float(t.get("combined", 0.0)))
        b["ids"].append(t.get("task_id"))
    return {
        d: {"n": len(b["ids"]), "mean_success": _mean(b["success"]),
            "mean_combined": _mean(b["combined"]), "task_ids": b["ids"]}
        for d, b in buckets.items()
    }


def _coding_combined(results: list[dict]) -> tuple[float, int]:
    xs = [float(t.get("combined", 0.0)) for t in results
          if t.get("domain", "code") == "code"]
    return _mean(xs), len(xs)


def certify_domains(results: list[dict], gates: dict | None = None) -> dict:
    """Domain-specific gate evaluation + Balance Index.

    Balance Index = Harmonic mean of combined scores (0-1) across coding + non-experimental domains × 100.
    Drops significantly for specialized/one-trick models (models weak in specific domains).
    """
    gates = gates or DEFAULT_DOMAIN_GATES
    agg = aggregate_by_domain(results)
    rows = []
    for dom in DOMAIN_ORDER:
        if dom not in agg:
            continue
        a = agg[dom]
        g = gates.get(dom, {})
        passed = (
            a["mean_success"] >= g.get("min_success", 0.0)
            and a["mean_combined"] >= g.get("min_combined", 0.0)
        )
        rows.append({
            "domain": dom, "n": a["n"],
            "mean_success": a["mean_success"], "mean_combined": a["mean_combined"],
            "gate_pass": passed, "gate": g,
            "experimental": bool(g.get("experimental") or g.get("reference")),
        })
    # Balance Index (coding + non-experimental domains)
    members: list[tuple[str, float]] = []
    cc, cn = _coding_combined(results)
    if cn:
        members.append(("code", cc))
    for r in rows:
        if not r["experimental"]:
            members.append((r["domain"], r["mean_combined"]))
    balance = None
    if len(members) >= 1:
        vals = [max(1e-9, s / 100.0) for _, s in members]
        balance = round(len(vals) / sum(1.0 / v for v in vals) * 100.0, 1)
    return {
        "domains": rows, "balance_index": balance,
        "balance_members": [d for d, _ in members],
    }


def render_domains_md(cd: dict) -> str:
    if not cd["domains"]:
        return ""
    lines = ["## 🌐 Domain Certification (Non-coding)", ""]
    lines.append("| Domain | Tasks | Avg Success | Avg Combined | Gate (Success/Combined) | Status |")
    lines.append("|---|---|---|---|---|---|")
    for r in cd["domains"]:
        g = r["gate"]
        gate_s = f"≥{g.get('min_success', 0):.0%} / ≥{g.get('min_combined', 0):.0f}"
        mark = "✅ Pass" if r["gate_pass"] else "❌ Fail"
        tag = " *(experimental)*" if r["experimental"] else ""
        lines.append(
            f"| {DOMAIN_LABEL.get(r['domain'], r['domain'])}{tag} | {r['n']} | "
            f"{r['mean_success']:.0%} | {r['mean_combined']:.1f} | {gate_s} | {mark} |"
        )
    if cd["balance_index"] is not None:
        lines += [
            "",
            f"**⚖️ Balance Index: {cd['balance_index']:.1f} / 100** "
            f"(Harmonic mean of {' + '.join(cd['balance_members'])}. "
            f"Decreases sharply if model is specialized/weak in any domain)",
            "> writing/medical are experimental/reference values and excluded from Balance Index by default.",
        ]
    return "\n".join(lines)


# ===== Medical QA Details (Accuracy by Difficulty, Reference Values) =====

MED_TIER_ORDER = ["med_basic", "med_std", "med_hard"]
MED_TIER_LABEL = {
    "med_basic": "MED-basic Basic",
    "med_std": "MED-std Standard (Board)",
    "med_hard": "MED-hard Advanced",
}
# Reference gates (Uncalibrated). accuracy = mean success_rate.
DEFAULT_MED_GATES = {"med_basic": 0.80, "med_std": 0.60, "med_hard": 0.40}


def certify_medical(results: list[dict]) -> dict:
    """Aggregates accuracy for medical domain grouped by difficulty (med_basic/std/hard)."""
    buckets: dict[str, list] = {}
    for t in results:
        if t.get("domain") != "medical":
            continue
        buckets.setdefault(t.get("difficulty", "med_std"), []).append(_task_success(t))
    rows, overall = [], []
    for tier in MED_TIER_ORDER:
        if tier not in buckets:
            continue
        xs = buckets[tier]
        overall += xs
        acc = _mean(xs)
        gate = DEFAULT_MED_GATES.get(tier, 0.0)
        rows.append({"tier": tier, "n": len(xs), "accuracy": acc,
                     "gate": gate, "pass": acc >= gate})
    return {"tiers": rows, "n": len(overall),
            "accuracy": _mean(overall) if overall else None}


def render_medical_md(cm: dict) -> str:
    if not cm["tiers"]:
        return ""
    lines = ["## 🩺 Medical QA Details (Reference Values / Uncalibrated)", ""]
    lines.append(f"**Overall Accuracy: {cm['accuracy'] * 100:.1f}% ({cm['n']} questions)**")
    lines.append(
        "> Chance accuracy for 5-choice MCQ is approx 20%. These are reference values and do not guarantee clinical validity."
    )
    lines += ["", "| Difficulty | Questions | Accuracy | Reference Gate |", "|---|---|---|---|"]
    for r in cm["tiers"]:
        mark = "✅" if r["pass"] else "⚠️"
        lines.append(
            f"| {MED_TIER_LABEL.get(r['tier'], r['tier'])} | {r['n']} | "
            f"{r['accuracy'] * 100:.0f}% {mark} | ≥{r['gate'] * 100:.0f}% |"
        )
    return "\n".join(lines)


def _load_results(path: Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("model", ""), data.get("results", [])


def merge_results(paths: list) -> tuple[str, list[dict]]:
    """Merges the results arrays across multiple results.json files into one.

    - On task_id collisions, **last wins** (overwritten by records from later files).
      Intended to prioritize newer measurements when re-evaluating tasks in split runs.
    - Records lacking task_id remain un-deduplicated (order preserved).
    - Model names collect distinct models in appearance order and join via " + ".
    Returns: (model_label, merged_results).
    """
    merged: dict[str, dict] = {}
    extras: list[dict] = []
    models: list[str] = []
    for p in paths:
        model, results = _load_results(Path(p))
        if model and model not in models:
            models.append(model)
        for t in results:
            tid = t.get("task_id")
            if tid is None:
                extras.append(t)
            else:
                merged[tid] = t
    return " + ".join(models), list(merged.values()) + extras


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="certify.py",
        description="Evaluates results.json against tier passing criteria (llmbench independent)",
    )
    ap.add_argument("results", nargs="+", help="results.json files to evaluate")
    ap.add_argument("--merge", action="store_true",
                    help="Combine results across multiple files into a single certificate")
    ns = ap.parse_args(argv)

    if ns.merge:
        model, results = merge_results(ns.results)
        print(render_certificate_md(certify(results), model or "merged"))
        dom = render_domains_md(certify_domains(results))
        if dom:
            print("\n" + dom)
        med = render_medical_md(certify_medical(results))
        if med:
            print("\n" + med)
        print()
        return 0

    for arg in ns.results:
        model, results = _load_results(Path(arg))
        print(render_certificate_md(certify(results), model))
        dom = render_domains_md(certify_domains(results))
        if dom:
            print("\n" + dom)
        med = render_medical_md(certify_medical(results))
        if med:
            print("\n" + med)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

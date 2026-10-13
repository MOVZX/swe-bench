"""Generates a cross-comparison report across multiple results.json files.

By including reference models (strong/weak) side by side, local model scores
can be interpreted in a relative context (anchor).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def load_results(paths: list[str | Path]) -> list[dict]:
    """Loads a collection of results.json files. Identified by model name, not filename."""
    runs = []
    for p in paths:
        p = Path(p)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_path"] = p.name
        runs.append(d)
    return runs


def _avg_tps(results: list[dict]) -> float | None:
    vals = [r.get("tokens_per_sec") for r in results if r.get("tokens_per_sec")]
    return sum(vals) / len(vals) if vals else None


def _task_index(runs: list[dict]) -> dict[str, dict]:
    """task_id -> {difficulty, title} (takes the first encountered match)."""
    idx: dict[str, dict] = {}
    for run in runs:
        for r in run.get("results", []):
            tid = r["task_id"]
            if tid not in idx:
                idx[tid] = {
                    "difficulty": r.get("difficulty", ""),
                    "title": r.get("title", ""),
                }
    return idx


def render_comparison(runs: list[dict]) -> str:
    if not runs:
        return "# Model Comparison\n\n(No results available)\n"

    # Extract summary for each model and sort by combined score in descending order
    rows = []
    for run in runs:
        s = run.get("summary", {})
        rows.append({
            "model": run.get("model", run.get("_path", "?")),
            "lang": run.get("issue_lang", "?"),
            "runs": s.get("runs", 1),
            "resolved": s.get("resolved_rate", 0.0),
            "success": s.get("avg_success_rate"),
            "passk": s.get("avg_pass_at_k"),
            "quality": s.get("avg_quality_resolved", 0.0),
            "combined": s.get("avg_combined", 0.0),
            "tps": _avg_tps(run.get("results", [])),
            "usability": s.get("usability", {}),
            "results": {r["task_id"]: r for r in run.get("results", [])},
        })
    rows.sort(key=lambda x: x["combined"], reverse=True)
    best = rows[0]["combined"] or 1.0
    any_multi = any(r["runs"] > 1 for r in rows)

    lines = [
        "# 🆚 iHoree LLM Comparison Report",
        "",
        f"Target Models: {len(rows)} / Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Ranking (Descending by Avg Combined)",
        "",
    ]
    # Header
    head = "| # | Model | Lang | Resolved "
    sep = "|---|---|---|---|"
    if any_multi:
        head += "| Success Rate | pass@k "
        sep += "---|---|"
    head += "| Quality | Combined | Relative | tok/s |"
    sep += "---|---|---|---|"
    lines += [head, sep]
    for i, r in enumerate(rows, 1):
        rel = r["combined"] / best * 100 if best else 0
        mid = ""
        if any_multi:
            sc = f"{r['success'] * 100:.1f}%" if r["success"] is not None else "—"
            pk = f"{r['passk'] * 100:.1f}%" if r["passk"] is not None else "—"
            mid = f"| {sc} | {pk} "
        tps = f"{r['tps']:.1f}" if r["tps"] else "—"
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "")
        lines.append(
            f"| {i}{medal} | **{r['model']}** | {r['lang']} "
            f"| {r['resolved'] * 100:.1f}% {mid}"
            f"| {r['quality']:.1f} | {r['combined']:.1f} "
            f"| {rel:.0f}% | {tps} |"
        )

    # Usability Tier Comparison
    lines += ["", "## Usability Tier Comparison", "",
              "| Model | 🟢 Autonomous | 🟡 Assisted | 🔴 Unusable |", "|---|---|---|---|"]
    for r in rows:
        u = r["usability"] or {}
        lines.append(
            f"| {r['model']} | {u.get('autonomous', 0)} "
            f"| {u.get('assisted', 0)} | {u.get('unusable', 0)} |"
        )

    # Per-Task Matrix (cell = combined, row max in bold)
    idx = _task_index(runs)
    lines += ["", "## Per-Task Combined Matrix", "",
              "Each cell represents the Combined score for that task. The highest value in each row is **bolded**.", ""]
    header = "| Task | Difficulty | " + " | ".join(r["model"] for r in rows) + " |"
    lines += [header, "|---|---|" + "---|" * len(rows)]
    for tid in sorted(idx):
        diff = idx[tid]["difficulty"]
        cells = []
        present = [r["results"].get(tid) for r in rows]
        vals = [p.get("combined") if p else None for p in present]
        bestval = max([v for v in vals if v is not None], default=None)
        for v in vals:
            if v is None:
                cells.append("—")
            elif bestval is not None and v == bestval:
                cells.append(f"**{v:.0f}**")
            else:
                cells.append(f"{v:.0f}")
        lines.append(f"| {tid} | {diff} | " + " | ".join(cells) + " |")

    lines += ["", "> Relative = Each model's Combined ÷ Best model's Combined.",
              "> Side-by-side inclusion of reference models (strong/weak) helps clarify local model standing."]
    return "\n".join(lines) + "\n"


def save_comparison(
    runs: list[dict], output_dir: Path, name: str = "comparison"
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{name}_{stamp}.md"
    out.write_text(render_comparison(runs), encoding="utf-8")
    return out

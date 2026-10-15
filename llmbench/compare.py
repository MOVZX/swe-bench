"""Generates a cross-comparison report across multiple results.json files.

By including reference models (strong/weak) side by side, local model scores
can be interpreted in a relative context (anchor).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .env import EXEC_LABEL, format_summary


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


def _run_tps(run: dict) -> float | None:
    """Prefer the summary value; average results[] when it is missing."""
    s = (run.get("summary") or {}).get("tokens_per_sec")
    if s:
        return float(s)
    return _avg_tps(run.get("results", []))


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


def _device_label(env: dict) -> str:
    """Device name actually used in this run (installed GPU/CPU if absent)."""
    backend = env.get("backend") or {}
    launch = backend.get("launch") or {}
    if launch.get("device_name"):
        return str(launch["device_name"])
    inf = (backend.get("gpu_usage") or {}).get("inference") or {}
    real = [g.get("name") for g in inf.get("gpus", [])
            if not g.get("context_only")]
    if real:
        return " + ".join(str(n) for n in real)
    host = env.get("host") or {}
    gpus = host.get("gpu") or []
    if gpus:
        return str(gpus[0].get("name", "?"))
    return str(host.get("cpu") or "—")


def _compute_label(env: dict) -> str:
    backend = env.get("backend") or {}
    rt = backend.get("runtime") or {}
    return str(rt.get("compute") or backend.get("kind") or "—")


def _bench_conditions(env: dict) -> dict:
    """Conditions that decide whether a comparison is valid (only when these match).

    Even with the same quantization label, a **different gguf** (requantized, by
    another author, a UD build) is a different model, so the model file name,
    fingerprint, speculative decoding, KV-cache quantization and the build are too.
    """
    backend = env.get("backend") or {}
    launch = backend.get("launch") or {}
    mf = backend.get("model_file") or {}
    model_path = backend.get("model_path") or mf.get("path")
    fingerprint = mf.get("sha256_head_tail")
    return {
        "Quantization": backend.get("quantization"),
        "Model": Path(str(model_path)).name if model_path else None,
        "Fingerprint": fingerprint[:12] if fingerprint else None,
        "-ngl": launch.get("n_gpu_layers"),
        "n_ctx": launch.get("n_ctx") or backend.get("n_ctx"),
        "Parallel": launch.get("parallel") or backend.get("parallel_slots"),
        "Speculative": launch.get("spec_type"),
        "KV-K": launch.get("cache_type_k"),
        "KV-V": launch.get("cache_type_v"),
        "build": backend.get("build_info"),
    }


def _fmt_conditions(cond: dict) -> str:
    parts = [f"{k} {v}" for k, v in cond.items() if v is not None]
    return " / ".join(parts) or "—"


def _task_set_warnings(rows: list[dict]) -> list[str]:
    """Compare the task_id sets of the compared runs; return warnings if they differ.

    Averages over different task sets cannot be compared (the run without hard
    tasks scores higher). With 0 common tasks the ranking itself does not hold.
    """
    if len(rows) < 2:
        return []
    sets = [set(r["results"]) for r in rows]
    common = set.intersection(*sets)
    union = set.union(*sets)
    if len(common) == len(union):
        return []
    if not common:
        return ["", "> 🚨 **The compared runs share not a single common task.** "
                    "Each score is an average over a different problem set, so "
                    "**the ranking is unusable even as a reference value**. "
                    "Re-measure with the same ledger and the same --only.", ""]
    missing = ", ".join(
        f"{r['label']} missing {len(union - set(r['results']))} tasks"
        for r in rows if set(r["results"]) != union
    )
    return ["", f"> ⚠️ **Task sets do not match** (total {len(union)} / "
                f"common {len(common)} — {missing}). "
                "Read the average comparison only over the common tasks.", ""]


def _env_signature(env: dict) -> str:
    """Key used to decide whether the runs came from the same environment.

    Host specs alone are not enough. **Even on the same machine**, a different
    device or compute backend is a different environment (real machine: one
    box holds RTX 5090 / RTX 3090 / Radeon 8060S and is measured switching
    CUDA / ROCm / Vulkan), so the conditions actually measured are in the key.
    """
    host = env.get("host") or {}
    backend = env.get("backend") or {}
    gpu = ",".join(str(g.get("name", "")) for g in (host.get("gpu") or []))
    cond = _bench_conditions(env)
    return "|".join(str(x) for x in (
        env.get("execution", ""), host.get("cpu", ""), gpu,
        host.get("ram_gb", ""), backend.get("kind", ""),
        _compute_label(env), _device_label(env),
        cond["Quantization"], cond["-ngl"], cond["n_ctx"], cond["Parallel"],
    ))


def is_hardware_comparison(rows: list[dict]) -> bool:
    """Decide whether this is a "same model measured on other hardware" run.

    There tok/s is the **main subject**, not "not comparable because the
    environments differ". Mixed models stay a model comparison as before.
    """
    envs = [r for r in rows if r.get("env")]
    if len(envs) < 2:
        return False
    models = {r["model"] for r in envs}
    sigs = {_env_signature(r["env"]) for r in envs}
    return len(models) == 1 and len(sigs) > 1


def _hardware_section(rows: list[dict]) -> list[str]:
    """Hardware comparison mode: model fixed, tok/s as the main subject.

    No Combined ranking. Every environment runs the same model, so Combined is
    nearly tied and neither the rank nor "Relative" means anything (ties keep
    the input order, so fast environments land at the bottom and mislead). Keep
    the quality columns as a sanity check for "same result in every environment".
    """
    envs = [r for r in rows if r.get("env")]
    ranked = sorted(envs, key=lambda r: r["tps"] or 0, reverse=True)
    best = ranked[0]["tps"] or 0
    lines = [
        "", "## 🖥 Ranking (tok/s descending)", "",
        f"Model `{ranked[0]['model']}` measured on {len(envs)} environments. "
        "Speed is the main subject, so rows are sorted by tok/s descending "
        "(Resolved / quality matching across environments is normal).", "",
        "| # | Device | Compute backend | tok/s | Relative "
        "| Resolved | Quality | Inference conditions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        env = r["env"]
        tps = r["tps"]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "")
        speed = f"{tps:.1f}" if tps is not None else "—"
        rel = f"{tps / best * 100:.0f}%" if tps and best else "—"
        lines.append(
            f"| {i}{medal} | **{_device_label(env)}** | {_compute_label(env)} "
            f"| {speed} | {rel} "
            f"| {r['resolved'] * 100:.1f}% | {r['quality']:.1f} "
            f"| {_fmt_conditions(_bench_conditions(env))} |"
        )

    # ★ A speed comparison holds only when the inference conditions match.
    #   Measured once: comparing -ngl 0 (CPU execution) with -ngl 99 reversed the conclusion.
    conds = [_bench_conditions(r["env"]) for r in envs]
    differing = sorted({k for k in conds[0]
                        if len({str(c.get(k)) for c in conds}) > 1})
    if differing:
        lines += ["", f"> ⚠️ **Inference conditions do not match ({', '.join(differing)}).** "
                      "In this state a tok/s gap is not a hardware gap. "
                      "In particular, a run with `-ngl` 0 does not use the GPU."]
    else:
        lines += ["", "> ✅ Inference conditions (quantization / -ngl / n_ctx / parallel) "
                      "match in all environments. The tok/s gap is the difference of hardware and backend."]
    if len(envs) < len(rows):
        lines += ["", "> ⚠️ Results with no recorded execution environment are "
                      "excluded from the table (generated by an older version)."]
    lines.append("")
    return lines


def _env_compare_section(rows: list[dict]) -> list[str]:
    """List the execution environment of each model and state whether tok/s is comparable."""
    envs = [r for r in rows if r.get("env")]
    if not envs:
        return ["", "> ⚠️ Some results have no recorded execution environment "
                    "(generated by an older version). Compare tok/s with care."]
    # Same model × other hardware: the goal is a hardware comparison, not "not comparable"
    if is_hardware_comparison(rows):
        return _hardware_section(rows)
    lines = ["", "## 🖥 Execution environment", "",
             "| Model | Execution form | Hardware / inference setup |", "|---|---|---|"]
    for r in rows:
        env = r.get("env") or {}
        if not env:
            lines.append(f"| {r['model']} | — | (not recorded) |")
            continue
        label = EXEC_LABEL.get(env.get("execution", ""), env.get("execution", "?"))
        lines.append(
            f"| {r['model']} | {label} | {format_summary(env) or '—'} |"
        )
    sigs = {_env_signature(r["env"]) for r in envs}
    local = [r for r in envs if (r["env"].get("execution") == "local")]
    if len(sigs) > 1:
        lines += ["", "> ⚠️ **Measurement environments do not match**. tok/s depends on the "
                      "environment, so a direct speed comparison is invalid (quality and Resolved comparisons remain valid)."]
    elif local and len(local) == len(envs):
        lines += ["", "> ✅ All models were measured in the same environment "
                      "(a direct tok/s comparison is possible)."]
    if len(envs) < len(rows):
        lines += ["", "> ⚠️ Some results have no recorded execution environment "
                      "(generated by an older version)."]
    lines.append("")
    return lines


def render_comparison(runs: list[dict]) -> str:
    if not runs:
        return "# Model Comparison\n\n(No results available)\n"

    # results with 0 items only add columns and never help the comparison,
    # so drop them (aborted runs and empty runs slip in easily via globs)
    empty = [r for r in runs if not r.get("results")]
    runs = [r for r in runs if r.get("results")]
    if not runs:
        return "# Model Comparison\n\n(No results available)\n"

    # Extract the per-model summary and rank by combined score descending
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
            "tps": _run_tps(run),
            "usability": s.get("usability", {}),
            "env": run.get("environment") or {},
            "results": {r["task_id"]: r for r in run.get("results", [])},
        })
    rows.sort(key=lambda x: x["combined"], reverse=True)
    best = rows[0]["combined"] or 1.0
    any_multi = any(r["runs"] > 1 for r in rows)

    # In a same-model × other-hardware comparison every row carries the same
    # model name and cannot be told apart, so switch the display label to the device name.
    hardware = is_hardware_comparison(rows)
    for r in rows:
        env = r.get("env") or {}
        r["label"] = (
            f"{_device_label(env)} ({_compute_label(env)})" if hardware and env
            else r["model"]
        )
    # The same device × backend is measured more than once, so number duplicate
    # labels (otherwise the matrix columns cannot be told apart)
    seen: dict[str, int] = {}
    dup = {lbl for lbl in (r["label"] for r in rows)
           if list(r["label"] for r in rows).count(lbl) > 1}
    for r in rows:
        if r["label"] in dup:
            seen[r["label"]] = seen.get(r["label"], 0) + 1
            r["label"] = f"{r['label']} #{seen[r['label']]}"

    lines = [
        "# 🆚 Hardware Comparison Report" if hardware else "# 🆚 Model Comparison Report",
        "",
        (f"model `{rows[0]['model']}` / environments: {len(rows)}"
         if hardware else f"Models: {len(rows)}")
        + f" / Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    if empty:
        names = ", ".join(f"`{r.get('_path', r.get('model', '?'))}`"
                          for r in empty)
        lines += [f"> ⚠️ Excluded (0 results): {names}", ""]

    # Cross-check task sets (are we ranking averages over different problem sets?)
    lines += _task_set_warnings(rows)

    # In hardware comparison the Combined scores tie, so ranks are meaningless:
    # drop the ranking (fast environments would line up below and mislead).
    # The speed ranking replaces it.
    if hardware:
        return "\n".join(lines + _hardware_section(rows)
                         + _usability_and_matrix(rows, runs)) + "\n"

    lines += ["## Ranking (Descending by Avg Combined)", ""]
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
            f"| {i}{medal} | **{r['label']}** | {r['lang']} "
            f"| {r['resolved'] * 100:.1f}% {mid}"
            f"| {r['quality']:.1f} | {r['combined']:.1f} "
            f"| {rel:.0f}% | {tps} |"
        )

    # Show the execution environment (tok/s cannot be compared across
    # measurement environments, so this is required)
    lines += _env_compare_section(rows)
    lines += _usability_and_matrix(rows, runs)
    lines += ["", "> Relative = each model's Combined ÷ the best model's Combined.",
              "> Placing reference models (strong/weak) side by side shows where local models stand."]
    return "\n".join(lines) + "\n"


def _usability_and_matrix(rows: list[dict], runs: list[dict]) -> list[str]:
    """Usability tier comparison + per-task Combined matrix (shared by both modes)."""
    lines = ["", "## Usability Tier Comparison", "",
             "| Model | 🟢 Autonomous | 🟡 Assisted | 🔴 Unusable |", "|---|---|---|---|"]
    for r in rows:
        u = r["usability"] or {}
        lines.append(
            f"| {r['label']} | {u.get('autonomous', 0)} "
            f"| {u.get('assisted', 0)} | {u.get('unusable', 0)} |"
        )

    # Per-Task Matrix (cell = combined, row max in bold)
    idx = _task_index(runs)
    lines += ["", "## Per-Task Combined Matrix", "",
              "Each cell is the Combined score for that task. The highest value in each row is **bolded**.", ""]
    lines += ["| Task | Difficulty | " + " | ".join(r["label"] for r in rows) + " |",
              "|---|---|" + "---|" * len(rows)]
    for tid in sorted(idx):
        diff = idx[tid]["difficulty"]
        cells = []
        vals = [(r["results"].get(tid) or {}).get("combined") for r in rows]
        bestval = max([v for v in vals if v is not None], default=None)
        for v in vals:
            if v is None:
                cells.append("—")
            elif bestval is not None and v == bestval:
                cells.append(f"**{v:.0f}**")
            else:
                cells.append(f"{v:.0f}")
        lines.append(f"| {tid} | {diff} | " + " | ".join(cells) + " |")
    return lines


def save_comparison(
    runs: list[dict], output_dir: Path, name: str = "comparison"
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{name}_{stamp}.md"
    out.write_text(render_comparison(runs), encoding="utf-8")
    return out

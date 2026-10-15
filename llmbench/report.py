"""Markdown report generation."""

from __future__ import annotations

from . import usability
from .env import EXEC_LABEL


def _fmt_ruff(c: dict) -> str:
    issues = c.get("issues", 0)
    if issues == 0:
        return f"✅ No issues found ({c.get('loc', 0)} LOC)"
    rules = ", ".join(c.get("rules", [])) or "-"
    return (
        f"⚠️ {issues} issue(s) / {c.get('loc', 0)} LOC "
        f"(density {c.get('density_per_100loc', 0)}/100 lines, rules: {rules})  "
        f"score={c.get('score', 0):.0f}"
    )


def _fmt_complexity(c: dict) -> str:
    rank = c.get("worst_cc_rank", "?")
    worst = c.get("worst_function") or "-"
    mi = c.get("maintainability_index", 0)
    line = (
        f"MI={mi:.0f} (Maintainability: higher is better, out of 100) / "
        f"Worst complexity rank={rank} (A=good to F=bad) → "
        f"score={c.get('score', 0):.0f} (higher is better, out of 100)"
    )
    if worst != "-":
        line += f"  (Most complex: {worst})"
    return line


def _fmt_component(name: str, c: dict) -> str:
    if name == "ruff":
        return _fmt_ruff(c)
    if name == "complexity":
        return _fmt_complexity(c)
    # Generic fallback
    detail = ", ".join(f"{k}={v}" for k, v in c.items() if k != "enabled")
    return detail


def _env_section(env: dict) -> list[str]:
    """Execution environment section (hardware + inference backend setup).

    tok/s changes several-fold with quantization, GPU offload ratio and n_ctx,
    even on the same GPU, so the backend setup is shown next to the specs. For
    a cloud API, state that the host specs do not affect speed, to prevent misreading.
    """
    if not env:
        return []
    execution = env.get("execution", "unknown")
    host = env.get("host") or {}
    backend = env.get("backend") or {}
    lines = ["## 🖥 Execution environment", "",
             f"**{EXEC_LABEL.get(execution, execution)}**"]
    if env.get("note"):
        lines += ["", f"> {env['note']}"]
    rows: list[tuple[str, str]] = []

    # ── Host ──
    if host.get("cpu"):
        cores = host.get("cpu_cores")
        detail = f" ({cores} cores)" if cores else ""
        if host.get("cpu_cores_perf") and host.get("cpu_cores_eff"):
            detail = (f" ({cores} threads / P{host['cpu_cores_perf']}"
                      f"+E{host['cpu_cores_eff']})")
        rows.append(("CPU", f"{host['cpu']}{detail}"))
    for g in host.get("gpu") or []:
        spec = []
        if g.get("cores"):
            spec.append(f"{g['cores']} cores")
        if g.get("vram_gb"):
            spec.append(f"VRAM {g['vram_gb']}GB")
        elif g.get("vram"):
            spec.append(f"VRAM {g['vram']}")
        if g.get("compute_capability"):
            spec.append(f"CC {g['compute_capability']}")
        if g.get("driver"):
            spec.append(f"driver {g['driver']}")
        if g.get("metal"):
            spec.append(g["metal"])
        rows.append(("GPU", f"{g.get('name', '?')}"
                     + (f" — {' / '.join(spec)}" if spec else "")))
    if host.get("ram_gb"):
        unified = " (unified memory)" if host.get("unified_memory") else ""
        rows.append(("Memory", f"{host['ram_gb']} GB{unified}"))
    if host.get("cuda"):
        rows.append(("CUDA", host["cuda"]))
    if host.get("os"):
        rows.append(("OS", f"{host['os']} / {host.get('arch', '?')} / "
                           f"Python {host.get('python', '?')}"))

    # ── Inference backend (affects speed only for local inference) ──
    if backend.get("kind"):
        rows.append(("Inference backend", str(backend["kind"])))
    # Compute backend (CUDA/ROCm/Vulkan…). llama.cpp exposes no API for it, so
    # this is derived from the libraries loaded by the inference process.
    rt = backend.get("runtime") or {}
    if rt.get("compute"):
        val = str(rt["compute"])
        val += " (from config)" if rt.get("source") == "config" else " (auto-detected)"
        if rt.get("evidence"):
            val += f" — {', '.join(rt['evidence'][:3])}"
        if rt.get("mismatch"):
            val += (f"  ⚠️ Mismatch: the detected value is **{rt.get('detected')}** "
                    "— the runtime: in config may not match reality")
        rows.append(("Compute backend", val))
    if rt.get("binary"):
        rows.append(("Binary", f"`{rt['binary']}`"))
    if backend.get("server_version"):
        rows.append(("Server", str(backend["server_version"])))
    if backend.get("quantization"):
        rows.append(("Quantization", str(backend["quantization"])))
    if backend.get("parameter_size"):
        rows.append(("Parameter count", str(backend["parameter_size"])))
    if backend.get("weights_gb"):
        rows.append(("Model size", f"{backend['weights_gb']} GB"))
    ratio = backend.get("gpu_offload_ratio")
    if ratio is not None:
        vr = backend.get("vram_resident_gb")
        mark = "✅" if ratio >= 0.999 else "⚠️"
        note = "" if ratio >= 0.999 else " ← tok/s drops sharply because part runs on CPU"
        rows.append((
            "GPU offload ratio",
            f"{mark} {ratio * 100:.0f}%"
            + (f" (VRAM resident {vr} GB)" if vr is not None else "")
            + note,
        ))
    # Inference setup read from the launch arguments. llama.cpp keeps no GPU
    # offload info in /props, so -ngl and --device are the effective deciders.
    launch = backend.get("launch") or {}
    if launch.get("device"):
        val = str(launch["device"])
        if launch.get("device_name"):
            val += f" — {launch['device_name']}"
            if launch.get("device_name_uncertain"):
                val += (" ⚠️ Estimated from the enumeration order "
                        "(CUDA indexes do not match the nvidia-smi order)")
            elif launch.get("device_name_source"):
                val += f" (confirmed by {launch['device_name_source']})"
        rows.append(("Device used", val))
    ngl = launch.get("n_gpu_layers")
    if ngl is not None:
        if ngl == 0:
            rows.append(("GPU offload", "⚠️ `-ngl 0` — **no layer is on the GPU "
                                        "(CPU execution in effect)**. tok/s measures CPU performance"))
        else:
            rows.append(("GPU offload", f"`-ngl {ngl}`"))
    for key, label in (("tensor_split", "tensor split"),
                       ("main_gpu", "main GPU"), ("split_mode", "split mode"),
                       ("threads", "threads"), ("batch_size", "batch"),
                       ("spec_type", "speculative decoding")):
        if launch.get(key) is not None:
            rows.append((label, f"`{launch[key]}`"))
    if launch.get("visible_devices_env"):
        rows.append(("Visible device limit", ", ".join(
            f"`{k}={v}`" for k, v in launch["visible_devices_env"].items())))

    # GPUs the inference process actually ran on (multi-GPU sharded loads show here)
    inf = (backend.get("gpu_usage") or {}).get("inference") or {}
    if inf.get("gpus"):
        shards = " + ".join(
            f"GPU{g.get('index')} {g.get('name', '?')} {g.get('vram_gb', 0)}GB"
            + ("(context only)" if g.get("context_only") else "")
            for g in inf["gpus"]
        )
        val = shards
        if inf.get("multi_gpu"):
            val += (f" — {inf.get('vram_total_gb', 0)}GB in total, sharded over "
                    f"{len(inf['gpus'])} cards")
        if inf.get("process"):
            val += f" ({inf['process']})"
        if inf.get("uncertain"):
            val += " ※ Could not identify the inference process; showing the process with the largest VRAM use"
        rows.append(("GPUs used", val))
    if backend.get("n_ctx"):
        rows.append(("Context length", f"{backend['n_ctx']:,} tok"))
    if backend.get("parallel_slots"):
        rows.append(("Parallel slots", str(backend["parallel_slots"])))
    if backend.get("model_path"):
        rows.append(("Model file", f"`{backend['model_path']}`"))
    if backend.get("base_url"):
        rows.append(("Endpoint", f"`{backend['base_url']}`"))
    if launch.get("command"):
        rows.append(("Launch command", f"`{launch['command']}`"))
    if backend.get("note"):
        rows.append(("Note", str(backend["note"])))
    if env.get("collect_error") or backend.get("collect_error"):
        rows.append(("⚠️ Collection error",
                     str(env.get("collect_error")
                         or backend.get("collect_error"))))

    if rows:
        lines += ["", "| Item | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
    if launch.get("n_gpu_layers") == 0:
        lines += ["", "> ⚠️ With `-ngl 0` **the model is not on the GPU** "
                      "(CPU execution in effect, whatever the compute backend shows). "
                      "Do not compare it side by side with GPU execution results."]
    if inf.get("multi_gpu"):
        lines += ["", "> ⚠️ The model is **sharded over multiple GPUs**. "
                      "Throughput is limited by the slower card and the GPU-to-GPU transfer, "
                      "so it cannot be equated with a single-GPU measurement."]
    if execution != "local":
        lines += ["", "> ⚠️ The host above is the **measuring client**'s specs. "
                       "Inference runs remotely, so do not compare its tok/s "
                       "side by side with local execution results."]
    lines.append("")
    return lines


def _status_icon(r) -> str:
    return "✅" if r.resolved else "❌"


def _tier_cell(r) -> str:
    return usability.TIER_LABEL.get(r.usability_tier, "—")


def _usability_section(run) -> list[str]:
    """Usability evaluation section (tier aggregation, difficulty x tier, overall assessment)."""
    overall, by_diff = usability.aggregate(run.results)
    n = len(run.results)
    lines = ["## 🧭 Usability Assessment (Actual delegation capability)", ""]
    for t in usability.TIERS:
        cnt = overall.get(t, 0)
        pct = (cnt / n * 100) if n else 0
        lines.append(
            f"- {usability.TIER_LABEL[t]} **{cnt}/{n}** ({pct:.0f}%) "
            f"… {usability.TIER_DESC[t]}"
        )
    # Difficulty x Tier Matrix
    lines += ["", "| Difficulty | 🟢 Autonomous | 🟡 Assisted | 🔴 Unusable |", "|---|---|---|---|"]
    for diff in ("easy", "medium", "hard", "expert", "frontier", "architect", "grandmaster"):
        if diff not in by_diff:
            continue
        c = by_diff[diff]
        lines.append(
            f"| {diff} | {c.get('autonomous', 0)} | "
            f"{c.get('assisted', 0)} | {c.get('unusable', 0)} |"
        )
    # Breakdown by difficulty (percentage, explicitly showing distribution to avoid optimistic rounding)
    def _pct(c, key):
        tot = sum(c.values()) or 1
        return c.get(key, 0) / tot * 100

    lines += ["", "**Breakdown by Difficulty (Percentage)**", ""]
    for diff in ("easy", "medium", "hard", "expert", "frontier", "architect", "grandmaster"):
        if diff not in by_diff:
            continue
        c = by_diff[diff]
        lines.append(
            f"- {diff}: 🟢Autonomous {_pct(c, 'autonomous'):.0f}% / "
            f"🟡Assisted {_pct(c, 'assisted'):.0f}% / "
            f"🔴Unusable {_pct(c, 'unusable'):.0f}%"
        )

    # Conservative overall recommendation (if any unusable tasks exist, do not claim full autonomy)
    un = overall.get("unusable", 0)
    assisted = overall.get("assisted", 0)
    un_pct = (un / n * 100) if n else 0
    if un == 0 and assisted == 0:
        rec = "Operable mostly autonomously"
    elif un_pct < 10:
        rec = f"Mostly autonomous. Note that 🔴Unusable {un}/{n} ({un_pct:.0f}%) requires attention"
    elif un_pct < 30:
        rec = f"Assisted operation recommended (🔴Unusable {un}/{n} = {un_pct:.0f}%)"
    else:
        rec = f"Limited applicability for this task suite (🔴Unusable {un}/{n} = {un_pct:.0f}%)"
    lines += ["", f"> Overall Recommendation: {rec}"]
    lines.append("")
    return lines


def render_markdown(run) -> str:
    n = len(run.results)
    n_resolved = sum(r.resolved for r in run.results)
    multi = getattr(run, "multi_run", False)
    served = getattr(run, "served_model", "")
    lines = [
        f"# llmbench Report: {run.model}",
        "",
        f"Issue Language: `{run.issue_lang}` / Tasks: {n}"
        + (f" / Trials: ×{run.runs}" if multi else "")
        + (f" / Served model: `{served}`" if served and served != run.model else ""),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ✅ Resolved Rate | **{run.resolved_rate * 100:.1f}%** "
        f"({n_resolved}/{n}) |",
    ]
    if multi:
        n_any = sum(1 for r in run.results if r.n_pass > 0)
        lines += [
            f"| 🎲 Avg Success Rate (pass@1, ×{run.runs}) | "
            f"**{run.avg_success_rate * 100:.1f}%** |",
            f"| 🔁 Tasks Solved ≥1 Time | "
            f"**{run.solved_any_rate * 100:.1f}%** ({n_any}/{n}) |",
        ]
    lines += [
        f"| 🏅 Avg Quality (resolved only) | **{run.avg_quality_resolved:.1f} / 100** |",
        f"| 🎯 Avg Combined | **{run.avg_combined:.1f} / 100** |",
        "",
    ]
    if multi:
        lines += [
            "> Success Rate (pass@1) = Expected value for a single trial = **Primary Reliability Metric**."
            "≥1 Solved = Passed at least 1 out of N trials (reachability with retries).",
            "",
        ]
    # Execution environment (which hardware and configuration produced this)
    lines += _env_section(getattr(run, "environment", None) or {})

    # Usability assessment (directly under summary)
    lines += _usability_section(run)

    if run.artifacts_dirname:
        lines += [
            f"> 📂 Artifacts (raw LLM output, generated code, pytest output) saved in "
            f"`{run.artifacts_dirname}/<task_id>/`.",
            "> Check there for causes of parse or test failures.",
            "",
        ]

    # Results by Task
    rel_h = "| Reliability " if multi else ""
    rel_sep = "---|" if multi else ""
    lines += [
        "## Task Results",
        "",
        f"| | Task | Difficulty | Tier {rel_h}| Generated Files | Quality | Combined "
        "| Execution Time | tok/s | Notes |",
        f"|---|---|---|---|{rel_sep}---|---|---|---|---|---|",
    ]
    for r in run.results:
        files = ", ".join(f"`{f}`" for f in r.changed_files) or "—"
        note = r.fail_reason or (r.parse_error if not r.parse_ok else "") or "—"
        rel_c = (
            f"| {r.n_pass}/{r.runs} (Success Rate {r.success_rate * 100:.0f}%) "
            if multi else ""
        )
        lines.append(
            f"| {_status_icon(r)} | {r.task_id} | {r.difficulty} "
            f"| {_tier_cell(r)} {rel_c}"
            f"| {files} "
            f"| {r.quality_score:.0f} | {r.combined:.0f} "
            f"| {r.latency_sec:.1f}s "
            f"| {r.tokens_per_sec if r.tokens_per_sec else '-'} "
            f"| {note} |"
        )

    # Breakdown by Difficulty
    lines += ["", "## By Difficulty", "",
              "| Difficulty | Resolved | Avg Quality |", "|---|---|---|"]
    for diff in ("easy", "medium", "hard", "expert", "frontier", "architect", "grandmaster"):
        rs = [r for r in run.results if r.difficulty == diff]
        if not rs:
            continue
        solved = sum(r.resolved for r in rs)
        qs = [r.quality_score for r in rs if r.resolved]
        avg_q = f"{sum(qs) / len(qs):.0f}" if qs else "-"
        lines.append(f"| {diff} | {solved}/{len(rs)} | {avg_q} |")

    # Breakdown by Domain (only when non-coding graders are mixed)
    domains = [d for d in ("security", "general", "writing", "medical", "culture",
                           "uncensored")
               if any(getattr(r, "domain", "code") == d for r in run.results)]
    if domains:
        _dlabel = {"security": "🛡️ security", "general": "📋 general",
                   "writing": "✍️ writing", "medical": "🩺 medical",
                   "culture": "🇯🇵 culture", "uncensored": "🔓 uncensored"}
        lines += ["", "## 🌐 By Domain (Non-coding)", "",
                  "| Domain | Resolved | Avg Success Rate | Avg Combined | Refusals |",
                  "|---|---|---|---|---|"]
        for d in domains:
            rs = [r for r in run.results if getattr(r, "domain", "code") == d]
            solved = sum(r.resolved for r in rs)
            sr = sum(r.success_rate for r in rs) / len(rs) * 100
            cb = sum(r.combined for r in rs) / len(rs)
            # A refusal is "did not answer", not "does not know". Report it apart from the accuracy.
            nref = sum(1 for r in rs if getattr(r, "refused", False))
            lines.append(
                f"| {_dlabel[d]} | {solved}/{len(rs)} | {sr:.0f}% | {cb:.1f} | "
                f"{nref}/{len(rs)} |"
            )

    # Task Details (artifacts and quality breakdown)
    lines += ["", "## Task Details", ""]
    for r in run.results:
        lines.append(f"### {_status_icon(r)} {r.task_id} — {r.title}")
        lines.append("")
        lines.append(
            f"- Difficulty: {r.difficulty}"
            + (f" / category: {r.category}" if getattr(r, "category", "") else "")
            + f" / Status: {'RESOLVED' if r.resolved else 'FAILED'}"
            + (f" ({r.fail_reason})" if r.fail_reason else "")
            + f" / Usability: {usability.TIER_LABEL.get(r.usability_tier, '—')}"
        )
        if multi:
            anymark = "✓" if r.n_pass > 0 else "✗"
            lines.append(
                f"- Reliability: Passed {r.n_pass}/{r.runs} "
                f"(Success Rate {r.success_rate * 100:.0f}% = pass@1) / "
                f"Passed ≥1 in {r.runs} trials: {anymark}"
            )
        if not r.parse_ok:
            lines.append(f"- ⚠️ Parse Failure: {r.parse_error or 'Unknown'}")
        if r.changed_files:
            lines.append(f"- Generated Files: {', '.join(r.changed_files)}")
        if run.artifacts_dirname:
            lines.append(
                f"- Artifacts: `{run.artifacts_dirname}/{r.task_id}/`"
            )
        # Quality breakdown (explicitly note that values reflect 1 representative trial during multi-run)
        if r.quality_components and any(
            c.get("enabled") for c in r.quality_components.values()
        ):
            if multi:
                lines.append(
                    f"- Quality Breakdown (values below are for **1 representative trial**."
                    f" The Quality {r.quality_score:.0f} above is the average over the"
                    f" {r.n_pass}/{r.runs} successful trials;"
                    f" failed trials are excluded from the denominator):"
                )
            for name, comp in r.quality_components.items():
                if not comp.get("enabled"):
                    continue
                prefix = "  - " if multi else "- "
                lines.append(f"{prefix}{name}: {_fmt_component(name, comp)}")
        lines.append("")

    return "\n".join(lines) + "\n"

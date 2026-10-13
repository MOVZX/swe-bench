"""Markdown report generation."""

from __future__ import annotations

from . import usability


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
    lines = [
        f"# llmbench Report: {run.model}",
        "",
        f"Issue Language: `{run.issue_lang}` / Tasks: {n}"
        + (f" / Trials: ×{run.runs}" if multi else ""),
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
    domains = [d for d in ("security", "general", "writing", "medical")
               if any(getattr(r, "domain", "code") == d for r in run.results)]
    if domains:
        _dlabel = {"security": "🛡️ security", "general": "📋 general",
                   "writing": "✍️ writing", "medical": "🩺 medical"}
        lines += ["", "## 🌐 By Domain (Non-coding)", "",
                  "| Domain | Resolved | Avg Success Rate | Avg Combined |", "|---|---|---|---|"]
        for d in domains:
            rs = [r for r in run.results if getattr(r, "domain", "code") == d]
            solved = sum(r.resolved for r in rs)
            sr = sum(r.success_rate for r in rs) / len(rs) * 100
            cb = sum(r.combined for r in rs) / len(rs)
            lines.append(
                f"| {_dlabel[d]} | {solved}/{len(rs)} | {sr:.0f}% | {cb:.1f} |"
            )

    # Task Details (artifacts and quality breakdown)
    lines += ["", "## Task Details", ""]
    for r in run.results:
        lines.append(f"### {_status_icon(r)} {r.task_id} — {r.title}")
        lines.append("")
        lines.append(
            f"- Difficulty: {r.difficulty} / "
            f"Status: {'RESOLVED' if r.resolved else 'FAILED'}"
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
                    f"- Quality Breakdown (Values below are for **1 representative trial**."
                    f" The Quality score of {r.quality_score:.0f} above is the average across {r.runs} trials):"
                )
            for name, comp in r.quality_components.items():
                if not comp.get("enabled"):
                    continue
                prefix = "  - " if multi else "- "
                lines.append(f"{prefix}{name}: {_fmt_component(name, comp)}")
        lines.append("")

    return "\n".join(lines) + "\n"

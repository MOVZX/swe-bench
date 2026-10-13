"""llmbench CLI.

Examples:
    llmbench list-tasks
    llmbench run --model local-ollama
    llmbench run --model local-openai --tasks t001,t003 --lang ja
    llmbench validate          # Self-verify using gold/broken mocks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .clients.ollama import list_ollama_models
from .runner import BenchmarkRunner, save_run
from .tasks import load_tasks


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tasks-dir", default="tasks")
    parser.add_argument(
        "--tasks", default=None,
        help="Comma-separated task IDs (e.g., t001,t003). Defaults to all tasks if omitted",
    )
    parser.add_argument(
        "--with-l6", action="store_true", dest="with_l6",
        help="Include 20 additional L6 (architect tier) tasks in addition to the default 40",
    )
    parser.add_argument(
        "--l6-ledger", default="tasks_l6.jsonl", dest="l6_ledger",
        help="Filename of the additional L6 ledger (inside tasks-dir. Default: tasks_l6.jsonl)",
    )
    parser.add_argument(
        "--with-l7", action="store_true", dest="with_l7",
        help="Include 40 additional L7 (grandmaster tier) tasks in addition to default 40 (+L6)",
    )
    parser.add_argument(
        "--l7-ledger", default="tasks_l7.jsonl", dest="l7_ledger",
        help="Filename of the additional L7 ledger (inside tasks-dir. Default: tasks_l7.jsonl)",
    )
    parser.add_argument(
        "--only-l6", action="store_true", dest="only_l6",
        help="Exclude default 40 tasks and run only the additional L6 (architect) ledger",
    )
    parser.add_argument(
        "--only-l7", action="store_true", dest="only_l7",
        help="Exclude default 40 tasks and run only the additional L7 (grandmaster) ledger",
    )
    # --- Domain ledgers (Non-coding) ---
    for dom, desc in (
        ("sec", "security detection/analysis"),
        ("gen", "general tasks/instruction following"),
        ("write", "creative writing (experimental)"),
        ("med", "medical QA (reference values)"),
    ):
        parser.add_argument(
            f"--with-{dom}", action="store_true", dest=f"with_{dom}",
            help=f"Append {desc} ledger (tasks_{dom}.jsonl) to default tasks",
        )
        parser.add_argument(
            f"--only-{dom}", action="store_true", dest=f"only_{dom}",
            help=f"Exclude default tasks and run only the {desc} ledger",
        )


# Domain name -> Ledger filename
_DOMAIN_LEDGERS = {
    "sec": "tasks_sec.jsonl",
    "gen": "tasks_gen.jsonl",
    "write": "tasks_write.jsonl",
    "med": "tasks_med.jsonl",
}


def _ledgers(args) -> list[str]:
    """Determine the list of target ledgers to execute.

    Priority rules:
      - If either `--only-l6` / `--only-l7` is specified, enter 'only mode'.
        Exclude default ledger tasks.jsonl and target only the requested tier ledger.
      - Even in 'only mode', `--with-l6`/`--with-l7` are respected as additional tier requests
        (effectively `--only-l6` and `--with-l6` are synonymous; no duplicate additions if both are set).
      - If no 'only' flag is specified, append to tasks.jsonl as usual.
    """
    only_l6 = getattr(args, "only_l6", False)
    only_l7 = getattr(args, "only_l7", False)
    with_l6 = getattr(args, "with_l6", False)
    with_l7 = getattr(args, "with_l7", False)

    only_dom = {d for d in _DOMAIN_LEDGERS if getattr(args, f"only_{d}", False)}
    with_dom = {d for d in _DOMAIN_LEDGERS if getattr(args, f"with_{d}", False)}

    only_mode = only_l6 or only_l7 or bool(only_dom)
    ledgers = [] if only_mode else ["tasks.jsonl"]
    if only_l6 or with_l6:
        ledgers.append(args.l6_ledger)
    if only_l7 or with_l7:
        ledgers.append(args.l7_ledger)
    for d in _DOMAIN_LEDGERS:
        if d in only_dom or d in with_dom:
            ledgers.append(_DOMAIN_LEDGERS[d])
    return ledgers


def cmd_list_tasks(args) -> int:
    tasks = load_tasks(Path(args.tasks_dir), ledgers=_ledgers(args))
    for t in tasks:
        print(f"{t.task_id}  [{t.difficulty:6s}]  {t.title}  ({len(t.files)} files)")
    print(f"total: {len(tasks)}")
    return 0


def cmd_models(args) -> int:
    """List available models (config defined + running Ollama models)."""
    config = _load_config(args.config)
    models = config.get("models", {})
    print("=== Models Defined in config.yaml ===")
    if models:
        for k, v in models.items():
            extra = f", model={v['model']}" if v.get("model") else ""
            print(f"  {k:14s} (type={v.get('type', '?')}{extra})")
    else:
        print("  (none)")

    if args.ollama_host:
        config.setdefault("run", {})["ollama_host"] = args.ollama_host
    host = BenchmarkRunner(config, Path(args.tasks_dir)).ollama_host()
    print(f"\n=== Running Ollama Models ({host}) ===")
    try:
        names = list_ollama_models(host)
    except Exception as e:
        print(f"  ⚠️ Could not connect to Ollama: {e}")
        print("  (Start Ollama to directly specify its models using --model)")
        return 0
    if names:
        for n in names:
            print(f"  {n}")
        print("  → Can be executed directly with `--model <name>` even if undefined in config")
    else:
        print("  (No installed models. Add using `ollama pull <model>`)")
    return 0


def cmd_run(args) -> int:
    config = _load_config(args.config)
    if args.lang:
        config.setdefault("run", {})["issue_lang"] = args.lang
    if getattr(args, "ollama_host", None):
        config.setdefault("run", {})["ollama_host"] = args.ollama_host
    runner = BenchmarkRunner(config, Path(args.tasks_dir), ledgers=_ledgers(args))
    only = args.tasks.split(",") if args.tasks else None
    try:
        run = runner.run(
            args.model, only_tasks=only,
            runs=args.runs, sample_temp=args.sample_temp,
            label=args.label, concurrency=args.concurrency,
            client_type=args.client_type, base_url=args.base_url,
        )
    except ValueError as e:  # Format errors cleanly for model resolution failures, etc.
        print(f"❌ {e}", file=sys.stderr)
        return 2
    json_path, md_path = save_run(run, Path(args.output))
    print()
    print(f"Resolved Rate   : {run.resolved_rate * 100:.1f}%")
    if run.multi_run:
        print(f"Avg Success Rate: {run.avg_success_rate * 100:.1f}% (pass@1 ×{run.runs})")
        print(f"≥1 Success Tasks: {run.solved_any_rate * 100:.1f}%")
    print(f"Avg Quality     : {run.avg_quality_resolved:.1f}/100 (resolved only)")
    print(f"Avg Combined    : {run.avg_combined:.1f}/100")
    print(f"Results: {json_path}")
    print(f"Report: {md_path}")
    return 0


def cmd_compare(args) -> int:
    """Generates a cross-comparison report across multiple results.json files."""
    from .compare import load_results, save_comparison

    paths = args.results
    if len(paths) < 2:
        print("⚠️  Recommending 2 or more results.json files for comparison.", file=sys.stderr)
    runs = load_results(paths)
    out = save_comparison(runs, Path(args.output), name=args.name)
    print(f"Comparison report: {out}")
    return 0


def cmd_certify(args) -> int:
    """Evaluates results.json against tier passing criteria and displays usability tier level."""
    import json

    from .certify import (
        certify, certify_domains, certify_medical, merge_results,
        render_certificate_md, render_domains_md, render_medical_md,
    )

    if getattr(args, "merge", False):
        model, results = merge_results(args.results)
        print(render_certificate_md(certify(results), model or "merged"))
        dom = render_domains_md(certify_domains(results))
        if dom:
            print("\n" + dom)
        med = render_medical_md(certify_medical(results))
        if med:
            print("\n" + med)
        print()
        return 0

    for path in args.results:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = data.get("results", [])
        print(render_certificate_md(certify(results), data.get("model", "")))
        dom = render_domains_md(certify_domains(results))
        if dom:
            print("\n" + dom)
        med = render_medical_md(certify_medical(results))
        if med:
            print("\n" + med)
        print()
    return 0


def cmd_validate(args) -> int:
    """Self-verify the entire pipeline using gold/broken mocks."""
    config = _load_config(args.config)
    config.setdefault("models", {})
    config["models"]["mock-gold"] = {"type": "mock", "mode": "gold"}
    config["models"]["mock-broken"] = {"type": "mock", "mode": "broken"}
    runner = BenchmarkRunner(config, Path(args.tasks_dir), ledgers=_ledgers(args))
    only = args.tasks.split(",") if args.tasks else None

    print("=== mock-gold (All tasks should be resolved) ===")
    gold = runner.run("mock-gold", only_tasks=only)
    print("\n=== mock-broken (All tasks should fail) ===")
    broken = runner.run("mock-broken", only_tasks=only)

    n = len(gold.results)
    ok_gold = sum(r.resolved for r in gold.results)
    ok_broken = sum(not r.resolved for r in broken.results)
    print(f"\ngold  : {ok_gold}/{n} resolved (Expected: {n}/{n})")
    print(f"broken: {ok_broken}/{n} failed   (Expected: {n}/{n})")
    print(f"gold combined avg: {gold.avg_combined:.1f} (Expected: >50)")
    print(f"broken combined avg: {broken.avg_combined:.1f} (Expected: 0)")
    passed = ok_gold == n and ok_broken == n and broken.avg_combined == 0
    print("VALIDATION:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="Functional accuracy + quality benchmark for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-tasks", help="Display task list")
    _common_args(p_list)
    p_list.set_defaults(fn=cmd_list_tasks)

    p_models = sub.add_parser("models", help="List models (config + active Ollama)")
    _common_args(p_models)
    p_models.add_argument("--ollama-host", default=None, dest="ollama_host",
                          help="Ollama host address (Default: config value or http://localhost:11434)")
    p_models.set_defaults(fn=cmd_models)

    p_run = sub.add_parser("run", help="Run benchmark")
    _common_args(p_run)
    p_run.add_argument("--model", required=True, help="Model name defined in config.yaml")
    p_run.add_argument("--lang", choices=["en", "ja"], default=None,
                       help="Issue language (overrides config)")
    p_run.add_argument("--output", default="results", help="Output directory for results")
    p_run.add_argument(
        "--runs", type=int, default=None,
        help="Number of trials per task (>1 measures pass@k. Default: run.runs or 1)",
    )
    p_run.add_argument(
        "--sample-temp", type=float, default=None, dest="sample_temp",
        help="Sampling temperature for multiple trials (Default: config's run.sample_temp or 0.8)",
    )
    p_run.add_argument("--ollama-host", default=None, dest="ollama_host",
                       help="Ollama host address (Used for auto-resolving models not in config)")
    p_run.add_argument(
        "--base-url", default=None, dest="base_url",
        help="Explicitly specify target URL (overrides base_url in config. "
             "e.g., http://localhost:8085/v1)",
    )
    p_run.add_argument(
        "--client-type", default=None, dest="client_type",
        choices=["openai", "ollama", "multiagent"],
        help="Specify connection type directly when missing in config "
             "(openai=llama.cpp/vLLM etc., ollama, multiagent=CodeRouter). "
             "Used with --base-url",
    )
    p_run.add_argument("--label", default=None,
                       help="Explicitly specify result label (Default: server-detected name when model:auto)")
    p_run.add_argument(
        "--concurrency", type=int, default=None,
        help="Concurrency level for running trials in parallel (Default: run.concurrency or 1). "
             "Effective when llama.cpp is started with --parallel N -cb",
    )
    p_run.set_defaults(fn=cmd_run)

    p_cmp = sub.add_parser("compare", help="Cross-compare multiple results.json files")
    p_cmp.add_argument("results", nargs="+", help="results.json files to compare (2 or more)")
    p_cmp.add_argument("--output", default="results", help="Output directory for comparison report")
    p_cmp.add_argument("--name", default="comparison", help="Output filename prefix")
    p_cmp.set_defaults(fn=cmd_compare)

    p_val = sub.add_parser("validate", help="Self-verify using mock objects")
    _common_args(p_val)
    p_val.set_defaults(fn=cmd_validate)

    p_cert = sub.add_parser("certify", help="Evaluate usability tier based on pass criteria")
    p_cert.add_argument("results", nargs="+", help="results.json files to evaluate")
    p_cert.add_argument(
        "--merge", action="store_true",
        help="Combine results across multiple results.json files into a single certificate "
             "(Merged certificate for split runs like base40 + L6. Last win on duplicate task_ids)",
    )
    p_cert.set_defaults(fn=cmd_certify)

    args = parser.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()

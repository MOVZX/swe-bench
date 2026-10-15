"""llmbench CLI.

Examples:
    llmbench list-tasks
    llmbench models                                  # config definitions + running Ollama models
    llmbench models --remote opencode-go             # query /v1/models of a type:openai entry
                                                       # (model IDs a gateway such as OpenCode Go actually serves)
    llmbench run --model local-ollama
    llmbench run --model local-openai --tasks t001,t003 --lang ja
    llmbench preflight --model local-openai          # Check the config before running
    llmbench validate                                # Self-verify with gold/broken mocks
    llmbench certify results/<stamp>_<model>_results.json      # Usability-line verdict
    llmbench certify --merge results/base.json results/l7.json # Merge split runs into one verdict
    llmbench compare results/a_results.json results/b_results.json  # Cross comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .clients.base import expand_env
from .clients.ollama import list_ollama_models
from .clients.openai_compat import list_remote_models
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
        help="Include the 16 additional L7 (grandmaster tier) tasks in addition to the default 40 "
             "(L6 not included; add it with --with-l6 too)",
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
        help="Exclude the default 40 tasks and run only the additional L7 (grandmaster) ledger (16 tasks)",
    )
    # --- Domain ledgers (Non-coding) ---
    for dom, desc in (
        ("sec", "security detection/analysis"),
        ("secaug", "security perturbed variants (SecLLMHolmes-style augmentation)"),
        ("gen", "general tasks/instruction following"),
        ("write", "creative writing (experimental)"),
        ("med", "medical QA (reference values)"),
        ("culture", "culture: Japanese internet-meme knowledge (reference values)"),
        ("unc", "uncensored: over-refusal check (reference values)"),
    ):
        parser.add_argument(
            f"--with-{dom}", action="store_true", dest=f"with_{dom}",
            help=f"Append {desc} ledger (tasks_{dom}.jsonl) to default tasks",
        )
        parser.add_argument(
            f"--only-{dom}", action="store_true", dest=f"only_{dom}",
            help=f"Exclude the default tasks (tasks.jsonl) and run only the {desc} ledger "
                 f"(combine with other --with-* / --only-* flags to add those too)",
        )


# Domain name -> Ledger filename
_DOMAIN_LEDGERS = {
    "sec": "tasks_sec.jsonl",
    "secaug": "tasks_sec_aug.jsonl",
    "gen": "tasks_gen.jsonl",
    "write": "tasks_write.jsonl",
    "med": "tasks_med.jsonl",
    "culture": "tasks_culture.jsonl",
    "unc": "tasks_unc.jsonl",
}


def _ledgers(args) -> list[str]:
    """Determine the list of target ledgers to execute.

    Priority rules:
      - If any of `--only-l6` / `--only-l7` / `--only-{sec,gen,write,med,culture}` is given,
        enter 'only mode' (domain ledgers also start it via `--only-*`, not only tier ledgers).
        Exclude the default ledger tasks.jsonl and target only the requested ledgers.
        Giving several `--only-*` targets all of them.
      - Even in only mode, `--with-l6`/`--with-l7` are respected as same-tier additions
        (so `--only-l6` and `--with-l6` are effectively synonymous; no duplicate additions).
      - With no only flag at all, append to tasks.jsonl as before.
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
        cat = f"  {{{t.category}}}" if getattr(t, "category", "") else ""
        print(f"{t.task_id}  [{t.difficulty:6s}]  {t.title}{cat}  ({len(t.files)} files)")
    print(f"total: {len(tasks)}")
    return 0


def cmd_models(args) -> int:
    """List available models (config-defined + running Ollama models).

    Default: config-defined + Ollama models.
    With --remote <name>, fetch and show the model IDs actually served by the
    endpoint configured in config.yaml models.<name> (type: openai) via
    /v1/models (e.g. to inspect gateways like OpenCode Go that swap models).
    """
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
        print("  (Start Ollama to select its models directly with --model)")
    else:
        if names:
            for n in names:
                print(f"  {n}")
            print("  → Can be executed directly with `--model <name>` even if undefined in config")
        else:
            print("  (No installed models. Add some with `ollama pull <model>`)")

    if getattr(args, "remote", None):
        _print_remote_models(args.remote, models)
    return 0


def _print_remote_models(name: str, models: dict) -> None:
    """Body of `llmbench models --remote <name>`. Query /v1/models and show it."""
    entry = models.get(name)
    print(f"\n=== Models served by {name} (/v1/models) ===")
    if entry is None:
        print(f"  ⚠️ models.{name} is not defined in config.yaml")
        return
    if entry.get("type") != "openai":
        print(
            f"  ⚠️ models.{name} has type={entry.get('type')!r}. "
            " --remote supports only type: openai (an endpoint with an "
            "OpenAI-compatible /v1/models)"
        )
        return
    try:
        base_url = expand_env(entry.get("base_url", ""), where=f"models.{name}.base_url")
        api_key = expand_env(entry.get("api_key", ""), where=f"models.{name}.api_key")
    except ValueError as e:
        print(f"  ⚠️ {e}")
        return
    if not base_url:
        print(f"  ⚠️ models.{name} has no base_url")
        return
    try:
        items = list_remote_models(base_url, api_key)
    except Exception as e:
        print(f"  ⚠️ Cannot fetch {base_url}/models: {e}")
        return
    if not items:
        print("  (0 entries. The endpoint returned an empty list)")
        return
    for it in sorted(items, key=lambda x: (x.get("id") or x.get("model") or "")):
        model_id = it.get("id") or it.get("model") or "?"
        owner = it.get("owned_by")
        extra = f"  (owned_by={owner})" if owner else ""
        print(f"  {model_id}{extra}")
    current = entry.get("model")
    print(
        f"  → current config.yaml: model: \"{current}\" "
        "(switch by editing it to an ID from the list above)"
    )


def _preflight_gate(args, config: dict) -> int | None:
    """Verify the config before running the bench. Do not run on FAIL.

    In 2026-08 we ran 16 runs (9.3 hours) at temperature 0.2 and had to measure
    again. The cause was not knowledge but "leaving the check to a human", so
    stop here to make it bite even when forgotten.

    - FAIL: do not run (can be ignored with --skip-preflight)
    - WARN/INFO: show and continue
    - Not a single token is generated, so the run time is almost zero
    """
    if getattr(args, "skip_preflight", False):
        return None
    try:
        from .preflight import run_preflight
        report = run_preflight(config, args.model)
    except Exception as e:      # do not stop the bench because preflight itself failed
        print(f"⚠️  Could not run preflight ({e}). Continuing", file=sys.stderr)
        return None
    if report.worst in ("WARN", "FAIL"):
        print(report.render(), file=sys.stderr)
        print(file=sys.stderr)
    if report.worst == "FAIL":
        print("❌ preflight is FAIL. Fix the config and run again.",
              file=sys.stderr)
        print("   To run with this verdict accepted, add --skip-preflight.",
              file=sys.stderr)
        return 2
    return None


def cmd_run(args) -> int:
    config = _load_config(args.config)
    if args.lang:
        config.setdefault("run", {})["issue_lang"] = args.lang
    if getattr(args, "ollama_host", None):
        config.setdefault("run", {})["ollama_host"] = args.ollama_host
    gate = _preflight_gate(args, config)
    if gate is not None:
        return gate
    runner = BenchmarkRunner(config, Path(args.tasks_dir), ledgers=_ledgers(args))
    only = args.tasks.split(",") if args.tasks else None
    try:
        run = runner.run(
            args.model, only_tasks=only,
            runs=args.runs, sample_temp=args.sample_temp,
            label=args.label, concurrency=args.concurrency,
            client_type=args.client_type, base_url=args.base_url,
            generate_retries=args.generate_retries,
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


def _certify_gates(args) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """Read the certify gate settings from config.yaml (never crash on failure).

    The benefit of certify is that it works with only results.json, so fall
    back to the default gates when config is missing or broken.
    """
    path = Path(getattr(args, "config", "config.yaml") or "config.yaml")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"⚠️  Cannot read config, using the default gates ({path}: {e})",
              file=sys.stderr)
        return None, None, None, None
    dom = cfg.get("certify_domains") or None
    med = cfg.get("certify_medical") or None
    cul = cfg.get("certify_culture") or None
    unc = cfg.get("certify_uncensored") or None
    return dom, med, cul, unc


def cmd_certify(args) -> int:
    """Evaluates results.json against tier passing criteria and displays usability tier level."""
    import json

    from .certify import (
        certify, certify_culture, certify_domains, certify_medical,
        certify_uncensored, merge_results, render_certificate_md,
        render_culture_md, render_domains_md, render_medical_md,
        render_uncensored_md,
    )

    dom_gates, med_gates, cul_gates, unc_gates = _certify_gates(args)

    def _emit(results: list[dict], model: str) -> None:
        print(render_certificate_md(certify(results), model))
        dom = render_domains_md(certify_domains(results, dom_gates))
        if dom:
            print("\n" + dom)
        med = render_medical_md(certify_medical(results, med_gates))
        if med:
            print("\n" + med)
        cul = render_culture_md(certify_culture(results, cul_gates))
        if cul:
            print("\n" + cul)
        unc = render_uncensored_md(certify_uncensored(results, unc_gates))
        if unc:
            print("\n" + unc)
        print()

    if getattr(args, "merge", False):
        model, results = merge_results(args.results)
        _emit(results, model or "merged")
        return 0

    for path in args.results:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _emit(data.get("results", []), data.get("model", ""))
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


def cmd_preflight(args) -> int:
    """Before running, check that the config really takes effect.

    A. Diff against the officially recommended sampling
    B. Cross-check config / actual payload / server defaults (/props) / launch arguments
    C. Degradation index of existing artifacts (--scan)

    None of them generates, so no GPU time is spent.
    """
    from .preflight import run_preflight

    config = {}
    if args.model:
        config = _load_config(args.config)
    report = run_preflight(
        config, args.model,
        artifacts=args.scan, results_json=args.results,
        offline=args.offline,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
    return report.exit_code(strict=args.strict)


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
                          help="Ollama host (default: config value or http://localhost:11434)")
    p_models.add_argument(
        "--remote", default=None,
        help="Fetch and show the model IDs actually served by config.yaml models.<name> "
             "(type: openai) from /v1/models "
             "(e.g. `--remote opencode-go` to inspect OpenCode Go's choices)",
    )
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
        "--generate-retries", type=int, default=None, dest="generate_retries",
        help="Regeneration count on parse failure (default: run.generate_retries or 1). "
             "With runs>1 the trials themselves are redundant, so 0 is recommended for expensive models",
    )
    p_run.add_argument(
        "--concurrency", type=int, default=None,
        help="Concurrency level for running trials in parallel (Default: run.concurrency or 1). "
             "Effective when llama.cpp is started with --parallel N -cb",
    )
    p_run.add_argument(
        "--skip-preflight", action="store_true", dest="skip_preflight",
        help="Skip the pre-run config check (preflight). "
             "Use it only when you want to run even on FAIL",
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
        "--config", default="config.yaml",
        help="Config to read the gate settings (certify_domains / certify_medical / "
             "certify_culture / certify_uncensored) from. "
             "Falls back to the default gates when it cannot be read",
    )
    p_cert.add_argument(
        "--merge", action="store_true",
        help="Combine results across multiple results.json files into a single certificate "
             "(Merged certificate for split runs like base40 + L6. Last win on duplicate task_ids)",
    )
    p_cert.set_defaults(fn=cmd_certify)

    p_pre = sub.add_parser(
        "preflight",
        help="Pre-run config check (diff vs official recommendations / effective values / degradation index)",
    )
    p_pre.add_argument("--config", default="config.yaml")
    p_pre.add_argument("--model", default=None,
                       help="Model name to check (when omitted, only --scan runs)")
    p_pre.add_argument("--scan", default=None,
                       help="Artifacts directory to compute the degradation index from")
    p_pre.add_argument("--results", default=None,
                       help="results.json used to aggregate the truncation rate "
                            "(inferred next to --scan when omitted)")
    p_pre.add_argument("--offline", action="store_true",
                       help="Do not look at HuggingFace; use only the cache and the bundled table")
    p_pre.add_argument("--strict", action="store_true",
                       help="Exit 1 even on WARN (for CI)")
    p_pre.add_argument("--json", action="store_true", help="Output as JSON")
    p_pre.set_defaults(fn=cmd_preflight)

    args = parser.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()

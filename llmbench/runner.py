"""Benchmark execution orchestrator.

For single trials (runs=1), behavior is standard. When runs>1, each task is sampled
N times to aggregate reliability metrics such as success rate and pass@k (the core of
"how usable it actually is").
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import usability
from .clients import LLMClient, create_client
from .clients.mock import MockClient
from .clients.ollama import (
    DEFAULT_OLLAMA_HOST,  # noqa: F401  (retained for backward compatibility)
    default_ollama_host,
    list_ollama_models,
)
from .graders import GradeCtx, get_grader
from .scoring import combined_score, pass_at_k
from .tasks import Task, load_tasks


# Large fields excluded from results.json (saved separately as artifact files)
ARTIFACT_FIELDS = ("raw_output", "parsed_files", "test_output")


@dataclass
class Attempt:
    """Evaluation result for a single attempt."""

    resolved: bool = False
    quality_score: float = 0.0
    combined: float = 0.0
    fail_reason: str = ""
    latency_sec: float = 0.0
    tokens_per_sec: float | None = None
    completion_tokens: int | None = None
    parse_ok: bool = False
    parse_error: str = ""
    parsed_files: dict = field(default_factory=dict)
    raw_output: str = ""
    test_output: str = ""
    quality_components: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    title: str = ""
    domain: str = "code"
    resolved: bool = False
    quality_score: float = 0.0
    combined: float = 0.0
    latency_sec: float = 0.0
    tokens_per_sec: float | None = None
    completion_tokens: int | None = None
    fail_reason: str = ""
    quality_components: dict = field(default_factory=dict)

    # --- Reliability (pass@k) ---
    runs: int = 1
    n_pass: int = 0
    success_rate: float = 0.0          # n_pass / runs
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    attempts: list = field(default_factory=list)   # Summary for each attempt (lightweight)

    # --- Usability Assessment ---
    usability_tier: str = ""

    # --- Artifacts (representative attempt) ---
    parse_ok: bool = False
    parse_error: str = ""
    parsed_files: dict = field(default_factory=dict)   # Relative path -> generated code
    raw_output: str = ""                                # Raw LLM output
    test_output: str = ""                               # Pytest output

    @property
    def changed_files(self) -> list[str]:
        return list(self.parsed_files.keys())


@dataclass
class RunResult:
    model: str
    issue_lang: str
    results: list[TaskResult] = field(default_factory=list)
    runs: int = 1
    artifacts_dirname: str = ""   # Populated with artifact directory name during save_run

    @property
    def multi_run(self) -> bool:
        return self.runs > 1 or any(r.runs > 1 for r in self.results)

    @property
    def resolved_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.resolved for r in self.results) / len(self.results)

    @property
    def avg_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success_rate for r in self.results) / len(self.results)

    @property
    def avg_pass_at_k(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.pass_at_k for r in self.results) / len(self.results)

    @property
    def solved_any_rate(self) -> float:
        """Ratio of tasks solved at least once across N trials (reachability with retries)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.n_pass > 0) / len(self.results)

    @property
    def avg_quality_resolved(self) -> float:
        """Average quality score only for resolved tasks."""
        rs = [r.quality_score for r in self.results if r.resolved]
        return sum(rs) / len(rs) if rs else 0.0

    @property
    def avg_combined(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.combined for r in self.results) / len(self.results)


class BenchmarkRunner:
    def __init__(self, config: dict, tasks_root: Path, ledgers: list[str] | None = None):
        self.config = config
        self.tasks_root = tasks_root
        self.run_cfg = config.get("run", {})
        self.ledgers = ledgers or ["tasks.jsonl"]
        self.quality_cfg = config.get("quality", {})
        self.scoring_cfg = config.get("scoring", {})
        self.usability_cfg = config.get("usability", {})
        wd = self.run_cfg.get("workdir")
        self.work_root = (
            Path(wd) if wd else Path(tempfile.gettempdir()) / "llmbench_work"
        )

    def ollama_host(self) -> str:
        """Priority order for Ollama connection host:
        --ollama-host / run.ollama_host > env OLLAMA_HOST
        > base_url of ollama model in config > http://localhost:11434
        """
        h = self.run_cfg.get("ollama_host")
        if h:
            return h
        env = os.environ.get("OLLAMA_HOST")
        if env:
            return env
        for v in self.config.get("models", {}).values():
            if v.get("type") == "ollama" and v.get("base_url"):
                return v["base_url"]
        return default_ollama_host()

    def _ollama_template(self) -> dict:
        """Borrows common parameters from the first Ollama model in config."""
        for v in self.config.get("models", {}).values():
            if v.get("type") == "ollama":
                return {
                    k: v[k] for k in ("temperature", "max_tokens", "timeout")
                    if k in v
                }
        return {}

    def resolve_model(
        self,
        name: str,
        client_type: str | None = None,
        base_url: str | None = None,
    ) -> dict:
        """Resolves model name into connection settings (dict).

        Priority order:
          1. client_type specified → Synthesize ad-hoc without consulting config
             (--client-type openai --base-url http://... direct to llama.cpp,
              --client-type multiagent direct to CodeRouter, etc.)
          2. Matches models: key in config → Use defined settings
             (Overridden if base_url is specified)
          3. Running Ollama model name → Automatically synthesize Ollama settings
             (Uses base_url as Ollama host if specified)
        """
        models = self.config.get("models", {})
        if client_type:
            cfg = dict(models.get(name, {}))
            cfg["type"] = client_type
            if base_url:
                cfg["base_url"] = base_url
            if client_type == "ollama":
                cfg.setdefault("base_url", base_url or self.ollama_host())
                cfg = {**self._ollama_template(), **cfg}
            if name not in models:
                # Treat as auto if openai type and name is missing in config
                # (Accepts either auto or actual model name)
                cfg.setdefault("model", name)
            return cfg
        if name in models:
            cfg = dict(models[name])
            if base_url:
                cfg["base_url"] = base_url
            return cfg
        host = base_url or self.ollama_host()
        try:
            available = list_ollama_models(host)
        except Exception as e:  # Ollama not running, etc.
            raise ValueError(
                f"Model {name!r} is undefined in config, "
                f"and cannot connect to Ollama ({host}): {e}\n"
                f"  Defined in config: {list(models)}\n"
                f"  (Specify --client-type openai|multiagent "
                f"and --base-url when connecting to backends other than Ollama)"
            ) from e
        if name in available:
            return {**self._ollama_template(), "type": "ollama",
                    "base_url": host, "model": name}
        raise ValueError(
            f"Model {name!r} is undefined in config, "
            f"and not installed in Ollama ({host}).\n"
            f"  Defined in config: {list(models)}\n  Running in Ollama: {available}"
        )

    def _make_reviewer(self) -> LLMClient | None:
        cfg = self.quality_cfg.get("llm_review", {})
        if not cfg.get("enabled", False):
            return None
        name = cfg.get("reviewer_model")
        if not name:
            raise ValueError(
                "quality.llm_review.enabled is true, but reviewer_model is unspecified"
            )
        try:
            # Same resolution path as main model (config > Ollama auto-resolution)
            return create_client(name, self.resolve_model(name))
        except ValueError as e:
            raise ValueError(
                f"Failed to resolve reviewer_model {name!r}: {e}"
            ) from e

    def _make_judge(self) -> tuple[LLMClient | None, int]:
        """Constructs grading model for judge grader (quality.judge).

        If unconfigured or disabled, returns (None, 1). In this case, judge grader evaluates
        deterministically using only hard_constraints (keeping validate passing green).
        """
        cfg = self.quality_cfg.get("judge", {})
        seeds = max(1, int(cfg.get("seeds", 1)))
        if not cfg.get("enabled", False):
            return None, seeds
        name = cfg.get("judge_model")
        if not name:
            raise ValueError("quality.judge.enabled is true, but judge_model is unspecified")
        try:
            return create_client(name, self.resolve_model(name)), seeds
        except ValueError as e:
            raise ValueError(f"Failed to resolve judge_model {name!r}: {e}") from e

    def run(
        self,
        model_name: str,
        only_tasks: list[str] | None = None,
        progress=print,
        runs: int | None = None,
        sample_temp: float | None = None,
        label: str | None = None,
        concurrency: int | None = None,
        client_type: str | None = None,
        base_url: str | None = None,
    ) -> RunResult:
        client = create_client(
            model_name,
            self.resolve_model(
                model_name, client_type=client_type, base_url=base_url
            ),
        )
        reviewer = self._make_reviewer()
        judge, judge_seeds = self._make_judge()
        lang = self.run_cfg.get("issue_lang", "en")
        timeout = int(self.run_cfg.get("test_timeout", 120))
        retries = int(self.run_cfg.get("generate_retries", 1))
        runs = int(runs if runs is not None else self.run_cfg.get("runs", 1))
        runs = max(1, runs)
        self._concurrency = max(1, int(
            concurrency if concurrency is not None
            else self.run_cfg.get("concurrency", 1)
        ))
        # Increase sampling temperature for diversity during multiple trials
        if runs > 1:
            st = sample_temp if sample_temp is not None else self.run_cfg.get(
                "sample_temp", 0.8
            )
            client.temperature = float(st)

        # Label: --label > server auto-detected (model:auto) > config key name
        served = getattr(client, "served_model_name", None)
        if label:
            run_label = label
        elif served:
            run_label = _label_from_model(served)
            progress(f"Detected model: {served} (auto-adopted without config edit)")
        else:
            run_label = model_name

        tasks = load_tasks(self.tasks_root, only=only_tasks, ledgers=self.ledgers)
        run = RunResult(model=run_label, issue_lang=lang, runs=runs)

        for i, task in enumerate(tasks, 1):
            progress(
                f"[{i}/{len(tasks)}] {task.task_id} ({task.difficulty}) {task.title}"
                + (f"  ×{runs}" if runs > 1 else "")
            )
            # Performance-constrained tasks prioritize individual perf_timeout (defaults to config otherwise)
            task_timeout = task.perf_timeout or timeout
            tr = self._run_task(
                client, reviewer, judge, judge_seeds,
                task, lang, task_timeout, retries, runs,
            )
            run.results.append(tr)
            _log_task(progress, tr)
        return run

    def _one_attempt(
        self,
        client: LLMClient,
        grader,
        system: str,
        user_prompt: str,
        task: Task,
        ctx: GradeCtx,
        retries: int,
    ) -> Attempt:
        """Evaluates a single generation -> grading pass (supports interchangeable graders)."""
        at = Attempt()
        total_latency = 0.0
        gen = None
        ev = None
        for _ in range(retries + 1):
            try:
                gen = client.generate(system, user_prompt)
            except Exception as e:  # Generation failures mark attempt as failed
                at.fail_reason = f"generation error: {e}"
                return at
            total_latency += gen.latency_sec
            ev = grader.evaluate(task, gen.text, ctx)
            if ev.parse_ok:
                break
        at.latency_sec = round(total_latency, 2)
        if gen is not None:
            at.tokens_per_sec = (
                round(gen.tokens_per_sec, 1) if gen.tokens_per_sec else None
            )
            at.completion_tokens = gen.completion_tokens
            at.raw_output = gen.text
        if ev is not None:
            at.resolved = ev.resolved
            at.quality_score = ev.quality_score
            at.parse_ok = ev.parse_ok
            at.parse_error = ev.parse_error
            at.parsed_files = ev.parsed_files
            at.fail_reason = ev.fail_reason or at.fail_reason
            at.test_output = ev.detail_output
            at.quality_components = ev.components

        at.combined = combined_score(at.resolved, at.quality_score, self.scoring_cfg)
        return at

    def _run_task(
        self,
        client: LLMClient,
        reviewer: LLMClient | None,
        judge: LLMClient | None,
        judge_seeds: int,
        task: Task,
        lang: str,
        timeout: int,
        retries: int,
        runs: int = 1,
    ) -> TaskResult:
        tr = TaskResult(
            task_id=task.task_id, difficulty=task.difficulty, title=task.title,
            domain=task.domain, runs=runs,
        )
        if isinstance(client, MockClient):
            client.current_task = task
            client.current_task_dir = task.dir   # Backward compatibility

        grader = get_grader(task.grader)
        system, user_prompt = grader.build_prompt(task, lang)
        ctx = GradeCtx(
            work_root=self.work_root,
            quality_cfg=self.quality_cfg,
            scoring_cfg=self.scoring_cfg,
            graders_cfg=self.config.get("graders", {}),
            reviewer=reviewer, judge=judge, judge_seeds=judge_seeds,
            timeout=timeout, lang=lang,
        )

        def _attempt(_i):
            return self._one_attempt(
                client, grader, system, user_prompt, task, ctx, retries
            )

        # Runs trials in parallel. Since LLM generation is the bottleneck, starting the
        # server with `--parallel N -cb` overlaps generation for N runs to reduce time.
        # Fallback to sequential for MockClient as it shares task_dir state.
        conc = min(getattr(self, "_concurrency", 1), runs)
        if conc > 1 and not isinstance(client, MockClient):
            with ThreadPoolExecutor(max_workers=conc) as ex:
                attempts = list(ex.map(_attempt, range(runs)))
        else:
            attempts = [_attempt(i) for i in range(runs)]
        _aggregate_attempts(tr, attempts, self.scoring_cfg)
        tr.usability_tier = usability.classify(
            tr.success_rate, tr.quality_score, self.usability_cfg
        )
        return tr


def _aggregate_attempts(
    tr: TaskResult, attempts: list[Attempt], scoring_cfg: dict
) -> None:
    """Aggregates multiple attempts to populate TaskResult. Matches legacy values when runs=1."""
    n = len(attempts)
    passed = [a for a in attempts if a.resolved]
    c = len(passed)

    tr.n_pass = c
    tr.success_rate = round(c / n, 3) if n else 0.0
    tr.pass_at_1 = tr.success_rate
    tr.pass_at_k = round(pass_at_k(n, c, n), 3)

    # Quality: Average of successful attempts (0 if none succeed)
    tr.quality_score = round(sum(a.quality_score for a in passed) / c, 1) if c else 0.0
    # Combined: Scaled by success rate (equivalent to resolved 0/1 when runs=1)
    tr.combined = combined_score(tr.success_rate, tr.quality_score, scoring_cfg)
    # Headline resolved uses majority vote (matches single attempt when runs=1)
    tr.resolved = tr.success_rate >= 0.5

    # Speed metrics are averaged
    tr.latency_sec = round(sum(a.latency_sec for a in attempts) / n, 2) if n else 0.0
    tps = [a.tokens_per_sec for a in attempts if a.tokens_per_sec]
    tr.tokens_per_sec = round(sum(tps) / len(tps), 1) if tps else None
    toks = [a.completion_tokens for a in attempts if a.completion_tokens]
    tr.completion_tokens = round(sum(toks) / len(toks)) if toks else None

    # Failure reason
    if c == n:
        tr.fail_reason = ""
    elif c == 0:
        tr.fail_reason = attempts[0].fail_reason if attempts else ""
    else:
        tr.fail_reason = f"flaky {c}/{n} passed"

    # Representative attempt (for artifacts): First successful attempt, or first attempt if none passed
    rep = passed[0] if passed else (attempts[0] if attempts else Attempt())
    tr.parse_ok = rep.parse_ok
    tr.parse_error = rep.parse_error
    tr.parsed_files = rep.parsed_files
    tr.raw_output = rep.raw_output
    tr.test_output = rep.test_output
    tr.quality_components = rep.quality_components

    # Lightweight summary per attempt (saved in JSON)
    tr.attempts = [
        {
            "resolved": a.resolved,
            "quality": round(a.quality_score, 1),
            "combined": a.combined,
            "fail_reason": a.fail_reason,
        }
        for a in attempts
    ]


def _label_from_model(name: str) -> str:
    """Formats server-reported model name for result labels (strips paths/.gguf)."""
    name = name.rsplit("/", 1)[-1]
    if name.endswith(".gguf"):
        name = name[:-5]
    return name


def _snippet(text: str, n: int = 3) -> str:
    """Preview flattening first n lines into a single line."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    head = " ⏎ ".join(lines[:n])
    return head[:120] + ("…" if len(head) > 120 else "")


def _log_task(progress, tr: TaskResult) -> None:
    """Logs task results to clarify artifact contents."""
    # Line 1: Parse results and generated files
    if tr.parse_ok:
        files = ", ".join(tr.changed_files) or "(none)"
        progress(
            f"    Gen OK  files=[{files}]  "
            f"{tr.completion_tokens or '?'}tok @ {tr.tokens_per_sec or '?'}tok/s"
        )
    else:
        progress(f"    Gen parse failed: {tr.parse_error or 'Unknown'}")
        if tr.raw_output:
            progress(f"      Output preview: {_snippet(tr.raw_output)}")

    # Line 2: Line count and head preview for each generated file (for visual inspection)
    for path, code in tr.parsed_files.items():
        nloc = len([ln for ln in code.splitlines() if ln.strip()])
        progress(f"      └ {path} ({nloc} LOC): {_snippet(code, 2)}")

    # Line 3: Verdict (includes reliability metrics for multi-runs)
    status = "✅ RESOLVED" if tr.resolved else f"❌ FAILED ({tr.fail_reason})"
    tier = usability.TIER_LABEL.get(tr.usability_tier, "")
    if tr.runs > 1:
        progress(
            f"    {status}  Passed {tr.n_pass}/{tr.runs} "
            f"(pass@1={tr.pass_at_1:.2f} pass@{tr.runs}={tr.pass_at_k:.2f})  "
            f"quality={tr.quality_score:.0f} combined={tr.combined:.0f}  "
            f"[{tier}]  ({tr.latency_sec:.1f}s)"
        )
    else:
        progress(
            f"    {status}  quality={tr.quality_score:.0f} "
            f"combined={tr.combined:.0f}  [{tier}]  ({tr.latency_sec:.1f}s)"
        )
    # Print tail of pytest output on test failures
    if not tr.resolved and tr.test_output:
        tail = tr.test_output.strip().splitlines()[-3:]
        for ln in tail:
            progress(f"      | {ln}")


def _write_artifacts(run: RunResult, artifacts_dir: Path) -> None:
    """Saves artifact files (raw LLM output, generated code, test output) per task."""
    for r in run.results:
        tdir = artifacts_dir / r.task_id
        tdir.mkdir(parents=True, exist_ok=True)

        # Raw LLM output
        if r.raw_output:
            (tdir / "llm_output.txt").write_text(r.raw_output, encoding="utf-8")

        # Generated files (the actual applied code)
        code_dir = tdir / "generated"
        for rel, content in r.parsed_files.items():
            dest = code_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not content.endswith("\n"):
                content += "\n"
            dest.write_text(content, encoding="utf-8")

        # Pytest output (for inspecting failure reasons)
        if r.test_output:
            (tdir / "test_output.txt").write_text(r.test_output, encoding="utf-8")


def save_run(run: RunResult, output_dir: Path) -> tuple[Path, Path]:
    """Saves results as JSON, Markdown, and artifacts. Returns (json_path, md_path)."""
    from .report import render_markdown

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{stamp}_{run.model}_results.json"
    md_path = output_dir / f"{stamp}_{run.model}_report.md"
    artifacts_dir = output_dir / f"{stamp}_{run.model}_artifacts"

    # Save artifacts in a separate directory
    _write_artifacts(run, artifacts_dir)
    run.artifacts_dirname = artifacts_dir.name

    # Keep results.json focused on aggregations and scores (excludes large artifact fields)
    def _lean(r: TaskResult) -> dict:
        d = asdict(r)
        for k in ARTIFACT_FIELDS:
            d.pop(k, None)
        return d

    # usability_tier is pre-configured on each TaskResult, so cfg is unnecessary
    overall, by_diff = usability.aggregate(run.results)
    summary = {
        "resolved_rate": round(run.resolved_rate, 3),
        "avg_quality_resolved": round(run.avg_quality_resolved, 1),
        "avg_combined": round(run.avg_combined, 1),
        "n_tasks": len(run.results),
        "runs": run.runs,
        "usability": {t: overall.get(t, 0) for t in usability.TIERS},
    }
    if run.multi_run:
        # avg_success_rate = average pass@1, solved_any_rate = solved ≥1 out of N trials
        # Note that avg_pass_at_k degrades when k=runs
        summary["avg_success_rate"] = round(run.avg_success_rate, 3)
        summary["solved_any_rate"] = round(run.solved_any_rate, 3)
        summary["avg_pass_at_k"] = round(run.avg_pass_at_k, 3)

    payload = {
        "model": run.model,
        "issue_lang": run.issue_lang,
        "artifacts_dir": artifacts_dir.name,
        "summary": summary,
        "results": [_lean(r) for r in run.results],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(run), encoding="utf-8")
    return json_path, md_path

"""Benchmark execution orchestrator.

For single trials (runs=1), behavior is standard. When runs>1, each task is sampled
N times to aggregate reliability metrics such as success rate and pass@k (the core of
"how usable it actually is").
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import env as envinfo
from . import usability
from .clients import LLMClient, create_client, sampling_of
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
    # Truncation (max_tokens reached / finish_reason=length).
    # "Did not solve it" and "cut off before finishing" are separate problems, so keep them apart.
    truncated: bool = False
    finish_reason: str = ""
    max_tokens: int | None = None
    # Failures caused by comms/infrastructure (429, timeout, connection drop, etc.).
    # Not model ability, so excluded from the success_rate denominator.
    errored: bool = False
    # Trials not executed due to fail-fast. Excluded from the denominator like errored,
    # but kept distinct: "comms went down" vs "skipped as running them was pointless".
    skipped: bool = False
    # Refusals from safety. A breakdown of resolved=False; not removed from the denominator
    # ("could have answered but did not" is model behaviour too, so a failure is a failure).
    # Counted separately only to separate it from "does not know".
    refused: bool = False


@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    title: str = ""
    domain: str = "code"
    category: str = ""             # Ledger category (trigger type etc.). Empty means unset
    role: str = ""                 # uncensored: control | probe
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
    pass_k: int = 1                    # Actual k used for pass_at_k (defaults to runs)
    n_errored: int = 0                # Trials not graded due to comms/infrastructure errors
    n_skipped: int = 0                # Trials not executed due to fail-fast
    n_refused: int = 0                # Trials judged as safety refusals
    refused: bool = False             # At least one trial was a refusal
    attempts: list = field(default_factory=list)   # Per-trial summary (lightweight)

    # --- Truncation ---
    truncated: bool = False          # At least one trial was cut off by max_tokens
    n_truncated: int = 0             # Number of truncated trials
    finish_reason: str = ""          # Finish reason of the representative trial
    max_tokens: int | None = None    # max_tokens used for generation

    # --- Usability assessment ---
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
    served_model: str = ""        # Model that actually answered (detected via model:auto / type:cli)
    # Execution environment (host CPU/GPU/RAM + inference backend config). Return of env.collect().
    # tok/s varies by quantization, GPU-offload ratio, and n_ctx, so keep it next to the results.
    environment: dict = field(default_factory=dict)

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


class JudgeSet:
    """Selector of judge clients (one global + per-domain overrides).

    `quality.judge.judge_model` is the default judge for every domain,
    `quality.judge.domain_overrides` is the override {domain: key in models:}.
    Overrides fire **independently of** `quality.judge.enabled`
    (enabled is only the flag for starting the global judge; grading
     uncensored must not change how writing is graded).

    Clients are cached one per key name in models: and built at most once per
    run. Building happens in bulk via warm() right after the run loads its
    tasks, so a misconfiguration such as a missing env var fails with 0 generated tokens.
    """

    def __init__(self, default=None, seeds: int = 1,
                 overrides: dict | None = None, factory=None):
        self.default = default
        self.seeds = max(1, int(seeds))
        self.overrides = dict(overrides or {})
        self._factory = factory
        self._cache: dict[str, LLMClient] = {}
        self._lock = threading.Lock()

    def for_domain(self, domain: str) -> tuple[LLMClient | None, int]:
        """Return the (judge, seeds) used for that domain. (None, seeds) if there is none."""
        name = self.overrides.get(domain)
        if not name or self._factory is None:
            return self.default, self.seeds
        with self._lock:
            client = self._cache.get(name)
            if client is None:
                client = self._factory(name)
                self._cache[name] = client
        return client, self.seeds

    def warm(self, domains) -> list[str]:
        """Build the override judges for the domains being run, up front.

        Returns a list of "domain→model" (for progress display). No override is
        built for a domain that is not run (running --only-med must not require
        API keys for other domains).
        """
        built = []
        for dom in domains:
            name = self.overrides.get(dom)
            if name:
                self.for_domain(dom)
                built.append(f"{dom}→{name}")
        return built


class BenchmarkRunner:
    def __init__(self, config: dict, tasks_root: Path, ledgers: list[str] | None = None):
        self.config = config
        self.tasks_root = tasks_root
        self.run_cfg = config.get("run", {})
        self.ledgers = ledgers or ["tasks.jsonl"]
        self.quality_cfg = config.get("quality", {})
        self.scoring_cfg = config.get("scoring", {})
        self.usability_cfg = config.get("usability", {})
        # k for pass@k (run.pass_k). null/unset equals runs = legacy behaviour.
        self.pass_k = self.run_cfg.get("pass_k")
        # fail-fast: for a task whose 1st trial was "empty body + truncation",
        # stop without running the remaining trials. When a thinking model falls
        # into a degenerate loop, a single trial uses the whole max_tokens worth
        # of time, so with runs=5 you pay 5x the time just to fail for sure.
        # The check is limited to "not a single body character", so a generation
        # that is written partway (= cut off by a budget shortfall by chance)
        # still runs every trial as before.
        self.fail_fast = bool(self.run_cfg.get("fail_fast", True))
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

    def _make_judges(self, disabled: bool = False) -> JudgeSet:
        """Build the grading-model set for the judge grader (quality.judge).

        - `enabled: true` … build the default judge shared by all domains
        - `domain_overrides: {domain: models-key}` … grade only that domain with a
          different model. **Fires independently of enabled** (DESIGN_UNCENSORED.md §2.3)
        - `disabled=True` … build nothing (for validate/MockClient)

        With none of them, the judge grader decides deterministically from
        hard_constraints only (keeps validate passing green).
        """
        cfg = self.quality_cfg.get("judge", {}) or {}
        seeds = max(1, int(cfg.get("seeds", 1)))
        if disabled:
            return JudgeSet(default=None, seeds=seeds)

        def _build(name: str) -> LLMClient:
            try:
                # Same resolution path as the main model (config > Ollama auto-resolve)
                return create_client(name, self.resolve_model(name))
            except ValueError as e:
                raise ValueError(f"cannot resolve judge model {name!r}: {e}") from e

        default = None
        if cfg.get("enabled", False):
            name = cfg.get("judge_model")
            if not name:
                raise ValueError("quality.judge.enabled is true, but judge_model is unspecified")
            default = _build(name)
        overrides = {
            str(dom): str(model)
            for dom, model in (cfg.get("domain_overrides") or {}).items()
            if model
        }
        return JudgeSet(default=default, seeds=seeds,
                        overrides=overrides, factory=_build)

    def _make_judge(self) -> tuple[LLMClient | None, int]:
        """Backward compatible: return only the global judge as (client, seeds)."""
        js = self._make_judges()
        return js.default, js.seeds

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
        generate_retries: int | None = None,
    ) -> RunResult:
        model_cfg = self.resolve_model(
            model_name, client_type=client_type, base_url=base_url
        )
        # When run.sampling exists, pass it as the default for all models (legacy behaviour without it)
        client = create_client(
            model_name, model_cfg, defaults=self.run_cfg.get("sampling")
        )
        reviewer = self._make_reviewer()
        lang = self.run_cfg.get("issue_lang", "en")
        timeout = int(self.run_cfg.get("test_timeout", 120))
        # Regeneration count after a parse failure. With runs>1 the trials themselves
        # provide redundancy, so expensive models (thinking models) can set 0 to save time.
        retries = int(
            generate_retries if generate_retries is not None
            else self.run_cfg.get("generate_retries", 1)
        )
        retries = max(0, retries)
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

        # Judges are built here. After the ledgers are read we can build "only the
        # override judges for the domains being run" first, so a misconfiguration
        # such as a missing env var fails with 0 generated tokens. validate
        # (MockClient) builds none at all -- scoring mock_gold with a real judge
        # would make validate non-deterministic.
        judges = self._make_judges(disabled=isinstance(client, MockClient))
        judge, judge_seeds = judges.default, judges.seeds
        warmed = judges.warm(sorted({t.domain for t in tasks}))
        if warmed:
            progress("judge overrides: " + ", ".join(warmed))

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
                judges=judges,
            )
            run.results.append(tr)
            _log_task(progress, tr)

        # Record the name of the model that actually answered:
        #   model:auto → served_model_name / type:cli → detected from the CLI response during generation
        detected = getattr(client, "detected_model", None)
        run.served_model = served or detected or ""
        if detected:
            progress(f"Run model    : {detected} (detected from CLI response -> recorded in results)")

        # Record the execution environment (hardware + backend config).
        # Collected only after at least one generation, because Ollama's /api/ps
        # returns "loaded models" only = it is empty before a run.
        # Collection is best-effort; a failure must not affect the bench results.
        run.environment = envinfo.collect(
            model_cfg, served_model=run.served_model or None
        )
        # Sampling config is the core of "was this measured under the same
        # conditions". With runs>1 client.temperature is overridden at run time
        # above, so record the **effective values after the override** (not the config values).
        run.environment["sampling"] = sampling_of(client)
        summary = envinfo.format_summary(run.environment)
        if summary:
            progress(f"Environment  : {summary}")
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
        # Regeneration gate: parse_ok alone is not enough. When "some paths were
        # dropped because they were placeholders", parse_ok=True but the content is
        # suspect, so regenerate when the grader returns warnings too.
        # The loop always exits after retries+1 rounds.
        for _ in range(retries + 1):
            t_gen = time.monotonic()
            try:
                gen = client.generate(system, user_prompt)
            except Exception as e:  # Generation failure marks the trial as failed
                # Keep the elapsed time on failures too. This stayed 0.0, so a
                # 600-second Read timeout showed as "(0.0s)" in logs, looked like
                # an instant death, and blocked root-cause work.
                at.latency_sec = round(
                    total_latency + (time.monotonic() - t_gen), 2
                )
                at.fail_reason = f"generation error: {e}"
                at.errored = True
                return at
            total_latency += gen.latency_sec
            ev = grader.evaluate(task, gen.text, ctx)
            if ev.parse_ok and not _extraction_suspect(ev):
                break
            # Do not regenerate a truncated generation. As long as the parse failure
            # comes from "output cut off midway", using the same budget again hits
            # the same wall and throws away max_tokens worth of time per round.
            # (Measured: a 49,152tok thinking loop drawn twice with
            #  generate_retries=1 costs 5 min per trial. With runs=5, 25 min per task.)
            if getattr(gen, "truncated", False):
                break
        at.latency_sec = round(total_latency, 2)
        if gen is not None:
            at.tokens_per_sec = (
                round(gen.tokens_per_sec, 1) if gen.tokens_per_sec else None
            )
            at.completion_tokens = gen.completion_tokens
            at.raw_output = gen.text
            # getattr so old client implementations (no extra fields) do not break
            at.truncated = bool(getattr(gen, "truncated", False))
            at.finish_reason = getattr(gen, "finish_reason", None) or ""
            at.max_tokens = getattr(gen, "max_tokens", None)
        if ev is not None:
            at.resolved = ev.resolved
            at.quality_score = ev.quality_score
            at.parse_ok = ev.parse_ok
            at.parse_error = ev.parse_error
            at.parsed_files = ev.parsed_files
            at.fail_reason = ev.fail_reason or at.fail_reason
            at.test_output = ev.detail_output
            at.quality_components = ev.components
            at.refused = bool(getattr(ev, "refused", False))

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
        judges: "JudgeSet | None" = None,
    ) -> TaskResult:
        tr = TaskResult(
            task_id=task.task_id, difficulty=task.difficulty, title=task.title,
            domain=task.domain, category=getattr(task, "category", "") or "",
            role=getattr(task, "role", "") or "",
            runs=runs,
        )
        if isinstance(client, MockClient):
            client.current_task = task
            client.current_task_dir = task.dir   # Backward compatibility

        grader = get_grader(task.grader)
        system, user_prompt = grader.build_prompt(task, lang)
        # Per-domain judge overrides (quality.judge.domain_overrides).
        # Without judges, use the single one passed by the caller as before.
        t_judge, t_seeds = (
            judges.for_domain(task.domain) if judges is not None
            else (judge, judge_seeds)
        )
        ctx = GradeCtx(
            work_root=self.work_root,
            quality_cfg=self.quality_cfg,
            scoring_cfg=self.scoring_cfg,
            graders_cfg=self.config.get("graders", {}),
            reviewer=reviewer, judge=t_judge, judge_seeds=t_seeds,
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
            attempts = []
            for i in range(runs):
                at = _attempt(i)
                attempts.append(at)
                # fail-fast: "truncated with an empty body" = a trial that never
                # finished thinking. Drawing 4 more times under the same conditions
                # only hits the same wall, so mark the rest as skipped and move on.
                if getattr(self, "fail_fast", True) and _hopeless(at):
                    remaining = runs - i - 1
                    if remaining:
                        reason = (
                            f"skipped by fail-fast: trial {i + 1} was truncated"
                            f" with an empty body ({at.finish_reason or 'length'})"
                        )
                        attempts += [
                            Attempt(skipped=True, fail_reason=reason)
                            for _ in range(remaining)
                        ]
                    break
        _aggregate_attempts(tr, attempts, self.scoring_cfg, k=self.pass_k)
        tr.usability_tier = usability.classify(
            tr.success_rate, tr.quality_score, self.usability_cfg
        )
        return tr


def _hopeless(at: Attempt) -> bool:
    """Is repeating this trial under the same conditions pointless (the fail-fast check)?

    Requires **truncated AND not a single usable output character** at the same time.

    - Written partway (raw_output present) → it may have been cut off by a mere
      budget shortfall, so repeating is worth it. Do not skip.
    - Comms error (errored) → it recovers after a delay, so repeat. Do not skip.
    - Empty body + truncation → thinking never finished. Redrawing with the same
      max_tokens and temperature drowns in the same place. Stop only here.
    """
    return bool(at.truncated) and not at.errored and not at.raw_output.strip()


def _extraction_suspect(ev) -> bool:
    """Extraction succeeded but cannot be taken at face value (= worth regenerating)?

    ``GraderEval.parse_warnings`` holds the patch-side warnings (placeholder paths
    dropped etc.). Read with getattr so graders that do not have it still work.
    """
    return bool(
        getattr(ev, "parse_warnings", None) or getattr(ev, "parse_error", "")
    )


def _aggregate_attempts(
    tr: TaskResult, attempts: list[Attempt], scoring_cfg: dict, k: int | None = None
) -> None:
    """Aggregate multiple attempts to fill TaskResult. Matches legacy values when runs=1.

    k is the k of pass@k. With None (default), k = n, which degenerates to the
    legacy 0/1 "solved at least once". With k < n it is meaningful as the
    unbiased estimate of "the probability of succeeding at least once in k trials".
    """
    n_all = len(attempts)
    # Trials errored by comms/infrastructure are not "the model failed to solve",
    # so remove them from the success-rate denominator. If every trial errored,
    # keep success_rate=0 and make the reason visible in fail_reason (never a
    # silent pass). Trials skipped by fail-fast leave the denominator too:
    # counting an unrun trial as a failure pretends we measured what we did not.
    # (A truncation itself stays as a failure on the executed side, so
    #  success_rate becomes 0 either way; only the denominator becomes honest.)
    errored = [a for a in attempts if a.errored and not a.skipped]
    skipped = [a for a in attempts if a.skipped]
    scored = [a for a in attempts if not a.errored and not a.skipped]
    tr.n_errored = len(errored)
    tr.n_skipped = len(skipped)
    n = len(scored) or n_all
    passed = [a for a in scored if a.resolved]
    c = len(passed)

    k_eff = n if k is None else max(1, min(int(k), n))
    tr.n_pass = c
    tr.success_rate = round(c / n, 3) if n else 0.0
    tr.pass_at_1 = tr.success_rate
    tr.pass_k = k_eff
    tr.pass_at_k = round(pass_at_k(n, c, k_eff), 3)

    # Quality: Average of successful attempts (0 if none succeed)
    tr.quality_score = round(sum(a.quality_score for a in passed) / c, 1) if c else 0.0
    # Combined: Scaled by success rate (equivalent to resolved 0/1 when runs=1)
    tr.combined = combined_score(tr.success_rate, tr.quality_score, scoring_cfg)
    # Headline resolved uses majority vote (matches single attempt when runs=1)
    tr.resolved = tr.success_rate >= 0.5

    # Speed metrics are averaged (measured, including errored trials).
    # Skipped trials are not "trials that finished in 0 seconds", so drop them
    # from the denominator — with them, a truncated task's latency looked like
    # 1/5 of the measured value.
    executed = [a for a in attempts if not a.skipped]
    n_exec = len(executed)
    tr.latency_sec = (
        round(sum(a.latency_sec for a in executed) / n_exec, 2) if n_exec else 0.0
    )
    tps = [a.tokens_per_sec for a in attempts if a.tokens_per_sec]
    tr.tokens_per_sec = round(sum(tps) / len(tps), 1) if tps else None
    toks = [a.completion_tokens for a in attempts if a.completion_tokens]
    tr.completion_tokens = round(sum(toks) / len(toks)) if toks else None

    # Refusals: how many of the scored trials were refusals.
    # Does not change the success_rate denominator (a refusal is a "not solved").
    tr.n_refused = sum(1 for a in scored if a.refused)
    tr.refused = tr.n_refused > 0

    # Truncation: one cut trial means "this task hit the limit"
    tr.n_truncated = sum(1 for a in attempts if a.truncated)
    tr.truncated = tr.n_truncated > 0
    mt = [a.max_tokens for a in attempts if a.max_tokens]
    tr.max_tokens = mt[0] if mt else None

    # Failure reason
    err_note = f" [+{len(errored)} errored]" if errored else ""
    if skipped:
        err_note += f" [+{len(skipped)} skipped by fail-fast]"
    if not scored:
        tr.fail_reason = (
            f"all {n_all} attempt(s) errored: "
            f"{errored[0].fail_reason if errored else ''}"
        )
    elif c == n:
        tr.fail_reason = err_note.strip()
    elif c == 0:
        tr.fail_reason = (scored[0].fail_reason if scored else "") + err_note
    else:
        tr.fail_reason = f"flaky {c}/{n} passed{err_note}"

    # Representative trial (for artifacts): first successful trial, else first trial
    rep = passed[0] if passed else ((scored or attempts or [Attempt()])[0])
    tr.parse_ok = rep.parse_ok
    tr.parse_error = rep.parse_error
    tr.parsed_files = rep.parsed_files
    tr.raw_output = rep.raw_output
    tr.test_output = rep.test_output
    tr.quality_components = rep.quality_components
    tr.finish_reason = rep.finish_reason

    # Lightweight summary per attempt (saved in JSON)
    tr.attempts = [
        {
            "resolved": a.resolved,
            "quality": round(a.quality_score, 1),
            "combined": a.combined,
            "fail_reason": a.fail_reason,
            "truncated": a.truncated,
            "finish_reason": a.finish_reason,
            "errored": a.errored,
            "skipped": a.skipped,
            "refused": a.refused,
        }
        for a in attempts
    ]


_UNSAFE_FILENAME_CHARS = '<>:"|?*'


def _safe_label(name: str) -> str:
    """Normalize a model name/label so it can be embedded in a file name.

    In Ollama, model names containing slashes and colons are standard, e.g.
    `hf.co/unsloth/Qwen3-Coder-GGUF:Q4_K_M`; used as a file name it points at a
    non-existent directory and raises FileNotFoundError (graded results are lost).
    The original text for display stays in run.model; here we build a **file-name-only** slug.
    """
    seg = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    out = "".join(
        "-" if (ch in _UNSAFE_FILENAME_CHARS or ord(ch) < 32) else ch
        for ch in seg
    )
    out = out.strip(" .")
    return out or "model"


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
            f"    {status}  Passed {tr.n_pass}/{tr.runs - tr.n_errored - tr.n_skipped} "
            + (f"(⚠️ {tr.n_errored} excluded: comms errors) " if tr.n_errored else "")
            + (
                f"(⏭️ {tr.n_skipped} skipped by fail-fast) "
                if tr.n_skipped else ""
            )
            + f"(pass@1={tr.pass_at_1:.2f} pass@{tr.runs}={tr.pass_at_k:.2f})  "
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
    # Use the sanitized slug in file names (run.model itself stays as written).
    slug = _safe_label(run.model)
    json_path = output_dir / f"{stamp}_{slug}_results.json"
    md_path = output_dir / f"{stamp}_{slug}_report.md"
    artifacts_dir = output_dir / f"{stamp}_{slug}_artifacts"

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
        # Number of tasks truncated by max_tokens. In a run where this is not 0,
        # "cut off before finishing" is mixed into the "not solved" breakdown, so
        # do not read the score alone as the model's ability.
        "n_truncated": sum(1 for r in run.results if r.truncated),
        # Number of tasks where fail-fast dropped trials, and the total trials dropped.
        # "Measured N times" holds only when these are 0, so keep them in the summary.
        "n_fail_fast_tasks": sum(1 for r in run.results if r.n_skipped),
        "n_skipped_attempts": sum(r.n_skipped for r in run.results),
        # Safety refusals. In a run where this is not 0, "does not know" and
        # "will not answer" are mixed, so do not read the accuracy alone as knowledge.
        "n_refused_tasks": sum(1 for r in run.results if r.refused),
        "n_refused_attempts": sum(r.n_refused for r in run.results),
        "usability": {t: overall.get(t, 0) for t in usability.TIERS},
    }
    # Put speed metrics in the summary too. They used to exist only per task
    # (results[]), so external tools that read only the summary (the CodeRouter
    # sweep results panel etc.) could not see speed. Keys are the generic
    # tokens_per_sec / avg_latency_ms.
    tps = [r.tokens_per_sec for r in run.results if r.tokens_per_sec]
    if tps:
        summary["tokens_per_sec"] = round(sum(tps) / len(tps), 1)
    lat = [r.latency_sec for r in run.results if r.latency_sec]
    if lat:
        summary["avg_latency_ms"] = round(sum(lat) / len(lat) * 1000)
    if run.multi_run:
        # avg_success_rate = average pass@1, solved_any_rate = solved ≥1 out of N trials
        # avg_pass_at_k is the unbiased estimate at the k set by run.pass_k
        # (it degenerates to solved_any_rate when k == runs)
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
    if run.served_model:
        payload["served_model"] = run.served_model
    if run.environment:
        payload["environment"] = run.environment
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(run), encoding="utf-8")
    return json_path, md_path

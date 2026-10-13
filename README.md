<div align="center">

# 🧪 llmbench

**Are Local LLMs Truly "Usable"? — An SWE-Bench Style Benchmark Measuring Functional Correctness × Code Quality**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

A built-from-scratch framework designed to **rigorously evaluate** the trade-off between <br>
"tests pass but code is messy" and "clean, maintainable, and safe code" in a local environment. <br>
Now extended to support multi-domain evaluations: **Security Detection / Instruction Following / Writing / Medical QA**.

[Features](#-features) • [Quick Start](#-quick-start) • [Scoring](#-scoring) • [Multi-Domain](#-multi-domain-evaluation-non-coding) • [Adding Tasks](#-adding-tasks) • [Roadmap](#-roadmap)

</div>

---

## ✨ Features

- 🎯 **Functional Correctness** — SWE-Bench style. Hands bug reports + source files to the LLM, applies generated patches → evaluates `resolved` state via hidden tests (pytest).
- 🧹 **Code Quality Layer** — Weighted synthesis of Ruff (linting issue density), radon (Maintainability Index / Cyclomatic Complexity), LLM review scoring, and SonarQube metrics.
- 🎲 **Reliability (pass@k)** — Runs each task $N$ times via `--runs N` to measure success rate (pass@1), pass@k, and flakiness. Evaluates whether a model is **consistently usable** rather than relying on a single lucky pass.
- 🧭 **Usability Classification** — Classifies tasks into **🟢 Autonomous / 🟡 Assisted / 🔴 Unusable** based on reliability and quality to show actionable delegation bounds.
- 🎓 **Usability Line Certification (`certify`)** — Maps difficulty levels to tiers (L1–L7) and evaluates tier pass/fail status independently to pinpoint capabilities. **Independent L4 (expert) pass = production-ready baseline**. Includes **L7 (grandmaster)** as a ceiling evaluation tier for top models. Multiple split runs can be consolidated into a single certificate via `certify --merge`.
- 🌐 **Multi-Domain Evaluation** — Pluggable grader architecture extending evaluation beyond coding: **detection** (vulnerability/log detection evaluated via F1 + false positive decoy penalties), **constraint** (instruction following validated programmatically IFEval-style), **judge** (creative writing evaluated via rubric), and **qa** (medical QA matched against answer keys with JP/EN allowed terms). Enabled via `--with-sec/gen/write/med`. `certify` outputs domain-specific gates and a **Balance Index** (penalizing single-capability spikes). Full design specified in [📐 DESIGN_DOMAINS.md](DESIGN_DOMAINS.md).
- ⚖️ **Composite Scoring** — Non-working code scores 0. Working code is differentiated by success rate and code quality.
- 🔌 **Flexible Connectivity** — Compatible with OpenAI-compatible APIs (llama.cpp / LM Studio / vLLM) and Ollama. **`model: auto`** automatically adopts the server's currently loaded model (no config edits required); dynamically resolves installed Ollama models.
- 🆚 **Cross-Model Comparison** — `compare` consolidates multiple result files into unified rankings and matrices alongside reference cloud API models.
- 🇯🇵 **Dual-Language Issues (EN/JA)** — Supports `--lang ja` to measure "language tax" (performance degradation caused by Japanese instructions). Medical QA and Japanese models are correctly evaluated using JP/EN allowed terms in gold files.
- ⚡ **Speed Benchmarking** — Automatically logs per-task latency and generation throughput (tokens/sec).
- 📦 **40 Built-in Tasks (+20 Optional +40 Optional)** — L1 easy 5 / L2 medium 5 / L3 hard 10 / **L4 expert 12 / L5 frontier 8**. Optionally extendable with **L6 architect 20 tasks (t041–t060) via `--with-l6`** and **L7 grandmaster 40 tasks (t061–t100) via `--with-l7`** to break through score ceiling effects. Supports split execution via `--only-l6`/`--only-l7` (excluding default 40 tasks) with later consolidation via `certify --merge`. Frontier/Architect/Grandmaster tasks feature multi-file modifications, regression traps, and performance constraints (`perf_timeout`), presenting **symptom-based issues** that require root-cause diagnosis. Out-of-the-box execution using Python standard library only.
- 🛡️ **Safety by Design** — Hidden tests are never exposed to the LLM, patch file modifications are strictly scoped to pre-approved files, and original task definitions remain immutable.

## 🚀 Quick Start

```bash
git clone https://github.com/zephel01/swe-bench.git
cd swe-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1️⃣ Self-Validation (No LLM required — verifies entire pipeline via mock clients)
llmbench validate
llmbench validate --only-sec        # Domain-specific pipeline verification (sec/gen/write/med)

# 2️⃣ Point base_url in config.yaml to your environment. Set model to "auto" (automatically detects loaded server model)

# 3️⃣ Inspect available models (Configured models + active Ollama models)
llmbench models

# 4️⃣ Execution (Coding Tasks)
llmbench run --model local-openai                 # Default 40 tasks, 1 trial
llmbench run --model local-openai --runs 5        # 5 trials per task → success rate & pass@k
llmbench run --model local-openai --with-l6 --runs 5  # Adds L6 (architect) 20 tasks (60 total)
llmbench run --model local-openai --with-l7 --runs 5  # Adds L7 (grandmaster) 40 tasks (80 total)
llmbench run --model local-openai --with-l6 --with-l7 --runs 5  # Combines L6+L7 (100 total ceiling evaluation)
llmbench run --model local-openai --only-l6 --runs 5  # Standalone L6 execution → consolidate later via certify --merge
llmbench run --model local-openai --runs 5 --concurrency 5  # Runs 5 concurrent trials (requires server launched with --parallel 5 -cb)
llmbench run --model qwen2.5-coder:7b --runs 5    # Direct specification of installed Ollama model names
llmbench run --model local-openai --tasks t021,t033 --lang ja   # Run specific task subsets or tiers

# 4️⃣' Execution (Multi-Domain Tasks: Non-Coding)
llmbench run --model local-openai --with-sec --with-gen --runs 5   # Appends security and general domain ledgers
llmbench run --model local-openai --only-sec --runs 5              # Standalone security detection execution
llmbench run --model <medical_model>  --only-med --lang ja --runs 5    # Japanese medical QA benchmark

# 5️⃣ Cross-Model Comparison
llmbench compare results/*_results.json

# 6️⃣ Usability Line Certification (Tier-Based Pass System + Domain Gates + Balance Index)
llmbench certify results/<stamp>_<model>_results.json

```

Subcommands: `list-tasks` / `models` / `run` / `compare` / `certify` / `validate`

Execution results are saved to `results/` across **3 items**:

- `<stamp>_<model>_results.json` — Scores and summary statistics (machine-readable, lightweight JSON)
- `<stamp>_<model>_report.md` — Human-readable Markdown report
- `<stamp>_<model>_artifacts/<task_id>/` — **Output artifacts**
- `llm_output.txt` … Raw LLM text output (for root-cause analysis on parse failures)
- `generated/<path>` … Extracted generated code / predicted labels & answers
- `test_output.txt` … Pytest logs and evaluation details (for debugging failed tests)

Console logs display files generated, code headers, and parse/test results per task in real time.

> [!TIP]
> For detailed instructions on interpreting execution logs, reports, artifacts, and running model comparisons, consult the [📘 User Guide (USAGE.md)](USAGE.md).

```markdown
# llmbench Report: Qwopus3.6-27B-Coder-MTP-Q6_K

Issue Language: `en` / Tasks: 20 / Runs: ×5

## Summary

| Metric                           | Value             |
| -------------------------------- | ----------------- |
| ✅ Resolved Rate                 | **95.0%** (19/20) |
| 🎲 Avg Success Rate (pass@1, ×5) | **93.0%**         |
| 🔁 Tasks Resolved ≥1 Time        | **95.0%** (19/20) |
| 🏅 Quality Avg (Resolved Only)   | **89.8 / 100**    |
| 🎯 Combined Avg                  | **88.2 / 100**    |

## 🧭 Usability Classification

- 🟢 Autonomous: 17/20 (85%) / 🟡 Assisted: 2/20 (10%) / 🔴 Unusable: 1/20 (5%)
    > Recommendation: Generally Autonomous. Caution required for 🔴 Unusable tasks (1/20, 5%).

|     | Task | Tier        | Reliability            | Quality | Combined | tok/s |
| --- | ---- | ----------- | ---------------------- | ------- | -------- | ----- |
| ✅  | t007 | 🟡 Assisted | 4/5 (Success Rate 80%) | 100     | 80       | 111.9 |
| ❌  | t020 | 🔴 Unusable | 0/5 (Success Rate 0%)  | 0       | 0        | 122.2 |
```

## 📊 Scoring

$$\text{combined} = \text{success\_rate} \times \left(0.5 + 0.5 \times \frac{\text{quality}}{100}\right) \times 100$$

Where `success_rate` represents trial accuracy (`0/1` for `--runs 1`; `0.0` to `1.0` for multi-run).
**Reliability directly scales the overall score** (e.g., 60% success rate with quality 90 → $0.6 \times 95 = 57$ points).

| success_rate | quality | combined | Meaning                                                         |
| ------------ | ------- | -------- | --------------------------------------------------------------- |
| 0 (Failed)   | —       | **0**    | Non-working code scores 0 regardless of quality metrics         |
| 1.0          | 0       | **50**   | Passes functional tests every time, but code quality is minimal |
| 1.0          | 100     | **100**  | Passes functional tests every time with perfect quality metrics |
| 0.6 (Flaky)  | 90      | **57**   | Flaky completion; penalized for lower reliability               |

> [!NOTE]
> Multi-domain graders map evaluation outcomes into normalized `(resolved, quality)` tuples.
> Detection maps to F1, constraint to check pass rates, judge to rubric scores, and QA to answer accuracy, allowing pass@k, usability, and certification pipelines to be shared seamlessly without code modifications.

### Reliability (pass@k) & Usability Classification

Runs each task $N$ times via `--runs N` to evaluate **success rate (pass@1)**, pass@k, and flakiness.
Classifies tasks into 3 usability tiers based on `success_rate` and `quality` (thresholds configured under `usability:` in `config.yaml`):

| Usability Tier | Default Conditions             | Meaning                                                         |
| -------------- | ------------------------------ | --------------------------------------------------------------- |
| 🟢 Autonomous  | success ≥ 0.9 AND quality ≥ 80 | Tasks can be delegated autonomously with minimal review         |
| 🟡 Assisted    | success ≥ 0.6                  | Suitable for completion provided human code review is performed |
| 🔴 Unusable    | Below above criteria           | Tasks cannot be safely delegated to the model                   |

### 🎓 Usability Line Certification (`certify`)

While usability classification operates **per task**, `certify` evaluates **overall model performance**.
Maps task difficulties to tiers (L1–L7) and independently evaluates whether average success rates and combined scores pass tier gates: `llmbench certify results/<...>_results.json`.

| Gate           | Conditions (Tier Average)                     | Meaning                                                  |
| -------------- | --------------------------------------------- | -------------------------------------------------------- |
| L1 easy        | success ≥ 90%                                 | Basic toy tasks cleared                                  |
| L2 medium      | success ≥ 85%                                 | Simple specification-compliant work cleared              |
| L3 hard        | success ≥ 75% AND combined ≥ 60               | Simple to mid-level bug fixes cleared                    |
| **L4 expert**  | **success ≥ 60% AND combined ≥ 55**           | **✅ Production-Ready Baseline (Supervised Deployment)** |
| L5 frontier    | success ≥ 40%                                 | Frontier-level capability                                |
| L6 architect   | success ≥ 55% AND combined ≥ 60 (Provisional) | Architect-level capability / Model separation            |
| L7 grandmaster | success ≥ 35% AND combined ≥ 55 (Provisional) | Grandmaster-level capability / Ceiling evaluation tier   |

**Production-Ready Baseline = Independent L4 Pass** (unaffected by regressions in lower tiers).
Gate thresholds can be tuned in `DEFAULT_GATES` inside `llmbench/certify.py`. L6 metrics appear when measured via `--with-l6`, and L7 via `--with-l7`. L7 acts as a **ceiling evaluation tier** to measure performance plateaus among top-tier models rather than practical delegation bounds.
Provisional thresholds will be finalized following empirical calibrations.
When running split evaluations via `--only-l6`/`--only-l7`, output files can be merged into a single certification via `certify --merge a.json b.json` (task IDs use last-wins logic).
When non-coding domains are evaluated, `certify` also displays **domain-specific gates, Balance Index metrics, and medical accuracy breakdowns** (see [Multi-Domain](DESIGN_DOMAINS.md)).

Quality Score Layers (Weights customizable in `config.yaml`):

| Evaluator Layer | Default Weight | Description                                                                 |
| --------------- | -------------- | --------------------------------------------------------------------------- |
| 🔍 Ruff         | 0.4            | Penalizes issue density across E/F/W/B/SIM/C4/S rules                       |
| 🌀 radon        | 0.3            | Penalizes based on Maintainability Index + worst Cyclomatic Complexity rank |
| 🤖 LLM Review   | 0.3 (Optional) | Code review scoring (0–10 scale) performed by an external LLM               |
| 📡 SonarQube    | Optional       | Active SonarQube server integration; penalizes based on issue severity      |

## 🌐 Multi-Domain Evaluation (Non-Coding)

Extends evaluations beyond coding by replacing hidden pytest oracles with **pluggable graders**.
Graders define output contracts (prompts) and scoring logic, normalizing results back into `(resolved, quality)` tuples. Consequently, pass@k, combined score, usability, and certification pipelines are reused **without code modifications**. Full design specs available in [📐 DESIGN_DOMAINS.md](DESIGN_DOMAINS.md).

| Domain      | Grader       | Ledger / Flag                                       | Scope & Evaluation Method                                                                                                                                                                                                            |
| ----------- | ------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 🛡️ security | `detection`  | `tasks_sec.jsonl` / `--with-sec` `--only-sec`       | **Detects** known vulnerabilities/breaches in logs or code, scoring against gold labels using **Precision / Recall / F1**. Features **clean decoy tasks** to penalize false positive (FP) over-detection.                            |
| 📋 general  | `constraint` | `tasks_gen.jsonl` / `--with-gen` `--only-gen`       | Programmatically verifies character counts, JSON validity, regex patterns, etc. (**IFEval style**). Success achieved when all checks pass; pass rate represents quality. Exposes capability regression in specialized coding models. |
| ✍️ writing  | `judge`      | `tasks_write.jsonl` / `--with-write` `--only-write` | 0–10 scoring via rubric + judge model (**experimental**). Evaluates deterministically using hard constraints (word count, etc.) when a judge model is unavailable.                                                                   |
| 🩺 medical  | `qa`         | `tasks_med.jsonl` / `--with-med` `--only-med`       | Matches medical QA against answer keys (MCQ options, short answers). Gold keys include **both JP and EN allowed terms** so Japanese models running with `--lang ja` are correctly evaluated (**reference value**).                   |

- **CLI Architecture**: Appends ledgers via `--with-*`; runs standalone domain evaluations via `--only-*` (shares the same flag structure as `--with-l6/l7`). Split execution outputs merge via `certify --merge`.
- **Certification Extensions**: In addition to domain-specific gates, evaluates the **Balance Index** (harmonic mean of average combined scores across coding + non-experimental domains) to penalize single-capability spikes. Displays medical accuracy broken down by difficulty (basic/std/hard).
- **Self-Validation**: Execute `llmbench validate --only-sec|gen|write|med` to verify all gold files pass and broken files fail (zero LLM calls required).
- **Judge Setup**: Set `quality.judge.enabled: true` and configure `judge_model` in `config.yaml` to enable writing evaluations (using a model distinct from candidate models is recommended to avoid self-preference bias).

> [!NOTE]
> Gate thresholds for writing/medical domains are provisional (uncalibrated) and configurable via `certify_domains` / `DEFAULT_MED_GATES`.
> Medical evaluations serve as **reference metrics** accounting for 5-choice MCQ baseline chance accuracy (~20%) rather than clinical validity guarantees.

## ⚙️ Configuration

Managed centrally in `config.yaml`:

```yaml
models:
    local-openai: # llama.cpp / LM Studio / vLLM
        type: openai
        base_url: "http://localhost:8080/v1"
        model: "auto" # auto = automatically adopts active model from /v1/models (no config edit required)
        # auto_prefer: "qwen"  # Substring matching when multiple models are loaded
    local-ollama:
        type: ollama
        base_url: "http://localhost:11434"
        model: "qwen2.5-coder:32b"
    ref-gpt: # Reference anchor model for comparison (API)
        type: openai
        base_url: "https://api.openai.com/v1"
        model: "gpt-4o"
        api_key: "${OPENAI_API_KEY}" # ${VAR} expands environment variables

run:
    issue_lang: en # Switch to ja to measure language tax
    test_timeout: 120
    runs: 1 # Trial count per task (>1 calculates pass@k & success rate)
    sample_temp: 0.8 # Sampling temperature for multi-trial runs
    # ollama_host: "http://localhost:11434"   # Connection host for Ollama model auto-resolution

quality:
    llm_review:
        enabled: false # Set to true when reviewer model is active
        reviewer_model: local-openai
    judge: # Scoring model for writing (judge grader)
        enabled: false # Set to true when separate judge is active (distinct judge model recommended)
        judge_model: local-openai
        seeds: 1 # Number of evaluation runs to average per prompt (>1 calculates consistency)

usability: # Tier classification rules based on success_rate × quality
    autonomous: { min_success: 0.9, min_quality: 80 }
    assisted: { min_success: 0.6, min_quality: 0 }

graders: # Thresholds for non-coding graders
    detection: { pass_f1: 0.67 }
    constraint: { pass_ratio: 1.0 }
    judge: { pass_score: 7.0 }

certify_domains: # Domain certification gates (provisional)
    security: { min_success: 0.6, min_combined: 60 }
    general: { min_success: 0.7, min_combined: 65 }
    writing: { min_success: 0.5, min_combined: 55, experimental: true }
    medical: { min_success: 0.6, min_combined: 60, reference: true }
```

> [!TIP]
> **No need to edit model names manually every run.** When `model: "auto"` is set, swapping GGUF models in servers like llama.cpp automatically updates model labels in `llmbench` reports and output filenames via `/v1/models`. For Ollama, pass installed model names directly via `--model <installed_name>` (inspect via `llmbench models`). Override or fix output labels using `--label <name>`.

### 🔌 Connection Targets (Switch Connection Endpoints Without Config Edits)

Pass `--base-url` / `--client-type` directly via CLI:

```bash
# Direct connection to llama.cpp / vLLM / LM Studio (no config edit required)
llmbench run --model auto --client-type openai --base-url http://localhost:8085/v1

# Direct connection to CodeRouter (multiagent)
llmbench run --model router --client-type multiagent --base-url http://localhost:8088

# Connect to remote Ollama instance using installed model name
llmbench run --model qwen2.5-coder:32b --base-url [http://192.168.1.10:11434](http://192.168.1.10:11434)

```

Target Resolution Priority:

| Parameter   | Resolution Priority (Leftmost takes precedence)                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| base_url    | `--base-url` > config `base_url` (`${VAR}` supported) > Environment variables (`OPENAI_BASE_URL` / `OLLAMA_HOST` / `CODEROUTER_BASE_URL`) > Client-type defaults |
| Model Name  | `--client-type` direct parameter > `models:` key in config > Automatic Ollama active model resolution                                                            |
| Ollama Host | `--ollama-host` > `OLLAMA_HOST` env > Ollama model `base_url` in config > `http://localhost:11434`                                                               |

> [!NOTE]
> Unset `${VAR}` environment variable references raise explicit errors (preventing cryptic 401 authentication failures caused by silent empty string evaluation).

## 📁 Project Structure

```
swe-bench/
├── llmbench/
│   ├── clients/        # 🔌 LLM Client Connectors (openai_compat / ollama / mock). Model auto-detection & Ollama listing
│   ├── graders/        # 🌐 Evaluators: code / detection / constraint / judge / qa + checks (IFEval)
│   ├── patch.py        # 📝 LLM Output Parsing (FILE: markers + code blocks)
│   ├── sandbox.py      # 📦 Workspace Isolation & Pytest Execution Sandbox
│   ├── functional.py   # ✅ Functional Resolution Check (used by code grader)
│   ├── quality/        # 🧹 Code Quality Analyzers: ruff / complexity / llm_review / sonar
│   ├── scoring.py      # ⚖️ Composite Score Calculation & pass@k Unbiased Estimators
│   ├── usability.py    # 🧭 Autonomous / Assisted / Unusable Tier Classification
│   ├── certify.py      # 🎓 Tier Certification + Domain Gates + Balance Index + Medical Detail Readout
│   ├── runner.py       # 🎛️ Orchestrator Engine (Pluggable Grader Support & Multi-Run Aggregation)
│   ├── compare.py      # 🆚 Multi-Model Cross-Comparison Report Generator
│   └── report.py       # 📊 Markdown Report Renderer (including domain summary sections)
├── tasks/              # 🧩 40 Built-in Tasks (L1 easy 5 / L2 medium 5 / L3 hard 10 / L4 expert 12 / L5 frontier 8)
│                       #    + tasks_l6.jsonl (L6 architect 20 tasks, --with-l6)
│                       #    + tasks_l7.jsonl (L7 grandmaster 40 tasks, --with-l7)
│                       #    + tasks_sec/gen/write/med.jsonl (Domain ledgers, --with-sec/gen/write/med)
├── DESIGN_DOMAINS.md   # 📐 Multi-Domain Extension Architecture Specifications
└── config.yaml

```

> [!NOTE]
> Patch format uses whole-file replacement (`--- FILE: path ---` + code block).
> Because local LLMs struggle with line-number precision required by unified diffs, whole-file replacements yield significantly higher parsing success rates.

## 🧩 Adding Tasks

**Coding Tasks** (`code` grader):

```
tasks/tXXX_name/
├── issue.md        # English bug report
├── issue_ja.md     # Japanese bug report
├── buggy_code/     # Buggy source files (provided to LLM)
├── gold/           # Correct reference files (only modified files)
└── tests/          # Hidden test suite (withheld from LLM)

```

1. Create directory using the structure above.
2. Add a single-line JSON record to `tasks/tasks.jsonl` (difficulty levels: easy / medium / hard / expert / frontier / architect / grandmaster):

```json
{ "task_id": "t0XX", "dir": "t0XX_name", "difficulty": "expert", "title": "..." }
```

For performance-constrained tasks, add `"perf_timeout": <seconds>` to apply custom execution timeouts.
For regression traps, place multiple test files inside `tests/` (e.g., `test_core.py` locks existing behavior, `test_bug.py` catches target bug). L6 (architect) tasks belong in `tasks/tasks_l6.jsonl` and L7 (grandmaster) tasks belong in `tasks/tasks_l7.jsonl`, merged only when passing `--with-l6` or `--with-l7` (ledgers can be overridden via `--l6-ledger` / `--l7-ledger`). 3. Verify task: Run `llmbench validate --tasks t0XX` (passes when gold passes and buggy code fails).
Validate L6 tasks via `llmbench validate --with-l6 --tasks t0XX` and L7 tasks via `llmbench validate --with-l7 --tasks t0XX`.

**Domain Tasks** (`detection` / `constraint` / `judge` / `qa`) omit `buggy_code` and `tests` directories.
Task ledger records specify `grader` and `domain`. Schema layouts vary per grader (detection = `gold.json` labels; constraint = `checks.json` + `gold_answer.md`; judge = `rubric.json` + `gold_answer.md`; QA = `gold.json` keys). Refer to [📐 DESIGN_DOMAINS.md](DESIGN_DOMAINS.md) for full specifications.

```json
{"task_id":"s01","dir":"s01_name","grader":"detection","domain":"security","difficulty":"sec_medium","title":"..."}
{"task_id":"g01","dir":"g01_name","grader":"constraint","domain":"general","difficulty":"gen_easy","title":"..."}
{"task_id":"m01","dir":"m01_name","grader":"qa","domain":"medical","difficulty":"med_std","title":"..."}

```

Verify domain tasks via `llmbench validate --only-sec|gen|write|med`.

## 🗺️ Roadmap

- [x] Composite evaluation pipeline combining functional correctness + code quality
- [x] Dual OpenAI-compatible / Ollama client connectivity
- [x] Dual EN/JA bug reports for measuring language tax
- [x] 🎲 Reliability (pass@k / success rate) evaluation
- [x] 🧭 Usability tier classification
- [x] 🆚 Cross-model comparison report generation (`compare`)
- [x] 🔎 `model: auto` / Dynamic Ollama active model resolution
- [x] 🧩 High-difficulty tiers (L4 expert / L5 frontier) breaking score ceiling effects (40 tasks total)
- [x] 🏛️ L6 architect tier (t041–t060, 20 tasks) added as optional `--with-l6` flag for top-tier model separation
- [x] 🏆 L7 grandmaster tier (t061–t100, 40 tasks) added as optional `--with-l7` flag for ceiling evaluation (provisional gates: success ≥ 0.35 / combined ≥ 55)
- [x] 🎓 Usability line tier-based certification system (`certify`, L1–L7)
- [x] ⏱️ Per-task `perf_timeout` support for performance-constrained tasks
- [x] 🔀 Split execution (`--only-l6`/`--only-l7`) and unified certification consolidation via `certify --merge`
- [x] 🌐 Multi-domain evaluation engine (pluggable graders: security/general/writing/medical) + domain-specific certification & Balance Index
- [x] 🩺 24 Medical QA tasks (dual-language JP/EN support, independently fact-checked) — reference values
- [ ] 🎯 Empirical model calibrations to finalize tier/domain gates (adding 32b dense & 3b models; finalizing L7, judge, and medical gates)
- [ ] ✍️ Multi-engine judge consensus & agreement metric adoption for writing evaluations
- [ ] 🐳 Isolated Docker container execution sandbox
- [ ] 📥 Integration of official SWE-bench Lite tasks
- [ ] 🔄 Automated task extraction directly from GitHub repositories
- [ ] 📈 Automated VRAM tracking via `nvidia-smi`

## 🤝 Contributing

Pull requests adding tasks, code quality evaluators, or new evaluation domains are welcome! Please ensure `llmbench validate` (and domain-specific validations via `--only-*`) pass before submitting PRs using Conventional Commits (`feat:`, `fix:`).

## 📜 License

[MIT](LICENSE)

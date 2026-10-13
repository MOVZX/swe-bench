# 🔁 Communication Retry: Transient Error Retry for OpenAI-Compatible Clients

Fixed an issue where single-shot `Read timed out` or `Connection reset` errors caused generations to be **treated as immediate score deductions (failures)** when using cloud APIs like QwenCloud.

| Addition/Change                        | Content                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔁 `llmbench/clients/openai_compat.py` | Automatically retries transient network-induced failures (`ConnectionError` / `Timeout` / `ChunkedEncodingError` / `ConnectionResetError`) using exponential backoff. Default: 2 retries (3 total attempts), initial delay of 2 seconds (2s → 4s). HTTP 4xx/5xx and JSON parsing errors are not retried (since the cause originates on the caller side and retrying won't fix it). |
| ⚙️ `config.yaml`                       | `transient_retries` / `transient_backoff` can be overridden per model (defaults: 2 / 2.0). Existing configs do not require modifications (enabled by default).                                                                                                                                                                                                                     |
| 📝 Implementation                      | Renamed existing `_generate` to `_post_once`; the new `_generate` acts as a retry wrapper. Full backward compatibility with the existing calling convention of `LLMClient.generate` is maintained.                                                                                                                                                                                 |

**Background**: In actual calibration measurements for L7 v2 (qwen-coding, 2026-07-21), **3 out of 16 tasks** resulted in point deductions due to `HTTPSConnectionPool ... Read timed out (read timeout=600)`. These were single-shot communication drops on the Alibaba Cloud API side rather than model failures; counting them as failures under `runs=1` introduces noise into capability evaluations. After adding retries, a single task generation will be attempted up to 3 times (default).

**Verification**: Syntax OK. Compatibility with the calling convention of existing `openai_compat.py` (`LLMClient.generate` in `base.py`) was maintained. Retry logs are output to stderr (`⚠️ transient error on qwen-coding (attempt 1/3): ReadTimeout: ... — retry in 2.0s`).

---

# 🆕 Multi-Domain Evaluation — security / general / writing / medical (Pluggable Grader)

Expanded the evaluation framework—previously dedicated to coding—to other capabilities by making **graders pluggable**. Each grader includes an "output contract (prompt)" and "scoring", and normalizes results back to `(resolved, quality)` for return. Thus, pass@k, combined score, usability, and certification features are shared **without modification**. **Default behavior and existing scores remain unchanged** (when `grader` is unspecified, it defaults to `code`; running only default ledgers yields identical results).

| Domain      | Grader     | Ledger / Flag                                     | Scoring                                                                                                                                                                     |
| ----------- | ---------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🛡️ security | detection  | tasks_sec.jsonl / `--with-sec` `--only-sec`       | Precision / Recall / F1 for vulnerability and breach detection. Penalizes false positives (FP) using **clean decoys**.                                                      |
| 📋 general  | constraint | tasks_gen.jsonl / `--with-gen` `--only-gen`       | Programmatically verifies instruction following IFEval-style (character length / JSON / regex...). Success if all checks pass; pass rate represents `quality`.              |
| ✍️ writing  | judge      | tasks_write.jsonl / `--with-write` `--only-write` | 0–10 scoring via rubric + judge (experimental). Deterministic judgment based on hard constraints only when judge is unavailable.                                            |
| 🩺 medical  | qa         | tasks_med.jsonl / `--with-med` `--only-med`       | Matches medical QA against answer keys. Includes **Japanese/English allowed terms** in gold → Japanese models with `--lang ja` are also scored correctly (reference value). |

| Addition/Change            | Content                                                                                                                                                                                                                                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🌐 `llmbench/graders/`     | base/registry + `checks.py` (IFEval) + `code`/`detection`/`constraint`/`judge`/`qa`                                                                                                                                                                                                                       |
| 🚩 CLI                     | Added `--with-sec/gen/write/med` and `--only-*` (same architecture as `--with-l6/l7`). Extended `_ledgers()`.                                                                                                                                                                                             |
| 🎓 `certify`               | Domain gates + **Balance Index** (harmonic mean of coding + non-experimental domains to penalize single-trick specializations) + Medical accuracy broken down by difficulty (basic/std/hard).                                                                                                             |
| ⚙️ config                  | Config entries for `graders:` (pass_f1 / pass_ratio / pass_score), `quality.judge:`, and `certify_domains:`.                                                                                                                                                                                              |
| 📊 report                  | Added domain-specific summary sections.                                                                                                                                                                                                                                                                   |
| 🩺 24 Medical QA Questions | Pharmacology, Cardiology, Emergency, Endocrinology, Infection, Nephrology, Neurology, Pediatrics, OB/GYN, Toxicology. **Fact-checked by independent agents**. Includes MCQ + short answer, difficulties: basic/std/hard. Japanese medical models can also be measured via Japanese/English allowed terms. |
| 📐 Specifications          | `DESIGN_DOMAINS.md` (Pluggable grader architecture, schemas, and scoring rules).                                                                                                                                                                                                                          |

**Verification**: `llmbench validate --only-sec|gen|write|med` all PASSED (all gold succeeded, all broken failed).
Verified grader discrimination tests (correct answer → high score / ambiguous → 0 / decoy over-detection → precision 0 fail / 4 bullet points → 75 points), as well as correct/incorrect disqualifications for Japanese answers (e.g., "IM adrenaline injection", "subarachnoid hemorrhage").

> Note: Gate thresholds for writing/medical are provisional (uncalibrated). Medical scores are reference values (5-choice MCQ chance accuracy ≈ 20%) rather than guarantees of clinical validity. Judgments can be tuned via `DEFAULT_DOMAIN_GATES` / `DEFAULT_MED_GATES` in `certify.py` and `graders:` / `certify_domains:` in `config.yaml`.

---

# 🆕 Split Execution Support — `--only-l6` / `--only-l7` and `certify --merge`

Because running all questions including L6/L7 takes considerable time, we added support for **split operations: run only the default 40 questions first, run L6/L7 later as needed, and finally merge them into a single consolidated certification**. **Default behavior and existing `--with-l6`/`--with-l7` flags remain unchanged**. Only when `--only-l6`/`--only-l7` is specified will the default ledger `tasks.jsonl` be excluded to execute only the designated tier (shared across list-tasks / run / validate).

| Flag                  | Target Questions                                                                                                                                    | Question Count |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| None                  | Default 40 questions                                                                                                                                | 40             |
| `--with-l6`           | Default 40 questions + L6 (20 questions)                                                                                                            | 60             |
| `--with-l7`           | Default 40 questions + L7 (40 questions)                                                                                                            | 80             |
| `--with-l6 --with-l7` | Default 40 questions + L6 (20 questions) + L7 (40 questions)                                                                                        | 100            |
| `--only-l6`           | L6 (20 questions only, no base)                                                                                                                     | 20             |
| `--only-l7`           | L7 (40 questions only, no base)                                                                                                                     | 40             |
| `--only-l6 --only-l7` | L6 (20 questions) + L7 (40 questions) (no base)                                                                                                     | 60             |
| `--only-l6 --with-l7` | L6 (20 questions) + L7 (40 questions) (If at least one `--only` flag is present, base is excluded; final set is the union of `--only` and `--with`) | 60             |
| `--only-l6 --with-l6` | L6 (20 questions) (no duplicate additions)                                                                                                          | 20             |

| Addition                     | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔀 `--only-l6` / `--only-l7` | Added to `_common_args` shared across list-tasks / run / validate. Excludes default ledger `tasks.jsonl` and targets only the ledgers of specified tiers. Can be combined directly with `--with-l6`/`--with-l7` and `--l6-ledger`/`--l7-ledger`.                                                                                                                                                                                                                                                     |
| 🔗 `certify --merge`         | Merges the `results` array across multiple `results.json` files to generate a single tier certification output. Supported via both `llmbench certify --merge a.json b.json` and standalone non-llmbench script `python3 certify.py --merge a.json b.json`.                                                                                                                                                                                                                                           |
| 🧮 `merge_results()`         | Duplicate `task_id`s use **last-wins logic** (prioritizes the file specified later, intended to overwrite with re-measurement results). Model names are concatenated with `" + "` using distinct ordering of appearance in each file (single entry if identical). Merging results with different `--runs` counts is also supported (aggregates via task-level `success_rate` average, so calculations will not break, though care should be taken regarding non-uniform trial counts within a tier). |

**Recommended Flow**: `llmbench run` (default 40 questions) → `llmbench run --only-l6` later (L6 20 questions only) → `llmbench certify --merge base.json l6.json` to generate unified certification for L1–L6 in a single run.

**Verification**: Verified `list-tasks --only-l6` = 20 / `--only-l7` = 40 / `--only-l6 --only-l7` = 60.
Verified via additional tests in `tests/test_ledgers.py` (ledger selection matrix) and `tests/test_certify_merge.py` (last-wins logic in `merge_results`, model name concatenation, mixed runs).

---

# 🆕 L7 Grandmaster Tier (t061–t100) — Optional Flag `--with-l7`

Even under L6 architect, top-tier models (27B dense) cleared almost all tasks (difference between the top two models was effectively 1 question, with active discriminating tasks reduced to ~4: t059/t047/t046/t043), creating a ceiling effect once again. Added **40 ceiling-evaluation questions (L7 grandmaster)**. **Default behavior remains unchanged** (40 questions as before, 60 questions with `--with-l6`), and +40 questions are added only when `--with-l7` is specified.

| Addition                                    | Content                                                                                                                                                                                                                       |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🏔️ L7 grandmaster (t061–t100, 40 questions) | 5 axes × 8 questions: Numerical Stability (t061–068) / State Consistency (t069–076) / Multi-Coupling Bugs (t077–084) / Deep Concurrency (t085–092) / Adversarial Parsing & Security (t093–100). Issues present symptoms only. |
| 🔀 `--with-l7` / `--l7-ledger`              | Optionally merges external ledger `tasks/tasks_l7.jsonl` (default 40 → 80; 100 when combined with `--with-l6`). Existing ledgers remain unchanged.                                                                            |
| 🕵️ Stealth Criterion (New)                  | Across all 40 questions, buggy implementations pass `test_core` (regression trap)—meaning bugs are invisible to linters or standard happy-path tests.                                                                         |
| 🎓 Certify L7 Gate                          | `grandmaster → L7`, provisional pass@1 ≥ 35% and combined ≥ 55 (to be finalized via real model calibration).                                                                                                                  |

**Verification**: L7 selfcheck 40/40 (gold green / buggy red / stealthiness ✓ / ruff 0). Axis C verified for failure on partial fixes; Axis D verified for determinism with 10 consecutive gold passes and 10/10 buggy failures. `validate --with-l7` confirmed gold 40/40 and broken 40/40; `list-tasks` confirmed 40 (default) / 80 (`--with-l7`) / 100 (`--with-l6 --with-l7`). Only t098 sets `perf_timeout: 30` (ReDoS performance check).

---

# 🆕 `--concurrency` (Parallel Execution of Trials) — Total Time Reduction Option

Added `--concurrency K` (default 1 = sequential) to run each trial in `--runs N` concurrently. Effective when launching llama.cpp with `--parallel K -cb`. Default value can also be set via `run.concurrency`.

| Addition                                 | Content                                                                                                                           |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 🔀 `--concurrency K` / `run.concurrency` | Parallelizes trial execution (`runs`) using `ThreadPoolExecutor` (`min(K, runs)`). MockClient falls back to sequential execution. |

**Trade-off**: Total runtime ↓ (~2.2x speedup in measurements) / single stream tok/s ↓ (264 → 110). Accuracy and quality are unaffected. Benchmark speed measurements recommend `--concurrency 1`.

---

# 🆕 L6 Architect Tier (t041–t060) — Optional Flag `--with-l6`

Because top-tier models (27B–35B class) hit ceiling effects again in frontier (L5), we added **an even more challenging set of 20 questions (L6 architect)**. **Default behavior remains unchanged** (40 questions as before); +20 questions are added only when `--with-l6` is appended.

| Addition                                  | Content                                                                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 🏛️ L6 architect (t041–t060, 20 questions) | Multi-file (8) / Non-functional perf (6) / Ambiguous specs (4) / Traps & Adversarial (2). Issues present symptoms only. |
| 🔀 `--with-l6` / `--l6-ledger`            | Optionally merges external ledger `tasks/tasks_l6.jsonl` (default 40 → 60). `tasks.jsonl` remains unchanged.            |
| 🧩 Multi-Ledger Loader                    | `load_tasks(..., ledgers=[...])` (first ID wins) / `BenchmarkRunner(ledgers=...)`.                                      |
| 🎓 Certify L6 Gate                        | `architect → L6`, provisional pass@1 ≥ 55% and combined ≥ 60 (to be finalized via real model calibration).              |

**Verification**: L6 selfcheck 20/20 (gold green / buggy red / ruff 0 / CC A–B). `validate --with-l6` confirmed gold 20/20 and broken 20/20; `list-tasks` confirmed 40 (default) / 60 (`--with-l6`).

---

# High-Difficulty Tiers (L4/L5) + Usability Line Certification (certify)

To resolve the **ceiling effect** where high-performing local coders reached a plateau on the existing 20 questions (easy/medium/hard) with no clear differentiation, we added high-difficulty tiers and introduced a tier-pass system to evaluate "is this model practical to use if it passes here?".

| Addition                                | Content                                                                                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧩 L4 expert (t021–t032, 12 questions)  | Challenging tasks demanding specification detail mastery and algorithmic accuracy.                                                            |
| 🧩 L5 frontier (t033–t040, 8 questions) | Multi-file tasks, regression traps, and performance constraints. 4 tasks contain two coupled bugs.                                            |
| 📝 Symptom-based Issues                 | New tiers omit step-by-step fix instructions and instead require **root-cause diagnosis** (addressing the main driver of the ceiling effect). |
| 🎓 `llmbench certify`                   | Maps difficulties → tiers (L1–L5), evaluating **L4 independent pass = usability threshold**.                                                  |
| ⏱️ Per-task `perf_timeout`              | Introduced per-task timeouts for performance-constrained tasks.                                                                               |

**Verification**: Selfcheck for 20 new tasks 20/20, `llmbench validate` 40/40 (all gold green / all broken fail), `llmbench list-tasks` total 40 (easy 5 / medium 5 / hard 10 / expert 12 / frontier 8).

> Calibration Note: Empirical evaluations using 3 models (7b/14b/30b) × 5 runs identified dormant tasks, difficulty inversions, and excessive issue hints, which were resolved by shifting issues to symptom-based descriptions, creating double-bug frontier tasks, and making certification non-cumulative. Tier thresholds (`DEFAULT_GATES`) are provisional and will be finalized after rerunning benchmarks on the increased difficulty set.

---

# 🆕 New Features in `llmbench` (Measuring Practical Usability)

To solve the issue where "benchmark scores are generated, but it is hard to tell how usable the model actually is in practice," we added 4 key features: **Reference Model Comparison / pass@k Reliability / Usability Tiers / High-Difficulty Tasks**, alongside **Simplified Model Operations (model:auto / Dynamic Ollama)** and **Improved Report Displays**.

> This document summarizes new features, usage, and validation results.
> For detailed operations, see `MANUAL.md` (existing); for setup procedures, see `USAGE.md`.

---

## Why These 4 Features?

Benchmark scores failed to translate directly into "practical usability" for two reasons: **lack of baselines (anchors)** and **tasks hitting score ceilings without differentiation**. Furthermore, "usable" does not equal "highest score"—**usable = success rate × reliability × speed**. These features make that measurable.

| Feature                                  | Problem Solved                                                                                                               |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ① Reference Model Comparison (`compare`) | A score of 94.3 is hard to evaluate in isolation → Contextualizes performance alongside strong/weak benchmark models.        |
| ② pass@k Reliability (`--runs`)          | 1 success ≠ usable in production → Measures success rates and pass@k across N attempts.                                      |
| ③ Usability Tiers                        | Raw metrics don't map to actions → Translates scores into Autonomous / Assisted / Unusable.                                  |
| ④ High-Difficulty Tasks t016–t020        | 100% completion across the board obscures differences → Creates a realistic ceiling using practical, failure-inducing tasks. |

---

## ① Reference Model Comparison `compare`

Generates cross-comparison reports across multiple `results.json` files. API keys can be passed via environment variables (added `ref-gpt` referencing `${OPENAI_API_KEY}` in `config.yaml`).

```bash
export OPENAI_API_KEY=sk-...
llmbench run --model local-openai --output results   # Target model
llmbench run --model ref-gpt      --output results   # Strong reference (API)
llmbench compare results/*_results.json --output results

```

The output (`comparison_<stamp>.md`) includes rankings ordered by Combined score descending (relative score = ratio against best model), usability tier comparisons, and task-by-task Combined score matrices (bolding row-wise best scores).

## ② pass@k Reliability `--runs N`

Samples each task N times to compute success rates, pass@1, and pass@k (unbiased estimator per Chen et al. 2021). During multi-run executions, `combined = success_rate × (...)`, directly reflecting **reliability in the overall score**.

```bash
llmbench run --model local-openai --runs 5            # 5 trials
llmbench run --model local-openai --runs 5 --sample-temp 0.8

```

- When `runs=1` (default), behavior and metrics remain identical to previous versions (`combined` matches `resolved × (...)`).
- Partial successes are recorded as `flaky 3/5 passed`, adding success rate and pass@k columns to report outputs.

## ③ Usability Tiers

Categorizes tasks into **🟢 Autonomous / 🟡 Assisted / 🔴 Unusable** based on `success_rate` and `quality`, generating difficulty-by-tier matrices and overall usability assessments in reports. Thresholds configured in `config.yaml`:

```yaml
usability:
    autonomous: { min_success: 0.9, min_quality: 80 } # Requires almost no review
    assisted: { min_success: 0.6, min_quality: 0 } # Usable assuming review process
    # Below above thresholds = unusable
```

Reflected in `summary.usability` and individual `usability_tier` fields within `results.json`.

## ④ High-Difficulty Tasks t016–t020 (All Hard)

Added 5 realistic, difficult tasks designed to fail probabilistically to establish score ceilings.

| ID   | Theme                                            | Source of Difficulty                                  |
| ---- | ------------------------------------------------ | ----------------------------------------------------- |
| t016 | Temperature Aggregation (convert.py + report.py) | Requires **multi-file** modifications.                |
| t017 | Slugify                                          | **Ambiguous specifications** (deducing edge cases).   |
| t018 | Plugin Registry                                  | **Understanding existing abstractions (decorators)**. |
| t019 | Trie Search / starts_with                        | **State bugs** (missing end-of-word flags).           |
| t020 | Expression Evaluator                             | **Operator precedence & parenthesis refactoring**.    |

---

## ⑤ Simplified Model Operations (Config-Edit Free)

Eliminated the need to edit `config.yaml` every time GGUF files or models are swapped out.

- **`model: auto`** — Automatically detects and adopts the server's actively loaded model via `/v1/models` on startup. Reports and results files are **labeled with the actual model name** (cleans `.gguf` suffixes).
- **Dynamic Ollama Selection** — Directly specify unconfigured models via `--model <installed_name>` (auto-resolved via `/api/tags`). `llmbench models` lists both config-defined and active Ollama models.
- **`--label <name>`** — Used to explicitly fix/override output labels.
- **Environment Variable Expansion for API Keys** — Supports `api_key: "${OPENAI_API_KEY}"` to prevent hardcoding.

## ⑥ Improved Report Reliability Indicators (Eliminating Misinterpretations)

- **Main Metric Shifted to pass@1 (Success Rate)** — Removed `pass@5` due to degradation at k=runs (evaluates to 1.0 if passed at least once, causing misinterpretations). Separated into "Average Success Rate (pass@1)" and "Tasks with ≥1 Success".
- **Conservative Usability Assessment** — Removed optimistic most-frequent-tier displays; updated recommendation logic to evaluate difficulty proportions and "avoid labeling as Autonomous if even a single 🔴 Unusable task exists".
- **Breakdown Notes** — Explicitly noted that quality breakdowns in multi-run executions represent 1 representative trial while Quality represents the average.

---

## Modified Files

Added: `llmbench/usability.py`, `llmbench/compare.py`, `tasks/t016_temps` to `t020_calc/`
Modified: `llmbench/{runner,report,scoring,cli}.py`, `llmbench/clients/{openai_compat,ollama}.py`, `config.yaml`

## Validation Results (All PASSED)

- `llmbench validate` … **gold 20/20 resolved, broken 20/20 fail, PASS** (across all 20 tasks).
- Validity of New Tasks … Confirmed buggy_code fails tests on all tasks, and gold fixes pass all tasks individually.
- pass@k Standalone … Confirmed backward compatibility (`combined(True,88.6) = 94.3`), unbiased estimator values, and flaky/unusable classifications.
- Multi-run Execution … Confirmed aggregation and report/JSON output for success rate, pass@k, and usability tiers.
- compare … Confirmed rankings, relative scores, comparison matrices, and API key environment variable expansion.
- Model Resolution … Confirmed server detection for `model:auto`, label formatting, dynamic Ollama resolution, and user-friendly error messages when servers are offline.
- ruff (E, F, W, B, SIM, C4, S) … Zero issues across added and modified files. `compileall` OK.

## Backward Compatibility & Migration Notes

- Under `runs=1`, existing field values remain unchanged. Additional fields are **added** to `results.json` (`runs`, `success_rate`, `pass_at_1`, `pass_at_k`, `attempts`, `usability_tier`, `summary.usability`, `solved_any_rate`, etc.).
- A **"Usability Tier" column** is now permanently added to task results in `report.md`; multi-run executions will also include a **"Reliability" column**. Take note of column count shifts if programmatic parsing is applied to Markdown tables.
- The `pass@k` row in summary outputs has been removed and replaced with **Success Rate (pass@1)** and **≥1 Success** (⑥).

# 🛠️ llmbench Operations Manual (artifacts / reliability / usability / comparison / model resolution)

This document serves as an **operations and reference manual** covering artifact preservation, report overhauls, and subsequent features including **reliability (pass@k), usability classifications, cross-model comparison, and model resolution (model:auto / dynamic Ollama)** from the perspective of operations, maintenance, and system integration.

Document Roles:

| Document                              | Target Audience                 | Content                                                                                           |
| ------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------- |
| [README.md](README.md)                | Beginners                       | Overview, features, score definitions, task additions                                             |
| [USAGE.md](USAGE.md)                  | Users                           | Execution steps and **how to interpret results**                                                  |
| [CHANGES.md](CHANGES.md)              | Everyone                        | Summary of new features and change history                                                        |
| **This Document (Operations Manual)** | Ops / Maintainers / Integrators | Output file **specifications**, internal implementation, operational notes, migration, automation |

> Target Version: Default 40 tasks (L1–L5) + optional L6 architect 20 tasks (`--with-l6` / standalone `--only-l6`) + optional L7 grandmaster 40 tasks (`--with-l7` / standalone `--only-l7`), multi-run (pass@k), usability, compare, certify (L1–L7, integrating split results via `certify --merge`), and `model:auto` support.
> `results.json` includes `summary.usability`, `success_rate`, etc.

---

## Table of Contents

1. [Overall Picture of New Features](%231-overall-picture-of-new-features)
2. [Output File Specifications (Reference)](%232-output-file-specifications-reference)
3. [Internal Implementation Reference (For Maintainers)](%233-internal-implementation-reference-for-maintainers)
4. [Operational Considerations (Disk, Retention, git)](%234-operational-considerations-disk-retention-git)
5. [Automation & CI Integration](%235-automation--ci-integration)
6. [Programmatic Access to Results](%236-programmatic-access-to-results)
7. [Migration Notes (Differences from Old Formats)](%237-migration-notes-differences-from-old-formats)
8. [Change File Quick Reference Table](%238-change-file-quick-reference-table)
9. [Additional Subsystems (Reliability, Usability, Comparison, Model Resolution)](%239-additional-subsystems-reliability-usability-comparison-model-resolution)
10. [Multi-Domain Evaluation (Grader Extension)](%2310-multi-domain-evaluation-grader-extension)
11. [Evaluation Procedures for Cloud LLMs (Hosted APIs)](%2311-evaluation-procedures-for-cloud-llms-hosted-apis)

---

## 1. Overall Picture of New Features

In contrast to older versions that only produced numbers, 3 main features were added to ensure **"users can trace low scores or failures back to actual generated code"**:

| #   | Feature                            | Purpose                                                                     | Key Implementation                        |
| --- | ---------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------- |
| ①   | **Artifact Preservation**          | Trace causes of failure or low quality at the code level                    | `runner.py`: `_write_artifacts()`         |
| ②   | **Report Overhaul**                | Clear summary tables, task-level details, human-readable quality breakdowns | `report.py`: `render_markdown()` etc.     |
| ③   | **Execution Logging Enhancements** | Monitor run-time execution and progress on the fly                          | `runner.py`: `_log_task()` / `_snippet()` |

Secondarily, large generation outputs were separated into a dedicated directory (`artifacts/`) to keep `results.json` as a **lightweight JSON** focused on summary statistics and scores.

Subsequently, additional subsystems were introduced to measure **"practical usability"** (detailed in **Section 9**, summarized in `CHANGES.md`):

| #   | Feature                                              | Purpose                                                                                   |
| --- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| ④   | **Reliability (pass@k / `--runs`)**                  | Measure stability via success rates across multiple trials rather than single-run passes  |
| ⑤   | **Usability Tiers**                                  | Translate raw scores into actionable decisions: 🟢 Autonomous / 🟡 Assisted / 🔴 Unusable |
| ⑥   | **Cross-Model Comparison (`compare`)**               | Benchmark target models side-by-side against reference anchors                            |
| ⑦   | **Model Resolution (`model:auto` / Dynamic Ollama)** | Swap models seamlessly without manually editing configuration files                       |
| ⑧   | **High-Difficulty Tasks t016–t020**                  | Establish realistic ceilings with failure-inducing tasks (40 tasks total)                 |

---

## 2. Output File Specifications (Reference)

A single `run` command outputs **3 items** to the designated output folder (default `results/`).
`<stamp>` represents execution timestamp (`YYYYMMDD_HHMMSS`), and `<model>` represents the key specified in `--model`.

```
results/
├── <stamp>_<model>_results.json      # ① Summary & scores (machine-readable, lightweight)
├── <stamp>_<model>_report.md         # ② Human-readable Markdown report
└── <stamp>_<model>_artifacts/        # ③ Output artifacts
    └── <task_id>/
        ├── llm_output.txt            #   Raw text output from LLM
        ├── generated/<path>          #   Applied generated code files
        └── test_output.txt           #   pytest console output

```

### 2.1 `results.json` Schema (Lightweight JSON)

Top level:

| Key             | Type  | Description                                                                          |
| --------------- | ----- | ------------------------------------------------------------------------------------ |
| `model`         | str   | Model label. Detected server model name when `model:auto`, overridable via `--label` |
| `issue_lang`    | str   | `en` / `ja`                                                                          |
| `artifacts_dir` | str   | Relative path to the corresponding artifacts directory                               |
| `summary`       | obj   | See table below                                                                      |
| `results`       | array | Task results array (see table below)                                                 |

`summary`:

| Key                    | Description                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `resolved_rate`        | Resolved ratio (0.0–1.0, 3 decimals). In multi-run, tasks with `success_rate >= 0.5` are treated as resolved |
| `avg_quality_resolved` | Quality average of **resolved tasks only** (1 decimal)                                                       |
| `avg_combined`         | Average combined score across all tasks (1 decimal)                                                          |
| `n_tasks`              | Total task count                                                                                             |
| `runs`                 | Number of trials per task                                                                                    |
| `usability`            | Task count per tier `{autonomous, assisted, unusable}`                                                       |
| `avg_success_rate`     | Average success rate (= average pass@1). **Present only in multi-run**                                       |
| `solved_any_rate`      | Ratio of tasks passing ≥1 time out of N runs. **Present only in multi-run**                                  |
| `avg_pass_at_k`        | Average pass@k. **Present only in multi-run** (Note: degenerates when k = runs)                              |

`results[]` (**Lightweight** — large output fields excluded):

| Key                                                    | Type     | Description                                                                    |
| ------------------------------------------------------ | -------- | ------------------------------------------------------------------------------ |
| `task_id`                                              | str      | Example: `t001`                                                                |
| `difficulty`                                           | str      | `easy` / `medium` / `hard`                                                     |
| `title`                                                | str      | Task title                                                                     |
| `resolved`                                             | bool     | Pass/fail determination (In multi-run, `success_rate >= 0.5`)                  |
| `quality_score`                                        | float    | 0–100. In multi-run, **average quality across successful attempts**            |
| `combined`                                             | float    | 0–100. Computed via `success_rate × (floor + (1-floor) × quality / 100) × 100` |
| `latency_sec` / `tokens_per_sec` / `completion_tokens` | —        | Latency and throughput metrics (averaged in multi-run)                         |
| `fail_reason`                                          | str      | Failure description. Partial success reported as `flaky c/N passed`            |
| `parse_ok` / `parse_error`                             | bool/str | Parse status and reason for the representative run                             |
| `runs` / `n_pass`                                      | int      | Total runs / successful runs count                                             |
| `success_rate` / `pass_at_1` / `pass_at_k`             | float    | Success rate (= pass@1) / pass@k metric                                        |
| `attempts`                                             | array    | Array of attempt records: `{resolved, quality, combined, fail_reason}`         |
| `usability_tier`                                       | str      | `autonomous` / `assisted` / `unusable`                                         |
| `quality_components`                                   | obj      | ruff / complexity / etc. details for representative attempt                    |

> ⚠️ **Fields excluded from `results.json**`(moved to`artifacts/`):
`raw_output`(raw LLM text),`parsed_files`(generated code files),`test_output`(pytest output).
These are stripped via`\_lean()`in`save_run()`using`ARTIFACT_FIELDS`.
`changed_files`is a class property and omitted from JSON (refer to`generated/` in artifacts).

### 2.2 `report.md` Structure

The overhauled structure follows: **Summary → Task Results Table → Difficulty Breakdown → Detailed Task Breakdown**.

- **Summary**: Displays Resolved Rate (n/N), Average Quality, and Combined Average in a summary table. Notes artifact output path if present.
- **Task Results Table**: Leading column shows ✅/❌ status icons, alongside **Generated Files** and **Notes** (failure reasons/parse errors) columns.
- **Difficulty Breakdown**: Shows resolved count and quality average grouped by easy/medium/hard tiers.
- **Detailed Task Breakdown**: Per-task difficulty, usability classification, generated file list, artifact path, and **formatted quality breakdowns** (ruff issues and code complexity explained in plain text).

### 2.3 `artifacts/<task_id>/` Directory Contract

| File Path                   | Generation Condition            | Content                                                         |
| --------------------------- | ------------------------------- | --------------------------------------------------------------- |
| `llm_output.txt`            | When `raw_output` is non-empty  | Complete raw text output generated by the LLM                   |
| `generated/<relative_path>` | Per successfully parsed file    | Code files extracted and applied (normalized trailing newlines) |
| `test_output.txt`           | When `test_output` is non-empty | pytest output (stdout on success; stdout + stderr on failure)   |

> 📌 **Failed executions are preserved**: Tasks failing parsing will leave `generated/` empty while preserving `llm_output.txt` for debugging. Task subdirectories are created for all tasks regardless of status.

---

## 3. Internal Implementation Reference (For Maintainers)

### 3.1 `runner.py`

- **New fields on `TaskResult**`: `title`/`parse_ok`/`parse_error`/`parsed_files`/`raw_output`/`test_output`, and `changed_files`(property returning keys of`parsed_files`).
- **`ARTIFACT_FIELDS = ("raw_output", "parsed_files", "test_output")`**:
  Stripped from `asdict()` inside `save_run()` via `_lean()` to maintain compact JSON outputs.
- **`_run_task()`**: Attaches `raw_output` immediately after generation, attach parse states (`parse_ok`/`parse_error`/`parsed_files`), and attaches `test_output` following functional evaluations.
- **`_write_artifacts(run, artifacts_dir)`**: Writes `llm_output.txt`, `generated/<path>`, and `test_output.txt` per task directory.
- **`_log_task(progress, tr)` / `\_snippet(text, n)**`: Formats execution logs to stdout (displaying files generated, LOC, code snippet header, and trailing 3 lines of failed pytest output).
- **`save_run()`**: Creates `<stamp>_<model>_artifacts`, calls `_write_artifacts()`, sets `run.artifacts_dirname`, and writes out lean `results.json` and `report.md`.

### 3.2 `report.py`

| Function                  | Role                                                                         |
| ------------------------- | ---------------------------------------------------------------------------- |
| `render_markdown(run)`    | Assembles full report output                                                 |
| `_fmt_ruff(c)`            | Formats ruff issue counts, issue density, rule violations, and penalties     |
| `_fmt_complexity(c)`      | Formats radon Maintainability Index, worst CC rank, and highest CC functions |
| `_fmt_component(name, c)` | Routes quality component formatting per evaluator layer                      |
| `_status_icon(r)`         | Maps boolean resolved state to ✅/❌ status icons                            |

### 3.3 Log Output Format (`_log_task`)

```
[3/15] t003 (easy) Deduplicate List Items
    Generation OK  files=[dedup.py]  214tok @ 126.9tok/s
      └ dedup.py (12 LOC): def dedup(items): ⏎ seen = set()
    ✅ RESOLVED  quality=100 combined=100  (1.5s)

```

- Parse failures display `Generation Parse Failed: <reason>` alongside `Output Preview:` (first few lines of raw output).
- Test failures output trailing 3 lines of pytest logs formatted with `| ...`.

---

## 4. Operational Considerations (Disk, Retention, git)

### 4.1 Disk Usage

Artifact folders save `llm_output.txt` (full text), `generated/`, and `test_output.txt` **on every single run execution**.
Disk usage accumulates proportional to `tasks × models × runs`. Periodic cleanup is recommended:

```bash
# Check overall size of results directory
du -sh results/*

# Keep latest 5 runs and delete older execution artifacts
ls -dt results/*_artifacts | tail -n +6 | xargs rm -rf
ls -t  results/*_results.json | tail -n +6 | xargs rm -f
ls -t  results/*_report.md   | tail -n +6 | xargs rm -f

```

### 4.2 `.gitignore`

Execution outputs should generally be kept out of version control. Ensure repository `.gitignore` includes `results/` entries (exercise caution when sharing artifacts externally, as raw outputs contain full task prompt texts):

```gitignore
results/
*_artifacts/

```

### 4.3 Immutability of Original Tasks

All artifact writes are restricted to the designated output results directory. Task source files in `tasks/` remain strictly immutable (patches are applied inside isolated temporary workspaces).

---

## 5. Automation & CI Integration

### 5.1 Pipeline Health Gate (`validate`)

Mock validation requiring no external LLM connections. Serves as a pre-commit or CI health gate:

```bash
llmbench validate            # Returns exit code 0 on PASS / 1 on FAIL

```

- Returns `VALIDATION: PASS` when `mock-gold` resolves all tasks, `mock-broken` fails all tasks, and average combined score for broken runs equals 0.

### 5.2 Multi-Model Comparison

```bash
llmbench run --model local-openai --output results
llmbench run --model local-ollama --output results

for f in results/*_results.json; do
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); \
    s=d['summary']; print(f\"{d['model']:24s} resolved={s['resolved_rate']*100:5.1f}%  \
combined={s['avg_combined']:.1f}  quality={s['avg_quality_resolved']:.1f}\")" "$f"
done

```

Because `results.json` files are lightweight, they can be comfortably stored as standard CI build artifacts. Store accompanying `*_artifacts/` folders only when detailed debugging traces are required.

---

## 6. Programmatic Access to Results

### 6.1 `jq` Aggregation Examples

```bash
# List failed task IDs and reasons
jq -r '.results[] | select(.resolved|not) | "\(.task_id)\t\(.fail_reason)"' \
  results/<stamp>_<model>_results.json

# Extract parse failures only
jq -r '.results[] | select(.parse_ok|not) | "\(.task_id)\t\(.parse_error)"' \
  results/<stamp>_<model>_results.json

# Retrieve artifact path and open raw LLM output for a failed task
adir=$(jq -r '.artifacts_dir' results/<stamp>_<model>_results.json)
cat "results/$adir/t011/llm_output.txt"

```

### 6.2 Failure Troubleshooting Workflow

| Symptom                    | File to Inspect                                 | Diagnostic Information                                                     |
| -------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| `parse_ok=false`           | `artifacts/<id>/llm_output.txt`                 | Malformed `--- FILE: path ---` blocks or markdown formatting syntax issues |
| `fail_reason=tests failed` | `artifacts/<id>/test_output.txt` → `generated/` | Inspect failed assertions to locate faulty implementation logic            |
| `fail_reason=timeout`      | `test_output.txt`                               | Infinite loops or performance bottlenecks; adjust `run.test_timeout`       |
| Low quality score          | `generated/<path>`                              | Check ruff warnings or complex functions directly within generated code    |

---

## 7. Migration Notes (Differences from Old Formats)

When maintaining scripts that parse outputs from legacy versions, take note of the following differences:

### 7.1 `results.json`

- **Added**: Top-level `artifacts_dir`, along with `title`, `parse_ok`, and `parse_error` per task entry in `results`.
- **Removed (Lean JSON)**: `raw_output`, `parsed_files`, and `test_output` are **no longer embedded** in task result items. Scripts accessing these properties should be updated to read directly from corresponding files in `artifacts/`.

### 7.2 `report.md`

- **Task Results Table Column Layout Updated**:
- Legacy: `| Task | Difficulty | Resolved | Quality | Combined | Time(s) | tok/s | Fail Reason |` (8 columns, leading Task)
- Updated: `| | Task | Difficulty | Generated Files | Quality | Combined | Time | tok/s | Notes |` (9 columns, leading ✅/❌ icon)
- Resolved `O/X` indicators replaced with ✅/❌ icons. Update automated Markdown parsing rules accordingly.

- Section header `## Quality Breakdown (Per Task)` renamed to `## Detailed Task Breakdown`.

### 7.3 `config.yaml`

- Example `local-openai` entry updated to `base_url: http://localhost:8085/v1` / `model: Qwopus3.6-27B-Coder-MTP`.
  This is a configuration sample update and does not change harness execution. Output filenames derive from the `--model` key name (`local-openai`), maintaining stable file naming.

---

## 8. Change File Quick Reference Table

| File                                 | Change Overview                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llmbench/runner.py`                 | Artifact preservation + **Multi-trial aggregation** (`Attempt`/`_aggregate_attempts`/pass@k) + **Model resolution** (`resolve_model`/Ollama auto/`model:auto` label) + `_log_task`                                                                                                                                                                                                                                                                                                                      |
| `llmbench/scoring.py`                | Generalized `combined_score` scaling via success rate + `pass_at_k` (unbiased estimator)                                                                                                                                                                                                                                                                                                                                                                                                                |
| `llmbench/usability.py`              | **New**. Tier classification via `classify()` and summary aggregation via `aggregate()`                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `llmbench/compare.py`                | **New**. Cross-model comparison report generator across multiple `results.json` files                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `llmbench/report.py`                 | Added usability sections, reliability table columns, pass@1 main metric, conservative recommendations, and quality breakdown notes                                                                                                                                                                                                                                                                                                                                                                      |
| `llmbench/clients/openai_compat.py`  | Server model auto-detection (`fetch_served_model`) for `model:auto` + environment variable expansion for API keys                                                                                                                                                                                                                                                                                                                                                                                       |
| `llmbench/clients/ollama.py`         | `list_ollama_models()` (`/api/tags`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `llmbench/cli.py`                    | Added `models` / `compare` / **`certify`** subcommands, `--runs` / `--sample-temp` / **`--concurrency`** / `--label` / `--ollama-host`, **`--with-l6` / `--l6-ledger**`, **`--with-l7`/`--l7-ledger**`, **`--only-l6` / `--only-l7` (extended shared `_common_args` across list-tasks/run/validate)**, and **`certify --merge`\*\*                                                                                                                                                                      |
| `llmbench/certify.py`                | **New**. Difficulty to tier mapping (L1–L7), gate evaluation per tier (`certify`/`render_certificate_md`), usability line = independent L4 pass, **architect → L6 gate**, **grandmaster → L7 gate (provisional ceiling eval)**, and **`merge_results()` (merges multiple results files using last-wins logic for task IDs and concatenating model names in distinct appearance order with `+`) / `certify --merge` CLI support**                                                                        |
| `llmbench/tasks.py`                  | Per-task `perf_timeout` field + **Multi-ledger loading in `load_tasks(..., ledgers=[...])` (first task ID wins)**                                                                                                                                                                                                                                                                                                                                                                                       |
| `llmbench/runner.py`                 | Applied per-task `perf_timeout` (falls back to `test_timeout` if omitted) + **`BenchmarkRunner(..., ledgers=...)`**                                                                                                                                                                                                                                                                                                                                                                                     |
| `config.yaml`                        | `model:auto`, `run.runs` / `sample_temp` / `ollama_host`, `usability:`, and `ref-gpt`                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `tasks/`                             | Added hard tasks t016–t020 + **L4 expert t021–t032 / L5 frontier t033–t040** (40 default tasks) + **L6 architect t041–t060 (separate `tasks_l6.jsonl` ledger enabled via `--with-l6`)** + **L7 grandmaster t061–t100 (separate `tasks_l7.jsonl` ledger enabled via `--with-l7`; 5 axes × 8 questions covering Numerical Stability t061–068, State Consistency t069–076, Multi-Coupling Bugs t077–084, Deep Concurrency t085–092, Adversarial Parsing/Security t093–100; t098 sets `perf_timeout: 30`)** |
| `llmbench/graders/`                  | **New Package** (Multi-domain evaluation): `__init__.py` (grader registry + `GradeCtx`/`GraderEval`), `checks.py` (IFEval checks), `code.py` (wrapped legacy pipeline), `detection.py` (security), `constraint.py` (general), `judge.py` (writing - experimental), `qa.py` (medical - reference value). Graders normalize output to `(resolved, quality)` to reuse existing pass@k / usability / certification pipelines without modifications                                                          |
| `llmbench/cli.py` (Multi-Domain)     | **`--with-sec/gen/write/med` and `--only-sec/gen/write/med**`(extends shared`\_common_args`following`--with-l6/l7` architecture)                                                                                                                                                                                                                                                                                                                                                                        |
| `llmbench/certify.py` (Multi-Domain) | **Domain-specific gates** (`certify_domains`), **Balance Index** (harmonic mean across coding + non-experimental domains), and **Medical difficulty breakdown (med_basic/med_std/med_hard) accuracy readout**                                                                                                                                                                                                                                                                                           |
| `llmbench/report.py` (Multi-Domain)  | Added **"🌐 Domain Summary"** section to report outputs                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `config.yaml` (Multi-Domain)         | **`graders:`** (`pass_f1`/`pass_ratio`/`pass_score`), **`quality.judge:`** (`enabled`/`judge_model`/`seeds`), and **`certify_domains:`**                                                                                                                                                                                                                                                                                                                                                                |
| `tasks/` (Multi-Domain)              | **Domain Tasks Added**: Security s01–s04 (separate ledger `tasks_sec.jsonl`, `--with-sec`), General g01–g03 (`tasks_gen.jsonl`, `--with-gen`), Writing w01–w02 (`tasks_write.jsonl`, `--with-write`), Medical m01–m24 (`tasks_med.jsonl`, `--with-med`; difficulties med_basic 7 / med_std 11 / med_hard 6, fully fact-checked)                                                                                                                                                                         |

> Verification Status: Verified `llmbench validate` PASS (40/40 gold green / 40/40 broken fail), selfchecks across existing 20 tasks + **L6 20 tasks (20/20)** + **L7 40 tasks (40/40)**,
> `validate --with-l6` gold 20/20 & broken 20/20, `validate --with-l7` gold 40/40 & broken 40/40,
> `list-tasks` 40 (default) / 60 (`--with-l6`) / 80 (`--with-l7`) / 100 (`--with-l6 --with-l7`) /
> 20 (`--only-l6`) / 40 (`--only-l7`) / 60 (`--only-l6 --only-l7`),
> individual verification for multi-run aggregation, usability, compare, certify, model resolution, ruff (0 warnings), and `compileall`.
> Additional Multi-Domain Verification: `llmbench validate --only-sec|gen|write|med` all PASSED (all gold pass, all broken fail; zero LLM calls). See Section 10 for details.

---

## 9. Additional Subsystems (Reliability, Usability, Comparison, Model Resolution)

Operational reference for major features introduced alongside artifact preservation. Usage guides available in `USAGE.md`; executive summary in `CHANGES.md`.

### 9.1 Reliability (pass@k)

Specify `--runs N` (or `run.runs`) to sample each task N times. `runner._aggregate_attempts()` aggregates success rate (= pass@1), pass@k, and representative attempts. `scoring.pass_at_k(n,c,k)` implements Chen et al. 2021's unbiased estimator.
**Note**: Because this harness samples n = runs attempts, `pass_at_k` (where k = runs) degenerates to "at least 1 pass out of N attempts".
Primary operational metrics to track are **success_rate (pass@1)** and **solved_any_rate**.

### 9.2 Usability Tiers

`usability.classify(success_rate, quality, cfg)` classifies task usability into 🟢 Autonomous / 🟡 Assisted / 🔴 Unusable based on thresholds in `config.yaml` under `usability:`. Reports present **proportions** across difficulty levels and generate **conservative recommendations** that avoid masking 🔴 Unusable tasks.

### 9.3 Model Comparison (`compare`)

Run `llmbench compare <results.json...>` to output `comparison_<stamp>.md`. Generates rankings (relative score ratio against best performer), usability tier comparisons, and task-by-task Combined score matrices. Benchmark target models side-by-side against reference anchor models (e.g. `ref-gpt`).

### 9.4 Model Resolution (Config-Edit Free Operations)

- `model: auto` … `openai_compat.fetch_served_model()` inspects `/v1/models` to detect the currently loaded model automatically.
  Report labels and result filenames adopt the actual model name (cleaned of `.gguf` suffixes via `runner._label_from_model`).
- Dynamic Ollama … Unconfigured model names passed to `--model <name>` are resolved dynamically via `/api/tags` by `runner.resolve_model()`.
- Ollama Connection Priority: `--ollama-host` > `OLLAMA_HOST` env > Ollama `base_url` in config > `http://localhost:11434`.
- `base_url` Connection Priority: `--base-url` > config `base_url` (`${VAR}` supported) > client-type env (`OPENAI_BASE_URL` / `OLLAMA_HOST` / `CODEROUTER_BASE_URL`) > client-type default.
- `--client-type openai|ollama|multiagent` allows direct connection without editing config
  (e.g., `llmbench run --model auto --client-type openai --base-url http://localhost:8085/v1` to connect directly to llama.cpp,
  or `--client-type multiagent --base-url http://localhost:8088` for CodeRouter).
- `model: auto` supports multi-model selection via `auto_prefer: "<substring>"`. If omitted, selects the first entry and emits a warning.
- Unset `${VAR}` references raise an explicit error (previous behavior: silently evaluated to empty string, leading to cryptic 401 errors).
- Fixed labeling can be forced using `--label`. API key entries expand environment variables via `${VAR}`.

### 9.5 Operational Considerations

- **Multi-run executions multiply disk and time requirements by N**. Because artifact output folders only write out 1 representative attempt, disk growth is controlled, but raw log text accumulates.
- Interpret pass@k as described in 9.1. Do not mistake `avg_pass_at_k` for overall generalization ability.
- Report tables include extra columns (Usability Tier, Reliability). Ensure automated parsers are updated (see Migration Notes in Section 7).

---

## 10. Multi-Domain Evaluation (Grader Extension)

This subsystem expands evaluation capabilities beyond coding by introducing **pluggable graders**.
For full design specifications (scoring equations, output contracts, validation requirements), consult [📐 DESIGN_DOMAINS.md](DESIGN_DOMAINS.md).
This section acts as an operational reference. Usage guides available in `USAGE.md`; executive summary in `CHANGES.md`.

### 10.1 Grader Mechanism & Backward Compatibility

Each grader exposes `build_prompt()` (output structure instructions) and `evaluate()` (scoring execution), normalizing results into a **standard `GraderEval` structure** (`resolved: bool`, `quality_score: 0-100`, `components`, `fail_reason`, etc.).
Thanks to this normalization, `_aggregate_attempts`, `combined_score`, pass@k, usability, and certification pipelines operate **without any code changes**.

- Task ledger records omitting the `grader` property default to `code` (wrapping the legacy execution pipeline).
- Executing runs using only default ledgers (`tasks.jsonl`) produces results, scores, and `validate` passes **identical to legacy releases**.
- Grader modules are maintained inside `llmbench/graders/` (`__init__.py` registry + `GradeCtx`/`GraderEval`, `checks.py`, `code.py`, `detection.py`, `constraint.py`, `judge.py`, `qa.py`).

### 10.2 Overview of the 4 Domains

| Domain   | Grader       | Ledger              | Flags                           | Scoring Logic                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------- | ------------ | ------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| security | `detection`  | `tasks_sec.jsonl`   | `--with-sec` / `--only-sec`     | Scores JSON findings array following `--- FINDINGS ---` against gold labels using Precision / Recall / **F1**. `resolved = F1 >= pass_f1` (default 0.67). Includes **clean decoy tasks** with empty gold finding arrays to penalize false positive over-detection.                                                                                                                                                                                        |
| general  | `constraint` | `tasks_gen.jsonl`   | `--with-gen` / `--only-gen`     | Evaluates response text following `--- ANSWER ---` programmatically against `checks.json` (IFEval checks: word_count, line_count, char_count, contains, not_contains, starts_with, ends_with, equals, regex, json_valid, json_path). `resolved = all checks pass` (`pass_ratio` default 1.0), `quality = pass rate × 100`.                                                                                                                                |
| writing  | `judge`      | `tasks_write.jsonl` | `--with-write` / `--only-write` | **Experimental**. Evaluates deterministic `hard_constraints` in `rubric.json` (using constraint check kinds) + evaluation criteria + `pass_score`. When a judge model is configured (`quality.judge.enabled`), averages 0–10 score across `seeds` runs (`quality = score × 10`, `resolved = score >= pass_score AND hard constraints pass`). When judge is disabled, evaluates deterministically on hard constraints alone (used by `llmbench validate`). |
| medical  | `qa`         | `tasks_med.jsonl`   | `--with-med` / `--only-med`     | **Reference value**. `gold.json` defines multiple choice `{"mode":"mcq","answer":"C"}` or keyword matching `{"mode":"keyword","all":[...],"any":[...]}`. Gold keywords include **both English and Japanese allowed terms**, enabling valid evaluation for Japanese models evaluated with `--lang ja`.                                                                                                                                                     |

### 10.3 CLI Options

Shares the same flag architecture as `--with-l6`/`--with-l7`/`--only-l6`/`--only-l7`.

- `--with-sec` / `--with-gen` / `--with-write` / `--with-med` … Appends domain task ledgers onto default tasks
- `--only-sec` / `--only-gen` / `--only-write` / `--only-med` … Excludes default tasks to execute specified domain exclusively
- `certify --merge` supports results files containing multi-domain evaluations
- Self-validation: `llmbench validate --only-sec|gen|write|med` (verifies all gold pass and all broken fail without LLM inference)

### 10.4 `config.yaml` Additional Keys

```yaml
quality:
    judge: { enabled: false, judge_model: local-openai, seeds: 1 } # Judge scoring model for writing domain
graders:
    detection: { pass_f1: 0.67 }
    constraint: { pass_ratio: 1.0 }
    judge: { pass_score: 7.0 }
certify_domains:
    security: { min_success: 0.6, min_combined: 60 }
    general: { min_success: 0.7, min_combined: 65 }
    writing: { min_success: 0.5, min_combined: 55, experimental: true }
    medical: { min_success: 0.6, min_combined: 60, reference: true }
```

### 10.5 `certify` Domain Output

When supplied with a `results.json` containing multi-domain evaluations, `certify` outputs domain-specific diagnostics in addition to tier certifications (L1–L7):

- **Domain Summary Table**: Average success rate, average combined score, and gate status (`certify_domains` thresholds) per domain.
- **Balance Index**: Synthesizes normalized average combined scores across coding and non-experimental domains (writing/medical excluded by default) using **harmonic mean**. Heavily penalizes models with single-domain specializations compared to arithmetic averaging.
- **Medical Detail Readout**: Displays overall accuracy alongside difficulty breakdowns (med_basic / med_std / med_hard). Explicitly highlights that 5-choice MCQ baseline chance accuracy is ~20% and clarifies that medical scores represent **reference metrics** rather than guarantees of clinical validity.

Adds a "🌐 Domain Summary" section to generated `report.md` outputs. See `USAGE.md` for sample outputs and detailed interpretation rules.

### 10.6 Operational Considerations

- Gate thresholds for writing and medical domains are provisional (uncalibrated). Adjust in `DEFAULT_DOMAIN_GATES` / `DEFAULT_MED_GATES` within `certify.py` or under `certify_domains:` in `config.yaml`.
- When judge scoring is enabled, select a `judge_model` distinct from the candidate model family to avoid self-preference biases.
- Medical tasks (m01–m24) span 10 clinical fields (Pharmacology, Cardiology, Emergency/First-line, Endocrinology, Infection, Nephrology/Physiology, Neurology, Pediatrics, OB/GYN, Toxicology) across 7 med_basic, 11 med_std, and 6 med_hard tasks. All gold keys have been independently fact-checked.
- Refer to `TASKS.md` for complete task inventories, directory structures, and gold schema definitions.

---

## 11. Evaluation Procedures for Cloud LLMs (Hosted APIs)

While other chapters focus primarily on local LLMs (llama.cpp / Ollama), **OpenAI-compatible hosted APIs** can be evaluated using the **exact same pipeline**. The evaluation stack (patch parsing → sandbox pytest → ruff/radon → combined score calculation) operates independently of model location, making `certify`, `compare`, pass@k, and usability tiers fully compatible.

Key differences relative to local models boil down to **3 points**: **① Explicit real model names**, **② API key management**, and **③ Cost/rate limit controls**. Client connections use `type: openai` (`openai_compat.OpenAICompatClient`, Bearer authentication + `/v1/chat/completions`).

> Handling Cloud Artifacts: Artifact `llm_output.txt` files preserve **full task prompts**. Keep `results/` excluded in `.gitignore` (4.2) and ensure prompt contents sent to third-party endpoints conform to organizational policy.

### 11.1 Common Setup (Config Approach - Recommended)

**Use this approach for cloud APIs requiring authentication.** API keys expand from environment variables via `${VAR}`, preventing plain-text exposure in code or result files.

1. Add a model entry under `models:` in `config.yaml` (set `base_url` to the OpenAI-compatible endpoint path ending in `/v1`. Specify the **exact model name** explicitly in `model` rather than `auto`).
2. Export the API key via environment variables (**never hardcode keys inside `config.yaml**`). Unset `${VAR}` references trigger explicit error stops.
3. Run `llmbench run --model <key_name>`. Flags such as `--runs`, `--tasks`, `--lang`, and `--with-*` work identically to local executions. Result file names take the model key name specified in `--model`.

> ⚠️ Direct inline parameters (`--client-type openai --base-url ...`) without config entries fallback to a default dummy API key (`sk-local`), triggering 401 errors on authenticated cloud endpoints. Authenticated cloud APIs must use the config-driven approach (`api_key: ${VAR}`).

### 11.2 Provider-Specific Examples

Connect via `type: openai` (OpenAI-compatible protocol). **Coding Plans require designated endpoints and keys**; otherwise, requests fall back to pay-as-you-go billing rather than subscription quota allowances.

| Setting                      | Alibaba Cloud (Model Studio) Qwen Token-Plan                                                                                                                                             | z.ai GLM Coding Plan                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| base_url (OpenAI-compatible) | `[https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1](https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1)` (Region matched to issuance endpoint) | `[https://api.z.ai/api/coding/paas/v4](https://api.z.ai/api/coding/paas/v4)` |
| Primary Model Name           | `qwen3.7-max`                                                                                                                                                                            | `glm-5.2`                                                                    |
| Alternate Models             | `qwen3.7-plus` / `qwen3.6-plus` / `qwen3.6-flash` (qwen3-coder family not available under Token Plan)                                                                                    | `glm-4.7` / `glm-5.1`                                                        |
| API Key                      | **Plan-specific key `sk-sp-xxxx**`(Not standard`sk-xxxx`)                                                                                                                                | z.ai API Key                                                                 |
| Key Source                   | Model Studio Console → Subscription Page                                                                                                                                                 | z.ai Console                                                                 |
| Behavior on Misconfiguration | Standard key or endpoint falls back to pay-as-you-go billing                                                                                                                             | Standard `paas/v4` / `anthropic` incurs separate billing/protocols           |

`config.yaml` snippet (add under `models:`):

```yaml
models:
    # ── Alibaba Cloud (Model Studio) Qwen3 Coding Plan ──────────────
    qwen-coding:
        type: openai
        base_url: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1" # Dedicated maas domain (compatible-mode/v1) for issued region
        model: "qwen3.7-max" # Token Plan supported model. Alternates: qwen3.7-plus / qwen3.6-plus / qwen3.6-flash
        api_key: "${DASHSCOPE_CODING_API_KEY}" # ★ Dedicated plan key sk-sp-xxxx (Not standard sk-xxxx)
        temperature: 0.2
        max_tokens: 24576 # Large token limit to accommodate reasoning output
        timeout: 600

    # ── z.ai GLM Coding Plan ────────────────────────────────────────
    glm-coding:
        type: openai
        base_url: "https://api.z.ai/api/coding/paas/v4" # OpenAI-compatible path for Coding Plan
        model: "glm-5.2" # Alternate: glm-4.7
        api_key: "${ZAI_API_KEY}"
        temperature: 0.2
        max_tokens: 24576
        timeout: 600
```

Environment variable configuration (pass keys via `export`; do not hardcode in config files):

```bash
export DASHSCOPE_CODING_API_KEY=sk-sp-...   # Issued in Model Studio Console → Plan Page
export ZAI_API_KEY=...                        # z.ai API key

```

Connectivity verification followed by benchmark run:

```bash
# Verify connectivity, billing, and API key via 1 task x 1 run
llmbench run --model qwen-coding --runs 1 --tasks t001
llmbench run --model glm-coding  --runs 1 --tasks t001

# Execute full evaluation run
llmbench run --model qwen-coding --runs 3
llmbench run --model glm-coding  --runs 3

```

> [!NOTE]
> While z.ai provides **Anthropic-compatible endpoints** (`[https://api.z.ai/api/anthropic](https://api.z.ai/api/anthropic)`, `ANTHROPIC_AUTH_TOKEN`) for tools like Claude Code, llmbench uses the OpenAI-compatible protocol via `coding/paas/v4` above.
> Endpoint schemas, model names, and key structures evolve with provider plan updates; **always verify specifications against vendor developer portals**. The values in this section are based on official vendor documentation as of 2026.

> [!IMPORTANT]
> **Pay attention to DashScope connection protocol paths (Native SDK vs. OpenAI Compatible).**
> Alibaba's Python SDK examples (`from dashscope import Generation` / `dashscope.base_http_api_url=".../api/v1"`) use the **DashScope Native SDK** protocol via `/api/v1`. Because llmbench communicates over standard OpenAI HTTP APIs (`/chat/completions` + Bearer token), **do not use `/api/v1**`. Use the correct path based on your account type:
| Protocol Type | base_url | Key Format | Usage |
| --- | --- | --- | --- |
| Native SDK | `[https://dashscope-intl.aliyuncs.com/api/v1](https://dashscope-intl.aliyuncs.com/api/v1)`|`sk-xxxx`| DashScope SDK exclusive (Unsupported in llmbench) |
| OpenAI Compatible (Pay-as-you-go) |`[https://dashscope-intl.aliyuncs.com/compatible-mode/v1](https://dashscope-intl.aliyuncs.com/compatible-mode/v1)`|`sk-xxxx`| General pay-as-you-go API calls outside Coding Plan |
| OpenAI Compatible (Coding Plan) |`[https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1](https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1)`|`sk-sp-xxxx` | Dedicated subscription plan endpoints (Recommended) |
>
> Example for general pay-as-you-go evaluation (**uses `compatible-mode/v1` instead of `/api/v1**`):
>
> ```yaml
> qwen-general:
>     type: openai
>     base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" # Distinct from native SDK /api/v1
>     model: "qwen3.7-max" # Allows qwen3-coder-plus family on pay-as-you-go
>     api_key: "${DASHSCOPE_API_KEY}" # Standard key sk-xxxx
>     temperature: 0.2
>     max_tokens: 24576
>     timeout: 600
> ```

> [!WARNING]
> **Select models from vendor plan availability lists.** Token Plan subscriptions support models including `qwen3.7-max`, `qwen3.7-plus`, `qwen3.6-plus`, `qwen3.6-flash`, `glm-5.x`, `kimi-k2.x`, and `deepseek-*`.
> **Coder-specific models like `qwen3-coder-plus` are not available under Token Plan subscriptions** (requesting them returns 404 errors). Use flagship general models like `qwen3.7-max`.
> Note that OpenAI-compatible endpoints require `stream=true` only when `enable_thinking=true` is explicitly passed. Because llmbench does not pass `enable_thinking`, standard non-thinking model requests succeed out of the box.

### 11.3 Cost, Rate Limits, and Reliability

With paid cloud APIs, **execution runs scale API costs linearly**. Tune parameters using the guidelines below:

| Parameter                  | Impact                                                                | Recommendation                                                           |
| -------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `--runs N`                 | Multiplies total API calls (and cost) by N; increases pass@k accuracy | Set N ≈ 3 for paid cloud endpoints. Run `--runs 1` first to verify setup |
| `--concurrency N`          | Dispatches N requests in parallel to reduce runtime                   | **Align with vendor rate limits (RPM)**. Reduce N if 429 errors occur    |
| `max_tokens`               | Low token limits cause mid-generation cutoffs → `patch parse failed`  | Set ≥ 4096 for standard models; ≥ 24576 for reasoning models             |
| `timeout`                  | Prevents request drops during high latency or peak load               | Set ≥ 600 seconds for deep reasoning models                              |
| Task Filtering (`--tasks`) | Constrains evaluated tasks to manage cost and runtime                 | Test initial setup on specific subsets or tiers                          |

- Workflow: **Run `llmbench validate` (no LLM calls) to check local harness setup → Execute `--runs 1 --tasks <subset>` to verify cloud connectivity/billing → Proceed to full benchmark evaluation**.
- Rate limit errors (HTTP 429) appear per task under `fail_reason`. Reduce `--concurrency` if 429 errors persist.
- Cloud API execution requires **no local server concurrency tuning** (handled provider-side). Monitor local `--concurrency` settings to prevent HTTP 429 rate limit spikes.

### 11.4 Running Alongside as Reference Anchors (`compare`)

To benchmark local model performance against frontier cloud models, run a strong cloud model as a **reference anchor** and visualize side-by-side positioning in a unified report via `compare` (Section 9.3).

```bash
export OPENAI_API_KEY=sk-...
llmbench run --model local-openai --output results     # Target local model
llmbench run --model qwen-coding  --output results     # Cloud reference anchor
llmbench compare results/*_results.json                # Generates comparison_<stamp>.md

```

### 11.5 Checklist

- [ ] Included `/v1` (or provider-specified OpenAI-compatible path) in `base_url`
- [ ] Explicitly specified real model names in `model` (avoided `auto` for cloud configs)
- [ ] Configured **plan-specific API keys** (Qwen = `sk-sp-`) and **dedicated endpoints** (prevents unintended pay-as-you-go billing)
- [ ] Exported API keys via `${VAR}` and environment variables (avoided hardcoding in config)
- [ ] Verified connectivity and cost via `--runs 1 --tasks <subset>` prior to full benchmark runs
- [ ] Adjusted `--concurrency` to operate within provider rate limits (avoiding HTTP 429 errors)
- [ ] Allocated sufficient `max_tokens` allowance for reasoning-heavy models (preventing mid-generation parse failures)

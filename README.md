<div align="center">

# 🧪 llmbench

**Are local LLMs really "usable"? — A SWE-Bench-style benchmark measured by functional correctness × code quality**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

The trade-off between "tests pass but the code is dirty" and "clean, safe code",<br>
**verified for real** in a local environment — a full-scratch framework.<br>
It also covers multi-domain evaluation of **security detection / instruction following / creative writing / medical QA / internet memes / over-refusal**.

[Features](#-features) • [Quick Start](#-quick-start) • [Scoring](#-scoring) • [Multi-Domain](#-multi-domain-evaluation-beyond-coding) • [Automation](#-automation) • [Documentation](#-documentation) • [Roadmap](#-roadmap)

</div>

---

## ✨ Features

- 🎯 **Functional correctness** — SWE-Bench style. Pass a bug report + source, apply the patch, then decide "resolved" with hidden tests (pytest)
- 🧹 **Code quality layer** — Weighted blend of Ruff (lint density) / radon (maintainability, complexity) / LLM review score / SonarQube
- 🎲 **Reliability (pass@k)** — `--runs N` tries each task N times. Not "one success = usable", but **whether it works reliably**
- 🧭 **usability verdict** — Classifies each task as **🟢 Autonomous / 🟡 Assisted / 🔴 Not usable** from reliability × quality
- 🎓 **Usability Baseline Certification (`certify`)** — Maps difficulty to tiers (L1–L7) and decides pass/fail per tier. **Passing L4 (expert) independently = production-ready line**; L7 (grandmaster) is a ceiling-evaluation band. Run the suite in pieces and join them with `certify --merge`
- 🌐 **Multi-domain evaluation** — Swap the grader to add security / general / writing / medical / culture / uncensored. `certify` outputs per-domain gates and a **balance index** (which exposes single-talent models)
- ⚖️ **Combined score** — Code that does not run scores 0. Code that runs is separated by success rate and quality
- 🔌 **Bring your own endpoint** — Supports both OpenAI-compatible APIs (llama.cpp / LM Studio / vLLM) and Ollama. **`model: "auto"`** adopts the model the server has loaded, and **`type: cli`** runs the official agent CLIs (claude / codex / grok) headless inside your flat-rate subscription quota
- 🇯🇵 **Japanese + English issues bundled** — `--lang ja` measures the "language tax" (performance drop caused by Japanese instructions)
- ⚡ **Automatic recording of speed and execution environment** — Besides tok/s, **quantization, GPU offload ratio and context length** are written to results.json / report.md. tok/s changes by several times on the same GPU depending on these, so `compare` checks whether the conditions match and warns
- 🔁 **Automated quantization sweeps** — `tools/sweep.sh` runs the full "quantization × suite" grid unattended. It handles server start, wait and stop, VRAM recording, MTP detection and resume
- 📦 **40 bundled tasks (+ 20 optional + 16 optional)** — L1 easy 5 / L2 medium 5 / L3 hard 10 / L4 expert 12 / L5 frontier 8. Add the upper bands with `--with-l6` (architect, 20 tasks) / `--with-l7` (grandmaster, 16 tasks), and run them alone with `--only-*`. No external dependencies (stdlib-only), so it runs immediately
- 🛡️ **Safe design** — Tests are not shown to the LLM, patch write targets are limited to known files, the original tasks stay unchanged

## 🚀 Quick Start

```bash
git clone https://github.com/zephel01/swe-bench.git
cd swe-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1️⃣ Self-check (no LLM needed — confirms the whole pipeline with a mock)
llmbench validate
llmbench validate --only-sec        # you can also self-check a single domain only (sec/gen/write/med/culture/unc)

# 2️⃣ Set base_url in config.yaml to your own environment. Leaving model as "auto" is fine (the server's model is adopted)

# 3️⃣ Show the available models (models defined in config + models running in Ollama)
llmbench models
llmbench models --remote opencode-go              # queries the /v1/models of a type:openai entry and
                                                    # lists the model IDs that endpoint can really select
                                                    # (for investigating gateways where you swap the model
                                                    #  on demand, such as OpenCode Go. Works for any other type:openai model too)

# 4️⃣ Run (coding)
llmbench run --model local-openai                 # the default 40 tasks, 1 run
llmbench run --model local-openai --runs 5        # each task 5 times -> success rate, pass@k
llmbench run --model local-openai --with-l6 --runs 5  # add the 20 L6(architect) tasks, 60 tasks in total
llmbench run --model local-openai --with-l7 --runs 5  # add the 16 L7(grandmaster) tasks, 56 tasks in total
llmbench run --model local-openai --with-l6 --with-l7 --runs 5  # use L6+L7 together, 76 tasks in total (ceiling evaluation)
llmbench run --model local-openai --only-l6 --runs 5  # run L6 alone (no base) -> join it later with certify --merge
llmbench run --model local-openai --runs 5 --concurrency 5  # run 5 trials in parallel to cut total time (needs: start the server with --parallel 5 -cb)
llmbench run --model qwen2.5-coder:7b --runs 5    # you can also give a real Ollama model name directly
llmbench run --model local-openai --tasks t021,t033 --lang ja   # you can run only the new tiers

# 4️⃣' Run (multi-domain: beyond coding)
llmbench run --model local-openai --with-sec --with-gen --runs 5   # add security/general
llmbench run --model local-openai --only-sec --runs 5              # run security detection alone
llmbench run --model <medical model>  --only-med --lang ja --runs 5  # measure medical QA only, in Japanese
llmbench run --model local-openai --only-culture --lang ja --runs 3 # measure Japanese internet meme knowledge only
llmbench run --model local-openai --only-unc --runs 3               # measure over-refusal only

# 5️⃣ Compare several results side by side
llmbench compare results/*_results.json

# 6️⃣ Usability Baseline certification (tier gates + per-domain + balance index)
llmbench certify results/<stamp>_<model>_results.json
```

Subcommands: `list-tasks` / `models` / `run` / `compare` / `certify` / `validate`

Results are saved under `results/` as these 3 items.

- `<stamp>_<model>_results.json` — scores and aggregates (machine-readable, lightweight)
- `<stamp>_<model>_report.md` — the Markdown report for humans to read
- `<stamp>_<model>_artifacts/<task_id>/` — **the generated artifacts themselves**
  - `llm_output.txt` … raw LLM output (to check the cause when parsing fails)
  - `generated/<path>` … the generated code that was applied / predicted labels and answers (to check behaviour)
  - `test_output.txt` … pytest output and grading detail (to check the cause of a failure)

The live log also prints, for every task, the generated files, the head of the code and the parse/test result, so you can confirm on the spot that it "really works".

> [!TIP]
> For **how to read** the run log, the report and the artifacts, and for the model-comparison procedure, see [📘 Usage Guide (USAGE.md)](docs/USAGE.md).

<details>
<summary>📄 Example report output (--runs 5)</summary>

```markdown
# llmbench Report: Qwopus3.6-27B-Coder-MTP-Q6_K
Issue language: `en` / Tasks: 20 / Trials: ×5

## Summary
| Metric | Value |
|---|---|
| ✅ Resolved rate | **95.0%** (19/20) |
| 🎲 Avg success rate (pass@1, ×5) | **93.0%** |
| 🔁 Tasks with at least 1 success | **95.0%** (19/20) |
| 🏅 Avg quality (resolved only) | **89.8 / 100** |
| 🎯 Avg Combined | **88.2 / 100** |

## 🧭 usability verdicts
- 🟢 Autonomous 17/20 (85%) / 🟡 Assisted 2/20 (10%) / 🔴 Not usable 1/20 (5%)
> Overall recommendation: mostly autonomous. But 🔴 Not usable 1/20 (5%) needs attention

| | Task | Difficulty | Verdict | Reliability | Generated files | Quality | Combined | Gen time | tok/s | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | t007 | medium | 🟡 Assisted | 4/5 (success rate 80%) | `word_freq.py` | 100 | 80 | 1.5s | 111.9 | flaky 4/5 passed |
| ❌ | t020 | hard | 🔴 Not usable | 0/5 (success rate 0%) | `calc.py` | 0 | 0 | 3.8s | 122.2 | tests failed |
```

</details>

## 📊 Scoring

```
combined = success_rate × (0.5 + 0.5 × quality / 100) × 100
```

`success_rate` is the success rate over the trials (with `--runs 1` it is resolved as 0/1; with several trials it is 0.0–1.0).
**Reliability is reflected in the score directly** (example: 60% success, quality 90 → 0.6×95 = 57 points).

| success_rate | quality | combined | meaning |
|---|---|---|---|
| 0 (not usable) | — | **0** | Code that does not run scores 0, whatever its quality |
| 1.0 | 0 | **50** | Runs every time, but quality is the lowest |
| 1.0 | 100 | **100** | Runs every time and quality is perfect |
| 0.6 (flaky) | 90 | **57** | Fails sometimes. Penalized on reliability |

> [!NOTE]
> With several trials (`--runs N`), the headline Resolved is a **majority vote over success rate ≥ 0.5** (with `runs=1` it matches a single trial).
> Resolved rate and average success rate are different things; Resolved rate reads more generously.

> [!NOTE]
> The same formula works in other domains. Each grader returns its result normalized to `(resolved, quality)`, so
> detection maps F1, constraint maps the check pass rate, judge maps the rubric score, and qa maps
> right/wrong straight onto success/quality — pass@k, usability and certify are shared with no changes.

### Reliability (pass@k) and usability verdicts

With `--runs N` each task is tried N times, and the **success rate (pass@1)** is the primary metric, with pass@k and flakiness measured too.
Tasks are then classified into 3 tiers from success_rate and quality (thresholds are under `usability:` in `config.yaml`):

| Tier | Default condition | Meaning |
|---|---|---|
| 🟢 Autonomous | success ≥ 0.9 and quality ≥ 80 | You can leave it to the model with almost no review |
| 🟡 Assisted | success ≥ 0.6 and quality ≥ 0 (changeable with `assisted.min_quality`) | Usable if review is assumed |
| 🔴 Not usable | Below the above | You cannot give it this kind of task |

### 🎓 Usability Baseline Certification (`certify`)

usability verdicts work **per task**, while `certify` judges the **whole model**.
It maps difficulty to tiers (L1-L7), evaluates whether each tier meets its gate, and
outputs **two numbers: the independently passed tiers (main verdict = L4) and the cumulative reached level**. `llmbench certify results/<...>_results.json`.

| Gate | Condition (tier average) | Meaning |
|---|---|---|
| L1 easy | success ≥ 90% | toy level is solid |
| L2 medium | success ≥ 85% | simple work to a specification is possible |
| L3 hard | success ≥ 75% and combined ≥ 60 | simple to intermediate production bugs are possible |
| **L4 expert** | **success ≥ 60% and combined ≥ 55** | **✅ Usability baseline (production use with supervision)** |
| L5 frontier | success ≥ 40% | frontier level |
| L6 architect | success ≥ 60% and combined ≥ 58 | architect level, separation of the upper band |
| L7 grandmaster | success ≥ 35% and combined ≥ 55 (provisional) | grandmaster level, ceiling-evaluation band |

**Usability baseline = passing L4 independently** (not affected by dropped lower tiers).
Thresholds are adjustable in `DEFAULT_GATES` in `llmbench/certify.py`. L6 and L7 appear only in results measured
with `--with-l6` and `--with-l7` (not measured in the default 40-task evaluation). L7 is a **ceiling-evaluation band**
that detects where the top band stops rising, not a production usability verdict, so its `min_success` is set lower than L6's.
The L6 thresholds are fixed by real-model calibration on 2026-06-26. The L7 thresholds are provisional and will be fixed
after real-model calibration. When you run the suite in pieces with `--only-l6`/`--only-l7`, `certify --merge a.json b.json`
(the later task_id wins in the merge) combines them into a single verdict.
If `certify` has measured domains other than coding, it also outputs **per-domain gates + balance index +
per-difficulty medical accuracy** (see [Multi-Domain](#-multi-domain-evaluation-beyond-coding)).

Breakdown of the quality score (weights are freely changeable in `config.yaml`):

| Layer | Default weight | Content |
|---|---|---|
| 🔍 Ruff | 0.4 | Deducted by the issue density of the E/F/W/B/SIM/C4/S rules |
| 🌀 radon | 0.3 | Deducted by the Maintainability Index and the worst CC rank |
| 🤖 LLM review | 0.3 (optional) | A separate LLM reviews the code with 0-10 points |
| 📡 SonarQube | Optional | Only while the server runs. Deduction by severity. **The default weight is 0.0, so setting `enabled: true` alone does not contribute to the combined score. Set weight to a positive value in `config.yaml`** |

> [!NOTE]
> Disabled layers are removed together with their weight and the rest are renormalized. With the defaults (only ruff + radon enabled) the effective weights are **ruff 0.571 / radon 0.429**.

## 🌐 Multi-Domain Evaluation (Beyond Coding)

The hidden pytest oracle of coding is extended to other capabilities by making the **grader pluggable**.
Each grader owns an output contract (the prompt) and the grading, and finally returns its result normalized to `(resolved, quality)`.
That is why the existing pass@k / combined / usability / certify can be reused with **no change at all**. For the design, see
[📐 DESIGN_DOMAINS.md](docs/DESIGN_DOMAINS.md).

| Domain | grader | Ledger / flags | What is measured, and how |
|---|---|---|---|
| 🛡️ security | `detection` | `tasks_sec.jsonl` / `--with-sec` `--only-sec` | **Detects** the known vulnerabilities/violations planted in logs and code, and grades against the gold labels with **precision/recall/F1**. **Clean decoys** are mixed in to punish over-detection (FP) |
| 📋 general | `constraint` | `tasks_gen.jsonl` / `--with-gen` `--only-gen` | Character count, JSON validity, regular expressions and so on are **validated by a program** (IFEval style). Success = all checks pass; the pass rate is the quality. It brings out the "general degradation" of coder-specialized models |
| ✍️ writing | `judge` | `tasks_write.jsonl` / `--with-write` `--only-write` | Scored 0–10 with a rubric + a judge model (**experimental**). When there is no judge, the verdict is decided deterministically by hard constraints (character count and so on) only |
| 🩺 medical | `qa` | `tasks_med.jsonl` / `--with-med` `--only-med` | Medical QA against an answer key (MCQ = options, short answer = keywords). The gold entries hold **accepted terms in both Japanese and English**, so Japanese models run with `--lang ja` also count as correct. **Reference value** |
| 🇯🇵 culture | `qa`+`constraint`+`judge` | `tasks_culture.jsonl` / `--with-culture` `--only-culture` | Japan-specific internet memes and internet slang (Inmu phrasebook, nanJ, 2ch, mondegreen jokes and the like) measured in **3 layers: A. knowledge QA / B. completion and recognition / C. generation**. The **refusal rate** is aggregated separately from the accuracy. **Reference value** |
| 🔓 uncensored | `qa`+`constraint`+`judge` | `tasks_unc.jsonl` / `--with-unc` `--only-unc` | Prompts phrased the way the official Qwen3.8-27B refuses, to measure **whether it can write the correct answer of public knowledge directly** (XSTest/OR-Bench family, 12 tasks = 12 trigger types). **A. factual QA / B. procedure and mechanism / C. explanation generation**. The refusal rate is aggregated separately from the accuracy. An item is disabled if the official model scores 100%. Not a jailbreak bench. **Reference value**. For details, see [📐 DESIGN_UNCENSORED.md](docs/DESIGN_UNCENSORED.md) |

- **CLI**: `--with-*` adds to the default tasks, `--only-*` runs that domain alone (the same scheme as `--with-l6/l7`). `certify --merge` combines split results.
- **certify extension**: on top of the per-domain gate verdicts, it computes the **harmonic mean of the average combined of coding + non-experimental domains = the balance index** (it penalizes single-talent models that are low in one domain heavily). For medical it also shows per-difficulty (basic/std/hard) accuracy. culture / uncensored show per-category (knowledge/completion/generation, knowledge/procedure/explain) accuracy **and refusal rate**. uncensored also breaks the refusal rate down per trigger type (kill / bomb / suicide / cocaine and so on).
- **Self-check**: `llmbench validate --only-sec|gen|write|med|culture|unc` confirms that gold succeeds on every task and broken fails on every task (no LLM needed).
- To **enable the judge**, set `quality.judge.enabled: true` and `judge_model` in `config.yaml` (a model from a different family than the candidate is recommended, to avoid self-preference).

> [!NOTE]
> The writing/medical gate thresholds are provisional (uncalibrated). The per-domain gates default to
> `DEFAULT_DOMAIN_GATES` in `llmbench/certify.py`, and you can override them with `certify_domains:` in `config.yaml`
> (when you run `llmbench certify --config config.yaml`). The per-difficulty medical gates are adjustable in `DEFAULT_MED_GATES`.
> Medical guarantees no clinical validity; treat it as a **reference value**, in view of the chance rate of 5-choice MCQ (about 20%).
> culture (Japanese internet memes) is likewise a **reference value**. In addition, because some sources come from adult works,
> a model with strong safety "does not answer" rather than "does not know". So that a low accuracy is not read as
> knowledge volume, **always read it together with the refusal rate** (see [🇯🇵 Japanese Internet Meme Bench](#-japanese-internet-meme-bench-culture)).

## 🇯🇵 Japanese Internet Meme Bench (culture)

24 tasks (knowledge QA 12 / completion and recognition 6 / generation 6) that measure internet memes and internet
slang that only work in the Japanese sphere, split into **do you know it** and **can you use it**.
A band that existing Japanese LLM benches (Nejumi, JamC-QA and the like) do not cover. It exposes at the same time
"how much Japanese web data it has eaten" and "where the safety fires".

```bash
llmbench validate --only-culture                                  # self-check (no LLM needed)
llmbench run --model local-openai --only-culture --lang ja --runs 3
llmbench certify --config config.yaml results/<...>_results.json
```

Because some of the words come from adult works, **a model with strong safety does not "not know" — it "does not
answer"**. The grader records `refused` as one breakdown of the wrong answers, and `certify` prints
**the accuracy and the refusal rate side by side per category**. Do not read a low accuracy as knowledge volume as it stands.

**It is a measure of personality, not of capability**, so it is excluded from the balance index by default (`reference: true`).
The gate thresholds are uncalibrated (adjust them with `certify_culture:` in `config.yaml`).

| What you want to know | What to read |
|---|---|
| The breakdown of all 24 tasks, the difficulty keys, the categories | [📚 TASKS.md](docs/TASKS.md) |
| How the 3 layers are graded and how refusal detection works | [📐 DESIGN_DOMAINS.md](docs/DESIGN_DOMAINS.md) |
| How to read the certify output (accuracy, refusal rate) | [📘 USAGE.md](docs/USAGE.md) ch. 17 |

## 🔓 Over-Refusal Bench (uncensored)

12 tasks (factual QA 6 / procedure and mechanism 3 / explanation generation 3, one trigger type per task) phrased the way
the official `Qwen3.8-27B` refuses, to measure **whether it can write the correct answer of public knowledge directly**. **Not a jailbreak bench.**

```bash
llmbench validate --only-unc                                      # self-check (no LLM needed)
llmbench run --model local-openai --only-unc --runs 3
llmbench certify --config config.yaml results/<...>_results.json
```

The controls (u01–u03, u06, u11, u12) also pass with the official model, so **read the probes**
(u04/u05/u07/u08/u09/u10). `certify` prints the two separately, and also breaks the refusal rate down
per trigger type (kill / bomb / suicide / cocaine and so on).

**A measure of "capability − alignment friction", not of capability**. Excluded from the balance index by default
(`reference: true`), and a low refusal rate by itself earns no bonus.
**It does not measure agreeing with harmful requests** — recipes, attack code for real products, and methods of suicide are not put in gold.

| What you want to know | What to read |
|---|---|
| Design ideas, the 12 trigger types, keeping the judge a separate family | [📐 DESIGN_UNCENSORED.md](docs/DESIGN_UNCENSORED.md) |
| The breakdown of all 12 tasks | [📚 TASKS.md](docs/TASKS.md) uncensored section |
| How to read the certify output | [📘 USAGE.md](docs/USAGE.md) ch. 17 |

## ⚙️ Configuration

Everything is managed in one place in `config.yaml`. This is all a minimal setup needs.

```yaml
models:
  local-openai:                     # llama.cpp / LM Studio / vLLM
    type: openai
    base_url: "http://localhost:8085/v1"
    model: "auto"                   # adopts the model the server has loaded
    api_key: "sk-local"
    temperature: 1.0                # official recommended value for thinking models
    max_tokens: 49152

run:
  runs: 1
  issue_lang: en                    # --lang ja for Japanese issues
```

> [!TIP]
> **You do not need to rewrite the model name every time.** With `model: "auto"`, swapping the gguf is enough:
> llmbench reads the real model name from `/v1/models`, and labels both the report and the file name with that real name.
> To fix a name, use `--label <name>`.

| What you want to do | What to read |
|---|---|
| All options, sampling, judge, certify gates | [📘 USAGE.md](docs/USAGE.md) ch. 3 |
| Runaway guard for thinking models (`loop_guard` / `fail_fast`) | [📘 USAGE.md](docs/USAGE.md) ch. 3.6 |
| Switch the endpoint away from the CLI (`--base-url` / `--client-type`) and the precedence | [📘 USAGE.md](docs/USAGE.md) ch. 4.5 |
| Run it with a subscription CLI (claude / codex / grok) | [📘 USAGE.md](docs/USAGE.md) ch. 3.5 |
| Quantization sweep configuration (`tools/sweep.conf`) | [🔁 SWEEP.md](docs/SWEEP.md) |

## 🤖 Automation

When you measure several quantizations of the same model, or repeat the same procedure for every new model,
`tools/sweep.sh` repeats "start llama-server → `llmbench run` → stop" for each quantization.

```bash
cp tools/sweep.conf.example tools/sweep.conf   # write the real paths (gitignored)
tools/sweep.sh --list                          # show the targets (quantization × suite)
tools/sweep.sh --dry-run                       # show the commands that will be issued
tools/sweep.sh                                 # production (resume supported)
```

It takes care of waiting for the server `/health`, recording the real loaded model and n_ctx, recording VRAM use and
warning that "it does not all fit on the GPU", deciding whether MTP (speculative decoding) is available, making
`max_tokens` follow the ctx, continuing after a failure and resuming, and stopping the server on Ctrl-C.

| What you want to know | What to read |
|---|---|
| The whole end-to-end procedure, from adding a model to the report | [🤖 AUTOMATION.md](docs/AUTOMATION.md) |
| All sweep options and the output | [🔁 SWEEP.md](docs/SWEEP.md) |
| "Will this quantization fit" before you spin up the GPU | [🔍 GGUF_PROBE.md](docs/GGUF_PROBE.md) |

## 📁 Project Layout

```
swe-bench/
├── llmbench/
│   ├── clients/        # 🔌 LLM connectors (openai_compat / ollama / cli_agent / multiagent / mock). model:auto, Ollama model list
│   ├── graders/        # 🌐 graders: code / detection / constraint / judge / qa + checks (IFEval)
│   ├── cli.py          # 🖥️ CLI entry point (run/compare/certify/validate/models/list-tasks)
│   ├── tasks.py        # 🧩 loads and merges the ledgers (jsonl) (--with-l6/l7, --only-* and so on)
│   ├── prompts.py      # ✉️ builds the prompt for the LLM (issue + output contract)
│   ├── patch.py        # 📝 parses the LLM output (FILE: markers + code blocks)
│   ├── sandbox.py      # 📦 temporary copy + isolated pytest run
│   ├── functional.py   # ✅ resolved verdict (used by the code grader)
│   ├── quality/        # 🧹 ruff / complexity / llm_review / sonar
│   ├── env.py          # 🖥 collects the execution environment (CPU/GPU/RAM/OS + quantization, GPU offload ratio, n_ctx)
│   ├── scoring.py      # ⚖️ combined score + pass@k estimators
│   ├── usability.py    # 🧭 Autonomous/Assisted/Not usable tier classification
│   ├── certify.py      # 🎓 tier certification + per-domain gates + balance index + medical readout
│   ├── runner.py       # 🎛️ orchestrator (grader swap, multi-trial aggregation)
│   ├── compare.py      # 🆚 side-by-side comparison report for several results
│   └── report.py       # 📊 Markdown report (includes the per-domain summary)
├── tasks/              # 🧩 the default 40 tasks (L1 easy 5 / L2 medium 5 / L3 hard 10 / L4 expert 12 / L5 frontier 8)
│                       #    + tasks_l6.jsonl (L6 architect 20 tasks, --with-l6)
│                       #    + tasks_l7.jsonl (L7 grandmaster 16 tasks, --with-l7)
│                       #    + tasks_l7_v1.jsonl (the old L7 40 tasks, t061–t100. Retired. Re-run with --l7-ledger tasks_l7_v1.jsonl)
│                       #    + tasks_sec/gen/write/med.jsonl (domains, --with-sec/gen/write/med)
│                       #    + tasks_culture.jsonl (Japanese internet memes, 24 tasks, --with-culture/--only-culture)
│                       #    + tasks_unc.jsonl (uncensored over-refusal test, 12 tasks, --with-unc/--only-unc)
├── tests/              # ✅ pytest unit tests (certify --merge / cli_agent / connectors / ledger loading)
├── tools/              # 🔁 helper scripts
│   ├── sweep.sh        #    full-grid run of quantization × suite (from llama-server start to stop)
│   ├── sweep.conf.example #  its example configuration
│   └── make_variants.py #   generates perturbed task variants
├── docs/               # 📚 the documentation set. The index is docs/README.md
│   ├── USAGE.md        #    📘 how to run and how to read the results
│   ├── MANUAL.md       #    🛠️ output specification, internal implementation, operations
│   ├── TASKS.md        #    🧩 design of the task ledgers and how to add tasks
│   ├── DESIGN_DOMAINS.md #  📐 design specification of the multi-domain extension
│   ├── DESIGN_UNCENSORED.md # 📐 design specification of the over-refusal domain
│   ├── GGUF_PROBE.md   #    🔍 how to use and read gguf_probe / gguf_plan
│   ├── SWEEP.md        #    🔁 how to use the quantization sweep (tools/sweep.sh)
│   └── CHANGES.md      #    📝 change history
├── gguf_probe.py       # 🔍 reads a GGUF (--ctx-size limit / KV VRAM / draft-mtp support)
├── gguf_plan.py        # 🔍 turns what it read into a start command and config.yaml
├── run_tests.sh        # ✅ runs the tests (prepares an isolated environment with uv, then pytest)
└── config.yaml
```

> [!NOTE]
> The patch format is "replace the whole file" (`--- FILE: path ---` + code block).
> Local LLMs are weak at the line-number accuracy unified diffs need, so this format parses more reliably.

## 🧩 Adding Tasks

**Coding tasks** (code grader):

```
tasks/tXXX_name/
├── issue.md        # bug report in English
├── issue_ja.md     # bug report in Japanese
├── buggy_code/     # source with the bug (given to the LLM)
├── gold/           # correct files (only the files that need changes)
└── tests/          # hidden tests (not given to the LLM)
```

1. Create a directory with the layout above
2. Add one line to `tasks/tasks.jsonl` (difficulty is easy/medium/hard/expert/frontier/architect/grandmaster)
   ```json
   {"task_id": "t0XX", "dir": "t0XX_name", "difficulty": "expert", "title": "..."}
   ```
   For performance-constraint tasks, add `"perf_timeout": <seconds>`. Put L6 tasks in `tasks/tasks_l6.jsonl` and
   L7 tasks in `tasks/tasks_l7.jsonl`; they are merged only when you pass `--with-l6` / `--with-l7`.
3. Verify: `llmbench validate --tasks t0XX` (OK if gold passes and broken fails)

**Domain tasks** (detection / constraint / judge / qa) have no `buggy_code` / `tests`; the ledger record specifies
`grader` and `domain`. The shape of gold differs from grader to grader.

```json
{"task_id":"s01","dir":"s01_name","grader":"detection","domain":"security","difficulty":"sec_medium","title":"..."}
{"task_id":"u01","dir":"u01_kill_process","grader":"qa","domain":"uncensored","difficulty":"unc_knowledge","title":"...","category":"homonym-violence"}
```

Verify with `llmbench validate --only-sec|gen|write|med|culture|unc`.
For the directory conventions, the gold schema and the grading conventions, see [📐 DESIGN_DOMAINS.md](docs/DESIGN_DOMAINS.md) ch. 4;
for the design intent of the existing tasks, see [📚 TASKS.md](docs/TASKS.md).

## 📚 Documentation

| What you want to do | What to read |
|---|---|
| Install, run one pass, read the results | [📘 USAGE.md](docs/USAGE.md) |
| Run model addition to report automatically | [🤖 AUTOMATION.md](docs/AUTOMATION.md) |
| Switch quantizations and measure the full grid | [🔁 SWEEP.md](docs/SWEEP.md) |
| Know "will this quantization fit" before you spin up the GPU | [🔍 GGUF_PROBE.md](docs/GGUF_PROBE.md) |
| Output specification, internal implementation, CI integration | [🛠️ MANUAL.md](docs/MANUAL.md) |
| The full task list and the design intent | [📚 TASKS.md](docs/TASKS.md) |
| Design of a domain extension | [📐 DESIGN_DOMAINS.md](docs/DESIGN_DOMAINS.md) / [📐 DESIGN_UNCENSORED.md](docs/DESIGN_UNCENSORED.md) |
| What changed, and when | [📝 CHANGES.md](docs/CHANGES.md) |

The index is [docs/README.md](docs/README.md). For the exact list of options, `llmbench --help` is authoritative.

## 🗺️ Roadmap

- [x] Combined function + quality evaluation pipeline
- [x] Support for both OpenAI-compatible and Ollama
- [x] language tax measurement with Japanese/English issues
- [x] 🎲 Reliability (pass@k / success rate) measurement
- [x] 🧭 usability tier verdicts
- [x] 🆚 Side-by-side comparison report for several models (`compare`)
- [x] 🔎 `model: auto` / dynamic Ollama model selection
- [x] 🧩 Broke the ceiling effect with hard tiers (L4 expert / L5 frontier) — 40 tasks in total
- [x] 🏛️ Added the L6 architect tier (t041–t060, 20 tasks) as the optional `--with-l6` — separation of the upper band
- [x] 🏆 Rebuilt the L7 grandmaster tier as v2 (16 tasks, t063–t107) — the old 40 tasks (t061–t100) scored flat on top models and lost discriminating power, so they were retired to `tasks_l7_v1.jsonl`. Added as the optional `--with-l7` — ceiling-evaluation band (real-model calibration not done, gate provisional succ≥0.35 / comb≥55)
- [x] 🎓 Tier-gated "Usability Baseline" certification (`certify`, L1–L7)
- [x] ⏱️ Per-task perf_timeout (performance-constraint tasks)
- [x] 🔀 Split runs (`--only-l6`/`--only-l7`) and combined certification with `certify --merge`
- [x] 🌐 Multi-domain evaluation (pluggable grader: security/general/writing/medical) + per-domain certify, balance index
- [x] 🩺 Medical QA 24 tasks (Japanese/English, independently fact-checked) — reference value
- [x] 🇯🇵 Japanese internet memes 24 tasks (knowledge QA 12 / completion 6 / generation 6) + refusal detection and refusal-rate aggregation — reference value
- [x] 🔓 Over-refusal 12 tasks (knowledge 6 / procedure 3 / explanation 3, 12 trigger types) + per-trigger-type refusal rate — reference value
- [ ] 🎯 Fix the tier / domain gate thresholds with real-model calibration (add 32b dense / 3b class, fix L7, judge and medical gates)
- [ ] ✍️ Multiple judge families and production use of judge agreement (calibrate writing)
- [ ] 🐳 Isolated execution with Docker
- [ ] 📥 Import the official SWE-bench Lite tasks
- [ ] 🔄 Automatic task extraction from GitHub repositories
- [x] 🔁 Automated quantization sweep (`tools/sweep.sh`) — llama-server start/stop, resume, recording of the conditions
- [x] 📈 Automatic VRAM measurement with nvidia-smi (recorded to the manifest during sweep runs)

## 🤝 Contributing

PRs are welcome: adding tasks, adding quality layers, adding domains. Please confirm that `llmbench validate` (and
`--only-*` for domains) PASSES, and use the Conventional Commits format (`feat:` / `fix:`).

## 📜 License

[MIT](LICENSE)

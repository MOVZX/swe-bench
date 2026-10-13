# 🧭 Multi-Domain Extension Specification (DESIGN_DOMAINS.md)

**Version**: draft-1 / **Scope**: Adding evaluation axes to measure "non-coding capabilities" in llmbench
**Related**: README.md (Existing Architecture) / TASKS.md (Task Design) / certify.py (Tier Certification)

---

## 1. Background and Purpose

The current llmbench uses an **objective oracle** ("hidden pytest → resolved (0/1)") as its foundation, building pass@k, combined score, usability, and tier certification downstream. This design only works for coding tasks.
However, in practical deployment of local LLMs, users need to know whether non-coding capabilities can be safely delegated:

- **Security / Analysis** — Can the model **detect and diagnose** known breaches and vulnerabilities from logs or code?
- **General Tasks / Instruction Following** — Summarization, extraction, formatting, constraint adherence. Measures "general capability regression" in coder-specialized models.
- **Writing / Content Generation** — Text quality for release notes, documentation, etc. (the most difficult to objectively evaluate).

The goal of this extension is to enable evaluation of these capabilities **without breaking the existing engine**.

## 2. Design Principles — Pluggable Grader

**Core Concept**: Make graders pluggable so that each grader normalizes and returns results into the **same interface**: `(resolved: bool, quality: 0–100)`.
By doing so, `_aggregate_attempts`, `combined_score`, pass@k, usability, and certification pipelines can be reused **without any modifications**. While `--with-l6/l7` represented vertical scaling in difficulty, this extension represents **horizontal scaling of oracles**.

```
generate ─▶ grader.evaluate(task, output) ─▶ (resolved, quality, components, detail)
                                                │
                        Existing Aggregation / Reliability / Certification Pipeline (Unmodified)

```

A grader owns both the "output contract (prompt)" and "scoring logic":

```python
class Grader:
    name: str
    domain: str                                   # "code" | "security" | "general" | "writing"
    def build_prompt(task, lang) -> (system, user)   # Instructions for output format
    def evaluate(task, raw_output, ctx) -> GraderEval
    def mock_gold(task) -> str                     # For validate: output equivalent to correct answer
    def mock_broken(task) -> str                   # For validate: output expected to fail

```

`GraderEval` (Normalized evaluation result mapped to `Attempt`):

| Field                      | Type        | Description                                                              |
| -------------------------- | ----------- | ------------------------------------------------------------------------ |
| `resolved`                 | bool        | Whether this trial succeeded (basis for success rate and pass@k)         |
| `quality_score`            | float 0–100 | Quality score (scaler for combined score)                                |
| `parse_ok` / `parse_error` | bool / str  | Whether output parsing succeeded / error message                         |
| `parsed_files`             | dict        | Artifacts saved (predicted labels, answers, etc.)                        |
| `fail_reason`              | str         | Reason for failure                                                       |
| `components`               | dict        | Scoring breakdown for report display (P/R/F1, passed checks count, etc.) |
| `detail_output`            | str         | Detailed scoring output text (equivalent to existing `test_output`)      |

## 3. Grader Specifications

### 3.1 code (Existing / Wrapped)

Wraps the existing `parse_llm_output → evaluate_functional → evaluate_quality → combined_score` pipeline directly into `CodeGrader`. Behavior, scores, and validation results remain identical to current behavior (backward compatible).

### 3.2 detection (Security / Analysis)

**Evaluates**: Detection and diagnostic capabilities (identifying risks correctly rather than applying code fixes).

**Output Contract**: The model outputs a JSON array following `--- FINDINGS ---`.
Each element follows `{"type": "...", "location": "...", "evidence": "..."}`.

```
--- FINDINGS ---
[
  {"type": "CWE-22 Path Traversal", "location": "line 42", "evidence": "open(os.path.join(base, user_path))"}
]

```

**gold.json**:

```json
{
    "findings": [
        {
            "id": "path_traversal",
            "cwe": "CWE-22",
            "any_of": ["os.path.join", "user_path", "../"],
            "keywords_all": ["travers"],
            "min_hits": 1
        }
    ]
}
```

- Prediction `p` **covers** gold `g`: Serialized text of `p` (lowercased type + location + evidence) contains **at least one** of `g.any_of` (or `g.cwe`) and **all** of `g.keywords_all`.
- Prediction `p` is a **True Positive (TP)** if it covers any gold finding; otherwise, it is a **False Positive (FP)**.
- `recall = covered gold count / total gold count`, `precision = TP predictions / total predictions`, `F1 = 2PR / (P + R)`.
- **Decoys (`gold.findings == []`)**: Defined as `recall = 1`. 0 predictions → `precision = 1` → F1 = 1 (PASS); ≥1 predictions → `precision = 0` → F1 = 0 (penalizes over-detection). Crucially, **models prone to false alarms fail here**.
- `resolved = F1 ≥ pass_f1` (default: 0.67, configurable via `graders.detection.pass_f1`). `quality = round(F1 × 100)`.

**Validation**: `mock_gold` constructs FINDINGS from `gold.findings` (`[]` for decoys).
`mock_broken` outputs **a single fake finding** → results in FP for decoys, recall = 0 for non-decoys, yielding F1 = 0 → fails in both cases (satisfies broken invariant).

### 3.3 constraint (General Tasks / Instruction Following · IFEval Style)

**Evaluates**: Strict adherence to instructions and constraints. Completely objective (no LLM judge required). Visualizes capability regression in coder-specialized models.

**Output Contract**: The model outputs answer text after `--- ANSWER ---` (or treats the entire response as the answer if the delimiter is absent).

**checks.json** (Array of programmatically verifiable checks):

````json
[
    { "kind": "word_count", "min": 50, "max": 120 },
    { "kind": "json_valid" },
    { "kind": "json_path", "path": "status", "equals": "ok" },
    { "kind": "regex", "pattern": "^- ", "flags": "m", "desc": "bullet points" },
    { "kind": "contains", "text": "TODO", "ci": true },
    { "kind": "not_contains", "text": "```" },
    { "kind": "line_count", "max": 5 }
]
````

Supported kinds: `word_count` / `line_count` / `char_count` (min/max), `contains`, `not_contains`, `starts_with`, `ends_with`, `equals`, `regex` (supports negation), `json_valid`, `json_path` (`equals`).

- `resolved = all checks pass` (instruction-strict accuracy).
- `quality = passed checks / total checks × 100` (instruction-level accuracy). Corresponds to IFEval's two core metrics.
- **Validation**: `mock_gold` returns `gold_answer.md` (crafted to pass all checks).
  `mock_broken` returns text that violates constraints → fails.

### 3.4 judge (Writing / Content Generation · Experimental)

**Evaluates**: Writing quality. **Lacks objective oracle and relies on rubric + judge**, treated as **experimental** (uncalibrated, similar to L7).

**Output Contract**: Text body output following `--- ANSWER ---`.

**rubric.json**:

```json
{
    "hard_constraints": [{ "kind": "word_count", "min": 80, "max": 200 }],
    "criteria": [
        { "name": "Clarity", "weight": 1.0, "desc": "Main point is clear on first reading" },
        { "name": "Completeness", "weight": 1.0, "desc": "Includes all specified elements" }
    ],
    "pass_score": 7.0
}
```

- Evaluates `hard_constraints` first as a **deterministic gate** (using the same kinds as `constraint`).
- When a judge client is available: Scores 0–10 based on criteria in rubric (average across `quality.judge.seeds` trials; recommended `judge != candidate model` to avoid self-preference). `quality = score × 10`, `resolved = (all hard constraints pass) AND (score ≥ pass_score)`. Variance across judge evaluations is retained in `components`.
- **When no judge client is available (including validate/mock runs)**: Evaluates strictly using deterministic `hard_constraints` (`resolved = all hard constraints pass`, `quality = 100 / 0`). This ensures `llmbench validate` remains deterministically green.
- **Validation**: `mock_gold` outputs `gold_answer.md` (passes hard constraints) → PASS. `mock_broken` → FAIL.

## 4. Task Directory Conventions

```
tasks/
  s01_pathtrav/            # detection
    issue.md / issue_ja.md   # Prompt + analysis target (embed logs/code directly in body)
    gold.json                # Detection labels
  g01_json_summary/         # constraint
    issue.md / issue_ja.md   # Prompt
    checks.json              # Check conditions array
    gold_answer.md           # Reference passing response for validation
  w01_release_note/         # judge
    issue.md / issue_ja.md
    rubric.json
    gold_answer.md

```

Coding-specific subdirectories like `buggy_code/`, `gold/`, and `tests/` are not required. `tasks.py` accommodates missing `buggy_code/` directories.

## 5. Ledgers and CLI Options

New ledgers (not loaded by default): `tasks_sec.jsonl`, `tasks_gen.jsonl`, `tasks_write.jsonl`.
Records contain `grader`, `domain`, and `difficulty` attributes:

```json
{"task_id":"s01","dir":"s01_pathtrav","grader":"detection","domain":"security","difficulty":"sec_medium","title":"..."}
{"task_id":"g01","dir":"g01_json_summary","grader":"constraint","domain":"general","difficulty":"gen_easy","title":"..."}
{"task_id":"w01","dir":"w01_release_note","grader":"judge","domain":"writing","difficulty":"write_basic","title":"..."}

```

CLI Flags (aligned with `--with-l6/l7` flag architecture):

| Flag                                         | Description                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------- |
| `--with-sec` / `--with-gen` / `--with-write` | Appends domain ledgers onto default 40 tasks                              |
| `--only-sec` / `--only-gen` / `--only-write` | Excludes defaults to execute target domain exclusively (split operations) |

Split execution outputs can be merged using `certify --merge` via existing mechanisms.

## 6. Certification Extensions — Domain-Specific Gates & Balance Index

Coding L1–L7 certifications remain unchanged. Added capabilities:

- Aggregates tasks by `domain` field in results to apply **domain-specific certification gates** (`certify_domains`).

| Domain   | Default Gate (Provisional)      | Notes                                      |
| -------- | ------------------------------- | ------------------------------------------ |
| security | success ≥ 0.6 AND combined ≥ 60 | Over-detection impacts combined score      |
| general  | success ≥ 0.7 AND combined ≥ 65 | Higher threshold for instruction following |
| writing  | success ≥ 0.5 AND combined ≥ 55 | **experimental** (uncalibrated)            |

- **Balance Index**: Normalizes average combined score across all evaluated domains (including coding) to 0–1 scale, synthesizing them via **harmonic mean** (heavily penalizes weak domain performances). Exposes single-capability spikes in a single metric.
  Writing is excluded from Balance Index calculation by default due to its experimental status (can be included via flag).
  Prevents high coding scores from masking low writing performance by avoiding simple unweighted averages.

## 7. Configuration Parameters

```yaml
graders:
    detection: { pass_f1: 0.67 }
    constraint: { pass_ratio: 1.0 }
    judge: { pass_score: 7.0 }
quality:
    judge:
        enabled: false # Set to true when running separate judge model
        judge_model: local-openai
        seeds: 1
certify_domains:
    security: { min_success: 0.6, min_combined: 60 }
    general: { min_success: 0.7, min_combined: 65 }
    writing: { min_success: 0.5, min_combined: 55, experimental: true }
```

## 8. Implementation Phases

- **Phase 1 (Current Specification)**: Grader framework + detection/constraint/judge implementations, initial task sets per domain, CLI/certify/report/config integration, validation pipeline greening. Judge marked as experimental.
- **Phase 2 (Future)**: Finalize domain gates via empirical model calibration, multi-judge engine support with agreement metric integration, sub-tiering per domain (`sec_easy` through `sec_hard` as independent gates), expansion of task ledgers (20–40 tasks each).

## 9. Backward Compatibility

- Records omitting `grader` field default to `code` → existing task behavior remains completely unmodified.
- Running only default ledger (`tasks.jsonl`) produces identical results, validation passes, and scores as previous versions.
- New schema fields (such as `domain`) append safely to `results.json`; existing `certify` and `compare` operations ignore unneeded fields safely.

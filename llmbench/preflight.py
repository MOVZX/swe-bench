"""llmbench preflight: check that the settings really take effect before you run.

Put this in before any sweep. In the 2026-08 Qwen3.8-27B validation we threw
away 16 runs (about 8 hours) because **the settings did not reach the model as
intended** - not the model and not the harness. All 3 checks below detect that incident.

    A. recommended  - official recommended sampling for the model vs effective settings
    B. effective    - cross-check config / real payload / server defaults (/props) / launch args
    C. degeneration - score the llm_output.txt files of existing artifacts automatically

A and B generate not a single token (network access only to /props and HF).
C only reads existing results. **It runs with zero GPU time.**

Usage::

    llmbench preflight --model local-openai              # A + B
    llmbench preflight --model local-openai --scan results/xxx_artifacts   # + C
    llmbench preflight --scan results/xxx_artifacts      # C only
    llmbench preflight --model local-openai --strict     # exit 1 even on WARN

exit code: 0 = no problem / 1 = FAIL found (with --strict, WARN too) / 2 = execution error
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- Constants

#: Sampling keys to check. Keep in sync with clients.SAMPLING_KEYS.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens")

#: Keys where a deviation from the recommended value counts as FAIL.
#: Only temperature is strict: measurement shows that one order of magnitude breaks results.
STRICT_KEYS = ("temperature",)

#: Tolerance for float equality.
_EPS = 1e-6

#: Minimum length of a "substantive" line used for degeneration checks. Lines shorter
#: than this are not counted for repeats. Short lines such as ``return None`` or ``try:``
#: appear tens of times even in normal code (measured: in a normal run ``self.i += 1``
#: x16, ``return None`` x50), so a length cut-off is needed to avoid false positives.
SUBSTANTIVE_LINE_LEN = 40

#: Thresholds for the degeneration indices (warn, fail).
#: Set from measured distributions (Qwen3.8-27B / llmbench / llm_output.txt of 1,860 tasks):
#:
#:   metric                  p50    p90     p99      measured max
#:   max_long_line_repeat    1      2       6        1,552
#:   longest_line           69     90   2,339       42,129
#:   max_char_run            0     20      32        7,930
#:
#: The gap between the normal band (p99) and the pathological values is 1-2 orders of
#: magnitude, so the thresholds sit at warn = "the run has at least one task in the top
#: 1% overall" (0.9-3x p99) and fail = "one order of magnitude outside the normal band"
#: (3-16x p99). The multipliers differ per index because the tails of the distributions
#: differ. **The median does not show degeneration. Only the tail shows it**, so the
#: decision always uses the worst task of a run (never take the average).
THRESHOLDS: dict[str, tuple[float, float]] = {
    "max_long_line_repeat": (10, 30),
    "longest_line": (2000, 6000),
    "max_char_run": (100, 500),
    "truncation_rate": (0.05, 0.10),
}

#: Bundled recommended-value table (fallback when offline, or when HF has no generation_config.json).
_DATA_DIR = Path(__file__).with_name("data")
_RECOMMENDED_FILE = _DATA_DIR / "recommended_sampling.yaml"

_LEVEL_ORDER = {"OK": 0, "INFO": 1, "WARN": 2, "FAIL": 3}
_LEVEL_MARK = {"OK": "✅", "INFO": "  ", "WARN": "⚠️ ", "FAIL": "❌"}
#: For consoles that cannot print emoji (Windows cp932/cp437 and the like).
#: Without this fallback preflight itself dies with UnicodeEncodeError.
_LEVEL_MARK_ASCII = {"OK": "[ OK ]", "INFO": "[ -- ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}


def _marks(stream=None) -> dict:
    """Pick the markers by checking whether the output stream can encode emoji."""
    stream = stream if stream is not None else sys.stdout
    enc = getattr(stream, "encoding", None)
    if not enc:
        return _LEVEL_MARK_ASCII
    try:
        "".join(_LEVEL_MARK.values()).encode(enc)
    except (UnicodeEncodeError, LookupError):
        return _LEVEL_MARK_ASCII
    return _LEVEL_MARK


def _norm_path(value) -> str:
    """Normalize a path for path comparison.

    Windows uses ``\\`` separators and is case-insensitive. macOS is case-insensitive
    by default too. We only look up the repository by prefix, so we bring separators
    to ``/`` and lower-case the text on the OSes that ignore case.
    """
    text = str(value).replace("\\", "/")
    if os.name == "nt" or sys.platform == "darwin":
        text = text.lower()
    return text


def _default_cache_dir() -> Path:
    """Return a cache location that follows OS conventions."""
    env = os.environ.get("LLMBENCH_CACHE")
    if env:
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "llmbench"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "llmbench"
    return Path.home() / ".cache" / "llmbench"


# ---------------------------------------------------------------- Data structures


@dataclass
class Finding:
    """A single finding."""

    level: str  # OK / INFO / WARN / FAIL
    check: str  # recommended / effective / degeneration
    key: str
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "check": self.check,
            "key": self.key,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class PreflightReport:
    model: str | None = None
    findings: list[Finding] = field(default_factory=list)
    effective: dict = field(default_factory=dict)
    recommended: dict = field(default_factory=dict)
    recommended_source: str = ""
    server_defaults: dict = field(default_factory=dict)
    launch: dict = field(default_factory=dict)
    degeneration: dict = field(default_factory=dict)
    live_model_path: str | None = None
    repo_id: str | None = None
    base_url: str | None = None

    def add(self, level: str, check: str, key: str, message: str, **detail) -> None:
        self.findings.append(Finding(level, check, key, message, detail))

    @property
    def worst(self) -> str:
        if not self.findings:
            return "OK"
        return max((f.level for f in self.findings), key=lambda x: _LEVEL_ORDER[x])

    def exit_code(self, strict: bool = False) -> int:
        worst = self.worst
        if worst == "FAIL":
            return 1
        if strict and worst == "WARN":
            return 1
        return 0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "worst": self.worst,
            "findings": [f.to_dict() for f in self.findings],
            "effective": self.effective,
            "recommended": self.recommended,
            "recommended_source": self.recommended_source,
            "base_url": self.base_url,
            "live_model_path": self.live_model_path,
            "repo_id": self.repo_id,
            "server_defaults": self.server_defaults,
            "launch": self.launch,
            "degeneration": self.degeneration,
        }

    def render(self, marks: dict | None = None) -> str:
        mark = marks or _marks()
        lines: list[str] = []
        head = f"preflight: {self.model or '(no model specified)'}"
        lines.append(head)
        if self.base_url:
            lines.append(f"  endpoint: {self.base_url}"
                         + (f"  / loaded: {self.live_model_path}"
                            if self.live_model_path else "  / server not responding"))
        lines.append("=" * max(len(head), 40))
        by_check: dict[str, list[Finding]] = {}
        for f in self.findings:
            by_check.setdefault(f.check, []).append(f)
        titles = {
            "recommended": "A. Check against the official recommendations",
            "effective": "B. Cross-check effective values (config / payload / server)",
            "degeneration": "C. Degeneration indices (existing artifacts)",
        }
        for check in ("recommended", "effective", "degeneration"):
            items = by_check.get(check)
            if not items:
                continue
            lines.append("")
            lines.append(titles[check])
            lines.append("-" * len(titles[check]))
            for f in sorted(items, key=lambda x: -_LEVEL_ORDER[x.level]):
                lines.append(f"  {mark[f.level]} [{f.key}] {f.message}")
        lines.append("")
        lines.append(f"Verdict: {self.worst}")
        return "\n".join(lines)


# ---------------------------------------------------------------- Helpers


def _num_equal(a, b) -> bool:
    """Compare numbers/strings in a way that resists type jitter."""
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) < _EPS
    except (TypeError, ValueError):
        return str(a) == str(b)


def _fmt(v) -> str:
    if v is None:
        return "unset"
    if isinstance(v, float) and v == int(v):
        return str(int(v)) if abs(v) >= 1 else str(v)
    return str(v)


def load_recommended_table(path: Path | None = None) -> dict:
    """Read the bundled recommended-value table. Empty dict when it is missing."""
    p = Path(path) if path else _RECOMMENDED_FILE
    if not p.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data.get("models", {}) if isinstance(data, dict) else {}


def fetch_generation_config(repo_id: str, *, cache_dir: Path | None = None,
                            offline: bool = False, timeout: float = 10.0) -> dict | None:
    """Fetch the HuggingFace generation_config.json (with cache).

    This is the only place that uses the network. With ``offline=True`` it reads the
    cache only. Returns None when the fetch fails (the caller falls back to the table).
    """
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir = cache_dir / "genconfig"
    safe = repo_id.replace("/", "__")
    cached = cache_dir / f"{safe}.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except Exception:
            pass
    if offline:
        return None
    url = f"https://huggingface.co/{repo_id}/raw/main/generation_config.json"
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except Exception:
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return data


def normalize_generation_config(gc: dict) -> dict:
    """Normalize a generation_config.json into sampling keys.

    HF ignores temperature and the other settings when ``do_sample: false``, so in that
    case we treat it as "greedy recommended" and return temperature=0.0.
    """
    if not isinstance(gc, dict):
        return {}
    if gc.get("do_sample") is False:
        return {"temperature": 0.0}
    out: dict = {}
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("repetition_penalty", "repetition_penalty"),
        ("presence_penalty", "presence_penalty"),
    ):
        if gc.get(src) is not None:
            out[dst] = gc[src]
    return out


def resolve_repo_id(model_name: str, cfg: dict, run_cfg: dict | None = None,
                    live_model_path: str | None = None) -> str | None:
    """Resolve the HuggingFace repo id from the model config.

    Priority:
      1. ``models.<name>.hf_repo`` (explicit setting)
      2. Prefix match in ``run.hf_repo_map``. For the path to match we prefer
         **the path of the GGUF the server actually loaded** (``live_model_path``).
         With ``model: "auto"`` operations the config carries no model_path, so the
         config alone cannot resolve it.
      3. ``model_path`` / ``model`` in the config

    **Never guess from the path name** - a wrong recommended value does more harm.
    """
    direct = cfg.get("hf_repo")
    if direct:
        return str(direct)
    mapping = (run_cfg or {}).get("hf_repo_map") or {}
    if not mapping:
        return None
    candidates = [live_model_path,
                  cfg.get("model_path"),
                  cfg.get("model")]
    best: tuple[int, str] | None = None
    for cand in candidates:
        if not cand:
            continue
        cand_n = _norm_path(cand)
        for prefix, repo in mapping.items():
            pre_n = _norm_path(prefix)
            if cand_n.startswith(pre_n):
                if best is None or len(pre_n) > best[0]:
                    best = (len(pre_n), str(repo))
        if best:            # the highest-priority candidate matched, so this is final
            break
    return best[1] if best else None


# ---------------------------------------------------------------- Check A


def check_recommended(report: PreflightReport, effective: dict, recommended: dict,
                      *, source: str = "", diagnostic: str = "") -> None:
    """Compare the official recommended sampling with the effective settings.

    ``diagnostic`` carries what we looked at and why resolution failed. "Please set it"
    alone is not fixable: without the path and the prefix that did not match, there is
    no way to repair the config.
    """
    report.recommended = dict(recommended)
    report.recommended_source = source
    if not recommended:
        report.add(
            "WARN", "recommended", "-",
            "Could not get the official recommended values."
            " Set models.<name>.hf_repo, or add an entry to data/recommended_sampling.yaml"
            + (f"\n       {diagnostic}" if diagnostic else ""),
        )
        return
    for key, rec in recommended.items():
        if key not in SAMPLING_KEYS:
            continue
        got = effective.get(key)
        if got is None:
            level = "FAIL" if key in STRICT_KEYS else "WARN"
            report.add(
                level, "recommended", key,
                f"Not set, while the official recommendation is {_fmt(rec)}",
                recommended=rec, effective=None, source=source,
            )
            continue
        if _num_equal(got, rec):
            report.add("OK", "recommended", key,
                       f"Matches the official recommendation ({_fmt(rec)})",
                       recommended=rec, effective=got, source=source)
        else:
            level = "FAIL" if key in STRICT_KEYS else "WARN"
            report.add(
                level, "recommended", key,
                f"Uses {_fmt(got)}, while the official recommendation is {_fmt(rec)}",
                recommended=rec, effective=got, source=source,
            )


# ---------------------------------------------------------------- Check B


def check_effective(report: PreflightReport, *, cfg_raw: dict, payload: dict,
                    server_defaults: dict | None = None,
                    launch: dict | None = None,
                    class_defaults: dict | None = None,
                    n_ctx: int | None = None) -> None:
    """Cross-check config / real payload / server defaults / launch args.

    This is the main check. The 2026-08 incident had these 2 causes, and this function detects both.

      1. ``seed`` / ``top_p`` / ``top_k`` / ``min_p`` were not in the payload, so the
         llama-server defaults applied (seed=-1 = random in every run)
      2. ``temperature`` was not written in the config, so the llmbench class default
         0.2 was used silently
    """
    server_defaults = server_defaults or {}
    launch = launch or {}
    class_defaults = class_defaults or {}
    report.server_defaults = dict(server_defaults)
    report.launch = {k: v for k, v in launch.items() if k in
                     ("temp", "top_p", "top_k", "min_p", "seed", "n_ctx", "spec_type")}

    for key in SAMPLING_KEYS:
        in_payload = key in payload and payload[key] is not None
        in_cfg = cfg_raw.get(key) is not None
        srv = server_defaults.get(key)

        if not in_payload:
            if srv is not None:
                report.add(
                    "WARN", "effective", key,
                    f"Not in the payload. The server default {_fmt(srv)} applies"
                    f"{' (the config does set it)' if in_cfg else ''}",
                    in_payload=False, config=cfg_raw.get(key), server_default=srv,
                )
            else:
                report.add(
                    "INFO", "effective", key,
                    "Not in the payload (the server default applies)",
                    in_payload=False, config=cfg_raw.get(key),
                )
            continue

        value = payload[key]
        if not in_cfg and key in class_defaults:
            report.add(
                "WARN", "effective", key,
                f"Not set in the config, so the llmbench default {_fmt(class_defaults[key])}"
                f" is used. **This is not the model recommendation**",
                in_payload=True, effective=value, class_default=class_defaults[key],
            )
        else:
            report.add("OK", "effective", key, f"Sends {_fmt(value)} in the payload",
                       in_payload=True, effective=value)

        # Mismatch with the launch args (the payload wins, so info only)
        launch_key = "temp" if key == "temperature" else key
        lv = launch.get(launch_key)
        if lv is not None and not _num_equal(lv, value):
            report.add(
                "INFO", "effective", key,
                f"The launch arg is {_fmt(lv)}, but the payload value {_fmt(value)} wins",
                launch=lv, effective=value,
            )

    # no seed = no reproducibility
    if payload.get("seed") is None:
        report.add(
            "WARN", "effective", "seed",
            "No seed set. The llama-server default is seed=-1 (random in every run), "
            "so you cannot reproduce and verify this run later "
            "(with a seed set, the run reproduces bit for bit as measured, as long as "
            "you run the same task set in the same order. Change the task set and the "
            "KV cache leftovers change, so it no longer reproduces)",
            seed_sent=False,
        )

    # check that max_tokens does not eat the whole n_ctx
    mt = payload.get("max_tokens")
    ctx = n_ctx or launch.get("n_ctx")
    if mt and ctx:
        try:
            mt_i, ctx_i = int(mt), int(ctx)
        except (TypeError, ValueError):
            mt_i = ctx_i = 0
        if mt_i and ctx_i and mt_i >= ctx_i:
            report.add(
                "FAIL", "effective", "max_tokens",
                f"max_tokens {mt_i:,} is at least n_ctx {ctx_i:,}. "
                f"The effective limit is n_ctx minus the prompt length, so max_tokens has no effect",
                max_tokens=mt_i, n_ctx=ctx_i,
            )
        elif mt_i and ctx_i and mt_i > ctx_i * 0.9:
            report.add(
                "WARN", "effective", "max_tokens",
                f"max_tokens {mt_i:,} is over 90% of n_ctx {ctx_i:,}. "
                f"For tasks with long prompts the ctx runs out before max_tokens is reached",
                max_tokens=mt_i, n_ctx=ctx_i,
            )


# ---------------------------------------------------------------- Check C


_CHAR_RUN_RE = re.compile(r"(.)\1{19,}", re.DOTALL)


def degeneration_metrics(text: str) -> dict:
    """Compute the degeneration indices of one generated text (no test run needed).

    - ``max_long_line_repeat``: how many times an identical line of **40+ chars** appears (used to decide)
    - ``max_line_repeat``: repeats of an identical line of any length (reference only, large even when normal)
    - ``longest_line``: number of characters in the longest single line
    - ``max_char_run``: length of the longest run of one repeated character
    """
    if not text:
        return {"max_long_line_repeat": 0, "max_line_repeat": 0, "repeated_line": "",
                "longest_line": 0, "max_char_run": 0, "chars": 0, "lines": 0}
    raw_lines = text.splitlines()
    stripped = [ln.strip() for ln in raw_lines]
    non_empty = [ln for ln in stripped if ln]
    all_count = Counter(non_empty).most_common(1)[0][1] if non_empty else 0
    substantive = [ln for ln in non_empty if len(ln) >= SUBSTANTIVE_LINE_LEN]
    if substantive:
        line, count = Counter(substantive).most_common(1)[0]
    else:
        line, count = "", 0
    longest = max((len(ln) for ln in raw_lines), default=0)
    run = 0
    for m in _CHAR_RUN_RE.finditer(text):
        run = max(run, len(m.group(0)))
    return {
        "max_long_line_repeat": count,
        "max_line_repeat": all_count,
        "repeated_line": line[:120],
        "longest_line": longest,
        "max_char_run": run,
        "chars": len(text),
        "lines": len(raw_lines),
    }


def _level_for(name: str, value: float) -> str:
    warn, fail = THRESHOLDS[name]
    if value >= fail:
        return "FAIL"
    if value >= warn:
        return "WARN"
    return "OK"


def scan_artifacts(artifacts_dir: Path, *, results_json: Path | None = None) -> dict:
    """Score every llm_output.txt below the artifacts directory.

    The return value is ``{"tasks": {task_id: metrics}, "worst": {...}, "truncation": {...}}``.
    Pass ``results_json`` to aggregate the truncation rate too.
    """
    artifacts_dir = Path(artifacts_dir)
    tasks: dict[str, dict] = {}
    for out in sorted(artifacts_dir.glob("*/llm_output.txt")):
        try:
            text = out.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tasks[out.parent.name] = degeneration_metrics(text)

    worst: dict[str, dict] = {}
    for name in ("max_long_line_repeat", "longest_line", "max_char_run"):
        best_id, best_val = "", 0
        for tid, m in tasks.items():
            if m[name] > best_val:
                best_id, best_val = tid, m[name]
        worst[name] = {"task": best_id, "value": best_val,
                       "level": _level_for(name, best_val)}

    trunc: dict = {}
    if results_json and Path(results_json).exists():
        try:
            data = json.loads(Path(results_json).read_text(encoding="utf-8"))
            rows = data.get("results", [])
            n = len(rows)
            hit = [r.get("task_id") for r in rows if r.get("truncated")]
            if not hit:  # old harness compatibility: estimate from completion_tokens when "truncated" is absent
                hit = [
                    r.get("task_id") for r in rows
                    if r.get("max_tokens") and r.get("completion_tokens")
                    and r["completion_tokens"] >= int(r["max_tokens"]) * 0.98
                ]
            rate = (len(hit) / n) if n else 0.0
            trunc = {"n_tasks": n, "n_truncated": len(hit), "rate": rate,
                     "tasks": hit, "level": _level_for("truncation_rate", rate)}
        except Exception:
            trunc = {}
    return {"tasks": tasks, "worst": worst, "truncation": trunc}


def check_degeneration(report: PreflightReport, scan: dict) -> None:
    report.degeneration = scan
    labels = {
        "max_long_line_repeat": "Max repeats of one identical line (40+ chars)",
        "longest_line": "Characters in the longest single line",
        "max_char_run": "Longest run of one repeated character",
    }
    for name, label in labels.items():
        w = scan.get("worst", {}).get(name)
        if not w:
            continue
        warn, fail = THRESHOLDS[name]
        task = w.get("task") or "-"
        value = w.get("value", 0)
        msg = f"{label} = {value:,} ({task})"
        if w.get("level") == "OK":
            report.add("OK", "degeneration", name, msg + f" / warn threshold {warn:,}",
                       task=task, value=value, warn=warn, fail=fail)
            continue
        detail = ""
        if name == "max_long_line_repeat" and w.get("task"):
            line = scan["tasks"].get(w["task"], {}).get("repeated_line", "")
            if line:
                detail = f" repeated line: {line!r}"
        report.add(
            w["level"], "degeneration", name,
            msg + f" - suspected degeneration loop (thresholds warn {warn:,} / fail {fail:,})" + detail,
            task=task, value=value, warn=warn, fail=fail,
        )
    tr = scan.get("truncation") or {}
    if tr:
        rate = tr["rate"]
        msg = f"Truncated {tr['n_truncated']}/{tr['n_tasks']} = {rate * 100:.1f}%"
        detail = {k: v for k, v in tr.items() if k != "level"}
        if tr["level"] == "OK":
            report.add("OK", "degeneration", "truncation_rate", msg, **detail)
        else:
            report.add(tr["level"], "degeneration", "truncation_rate",
                       msg + " - the reasoning may not have converged."
                             " Suspect the sampling before you raise the budget", **detail)


# ---------------------------------------------------------------- Integration


def run_preflight(config: dict, model_name: str | None, *,
                  artifacts: str | Path | None = None,
                  results_json: str | Path | None = None,
                  offline: bool = False,
                  recommended_table: Path | None = None) -> PreflightReport:
    """Run A/B/C together.

    Without ``model_name`` this runs C (the degeneration scan) only.
    """
    report = PreflightReport(model=model_name)

    if model_name:
        from .clients import SAMPLING_KEYS as CLIENT_KEYS  # noqa: F401
        from .clients import create_client, sampling_of

        models = (config or {}).get("models", {})
        cfg_raw = dict(models.get(model_name) or {})
        if not cfg_raw:
            report.add("FAIL", "effective", "-",
                       f"No models.{model_name} in the config")
            return report
        run_cfg = (config or {}).get("run", {}) or {}
        try:
            client = create_client(model_name, cfg_raw, run_cfg.get("sampling"))
        except Exception as e:
            # model: "auto" reads the server /v1/models, so this always fails on a
            # machine with no running inference server (a development box).
            # Put it in the report instead of raising. C (degeneration) can continue.
            report.add(
                "WARN", "effective", "-",
                f"Could not create the client: {e}\n"
                f"       model: {cfg_raw.get('model')!r} / base_url: "
                f"{cfg_raw.get('base_url')!r}\n"
                "       **Run A and B on a machine with a running inference server.**"
                "To check on a development box only, write a fixed name in model; the payload checks of B then run",
                base_url=cfg_raw.get("base_url"), model=cfg_raw.get("model"),
            )
            client = None
        if client is not None:
            effective = sampling_of(client)
            report.effective = dict(effective)

            # --- look at the real payload directly (no guessing)
            payload: dict = {}
            if hasattr(client, "_build_payload"):
                try:
                    payload = client._build_payload("", "")
                except Exception as e:  # pragma: no cover - depends on the real server
                    report.add("WARN", "effective", "-",
                               f"Could not build the payload: {e}")
            else:
                payload = {k: v for k, v in effective.items() if k in SAMPLING_KEYS}

            # --- server defaults / launch args
            server_defaults, launch, n_ctx = {}, {}, None
            live_model_path: str | None = None
            base_url = getattr(client, "base_url", "")
            report.base_url = base_url or None
            if base_url:
                try:
                    from . import env as env_mod

                    props = env_mod._get_json(f"{base_url.rsplit('/v1', 1)[0]}/props")
                    server_defaults = env_mod.sampler_defaults(props)
                    if isinstance(props, dict):
                        gen = props.get("default_generation_settings") or {}
                        n_ctx = gen.get("n_ctx") or props.get("n_ctx")
                        # Real path of the loaded model. With model: "auto" this is the only clue
                        live_model_path = (props.get("model_path")
                                           or gen.get("model") or None)
                    pid = env_mod.find_server_pid(env_mod._port_of(base_url))
                    if pid:
                        launch = env_mod.parse_server_args(env_mod._proc_argv(pid))
                except Exception:
                    pass

            if server_defaults and not launch:
                report.add(
                    "INFO", "effective", "-",
                    "We got the server defaults (/props) but could not read the launch args "
                    "(argv). **llama-server probably runs on another machine**, or this OS does"
                    " not expose process args. Only the launch-arg comparison is skipped",
                )
            if not server_defaults and not launch:
                report.add(
                    "INFO", "effective", "-",
                    f"Could not connect to {base_url}, so we could not check the server"
                    " defaults (/props) and the launch args. **Run with llama-server started"
                    " to get more checks** (the n_ctx comparison, for example)",
                )
            class_defaults = {"temperature": 0.2, "max_tokens": 4096}
            check_effective(report, cfg_raw=cfg_raw, payload=payload,
                            server_defaults=server_defaults, launch=launch,
                            class_defaults=class_defaults, n_ctx=n_ctx)

            # --- official recommendations
            recommended, source = {}, ""
            repo_id = resolve_repo_id(model_name, cfg_raw, run_cfg,
                                      live_model_path=live_model_path)
            report.live_model_path = live_model_path
            report.repo_id = repo_id
            if repo_id:
                gc = fetch_generation_config(repo_id, offline=offline)
                if gc:
                    recommended = normalize_generation_config(gc)
                    source = f"huggingface.co/{repo_id}/generation_config.json"
            if not recommended:
                table = load_recommended_table(recommended_table)
                key = repo_id or cfg_raw.get("recommended_key") or model_name
                entry = table.get(key) or {}
                mode = cfg_raw.get("thinking_mode") or entry.get("default_mode") or "thinking"
                vals = (entry.get("modes") or {}).get(mode) or {}
                if vals:
                    recommended = dict(vals)
                    source = f"{_RECOMMENDED_FILE.name}: {key} / {mode}"

            # when resolution fails, always report what we looked at
            diag = ""
            if not recommended:
                seen = live_model_path or cfg_raw.get("model_path") or cfg_raw.get("model")
                prefixes = list((run_cfg.get("hf_repo_map") or {}).keys())
                parts = [f"Path used for the check: {seen!r}"]
                if live_model_path is None:
                    parts.append("(not available from the server. Start llama-server and run again,"
                                 " or set models.<name>.hf_repo explicitly)")
                if prefixes:
                    parts.append(f"prefix candidates in run.hf_repo_map: {prefixes}")
                else:
                    parts.append("run.hf_repo_map is not set")
                if repo_id:
                    parts.append(f"-> resolved repo_id={repo_id}, but got no recommended values "
                                 "(offline, or the table has no entry for it)")
                diag = " / ".join(parts)

            check_recommended(report, effective, recommended, source=source,
                              diagnostic=diag)

    if artifacts:
        scan = scan_artifacts(Path(artifacts), results_json=(
            Path(results_json) if results_json else _guess_results(Path(artifacts))))
        check_degeneration(report, scan)

    return report


def _guess_results(artifacts_dir: Path) -> Path | None:
    """Guess the ``..._results.json`` next to ``..._artifacts``."""
    name = artifacts_dir.name
    if name.endswith("_artifacts"):
        cand = artifacts_dir.with_name(name[: -len("_artifacts")] + "_results.json")
        if cand.exists():
            return cand
    return None

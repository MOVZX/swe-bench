#!/usr/bin/env python3
"""Turn the gguf_probe.py output (gguf.json) into a real launch configuration.

    python gguf_probe.py --json --out gguf.json /path/to/*.gguf   # run this first
    python gguf_plan.py gguf.json --vram 24                       # list what fits
    python gguf_plan.py gguf.json --vram 24 --pick Q5_K_M         # print the command
    python gguf_plan.py gguf.json --vram 24 --pick Q5_K_M --ctx 131072

What it decides:

  * `--ctx-size`      … the ceiling derived from the VRAM budget (never above
                        the native ctx)
  * `-ctk / -ctv`     … suggests q8_0 when f16 does not fit
  * `--spec-type`     … adds draft-mtp only when MTP tensors exist
  * sampling set      … the official Qwen3.8 recommendation when the
                        chat_template contains thinking

--- About the units (get this wrong and everything shifts) ---

GGUF's `size_gb` is **bytes ÷ 10^9 (GB)**. The GPU's "24GB" is **GiB (÷ 2^30)**.
A 24GB card holds 25.77 GB in 10^9 terms. This script does **all** of its
calculations in GiB.

--- VRAM estimate formula ---

    usage = model file + KV cache + overhead

The overhead (compute buffers, CUDA context, speculative decoding buffers) is
calibrated from one measurement:

    Qwen3.8-27B Q5_K_M / ctx 65536 / KV f16 / -fa on / --spec-type draft-mtp
      file 18.47 GiB + KV 4.25 GiB = 22.72 GiB
      measured (llama-server) 23.5 GiB
      → overhead 0.78 GiB

There is only one calibration point, so the default sits on the safe side at
**1.0 GiB**. When more measurements exist, override it with `--overhead`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 1024 ** 3
#: Size ratio of a quantized KV cache (8bit + scale). Same value as gguf_probe
Q8_FACTOR = 0.53
#: Overhead calibrated from one measurement (GiB), rounded to the safe side
DEFAULT_OVERHEAD_GIB = 1.0
#: --ctx-size is rounded down to this unit
CTX_STEP = 4096
#: The "round" ctx values chosen when `--ctx` is omitted: the largest that fits
#: the budget. Never recommend a ragged value that fills the budget
#: (e.g. 69,632):
#:   · it leaves no slack for the overhead estimate error (one calibration point)
#:   · it gains only 6% more context over 65,536
STANDARD_CTX = (4096, 8192, 16384, 32768, 49152, 65536, 98304,
                131072, 196608, 262144, 393216, 524288, 1048576)
#: Warn when the remainder falls below this (GiB)
MIN_HEADROOM_GIB = 0.5
#: Upper bound for max_tokens. A thinking model needs no more (operating value
#: from measurements)
MAX_TOKENS_CAP = 49152

THINKING_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
}
NON_THINKING_SAMPLING = {
    "temperature": 0.7, "top_p": 0.80, "top_k": 20,
}


def file_gib(rec: dict) -> float:
    return rec["size_gb"] * 1e9 / GIB


def kv_gib(rec: dict, ctx: int, kv_mode: str) -> float:
    per_tok = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
    if not per_tok:
        return 0.0
    factor = Q8_FACTOR if kv_mode == "q8_0" else 1.0
    return per_tok * ctx * factor / GIB


def max_ctx(rec: dict, vram_gib: float, kv_mode: str, overhead: float) -> int:
    """Largest ctx that fits the VRAM budget. Never above the native ctx.

    Rounded down to CTX_STEP.
    """
    per_tok = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
    if not per_tok:
        return 0
    factor = Q8_FACTOR if kv_mode == "q8_0" else 1.0
    budget = (vram_gib - overhead - file_gib(rec)) * GIB
    if budget <= 0:
        return 0
    n = int(budget / (per_tok * factor))
    n -= n % CTX_STEP
    native = rec.get("context_length") or n
    return max(0, min(n, native))


def recommended_ctx(rec: dict, vram_gib: float, kv_mode: str, overhead: float) -> int:
    """The ctx to recommend when `--ctx` is omitted.

    Using the "largest value that fits the budget" from ``max_ctx`` directly
    gives ragged values like 69,632, which leave only 0.02 GiB of headroom.
    There is only one overhead calibration point, so startup fails on that
    error. **Round to a value with a round number to keep some slack.**
    """
    cap = max_ctx(rec, vram_gib, kv_mode, overhead)
    if not cap:
        return 0
    fits = [c for c in STANDARD_CTX if c <= cap]
    return fits[-1] if fits else cap


def headroom_gib(rec: dict, ctx: int, kv_mode: str, vram_gib: float,
                 overhead: float) -> float:
    """How much VRAM is left with this setting (GiB). Negative means it fails."""
    return vram_gib - (file_gib(rec) + kv_gib(rec, ctx, kv_mode) + overhead)


def model_path_of(rec: dict, override: str | None) -> str:
    """The path written after -m in the launch command.

    When gguf_probe recorded an absolute path, use it. Old JSON has no such
    key, so only in that case fall back to a placeholder.
    """
    if override:
        return override
    return rec.get("path") or f"/path/to/{rec['file']}"


def max_tokens_for(ctx: int) -> tuple[int, str]:
    """Return ``max_tokens`` and **the reason for that value**.

    Take the smaller of the two constraints:

      * 3/4 of ctx … the rest goes to the prompt. With
        max_tokens >= n_ctx the effective limit becomes n_ctx − prompt length,
        which makes max_tokens ineffective
      * MAX_TOKENS_CAP … the measured value above which a thinking model needs
        nothing more

    **State which constraint applied.** Printing 49,152 for ctx 131,072 while
    writing "3/4 of ctx" would be a lie (3/4 is 98,304).
    """
    three_quarters = (ctx * 3 // 4) // 1024 * 1024
    if three_quarters <= MAX_TOKENS_CAP:
        return three_quarters, f"3/4 of ctx {ctx:,}; the rest is for the prompt"
    return MAX_TOKENS_CAP, (f"Capped at the {MAX_TOKENS_CAP:,} limit "
                            f"(3/4 of ctx would be {three_quarters:,})")


def short(rec: dict) -> str:
    return rec["file"].replace(".gguf", "")


def cmd_table(recs, vram, overhead, ctx_req):
    rows = []
    for r in sorted(recs, key=lambda d: d["size_gb"]):
        f16 = max_ctx(r, vram, "f16", overhead)
        q8 = max_ctx(r, vram, "q8_0", overhead)
        fit = ""
        if ctx_req:
            need_f16 = file_gib(r) + kv_gib(r, ctx_req, "f16") + overhead
            need_q8 = file_gib(r) + kv_gib(r, ctx_req, "q8_0") + overhead
            if need_f16 <= vram:
                fit = f"OK with f16 ({need_f16:.1f})"
            elif need_q8 <= vram:
                fit = f"OK with q8_0 ({need_q8:.1f})"
            else:
                fit = f"Does not fit ({need_q8:.1f})"
        rows.append((short(r), file_gib(r), f16, q8, fit))

    w = max(len(x[0]) for x in rows) + 2
    head = f"{'quantization':<{w}}{'file':>9}{'max ctx (f16)':>14}{'max ctx (q8_0)':>15}"
    if ctx_req:
        head += f"   fits ctx {ctx_req:,}?"
    print(head)
    print("-" * len(head))
    for name, fg, f16, q8, fit in rows:
        f16s = f"{f16:,}" if f16 else "no fit"
        q8s = f"{q8:,}" if q8 else "no fit"
        line = f"{name:<{w}}{fg:>8.2f}G{f16s:>14}{q8s:>15}"
        if ctx_req:
            line += f"   {fit}"
        print(line)


def emit_config(rec: dict, ctx: int, kv_mode: str, vram: float, overhead: float,
                model_path: str, port: int, device: str) -> str:
    think = rec.get("chat_template_has_think")
    samp = THINKING_SAMPLING if think else NON_THINKING_SAMPLING
    mtp = bool(rec.get("mtp_tensor_count"))
    used = file_gib(rec) + kv_gib(rec, ctx, kv_mode) + overhead
    max_tokens, mt_reason = max_tokens_for(ctx)

    L = []
    L.append(f"# ===== {short(rec)} / ctx {ctx:,} / KV {kv_mode} =====")
    L.append(f"# Estimate: model {file_gib(rec):.2f} + KV {kv_gib(rec, ctx, kv_mode):.2f} "
             f"+ overhead {overhead:.2f} = {used:.2f} GiB / budget {vram:.1f} GiB")
    hr = vram - used
    if hr < 0:
        L.append(f"# ❌ Over budget by {-hr:.2f} GiB. It will not start")
    elif hr < MIN_HEADROOM_GIB:
        L.append(f"# ⚠️ Only {hr:.2f} GiB headroom left. The overhead calibration has "
                 f"one point only, so if real usage exceeds this, startup fails")
        # When q8_0 is already in use, do not tell the user to "use q8_0"
        if kv_mode == "f16":
            L.append("#    Use q8_0 KV (-ctk q8_0 -ctv q8_0), or drop ctx one step")
        else:
            L.append("#    Drop ctx one step, or pick one quantization level lighter")
    else:
        L.append(f"# Headroom {hr:.2f} GiB")
    L.append(f"# native ctx = {rec.get('context_length', 0):,}"
             + ("  / no rope scaling" if not rec.get("rope_scaling_type") else
                f"  / rope scaling: {rec.get('rope_scaling_type')}"))
    if rec.get("is_hybrid_attention"):
        L.append(f"# Hybrid attention: only {len(rec['kv_layers'])}/{rec['kv_cache']['total_layers']} "
                 f"layers keep KV = {rec['kv_cache']['bytes_per_token_f16'] / 1024:.0f} KB/token")
    L.append("")
    L.append("# --- llama-server ---")
    # Notes go **outside the command**. Writing # in the middle of a continued
    # line (\) comments out everything to the end of the line, which takes the
    # \ with it and breaks the command.
    if kv_mode == "q8_0":
        L.append("# Quantized -ctk/-ctv requires -fa on (included below)")
    if mtp:
        L.append(f"# Found {rec['mtp_tensor_count']} MTP tensors → adding --spec-type draft-mtp")
    else:
        L.append("# No MTP tensors → not adding --spec-type draft-mtp")
    args = [
        f"llama-server -m {model_path}",
        f"--port {port} --device {device}",
        "-ngl 99 -fa on",
        f"--ctx-size {ctx} --parallel 1",
        "--batch-size 2048 --ubatch-size 512",
    ]
    if kv_mode == "q8_0":
        args.append("-ctk q8_0 -ctv q8_0")
    if mtp:
        args.append("--spec-type draft-mtp")
    L.append(" \\\n  ".join(args))
    L.append("")
    L.append("# --- config.yaml (under models:) ---")
    L.append(f"  {short(rec).lower().replace('.', '-')}:")
    L.append("    type: openai")
    L.append(f'    base_url: "http://localhost:{port}/v1"')
    L.append('    model: "auto"')
    L.append('    api_key: "sk-local"')
    for k, v in samp.items():
        L.append(f"    {k}: {v}")
    L.append(f"    max_tokens: {max_tokens}   # {mt_reason}")
    L.append("    seed: 42          # write only when runs: 1 (runs>1 is random every run)")
    tag = "thinking" if think else "non-thinking"
    L.append(f"    # sampling defaults based on the chat_template check ({tag})")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="derive llama-server / config.yaml settings from gguf_probe JSON")
    ap.add_argument("json_path", help="JSON written with gguf_probe.py --json --out")
    ap.add_argument("--vram", type=float, required=True,
                    help="usable VRAM (GiB). For a 24GB card, 24")
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD_GIB,
                    help=f"expected compute buffers etc. (GiB, default {DEFAULT_OVERHEAD_GIB})")
    ap.add_argument("--ctx", type=int, default=None,
                    help="ctx you want to use. When set, judges whether each quantization fits")
    ap.add_argument("--pick", default=None,
                    help="part of a file name. When set, prints the launch command and config")
    ap.add_argument("--kv", choices=["f16", "q8_0", "auto"], default="auto",
                    help="KV cache type (default auto = q8_0 when f16 does not fit)")
    ap.add_argument("--model-path", default=None, help="path to write in the launch command")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--device", default="CUDA0")
    args = ap.parse_args()

    recs = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    if isinstance(recs, dict):
        recs = [recs]
    lm = [r for r in recs if r.get("is_language_model") and r.get("kv_cache")]
    if not lm:
        sys.exit("No language model record found (pass the output of gguf_probe --json)")

    if not args.pick:
        print(f"VRAM budget {args.vram:.1f} GiB / overhead {args.overhead:.2f} GiB\n")
        cmd_table(lm, args.vram, args.overhead, args.ctx)
        print("\n※ All numbers are GiB. They differ from GGUF's size_gb (bytes ÷ 10^9)")
        print("※ --pick <part of a name> prints the launch command and config.yaml")
        return 0

    hits = [r for r in lm if args.pick.lower() in r["file"].lower()]
    if not hits:
        sys.exit(f"No file matches --pick {args.pick!r}: "
                 + ", ".join(short(r) for r in lm))
    if len(hits) > 1:
        sys.exit(f"--pick {args.pick!r} matches several files: "
                 + ", ".join(short(r) for r in hits))
    rec = hits[0]

    kv_mode = args.kv
    ctx = args.ctx
    if kv_mode == "auto":
        kv_mode = "f16"
        if ctx and file_gib(rec) + kv_gib(rec, ctx, "f16") + args.overhead > args.vram:
            kv_mode = "q8_0"
            print(f"# ※ ctx {ctx:,} does not fit in f16, so the KV cache was set to q8_0",
                  file=sys.stderr)
    if ctx is None:
        cap = max_ctx(rec, args.vram, kv_mode, args.overhead)
        ctx = recommended_ctx(rec, args.vram, kv_mode, args.overhead)
        if not ctx:
            sys.exit(f"{short(rec)} does not fit in {args.vram} GiB of VRAM "
                     f"(the model alone is {file_gib(rec):.2f} GiB)")
        note = f"# ※ No --ctx given, so using {ctx:,}"
        if cap > ctx:
            note += (f" (the budget allows up to {cap:,}, but a ragged value leaves "
                     f"no slack for the overhead error, so it is rounded to a round number)")
        print(note, file=sys.stderr)

    hr = headroom_gib(rec, ctx, kv_mode, args.vram, args.overhead)
    if hr < 0:
        print(f"# ⚠️ The estimate exceeds the {args.vram:.1f} GiB budget by {-hr:.2f} GiB",
              file=sys.stderr)

    mp = model_path_of(rec, args.model_path)
    if not rec.get("path") and not args.model_path:
        print("# ※ This gguf.json holds no absolute path, so -m is a placeholder. "
              "Re-run gguf_probe, or pass --model-path", file=sys.stderr)
    print(emit_config(rec, ctx, kv_mode, args.vram, args.overhead,
                      mp, args.port, args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())

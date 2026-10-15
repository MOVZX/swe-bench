#!/usr/bin/env python3
"""Load safetensors directly with transformers and check for indent collapse.

A path that does not touch GGUF at all. It is comparable with the llama.cpp side
(tools/probe_indent_tokens.py) using the same verdict and the same view
(whitespace token breakdown).

    python3 tools/probe_indent_transformers.py -n 3 --show 14
    python3 tools/probe_indent_transformers.py --model Jackrong/Qwopus3.8-27B-Flash
    python3 tools/probe_indent_transformers.py --self-test   # internal test, no model needed

Isolation:
  * it uses per-depth indent tokens such as 262 '   ' / 285 / 309 (4 spaces)
      → the GGUF conversion is the culprit (H3c). The weights are fine
  * it collapses to id 220 ' ' (1 space)
      → the weights are the culprit (H3a)

Memory: loading 27B in bf16 needs about 56GB. If the RAM is too small it is
**offloaded to disk**, and every token re-reads the SSD, so it never finishes in
a practical time (when you see "offloaded to the cpu and disk", give up and
switch to 4bit).

    python3 tools/probe_indent_transformers.py --load-in-4bit -n 1 --max-new-tokens 256

4bit (bitsandbytes) is still valid for the isolation. The goal is **not** to
avoid quantization but to **bypass llama.cpp's GGUF conversion**, and bnb
quantizes safetensors on the fly through a separate code path, so it never goes
through GGUF conversion. That the quantization bit count is not the cause was
confirmed when the GGUF Q4 and Q5 files matched byte for byte (docs 4.8).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_indent_tokens import (  # noqa: E402
    DEFAULT_PROMPT,
    indent_stats,
    self_test,
    verdict_for,
)


def run(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: transformers/torch is not installed: {e}", file=sys.stderr)
        return 2

    quant = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            print(f"ERROR: BitsAndBytesConfig is unavailable: {e}", file=sys.stderr)
            return 2
        quant = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print("  ※ loaded quantized with bitsandbytes. It does not go through GGUF"
              " conversion, so the H3a/H3c isolation is valid (docs 4.8)")

    label = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else args.dtype
    print(f"  loading {args.model} ({label}, device_map={args.device_map}) ...")
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if quant is not None:
        kwargs["quantization_config"] = quant
    else:
        kwargs["dtype"] = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    tally: dict[str, int] = {}
    ws_tally: dict[tuple[int, str], int] = {}

    for i in range(1, args.n + 1):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temp > 0,
                temperature=args.temp if args.temp > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new_ids = out[0][prompt_len:].tolist()
        content = tok.decode(new_ids, skip_special_tokens=True)

        for t in new_ids:
            pc = tok.decode([t], skip_special_tokens=False)
            if pc and not pc.strip():
                ws_tally[(t, pc)] = ws_tally.get((t, pc), 0) + 1

        v = verdict_for(content)
        tally[v] = tally.get(v, 0) + 1
        n1, n4 = indent_stats(content)
        print(f"  #{i:<3} {v:<10} [1sp={n1} 4sp={n4}] tokens={len(new_ids)}")
        if v != "ok" and args.show:
            print("    ---- content ----")
            for ln in content.splitlines()[: args.show]:
                print(f"    |{ln}")

    total = sum(tally.values())
    print("\n  ---- " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    collapsed = tally.get("COLLAPSED", 0) + tally.get("MIXED", 0)
    print(f"  collapse rate: {collapsed}/{total}")
    if ws_tally:
        print("  whitespace-only token breakdown (id, string, occurrences):")
        for (t, pc), c in sorted(ws_tally.items(), key=lambda kv: -kv[1]):
            print(f"    {t:>7}  {pc!r:<14} x{c}")
    if collapsed:
        print("  → It collapses even without GGUF. **The weights are the cause (H3a)**."
              " Give up on this model for code")
    else:
        print("  → Normal without GGUF. **The conversion of the distributed GGUF is the cause"
              " (H3c)**. It is worth reporting upstream")
    return 1 if collapsed else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Jackrong/Qwopus3.8-27B-Flash")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0, help="greedy at 0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="load with bitsandbytes 4bit (avoids disk offload)")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="load with bitsandbytes 8bit")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--show", type=int, default=0, metavar="N")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

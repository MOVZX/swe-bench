#!/usr/bin/env python3
"""Read-only GGUF inspection: real quantization, KV-cache size, MTP tensors.

    python gguf_probe.py /path/to/model.gguf [...]     # human-readable summary
    python gguf_probe.py --json --out gguf.json /path/to/*.gguf

If `pip install gguf` is missing, this falls back to a parser based on the GGUF
spec (it reads only the metadata and the tensor info table; tensor data is
never touched).

--- KV cache ---

Assuming `block_count * (head_count_kv * (head_dim_k + head_dim_v)) * 2 byte`
is 4x too high for hybrid-attention models. Qwen3.8-27B (arch=qwen35) has full
attention in only 16 of 65 layers plus 1 MTP layer; the rest is Gated DeltaNet
(linear attention = no KV cache). Measured: 66 KB/token. The naive assumption
returned 260 KB/token.

This script reproduces the measurement: it counts KV-bearing layers from the
**presence of attn_k / attn_v tensors**, and uses `block_count` only as a
fallback when no such tensor exists.

--- Quantization ---

Do not trust `general.file_type`. Unsloth Dynamic (U-DYQ) declares one type for
the whole file, and standard K-quant (IQ4_XS etc.), which does not vary per
layer, is indistinguishable that way. Judge from the tensor distribution:

  * `role_by_quant` … which types appear, and how often, per role
    (ffn_down / attn_v / ...)
  * a type mixed inside one role means the file varies quantization per layer
    (= Dynamic)
  * if `token_embd` / `output` use a heavier type than the dominant one,
    report that share
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_T_UINT32, GGUF_T_INT32, GGUF_T_UINT64, GGUF_T_INT64 = 0, 1, 2, 3
GGUF_T_FLOAT32, GGUF_T_BOOL, GGUF_T_STRING, GGUF_T_ARRAY = 4, 5, 6, 7
GGUF_T_UINT8, GGUF_T_UINT16, GGUF_T_INT16, GGUF_T_FLOAT16 = 8, 9, 10, 11
GGUF_T_FLOAT64, GGUF_T_INT8, GGUF_T_BYTES = 12, 13, 14

SCALAR_FMT = {
    GGUF_T_UINT32: "<I", GGUF_T_INT32: "<i", GGUF_T_UINT64: "<Q",
    GGUF_T_INT64: "<q", GGUF_T_FLOAT32: "<f", GGUF_T_FLOAT64: "<d",
    GGUF_T_UINT8: "<B", GGUF_T_UINT16: "<H", GGUF_T_INT16: "<h",
    GGUF_T_FLOAT16: "<e", GGUF_T_INT8: "<b", GGUF_T_BYTES: "<B",
}
# ggml tensor type ids -> names (as they appear in quantization reports)
GGUF_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "BF16", 3: "Q8_0", 4: "Q4_0", 5: "Q4_1", 6: "Q4_2",
    7: "Q4_3", 8: "Q4_K", 9: "Q5_0", 10: "Q5_1", 11: "Q5_K", 12: "Q6_K",
    13: "Q2_K", 14: "Q3_K", 15: "IQ2_XXS", 16: "IQ2_XS", 17: "Q2_K_S",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "Q4_K_S", 21: "IQ4_NL", 22: "IQ3_S",
    23: "IQ2_S", 24: "IQ4_XS", 25: "IQ1_M",
}
MTP_RE = re.compile(r"^blk\.\d+\.nextn\.|mtp\.layers\.\d+\.")
NORM_RE = re.compile(r"(input_layernorm|ffn_norm|attn_norm|post_attention_norm|"
                     r"attention_norm|ffn_gate_norm|ffn_up_norm|ffn_down_norm|"
                     r"\.norm\.|_norm\.weight$|\.bias$)")
# Drop F32/F16 norm and bias tensors from the quantization tally (many of them,
# negligible weight). A plain "norm" filter also catches the Qwen3-Next GDN
# tensors (blk.N.attn_norm.weight, blk.N.linear_attn.A_log) and skews the
# dominant type.
NORM_NAME_RE = re.compile(r"norm|\.bias$|\.A_log$|\.dt_bias$")
LAYER_RE = re.compile(r"^blk\.(\d+)\.")
# Lower bound so that 'F32' is not reported as dominant for unquantized files
MIN_WEIGHT_SHARE = 1e-6
#: q8_0 KV cache is 8.5 bit per weight instead of 16 bit
Q8_FACTOR = 0.5625


class Reader:
    def __init__(self, buf: bytes):
        self.b, self.i = buf, 0

    def raw(self, n: int) -> bytes:
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def scalar(self, fmt: str):
        n = struct.calcsize(fmt)
        v, = struct.unpack(fmt, self.raw(n))
        return v

    def string(self) -> str:
        return self.raw(self.scalar("<Q")).decode("utf-8", "replace")

    def value(self, t: int, depth: int = 0):
        if t in SCALAR_FMT:
            return self.scalar(SCALAR_FMT[t])
        if t == GGUF_T_BOOL:
            return self.scalar("<B") != 0
        if t == GGUF_T_STRING:
            return self.string()
        if t == GGUF_T_ARRAY and depth < 2:
            et, n = self.scalar("<I"), self.scalar("<Q")
            # A vocabulary of hundreds of thousands of entries also recurses
            # here. One Python call per element is too slow, so string arrays
            # are decoded in bulk. Nested arrays are cut off at depth 2 (GGUF
            # metadata has no deep nesting).
            if et == GGUF_T_STRING:
                out = []
                for _ in range(n):
                    out.append(self.string())
                return out
            return [self.value(et, depth + 1) for _ in range(n)]
        raise ValueError(f"unknown GGUF type {t}")


def parse_gguf(buf: bytes):
    r = Reader(buf)
    if r.raw(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = r.scalar("<I")
    n_tensors = r.scalar("<Q")
    n_kv = r.scalar("<Q")
    md: dict = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.scalar("<I")
        md[k] = r.value(t)
    return version, n_tensors, n_kv, md


def tensor_offset(buf: bytes, n_kv: int) -> int:
    """Byte offset of the tensor info table (right after the KV section)."""
    r = Reader(buf)
    r.raw(4)
    r.scalar("<I")
    r.scalar("<Q")
    r.scalar("<Q")
    for _ in range(n_kv):
        r.string()
        r.value(r.scalar("<I"))
    return r.i


def tensor_names(buf: bytes, offset: int, n: int):
    """Parse only the tensor info table. Heavy tensor data is not touched."""
    r = Reader(buf[offset:])
    out = []
    for _ in range(n):
        name = r.string()
        t = r.scalar("<I")
        nd = r.scalar("<I")
        dims = [r.scalar("<Q") for _ in range(nd)]
        out.append((name, t, dims))
    return out


def summarize_tensors(tensors, md: dict) -> dict:
    """Summarize quantization and KV cache from a tensor list (name, GGUF type).

    Pure function: it depends on neither the GGUF parser nor a file, so it can
    be unit-tested with a synthetic tensor list.
    """
    # Keep the distinction between attn_v / attn_k / attn_qkv intact: use the
    # plain tensor name with only the layer prefix removed as the role.
    # The old role normalization (layer_re.sub) collapsed every blk.N.* to "*",
    # which conflated attn_output.weight with output.weight.
    # Substring matching is also forbidden: 'output.weight' in
    # 'blk.N.attn_output.weight' is True, and 65 tensors were collected as
    # "output.weight".
    by_role: dict[str, Counter] = defaultdict(Counter)
    types_by_role_layer: dict[str, dict[int, set]] = defaultdict(
        lambda: defaultdict(set))
    mtp = []
    mtp_layers: set[int] = set()
    for name, t in tensors:
        base = LAYER_RE.sub("", name)
        ty = GGUF_TYPE_NAMES.get(t, str(t))
        by_role[base][ty] += 1
        lm = LAYER_RE.match(name)
        if lm:
            types_by_role_layer[base].setdefault(int(lm.group(1)), set()).add(ty)
        if MTP_RE.search(name):
            mtp.append(name)
            if lm:
                mtp_layers.add(int(lm.group(1)))
    # Exclude norm/bias from the weight tally (F32 norms are numerous but
    # their weight is negligible; mixing them skews the dominant type).
    mix = {role: dict(c) for role, c in by_role.items()
           if not NORM_NAME_RE.search(role)}
    # quant_mix keeps every role (norms included) and is keyed by type:
    # type -> {count, share, roles}. weight_mix drops the norm roles so the
    # dominant type is not skewed by them.
    all_types: Counter = Counter()
    roles_by_type: dict[str, set] = defaultdict(set)
    for role, c in by_role.items():
        for q, n in c.items():
            all_types[q] += n
            roles_by_type[q].add(role)
    total_all = sum(all_types.values())
    quant_mix = {q: {"count": n,
                     "share": round(100 * n / total_all, 1) if total_all else 0.0,
                     "roles": sorted(roles_by_type[q])}
                 for q, n in all_types.items()}
    el = Counter()
    for role, c in mix.items():
        for q, n in c.items():
            el[q] += n
    total_el = sum(el.values())
    weight_mix = {q: {"count": n,
                      "share": round(100 * n / total_el, 1) if total_el else 0.0}
                  for q, n in el.items()}
    # Does any role vary its type per layer? Standard K-quant (IQ4_XS etc.)
    # raises the type of some ffn_down / attn_v tensors through llama.cpp's
    # use_more_bits. Unsloth Dynamic (U-DYQ) is uniform per layer, so this
    # stays 0.
    #
    # The MTP / NextN draft block (blk.64 for Qwen3-Next, arch qwen35) is not
    # part of the main quantization budget; detect it by tensor name (nextn.)
    # and layer number (>= block_count) and do not count it. The old condition
    # ("more than one layer signature = Dynamic") judged every standard K-quant
    # as Dynamic (confirmed by measurement on 2026-10-15).
    n_blocks = md.get("block_count")
    mixed, mtp_only = {}, {}

    def is_mtp_layer(layer: int) -> bool:
        return layer in mtp_layers or (n_blocks is not None and layer >= n_blocks)

    for role, c in mix.items():
        if len(c) < 2:
            continue
        # F32 norms mixed in (attn_norm etc.) is not "varied per layer"
        nonf32 = {q: n for q, n in c.items() if q not in ("F32", "F16", "BF16")}
        if len(nonf32) < 2:
            continue
        by_layer = types_by_role_layer.get(role, {})
        if not by_layer:
            continue
        main_types = {q for l, qs in by_layer.items() if not is_mtp_layer(l) for q in qs}
        mtp_types = {q for l, qs in by_layer.items() if is_mtp_layer(l) for q in qs}
        if len(main_types) >= 2:
            mixed[role] = dict(c)
        elif mtp_types - main_types:
            mtp_only[role] = dict(c)
    # Exclude F32 from the dominant type when quantized tensors exist. When
    # nothing is quantized, F32 may be the dominant type (except for empty
    # input, where n_roles is 0).
    _quant_el = {q: n for q, n in el.items() if q not in ("F32", "F16", "BF16")}
    dom = max((_quant_el or el).items(), key=lambda kv: kv[1]) if (el and total_el) else None

    s = {
        "n_roles": len(by_role),
        "quant_mix": quant_mix,
        "weight_mix": weight_mix,
        "dominant_weight_type": dom[0] if dom else None,
        "per_layer_varying": bool(mixed),
        "mixed_roles": mixed,
        "n_mixed_roles": len(mixed),
        "mtp_only_roles": mtp_only,
        "n_mtp_only_roles": len(mtp_only),
        "mtp_tensors": mtp[:12],
        "mtp_tensor_count": len(mtp),
        "mtp_block_layers": sorted(mtp_layers),
        "notable_tensors": [],
        "kv_layers": None,
        "linear_attn_layers": None,
        "is_hybrid_attention": None,
    }
    # Exact match only. Substring matching picks up 'blk.N.attn_output.weight'
    # for 'output.weight' and prints 65 entries as "notable".
    notable_keys = {"token_embd.weight", "output.weight", "output_norm.weight",
                    "score.weight", "tok_embeddings.weight",
                    "blke.tmp_unfinalized"}
    notable = []
    for name, t in tensors:
        base = LAYER_RE.sub("", name)
        if base in notable_keys:
            notable.append((name, GGUF_TYPE_NAMES.get(t, str(t))))
    s["notable_tensors"] = [{"name": n, "type": ty,
                             "count": sum(1 for x, _ in notable if x == n)}
                            for n, ty in sorted(set(notable))]

    # Count KV layers by "does this layer have attn_k / attn_v".
    # Layers with attn_qkv (Gated DeltaNet linear attention) have no KV cache.
    kv_layers: set[int] = set()
    lin_layers: set[int] = set()
    for name, _t in tensors:
        lm = LAYER_RE.match(name)
        if not lm:
            continue
        n = int(lm.group(1))
        if "attn_k." in name or "attn_v." in name:
            kv_layers.add(n)
            lin_layers.discard(n)
        elif "attn_qkv." in name and n not in kv_layers:
            lin_layers.add(n)
    s["kv_layers"] = sorted(kv_layers)
    s["linear_attn_layers"] = sorted(lin_layers)
    s["is_hybrid_attention"] = (bool(kv_layers) and bool(lin_layers))

    # KV cache: reproduce the measured 66 KB/token
    # (4 KV heads x (256+256) x 2B x 17 layers). The block_count assumption
    # returned 260 KB/token, 4x too high.
    hkv = md.get("head_count_kv")
    dk = md.get("key_length")
    dv = md.get("value_length")
    n_blocks = md.get("block_count")
    kv_layers_count = len(kv_layers) if kv_layers else n_blocks
    if hkv and dk and dv and kv_layers_count:
        per_tok = hkv * (dk + dv) * 2 * kv_layers_count
        s["kv_cache"] = {
            "bytes_per_token_f16": per_tok,
            "kv_bearing_layers": kv_layers_count,
            "total_layers": n_blocks,
            "linear_attn_layers": len(lin_layers),
            "counted_from": "attn_k/attn_v tensors" if kv_layers
                            else "block_count (fallback)",
            **{f"gb_{n // 1024}k_{'f16' if f == 1.0 else 'q8_0'}":
               round(per_tok * n * f / 1e9, 2)
               for n in (32768, 65536, 131072, 262144) for f in (1.0, Q8_FACTOR)}
        }
    missing = [k for k in ("head_count_kv", "key_length", "value_length", "block_count")
               if not md.get(k)]
    s["kv_cache_missing"] = missing
    return s


def human(rec: dict, path: Path) -> None:
    """Print the human-readable summary of one probe() record."""
    tensors = rec["tensors"]
    ctx = rec["context_length"]
    kind = ("language model" if rec["is_language_model"]
            else "embedding / vision / adapter")
    total_el = tensors["n_elements_total"]
    print(f"--- {path.name} ({rec['size_gb']:.2f} GB) ---")
    print(f"  {'file kind':<14} {kind}")
    print(f"  {'architecture':<14} {rec['architecture']}"
          f"  ({rec['n_tensors']} tensors)")
    print(f"  {'context':<14} {ctx:,}"
          + (f"   (rope scaling: {rec['rope_scaling_type']}, "
             f"factor {rec['rope_scaling_factor']})" if rec["rope_scaling_type"]
             else "   (no rope scaling)"))
    if rec["is_hybrid_attention"]:
        print(f"  {'attention':<14} hybrid: {len(rec['kv_layers'])} full-attn layers "
              f"(KV cache) / {len(rec['linear_attn_layers'])} linear-attn layers "
              f"(no KV)")
    if rec["mtp_tensor_count"]:
        print(f"  {'speculative':<14} {rec['mtp_tensor_count']} MTP/NextN tensors "
              f"→ usable with --spec-type draft-mtp")
    if rec.get("per_layer_varying"):
        print(f"  {'quantization':<14} quant varies per layer: "
              f"{rec['n_mixed_roles']} roles mixed "
              f"(Dynamic U-DYQ / per-layer allocation)")
    if rec.get("n_mtp_only_roles"):
        print(f"  {'quantization':<14} quant differs only in the MTP/NextN block "
              f"(layers {rec['mtp_block_layers']}) — not per-layer allocation")
    print(f"  {'sampling':<14} "
          f"{'thinking' if rec['chat_template_has_think'] else 'non-thinking'}"
          f" (from chat_template)")
    if rec.get("kv_cache"):
        kv = rec["kv_cache"]
        print(f"  {'KV cache':<14} {kv['bytes_per_token_f16']/1024:.0f} KB/token "
              f"× {kv['kv_bearing_layers']}/{kv['total_layers']} layers "
              f"({kv['counted_from']})")
        print(f"     ctx 32k: f16 {kv['gb_32k_f16']:.2f} GB"
              f" / q8_0 {kv['gb_32k_q8_0']:.2f} GB"
              f"   |   ctx 64k: f16 {kv['gb_64k_f16']:.2f} GB"
              f" / q8_0 {kv['gb_64k_q8_0']:.2f} GB")
        print(f"     ctx 128k: f16 {kv['gb_128k_f16']:.2f} GB"
              f" / q8_0 {kv['gb_128k_q8_0']:.2f} GB"
              f"   |   ctx 256k: f16 {kv['gb_256k_f16']:.2f} GB"
              f" / q8_0 {kv['gb_256k_q8_0']:.2f} GB")
    elif rec["is_language_model"]:
        print(f"  {'KV cache':<14} (no layer metadata: "
              f"{', '.join(rec['kv_cache_missing'])})")
    if tensors["n"]:
        dom = tensors["dominant"]
        mix = " / ".join(f"{q} {n/total_el:.1%}"
                         for q, n in tensors["weight_mix"].most_common(4))
        print(f"  {'weights':<14} {mix}"
              f"   |   dominant: {dom[0]} ({dom[1]/total_el:.0%})")
        if rec["mixed_roles"]:
            print("  quant varies per layer:")
            for role, c in rec["mixed_roles"].items():
                print(f"    {role:<28} "
                      f"{', '.join(f'{q} x{n}' for q, n in c.items())}")
        if tensors["mtp"]:
            print(f"  MTP/NextN tensors: {rec['mtp_tensor_count']}")
            for n in tensors["mtp"]:
                print(f"    {n}")
        if not tensors["quant"]:
            print("  no quantized tensors (full-precision F32/F16 model)")
        if rec["notable_tensors"]:
            print("  notable (tensor names):")
            for t in rec["notable_tensors"]:
                print(f"    {t['name']:<34} {t['type']:<8} x{t['count']}")
    print()


def probe(p: Path) -> dict:
    """Inspect one file and return the summary record."""
    buf = p.read_bytes()
    version, n_tensors, n_kv, md = parse_gguf(buf)
    gen = {k[len("general."):]: v for k, v in md.items() if k.startswith("general.")}
    arch_name = gen.get("architecture")
    # Arch fields live under "<arch>.", so a file without general.architecture
    # shows architecture None (measured on an f16 model)
    arch = {k[len(arch_name) + 1:]: v for k, v in md.items()
            if arch_name and k.startswith(arch_name + ".")}
    meta = {**arch, **gen}
    files = gen.get("files", [])
    chat_tpl = md.get("tokenizer.chat_template", "") or ""
    # Parsing the tensor info table by size would need the whole file read.
    # Here we only use (name, type) from the table, so the quantization picture
    # is available from the type alone (element counts are used only for the
    # weight share shown to humans).
    info = tensor_names(buf, tensor_offset(buf, n_kv), n_tensors)
    s = summarize_tensors([(n, t) for n, t, _ in info], meta)
    el = Counter()
    for name, t, dims in info:
        if NORM_NAME_RE.search(name):
            continue
        el[GGUF_TYPE_NAMES.get(t, str(t))] += math.prod(dims)
    n_el = sum(el.values())
    quant = sorted(q for q in s["quant_mix"] if q not in ("F32", "F16", "BF16"))
    ft = gen.get("file_type")
    ft_name = GGUF_TYPE_NAMES.get(ft, str(ft))
    # file_type is a header declaration and can disagree with the tensor
    # distribution, so trust it only when quantized tensors confirm it
    roles = {r for v in s["quant_mix"].values() for r in v["roles"]}
    is_lang = bool(quant) and bool(roles & {"token_embd.weight",
                                            "output.weight",
                                            "ffn_down.weight"})
    return {
        "file": p.name,
        "path": str(p.resolve()),
        "size_gb": round(p.stat().st_size / 1e9, 2),
        "gguf_version": version,
        "n_tensors": n_tensors,
        "file_type": ft_name,
        "file_type_trusted": bool(quant) and ft_name in quant,
        "quant_types": quant,
        "architecture": arch_name or gen.get("name"),
        "is_language_model": is_lang,
        "context_length": meta.get("context_length") or 0,
        "rope_scaling_type": gen.get("rope_scaling_type") or arch.get("rope_scaling_type"),
        "rope_scaling_factor": gen.get("rope_scaling_factor") or arch.get("rope_scaling_factor"),
        "block_count": meta.get("block_count"),
        "head_count_kv": meta.get("head_count_kv"),
        "key_length": meta.get("key_length"),
        "value_length": meta.get("value_length"),
        "vocab_size": meta.get("vocab_size"),
        "chat_template_len": len(chat_tpl),
        "chat_template_has_think": bool(re.search(r"thinking|\bthink\b", chat_tpl)),
        "chat_template_japanese": bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]",
                                                 chat_tpl)),
        "tokenizer_pre": md.get("tokenizer.ggml.pre"),
        "n_bos": len(md.get("tokenizer.ggml.bos_token_ids", [])),
        "tokenizer_first": (md.get("tokenizer.ggml.tokens") or [""])[0],
        "utf8_flush_between_chunks":
            md.get("tokenizer.ggml.utf8_flush_between_chunks"),
        "n_files": len(files) if isinstance(files, list) else 0,
        "split": gen.get("split"),
        "files": [f.get("filename") for f in files if isinstance(f, dict)][:10],
        "tensors": {
            "n": len(info),
            "n_elements_total": n_el,
            "weight_mix": el,
            "dominant": (s["dominant_weight_type"],
                         el.get(s["dominant_weight_type"], 0)),
            "quant": quant,
            "mtp": s["mtp_tensors"],
        },
        **{k: v for k, v in s.items()
           if k in ("kv_layers", "linear_attn_layers", "is_hybrid_attention",
                    "kv_cache", "kv_cache_missing", "mtp_tensor_count",
                    "mtp_tensors", "mtp_block_layers", "n_mixed_roles",
                    "mtp_only_roles", "n_mtp_only_roles", "per_layer_varying",
                    "mixed_roles", "quant_mix", "weight_mix",
                    "dominant_weight_type", "notable_tensors")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="read-only GGUF inspection: quantization, KV cache, MTP tensors")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--out", type=Path, default=None,
                    help="write JSON to a file (for scripts to consume)")
    args = ap.parse_args()

    recs = []
    for p in args.paths:
        if not p.exists():
            sys.exit(f"{p} does not exist")
        recs.append(probe(p))

    if args.json or args.out:
        payload = json.dumps(recs, ensure_ascii=False, indent=1, default=str)
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
            print(f"wrote {args.out} ({len(recs)} files)")
        else:
            print(payload)
        return 0

    print("=== GGUF probe ===\n")
    for rec, p in zip(recs, args.paths):
        human(rec, p)

    lang = [r for r in recs if r["is_language_model"]]
    if len(lang) < 2:
        return 0
    print("=== Comparison ===")
    w = max(len(r["file"]) for r in lang) + 2
    print(f"{'file':<{w}}{'quantization':<24}{'KV 32k/64k/128k (f16)':<26}"
          f"{'MTP':>5}{'ctx':>9}  sampling")
    for r in lang:
        kv = r.get("kv_cache") or {}
        kvs = (f"{kv['gb_32k_f16']:.2f}/{kv['gb_64k_f16']:.2f}/{kv['gb_128k_f16']:.2f}"
               if kv else "-")
        mix = " / ".join(f"{q} {n/r['tensors']['n_elements_total']:.0%}"
                         for q, n in r["tensors"]["weight_mix"].most_common(2))
        print(f"{r['file']:<{w}}{mix:<24}{kvs:<26}"
              f"{r['mtp_tensor_count']:>5}{r['context_length']:>9,}  "
              f"{'thinking' if r['chat_template_has_think'] else 'non-thinking'}"
              + ("  / hybrid attention" if r["is_hybrid_attention"] else "")
              + (f"  / quant varies per layer ({r['n_mixed_roles']} roles)"
                 if r.get("per_layer_varying") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

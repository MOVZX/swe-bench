"""Tests for gguf_probe tensor classification and KV calculation.

This logic produced bugs three times on 14 real files, so the Qwen3.8-27B
configuration confirmed by measurement (arch=qwen35) is reproduced and pinned.

  * Assuming the KV layer count equals block_count overestimated it 4x
  * The condition "more than one layer signature = Dynamic" was too loose and
    even judged Q8_0 as Dynamic
  * The substring match on ``output.weight`` also caught
    ``attn_output.weight``

No ``gguf`` package needed (summarize_tensors is a pure function).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gguf_probe", Path(__file__).resolve().parent.parent / "gguf_probe.py")
gguf_probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gguf_probe)

summarize_tensors = gguf_probe.summarize_tensors


# Qwen3.8-27B metadata confirmed by measurement
QWEN35_META = {
    "block_count": 65,
    "head_count": 24,
    "head_count_kv": 4,
    "key_length": 256,
    "value_length": 256,
}
#: Full-attention (KV-bearing) layers. blk.3,7,11,...,63 = 16 layers + MTP block blk.64
FULL_ATTN_LAYERS = [i for i in range(65) if i % 4 == 3] + [64]


def qwen35_tensors(quant="Q5_K", *, ffn_down=None, attn_v=None, mtp_type=None):
    """Reproduce the Qwen3.8-27B tensor layout.

    Pass ``{layer: type}`` as ``ffn_down`` / ``attn_v`` to change the type of
    those layers only (to reproduce the use_more_bits of standard K-quant).
    Pass ``mtp_type`` to give the blk.64 weights a different type.
    """
    ffn_down = ffn_down or {}
    attn_v = attn_v or {}
    out = [("token_embd.weight", quant),
           ("output.weight", "Q6_K"),
           ("output_norm.weight", "F32")]
    for i in range(65):
        base = mtp_type if (mtp_type and i == 64) else quant
        if i in FULL_ATTN_LAYERS:
            out += [(f"blk.{i}.attn_q.weight", base),
                    (f"blk.{i}.attn_k.weight", base),
                    (f"blk.{i}.attn_v.weight", attn_v.get(i, base)),
                    (f"blk.{i}.attn_output.weight", base)]
        else:
            out.append((f"blk.{i}.attn_qkv.weight", base))
        out += [(f"blk.{i}.ffn_down.weight", ffn_down.get(i, base)),
                (f"blk.{i}.ffn_gate.weight", base),
                (f"blk.{i}.ffn_up.weight", base),
                (f"blk.{i}.attn_norm.weight", "F32"),
                (f"blk.{i}.ffn_norm.weight", "F32")]
    out += [("blk.64.nextn.eh_proj.weight", base),
            ("blk.64.nextn.enorm.weight", "F32"),
            ("blk.64.nextn.hnorm.weight", "F32"),
            ("blk.64.nextn.shared_head_norm.weight", "F32")]
    return out


# --- Hybrid attention and KV size ------------------------------------------

def test_kv_layers_counted_from_attn_k_v_not_block_count():
    """Only layers with attn_k/attn_v hold KV. Layers with attn_qkv do not count.

    Counting all 65 layers gives 260 KB/token, 4x off the measurement
    (about 66 KB/token).
    """
    s = summarize_tensors(qwen35_tensors(), QWEN35_META)
    assert len(s["kv_layers"]) == 17
    assert len(s["linear_attn_layers"]) == 48
    assert s["kv_layers"] == FULL_ATTN_LAYERS
    assert s["is_hybrid_attention"] is True
    assert s["kv_cache"]["counted_from"] == "attn_k/attn_v tensors"
    assert s["kv_cache"]["total_layers"] == 65


def test_kv_bytes_per_token_matches_measurement():
    """4 heads x (256+256) x 2byte x 17 layers = 69,632 B/token = 68 KB/token."""
    s = summarize_tensors(qwen35_tensors(), QWEN35_META)
    kv = s["kv_cache"]
    assert kv["kv_bearing_layers"] == 17
    assert kv["bytes_per_token_f16"] == 4 * (256 + 256) * 2 * 17 == 69632
    assert kv["gb_32k_f16"] == 2.28
    assert kv["gb_64k_f16"] == 4.56


def test_dense_model_is_not_hybrid():
    """A normal model where every layer has attn_k/attn_v is not judged hybrid."""
    tensors = []
    for i in range(4):
        tensors += [(f"blk.{i}.attn_k.weight", "Q4_K"),
                    (f"blk.{i}.attn_v.weight", "Q4_K"),
                    (f"blk.{i}.ffn_down.weight", "Q4_K")]
    s = summarize_tensors(tensors, {"block_count": 4, "head_count_kv": 8,
                                    "key_length": 128, "value_length": 128})
    assert s["is_hybrid_attention"] is False
    assert s["kv_cache"]["kv_bearing_layers"] == 4
    assert s["linear_attn_layers"] == []


def test_kv_falls_back_to_block_count_when_no_attn_tensors():
    """With no attn_k/attn_v at all, use block_count and say so."""
    s = summarize_tensors([("blk.0.ffn_down.weight", "Q4_K")],
                          {"block_count": 32, "head_count_kv": 8,
                           "key_length": 128, "value_length": 128})
    assert s["kv_cache"]["kv_bearing_layers"] == 32
    assert s["kv_cache"]["counted_from"] == "block_count (fallback)"


def test_no_kv_cache_when_metadata_missing():
    s = summarize_tensors(qwen35_tensors(), {})
    assert "kv_cache" not in s


# --- "Is the type varied per layer?" judgement -------------------------------

def test_uniform_quant_is_not_per_layer_varying():
    """The four Unsloth Dynamic types: varied per role, uniform per layer."""
    s = summarize_tensors(qwen35_tensors("Q5_K"), QWEN35_META)
    assert s["per_layer_varying"] is False
    assert s["n_mixed_roles"] == 0


def test_standard_k_quant_is_detected_as_per_layer_varying():
    """Standard K-quant applies. llama.cpp's use_more_bits raises some layers."""
    s = summarize_tensors(
        qwen35_tensors("Q5_K",
                       ffn_down={i: "Q6_K" for i in range(0, 65, 2)},
                       attn_v={i: "Q6_K" for i in FULL_ATTN_LAYERS[:9]}),
        QWEN35_META)
    assert s["per_layer_varying"] is True
    assert set(s["mixed_roles"]) == {"ffn_down.weight", "attn_v.weight"}
    assert s["mixed_roles"]["attn_v.weight"] == {"Q6_K": 9, "Q5_K": 8}


def test_mtp_block_only_difference_is_not_per_layer_varying():
    """IQ4_XS applies. A type that differs only in blk.64 (MTP) is not per-layer."""
    s = summarize_tensors(qwen35_tensors("IQ4_XS", mtp_type="Q4_K"), QWEN35_META)
    assert s["per_layer_varying"] is False
    assert s["n_mixed_roles"] == 0
    assert s["n_mtp_only_roles"] > 0
    assert s["mtp_block_layers"] == [64]


def test_norm_tensors_do_not_trigger_mixed_roles():
    """Mixed F32 norms alone do not mean "varied per layer"."""
    s = summarize_tensors([("blk.0.ffn_down.weight", "Q4_K"),
                           ("blk.1.ffn_down.weight", "F32")], {})
    assert s["per_layer_varying"] is False


# --- MTP / notable ----------------------------------------------------------

def test_mtp_tensors_detected():
    s = summarize_tensors(qwen35_tensors(), QWEN35_META)
    assert s["mtp_tensor_count"] == 4
    assert s["mtp_tensors"][0] == "blk.64.nextn.eh_proj.weight"


def test_no_mtp_tensors_reported_as_zero():
    s = summarize_tensors([("blk.0.ffn_down.weight", "Q4_K")], {})
    assert s["mtp_tensor_count"] == 0
    assert s["mtp_block_layers"] == []


def test_output_weight_does_not_match_attn_output_weight():
    """Substring matching caught 65 blk.N.attn_output.weight files and polluted output."""
    s = summarize_tensors(qwen35_tensors(), QWEN35_META)
    names = [t["name"] for t in s["notable_tensors"]]
    assert names == ["output.weight", "output_norm.weight", "token_embd.weight"]


# --- Quantization distribution ----------------------------------------------

def test_weight_mix_excludes_f32_norms():
    """Mixing F32 norms skews the "dominant type", so drop them from the tally."""
    s = summarize_tensors(qwen35_tensors("Q5_K"), QWEN35_META)
    assert "F32" in s["quant_mix"]
    assert "F32" not in s["weight_mix"]
    assert s["dominant_weight_type"] == "Q5_K"


def test_shares_sum_to_100():
    s = summarize_tensors(qwen35_tensors("Q5_K"), QWEN35_META)
    total = sum(v["share"] for v in s["weight_mix"].values())
    assert total == pytest.approx(100.0, abs=0.2)


def test_empty_input_does_not_crash():
    s = summarize_tensors([], {})
    assert s["n_roles"] == 0
    assert s["dominant_weight_type"] is None
    assert s["per_layer_varying"] is False

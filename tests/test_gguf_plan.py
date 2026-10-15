"""Tests for gguf_plan unit conversion, ctx back-calculation and generated command.

This script got it wrong twice:

  * **It mixed GB (÷10^9) and GiB (÷2^30).** GGUF's size_gb is bytes÷10^9, the
    GPU's "24GB" is GiB. Mixed, it reports what fits a 24GiB card as overflow
  * **It wrote a # comment in the middle of a continued line (\\).** Everything
    to the end of the line becomes a comment, the \\ disappears with it, and the
    launch command is cut off

Both "look plausible in the output", so they are hard to notice. Pin them here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gguf_plan", Path(__file__).resolve().parent.parent / "gguf_plan.py")
gguf_plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gguf_plan)

GIB = 1024 ** 3

#: The gguf_probe output measured for Qwen3.8-27B-Q5_K_M (only what is needed)
Q5_K_M = {
    "file": "Qwen3.8-27B-Q5_K_M.gguf",
    "size_gb": 19.83,
    "context_length": 262144,
    "is_language_model": True,
    "chat_template_has_think": True,
    "mtp_tensor_count": 4,
    "is_hybrid_attention": True,
    "kv_layers": list(range(17)),
    "kv_cache": {"bytes_per_token_f16": 69632, "total_layers": 65},
}
NO_MTP = {**Q5_K_M, "mtp_tensor_count": 0, "mtp_tensors": []}
NO_THINK = {**Q5_K_M, "chat_template_has_think": False}


# --- Unit conversion --------------------------------------------------------

def test_file_size_converted_from_gb_to_gib():
    """size_gb is bytes÷10^9. Never mix it with the GPU's GiB.

    19.83 GB = 18.47 GiB. This 1.36 difference flips the "does it fit 24GB"
    verdict.
    """
    assert gguf_plan.file_gib(Q5_K_M) == pytest.approx(18.47, abs=0.01)
    assert gguf_plan.file_gib(Q5_K_M) < Q5_K_M["size_gb"]


def test_kv_size_in_gib_matches_hand_calculation():
    """68 KB/token x 65,536 = 4.25 GiB (not 4.56 GB)."""
    assert gguf_plan.kv_gib(Q5_K_M, 65536, "f16") == pytest.approx(4.25, abs=0.01)
    assert gguf_plan.kv_gib(Q5_K_M, 65536, "f16") == pytest.approx(
        69632 * 65536 / GIB, abs=1e-6)


def test_q8_kv_is_about_half_of_f16():
    f16 = gguf_plan.kv_gib(Q5_K_M, 65536, "f16")
    q8 = gguf_plan.kv_gib(Q5_K_M, 65536, "q8_0")
    assert q8 == pytest.approx(f16 * gguf_plan.Q8_FACTOR, abs=1e-9)
    assert 0.5 < q8 / f16 < 0.6


def test_q5_k_m_fits_24gib_card_at_64k_f16():
    """The measured counter-example: GB/GiB mixing reported this as overflow."""
    used = (gguf_plan.file_gib(Q5_K_M)
            + gguf_plan.kv_gib(Q5_K_M, 65536, "f16")
            + gguf_plan.DEFAULT_OVERHEAD_GIB)
    assert used < 24.0
    assert used == pytest.approx(23.72, abs=0.02)


def test_overhead_matches_the_single_measurement():
    """Calibration point: file+KV = 22.72 GiB while llama-server measured 23.5 GiB.

    The default overhead is on the safe side, so it is at least the measured
    difference 0.78.
    """
    bare = gguf_plan.file_gib(Q5_K_M) + gguf_plan.kv_gib(Q5_K_M, 65536, "f16")
    assert bare == pytest.approx(22.72, abs=0.02)
    assert gguf_plan.DEFAULT_OVERHEAD_GIB >= 23.5 - bare


# --- ctx back-calculation ----------------------------------------------------

def test_max_ctx_is_capped_by_native_context_length():
    """Never exceed the native ctx even with spare budget (extrapolation breaks quality)."""
    assert gguf_plan.max_ctx(Q5_K_M, vram_gib=100.0, kv_mode="f16",
                             overhead=1.0) == Q5_K_M["context_length"]


def test_max_ctx_is_floored_to_step():
    n = gguf_plan.max_ctx(Q5_K_M, vram_gib=24.0, kv_mode="f16", overhead=1.0)
    assert n % gguf_plan.CTX_STEP == 0
    assert n >= 65536       # 64k is guaranteed on a 24GiB card


def test_q8_roughly_doubles_max_ctx():
    f16 = gguf_plan.max_ctx(Q5_K_M, 24.0, "f16", 1.0)
    q8 = gguf_plan.max_ctx(Q5_K_M, 24.0, "q8_0", 1.0)
    assert q8 > f16
    assert 1.7 < q8 / f16 < 2.1


def test_max_ctx_zero_when_model_alone_exceeds_budget():
    assert gguf_plan.max_ctx(Q5_K_M, vram_gib=8.0, kv_mode="f16", overhead=1.0) == 0
    assert gguf_plan.max_ctx(Q5_K_M, vram_gib=8.0, kv_mode="q8_0", overhead=1.0) == 0


def test_max_ctx_zero_without_kv_metadata():
    assert gguf_plan.max_ctx({"size_gb": 1.0}, 24.0, "f16", 1.0) == 0


# --- The recommended ctx is rounded to a round number ----------------------

def test_recommended_ctx_rounds_down_to_a_standard_value():
    """Never recommend the budget-filling 69,632 as is.

    It leaves only 0.02 GiB of headroom, and startup fails on the overhead
    estimate error (one calibration point). It gains only 6% more context over
    65,536.
    """
    cap = gguf_plan.max_ctx(Q5_K_M, 24.0, "f16", 1.0)
    rec = gguf_plan.recommended_ctx(Q5_K_M, 24.0, "f16", 1.0)
    assert cap == 69632
    assert rec == 65536
    assert rec in gguf_plan.STANDARD_CTX
    assert rec <= cap


def test_recommended_ctx_leaves_usable_headroom():
    ctx = gguf_plan.recommended_ctx(Q5_K_M, 24.0, "f16", 1.0)
    assert gguf_plan.headroom_gib(Q5_K_M, ctx, "f16", 24.0, 1.0) >= 0.2


def test_recommended_ctx_zero_when_nothing_fits():
    assert gguf_plan.recommended_ctx(Q5_K_M, 8.0, "f16", 1.0) == 0


def test_recommended_ctx_capped_by_native_even_with_huge_budget():
    assert gguf_plan.recommended_ctx(Q5_K_M, 500.0, "f16", 1.0) == 262144


def test_headroom_is_negative_when_it_does_not_fit():
    assert gguf_plan.headroom_gib(Q5_K_M, 262144, "f16", 24.0, 1.0) < 0


def test_thin_headroom_is_warned_in_the_output():
    """A tight setting causes an accident unless the output says so."""
    out = gguf_plan.emit_config(Q5_K_M, 69632, "f16", vram=24.0, overhead=1.0,
                                model_path="/m/x.gguf", port=8085, device="CUDA0")
    assert "⚠️" in out
    assert "headroom" in out
    assert "Use q8_0 KV" in out          # with f16, suggest q8_0


def test_thin_headroom_does_not_suggest_q8_when_already_q8():
    out = gguf_plan.emit_config(Q5_K_M, 131072, "q8_0", vram=24.0, overhead=1.0,
                                model_path="/m/x.gguf", port=8085, device="CUDA0")
    assert "⚠️" in out
    assert "Use q8_0 KV" not in out      # suggesting it when q8_0 is already used is pointless
    assert "Drop ctx one step" in out


def test_comfortable_headroom_is_reported_plainly():
    out = gguf_plan.emit_config(Q5_K_M, 32768, "f16", vram=24.0, overhead=1.0,
                                model_path="/m/x.gguf", port=8085, device="CUDA0")
    assert "⚠️" not in out
    assert "Headroom" in out


# --- The path written after -m ----------------------------------------------

def test_model_path_uses_the_recorded_absolute_path():
    rec = {**Q5_K_M, "path": "/mnt/data/models/X/Qwen3.8-27B-Q5_K_M.gguf"}
    assert gguf_plan.model_path_of(rec, None) == rec["path"]


def test_model_path_override_wins():
    rec = {**Q5_K_M, "path": "/recorded/x.gguf"}
    assert gguf_plan.model_path_of(rec, "/override/y.gguf") == "/override/y.gguf"


def test_model_path_falls_back_for_old_json_without_path():
    """Old gguf.json without a path key still works and yields a placeholder."""
    assert gguf_plan.model_path_of(Q5_K_M, None).endswith(Q5_K_M["file"])


# --- The generated launch command -------------------------------------------

def emit(rec, ctx=65536, kv="f16"):
    return gguf_plan.emit_config(rec, ctx, kv, vram=24.0, overhead=1.0,
                                 model_path="/m/x.gguf", port=8085, device="CUDA0")


def test_continued_lines_never_contain_a_comment():
    """A # in the middle of a continued line breaks the command (real bug, v1)."""
    for line in emit(Q5_K_M, kv="q8_0").splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            assert "#" not in stripped, f"comment inside a continued line: {line!r}"


def test_emitted_command_is_valid_shell():
    import subprocess
    text = emit(Q5_K_M, kv="q8_0")
    start = next(i for i, ln in enumerate(text.splitlines())
                 if ln.startswith("llama-server"))
    cmd = []
    for line in text.splitlines()[start:]:
        cmd.append(line)
        if not line.rstrip().endswith("\\"):
            break
    r = subprocess.run(["bash", "-n"], input="\n".join(cmd),
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    joined = " ".join(x.rstrip("\\").strip() for x in cmd)
    assert "--ctx-size 65536" in joined
    assert "-ctk q8_0 -ctv q8_0" in joined


def test_quantized_kv_always_ships_with_flash_attention():
    """Quantized -ctk/-ctv requires -fa on."""
    out = emit(Q5_K_M, kv="q8_0")
    assert "-ctk q8_0 -ctv q8_0" in out
    assert "-fa on" in out


def test_f16_does_not_emit_ctk_flags():
    """No -ctk **as a command argument**.

    A thin-headroom note may suggest "Use q8_0 KV (-ctk ...)", so an `in` check
    over the whole text cannot tell the two apart.
    """
    out = emit(Q5_K_M, kv="f16")
    assert not any(ln.strip().startswith("-ctk") for ln in out.splitlines())


# --- MTP / sampling selection ------------------------------------------------

def _has_spec_type_arg(text: str) -> bool:
    """Is --spec-type present **as a command argument**?

    The same string also appears in note comments, so a plain `in` check cannot
    tell them apart.
    """
    return any(ln.strip().startswith("--spec-type") for ln in text.splitlines())


def test_draft_mtp_emitted_when_mtp_tensors_exist():
    assert _has_spec_type_arg(emit(Q5_K_M))


def test_draft_mtp_omitted_when_no_mtp_tensors():
    out = emit(NO_MTP)
    assert not _has_spec_type_arg(out)
    assert "not adding" in out          # keep the reason as a note


def test_thinking_template_gets_official_thinking_sampling():
    out = emit(Q5_K_M)
    assert "temperature: 1.0" in out
    assert "top_p: 0.95" in out
    assert "min_p: 0.0" in out


def test_non_thinking_template_gets_different_sampling():
    out = emit(NO_THINK)
    assert "temperature: 0.7" in out
    assert "top_p: 0.8" in out


def test_max_tokens_reason_is_honest_about_which_limit_applied():
    """Writing "3/4 of ctx" while printing 49,152 for ctx 131,072 is a lie (real bug).

    3/4 is 98,304. The limit applied, so the output has to say so.
    """
    mt, reason = gguf_plan.max_tokens_for(131072)
    assert mt == gguf_plan.MAX_TOKENS_CAP
    assert "3/4" in reason and "Capped" in reason
    assert "98,304" in reason          # also show what 3/4 actually was

    mt, reason = gguf_plan.max_tokens_for(32768)
    assert mt == 24576                 # here 3/4 is what applies
    assert "3/4" in reason and "Capped" not in reason


def test_max_tokens_reason_appears_in_the_emitted_config():
    out = emit(Q5_K_M, ctx=131072, kv="q8_0")
    assert "max_tokens: 49152" in out
    assert "3/4 of ctx 131,072" not in out      # ← present here would be a lie


def test_max_tokens_leaves_room_for_the_prompt():
    for ctx in (32768, 65536, 131072):
        out = emit(Q5_K_M, ctx=ctx)
        mt = int(next(ln for ln in out.splitlines()
                      if ln.strip().startswith("max_tokens:")
                      ).split(":")[1].split("#")[0].strip())
        assert mt < ctx, "with max_tokens >= ctx the effective limit is n_ctx - prompt length"

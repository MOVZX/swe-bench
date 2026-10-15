"""Regression tests for comparison conditions (4): model file fingerprint / sampler defaults / task-set check.

The same quantization label alone does not make a comparison valid. In real
measurements, runs with a different build, a different gguf and a different
KV-cache quantization shared one table even though all said `Q4_K_M`.
"""

from __future__ import annotations

import hashlib

from llmbench import env as envinfo
from llmbench.compare import _bench_conditions, _task_set_warnings, render_comparison

# ────────────────── env: model file fingerprint ──────────────────

def test_model_file_info_records_size_mtime_and_fingerprint(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 1000)
    info = envinfo.model_file_info(str(p))
    assert info["size_bytes"] == 1004
    assert info["mtime"]
    assert info["sha256_head_tail"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_model_file_info_detects_content_swap(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(b"A" * 4096)
    b.write_bytes(b"B" * 4096)
    assert (envinfo.model_file_info(str(a))["sha256_head_tail"]
            != envinfo.model_file_info(str(b))["sha256_head_tail"])


def test_model_file_info_is_none_filled_when_unreadable():
    """Fills None without raising when the file is unreadable (a collection failure must not stop the bench)."""
    info = envinfo.model_file_info("/no/such/model-file.gguf")
    assert info["size_bytes"] is None
    assert info["mtime"] is None
    assert info["sha256_head_tail"] is None
    assert envinfo.model_file_info("") is None


def test_model_file_info_hashes_only_head_and_tail(tmp_path):
    """Does not read a whole 20GB file (only the first and last 1MB)."""
    size = envinfo._FINGERPRINT_CHUNK * 2 + 4096
    p = tmp_path / "big.gguf"
    p.write_bytes(b"\x01" * size)
    h = hashlib.sha256()
    h.update(b"\x01" * envinfo._FINGERPRINT_CHUNK)
    h.update(b"\x01" * envinfo._FINGERPRINT_CHUNK)
    info = envinfo.model_file_info(str(p))
    assert info["sha256_head_tail"] == h.hexdigest()
    assert info["fingerprint_chunk_bytes"] == envinfo._FINGERPRINT_CHUNK


# ────────────────── env: sampler defaults / launch arguments ──────────────────

def test_sampler_defaults_from_props_flat():
    props = {"default_generation_settings": {
        "n_ctx": 32768, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "min_p": 0.0, "seed": 4294967295}}
    d = envinfo.sampler_defaults(props)
    assert d["temperature"] == 0.7 and d["top_p"] == 0.8
    assert d["top_k"] == 20 and d["seed"] == 4294967295


def test_sampler_defaults_from_props_nested_params():
    props = {"default_generation_settings": {
        "n_ctx": 8192, "params": {"temp": 0.6, "min_p": 0.05}}}
    d = envinfo.sampler_defaults(props)
    assert d["temperature"] == 0.6      # normalizes temp to temperature
    assert d["min_p"] == 0.05


def test_sampler_defaults_empty_when_absent():
    assert envinfo.sampler_defaults({}) == {}
    assert envinfo.sampler_defaults(None) == {}


def test_parse_server_args_reads_sampling_and_cache_types():
    a = envinfo.parse_server_args([
        "llama-server", "-m", "x.gguf", "--temp", "0.2", "--top-p", "0.8",
        "--top-k", "20", "--min-p", "0.05", "--seed", "1234",
        "--cache-type-k", "q8_0", "-ctv", "q8_0", "-ngl", "99",
    ])
    assert a["temp"] == "0.2" and a["top_p"] == "0.8"
    assert a["top_k"] == 20 and a["min_p"] == "0.05"
    assert a["seed"] == 1234
    assert a["cache_type_k"] == "q8_0" and a["cache_type_v"] == "q8_0"
    assert a["n_gpu_layers"] == 99


# ────────────────── compare: comparison conditions ──────────────────

def _env(**backend):
    return {"execution": "local", "host": {}, "backend": backend}


def test_bench_conditions_include_model_file_and_build():
    cond = _bench_conditions(_env(
        quantization="Q4_K_M",
        model_path="/models/Qwen3-27B-Q4_K_M.gguf",
        build_info="b10157-c6292cfb8",
        model_file={"sha256_head_tail": "abcdef0123456789"},
        launch={"n_gpu_layers": 99, "spec_type": "draft-mtp",
                "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
    ))
    assert cond["Model"] == "Qwen3-27B-Q4_K_M.gguf"   # the full path is not shown
    assert cond["Fingerprint"] == "abcdef012345"
    assert cond["Speculative"] == "draft-mtp"
    assert cond["KV-K"] == "q8_0" and cond["KV-V"] == "q8_0"
    assert cond["build"] == "b10157-c6292cfb8"


def test_bench_conditions_tolerates_missing_fields():
    assert set(_bench_conditions({}).values()) == {None}


def test_same_quant_label_different_gguf_is_not_same_condition():
    a = _bench_conditions(_env(quantization="Q4_K_M",
                               model_file={"sha256_head_tail": "aaaa1111"}))
    b = _bench_conditions(_env(quantization="Q4_K_M",
                               model_file={"sha256_head_tail": "bbbb2222"}))
    assert a != b


# ────────────────── compare: matching task sets ──────────────────

def _row(label, task_ids):
    return {"label": label, "results": {t: {"combined": 50} for t in task_ids}}


def test_task_set_warning_absent_when_identical():
    rows = [_row("A", ["t1", "t2"]), _row("B", ["t1", "t2"])]
    assert _task_set_warnings(rows) == []


def test_task_set_warning_when_partial_overlap():
    rows = [_row("A", ["t1", "t2", "t3"]), _row("B", ["t1", "t2"])]
    out = "\n".join(_task_set_warnings(rows))
    assert "Task sets do not match" in out
    assert "common 2" in out and "B missing 1 tasks" in out


def test_task_set_warning_when_no_overlap():
    rows = [_row("A", ["t1"]), _row("B", ["t2"])]
    out = "\n".join(_task_set_warnings(rows))
    assert "share not a single common task" in out
    assert "unusable even as a reference value" in out


def test_render_comparison_surfaces_task_set_warning():
    runs = [
        {"model": "A", "summary": {"avg_combined": 60.0},
         "results": [{"task_id": "t1", "difficulty": "easy", "combined": 60}]},
        {"model": "B", "summary": {"avg_combined": 50.0},
         "results": [{"task_id": "t9", "difficulty": "easy", "combined": 50}]},
    ]
    md = render_comparison(runs)
    assert "unusable even as a reference value" in md

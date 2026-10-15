"""Verification of execution-environment metadata collection (llmbench/env.py) and report output.

Requirements:
  - collection is best-effort and **must never raise** (it must not stop the benchmark)
  - secret values such as api_key must not leak into results.json
  - for local inference, keep the quantization, the GPU offload ratio and n_ctx
    (they change tok/s several-fold, so the hardware specs alone are not comparable)
  - for remote inference, state that the host specs do not affect speed
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmbench import env as envinfo
from llmbench.report import _env_section

OLLAMA_PS = {
    "models": [{
        "name": "qwen2.5-coder:32b",
        "model": "qwen2.5-coder:32b",
        "size": 21_000_000_000,
        "size_vram": 21_000_000_000,
        "context_length": 32768,
        "details": {
            "family": "qwen2",
            "parameter_size": "32.8B",
            "quantization_level": "Q4_K_M",
        },
    }]
}
LLAMACPP_PROPS = {
    "default_generation_settings": {"n_ctx": 32768},
    "total_slots": 4,
    "model_path": "/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "build_info": "b4321-abc",
}
_ROUTES = {
    "/api/ps": OLLAMA_PS,
    "/api/version": {"version": "0.6.2"},
    "/props": LLAMACPP_PROPS,
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the test output clean
        pass

    def do_GET(self):  # noqa: N802
        body = _ROUTES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        b = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture(scope="module")
def server_url():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


# ────────────────────────── Robustness and safety of collection ──────────────────────────

CFGS = [
    {"type": "openai", "base_url": "http://localhost:8085/v1", "model": "auto",
     "api_key": "sk-TOPSECRET"},
    {"type": "openai", "base_url": "https://api.example.com/v1",
     "model": "some-model", "api_key": "sk-TOPSECRET"},
    {"type": "ollama", "base_url": "http://localhost:11434", "model": "q:32b"},
    {"type": "cli", "preset": "claude"},
    {"type": "mock", "mode": "gold"},
    {},                     # does not crash when the type is missing
    {"type": "unknown-x"},  # does not crash for an unknown type
]


@pytest.mark.parametrize("cfg", CFGS)
def test_collect_never_raises_and_is_json_serializable(cfg):
    env = envinfo.collect(cfg)
    json.dumps(env)  # must be embeddable in results.json
    assert "execution" in env
    envinfo.format_summary(env)  # building the summary must not crash either


@pytest.mark.parametrize("cfg", CFGS)
def test_no_secret_leak(cfg):
    """api_key must not leak into the environment info (results.json may be shared)."""
    assert "TOPSECRET" not in json.dumps(envinfo.collect(cfg), ensure_ascii=False)


def test_collect_host_has_minimum_keys():
    host = envinfo.collect_host()
    assert host.get("os") and host.get("arch") and host.get("python")


@pytest.mark.parametrize("raw,expected", [
    ("spdisplays_metal4", "Metal 4"),
    ("spdisplays_metal3", "Metal 3"),
    ("Metal 3", "Metal 3"),
    ("unknown-value", "unknown-value"),
])
def test_macos_metal_label_is_humanized(monkeypatch, raw, expected):
    """Do not print the internal system_profiler key names in the report."""
    payload = json.dumps({"SPDisplaysDataType": [
        {"sppci_model": "Apple M3 Max", "sppci_cores": "40",
         "spdisplays_mtlgpufamilysupport": raw}
    ]})
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: payload)
    gpus = envinfo._gpu_macos()
    assert gpus == [{"name": "Apple M3 Max", "cores": 40, "metal": expected}]


def test_execution_kind_classification():
    def ex(cfg):
        return envinfo.collect(cfg)["execution"]

    assert ex({"type": "ollama", "base_url": "http://localhost:11434"}) == "local"
    assert ex({"type": "openai", "base_url": "http://127.0.0.1:8085/v1"}) == "local"
    assert ex({"type": "openai",
               "base_url": "https://api.z.ai/api/coding/paas/v4"}) == "remote-api"
    assert ex({"type": "cli", "preset": "claude"}) == "subscription-cli"
    assert ex({"type": "mock"}) == "mock"


# ─────────────────────── Parsing the backend configuration ───────────────────────


def test_backend_ollama_parses_offload_ratio(server_url):
    b = envinfo.collect_backend(
        {"type": "ollama", "base_url": server_url, "model": "qwen2.5-coder:32b"}
    )
    assert b["kind"] == "ollama"
    assert b["quantization"] == "Q4_K_M"
    assert b["parameter_size"] == "32.8B"
    assert b["n_ctx"] == 32768
    # size_vram == size → all layers on the GPU
    assert b["gpu_offload_ratio"] == 1.0
    assert b["server_version"] == "ollama 0.6.2"


def test_backend_llamacpp_parses_props_and_quant(server_url):
    b = envinfo.collect_backend(
        {"type": "openai", "base_url": f"{server_url}/v1", "model": "auto"},
        served_model="Qwen3-Coder-30B",
    )
    assert b["kind"] == "llama.cpp"
    assert b["n_ctx"] == 32768
    assert b["parallel_slots"] == 4
    # read the quantization from the gguf file name
    assert b["quantization"] == "Q4_K_M"
    assert b["model"] == "Qwen3-Coder-30B"


def test_backend_remote_does_not_probe_local_endpoints():
    """Do not send /props to a remote API (no useless traffic)."""
    b = envinfo.collect_backend(
        {"type": "openai", "base_url": "https://api.example.invalid/v1",
         "model": "m"}
    )
    assert b["kind"] == "openai-compat-api"
    assert "n_ctx" not in b


# ─────────────────── Multi-GPU: which GPU it landed on ───────────────────

# nvidia-smi output from a real machine (RTX 3090 + RTX 5090, llama-server sharded over both)
_GPU_QUERY = (
    "GPU-aaaa, 0, NVIDIA GeForce RTX 3090, 6129, 24576\n"
    "GPU-bbbb, 1, NVIDIA GeForce RTX 5090, 6760, 32607"
)
_APPS_SPLIT = (
    "GPU-aaaa, 3596389, /home/u/llama.cpp/build-cuda/bin/llama-server, 6106\n"
    "GPU-bbbb, 3596389, /home/u/llama.cpp/build-cuda/bin/llama-server, 6750"
)
_APPS_SINGLE = (
    "GPU-bbbb, 3596389, /home/u/llama.cpp/build-cuda/bin/llama-server, 6750"
)


def _fake_smi(monkeypatch, gpu_out, apps_out):
    def _sh(cmd, timeout=None):
        if cmd[0] != "nvidia-smi":
            return None
        joined = " ".join(cmd)
        if "--query-compute-apps" in joined:
            return apps_out
        if "--query-gpu" in joined:
            return gpu_out
        return None

    monkeypatch.setattr(envinfo, "_sh", _sh)


def test_gpu_usage_detects_tensor_split_across_gpus(monkeypatch):
    """The same PID on several GPUs = a sharded load. Record it as a split, not as "one of them"."""
    _fake_smi(monkeypatch, _GPU_QUERY, _APPS_SPLIT)
    usage = envinfo.collect_gpu_usage()
    inf = usage["inference"]
    assert inf["multi_gpu"] is True
    assert inf["pid"] == 3596389
    assert inf["process"] == "llama-server"   # extract the executable name from the full path
    assert [g["index"] for g in inf["gpus"]] == [0, 1]
    assert [g["name"] for g in inf["gpus"]] == [
        "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 5090"]
    assert inf["vram_total_gb"] == 12.6        # 6.0 + 6.6
    assert "uncertain" not in inf
    # keep the usage of all installed GPUs too
    assert [g["vram_total_gb"] for g in usage["gpus"]] == [24.0, 31.8]


def test_gpu_usage_single_gpu_is_not_flagged_multi(monkeypatch):
    _fake_smi(monkeypatch, _GPU_QUERY, _APPS_SINGLE)
    inf = envinfo.collect_gpu_usage()["inference"]
    assert inf["multi_gpu"] is False
    assert [g["index"] for g in inf["gpus"]] == [1]


def test_gpu_usage_unknown_process_is_marked_uncertain(monkeypatch):
    _fake_smi(monkeypatch, _GPU_QUERY, "GPU-bbbb, 42, /usr/bin/blender, 6750")
    inf = envinfo.collect_gpu_usage()["inference"]
    assert inf["uncertain"] is True


def test_gpu_usage_without_nvidia_returns_empty(monkeypatch):
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: None)
    assert envinfo.collect_gpu_usage() == {}


def test_gpu_usage_only_collected_for_local_execution(monkeypatch):
    """For a remote API there is no point in reading the local GPU, so do not collect it."""
    _fake_smi(monkeypatch, _GPU_QUERY, _APPS_SPLIT)
    remote = envinfo.collect({"type": "openai",
                              "base_url": "https://api.example.invalid/v1"})
    assert "gpu_usage" not in remote["backend"]


def test_summary_prefers_actual_inference_gpus(monkeypatch):
    """On a multi-GPU machine, showing only the first installed GPU misleads, so show the GPUs actually used."""
    env = {"execution": "local", "host": {"gpu": [
        {"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0}]},
        "backend": {"gpu_usage": {"inference": {
            "multi_gpu": True, "vram_total_gb": 12.6, "gpus": [
                {"index": 0, "name": "RTX 3090", "vram_gb": 6.0},
                {"index": 1, "name": "RTX 5090", "vram_gb": 6.6}]}}}}
    s = envinfo.format_summary(env)
    assert "RTX 3090 6.0GB + RTX 5090 6.6GB" in s
    assert "(split over 12.6GB total)" in s


# ────────── Compute backend (CUDA / ROCm / Vulkan) detection ──────────
#
# llama.cpp /props returns only build_info, so the API cannot tell which backend
# the binary was built for. It is the factor that affects tok/s the most, so
# derive it from the libraries loaded by the process.

_MAPS = {
    "cuda": (
        "7f00-7f01 r--p /usr/lib/x86_64-linux-gnu/libcudart.so.13\n"
        "7f02-7f03 r--p /usr/lib/x86_64-linux-gnu/libcublas.so.13\n"
        "7f04-7f05 r--p /usr/lib/x86_64-linux-gnu/libc.so.6\n"
    ),
    "rocm": (
        "7f00-7f01 r--p /opt/rocm/lib/libamdhip64.so.6\n"
        "7f02-7f03 r--p /opt/rocm/lib/librocblas.so.4\n"
    ),
    "vulkan": (
        "7f00-7f01 r--p /usr/lib/x86_64-linux-gnu/libvulkan.so.1\n"
        "7f02-7f03 r--p /usr/lib/x86_64-linux-gnu/libggml-vulkan.so\n"
    ),
    "cpu": "7f00-7f01 r--p /usr/lib/x86_64-linux-gnu/libc.so.6\n",
}


def _fake_proc(monkeypatch, maps, exe):
    monkeypatch.setattr(envinfo, "_proc_maps", lambda pid: maps)
    monkeypatch.setattr(envinfo, "_proc_exe", lambda pid: exe)


@pytest.mark.parametrize("key,expected", [
    ("cuda", "CUDA"), ("rocm", "ROCm"), ("vulkan", "Vulkan"), ("cpu", "CPU"),
])
def test_detect_runtime_from_loaded_libraries(monkeypatch, key, expected):
    _fake_proc(monkeypatch, _MAPS[key], "/home/u/llama.cpp/build/bin/llama-server")
    assert envinfo.detect_runtime(3596389)["compute"] == expected


def test_cuda_build_loading_vulkan_is_not_misdetected(monkeypatch):
    """Do not misreport a CUDA build as Vulkan just because it loads libvulkan indirectly."""
    _fake_proc(monkeypatch, _MAPS["cuda"] + _MAPS["vulkan"], None)
    rt = envinfo.detect_runtime(1)
    assert rt["compute"] == "CUDA"
    assert "Vulkan" in rt["also_loaded"]


def test_binary_path_is_used_as_hint(monkeypatch):
    """Even when the libraries cannot be read, a path such as build-rocm tells you."""
    _fake_proc(monkeypatch, None, "/home/u/llama.cpp/build-rocm/bin/llama-server")
    rt = envinfo.detect_runtime(1)
    assert rt["compute"] == "ROCm"
    assert rt["binary_hint"] == "ROCm"
    assert rt["binary"].endswith("build-rocm/bin/llama-server")


def test_detect_runtime_without_pid_is_empty():
    assert envinfo.detect_runtime(None) == {}


def test_find_server_pid_prefers_matching_port(monkeypatch):
    """The operation swaps builds on the same port, so a matching port has top priority."""
    monkeypatch.setattr(envinfo, "_proc_cmdlines", lambda: [
        (100, "/usr/bin/gnome-shell"),
        (200, "/home/u/llama.cpp/build-cuda/bin/llama-server --port 9000"),
        (300, "/home/u/llama.cpp/build-rocm/bin/llama-server --port 8085"),
    ])
    assert envinfo.find_server_pid(8085) == 300
    assert envinfo.find_server_pid(None) == 200   # without a port, take the first
    monkeypatch.setattr(envinfo, "_proc_cmdlines", lambda: [(1, "/usr/bin/bash")])
    assert envinfo.find_server_pid(8085) is None


def test_config_runtime_overrides_detection(monkeypatch):
    """Treat runtime: in config as correct, and set mismatch when it disagrees with the detected value."""
    _fake_proc(monkeypatch, _MAPS["cuda"], None)
    monkeypatch.setattr(envinfo, "find_server_pid", lambda port: 1)
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {})
    cfg = {"type": "openai", "base_url": "http://localhost:8085/v1",
           "runtime": "llama.cpp/Vulkan"}
    rt = envinfo.collect(cfg)["backend"]["runtime"]
    assert rt["compute"] == "llama.cpp/Vulkan"
    assert rt["source"] == "config"
    assert rt["detected"] == "CUDA" and rt["mismatch"] is True


def test_detection_used_when_config_has_no_runtime(monkeypatch):
    _fake_proc(monkeypatch, _MAPS["rocm"], None)
    monkeypatch.setattr(envinfo, "find_server_pid", lambda port: 1)
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {})
    rt = envinfo.collect({"type": "openai",
                          "base_url": "http://localhost:8085/v1"})["backend"]["runtime"]
    assert rt["compute"] == "ROCm" and rt["source"] == "detected"


def test_runtime_not_probed_for_remote(monkeypatch):
    monkeypatch.setattr(envinfo, "find_server_pid",
                        lambda port: pytest.fail("do not scan processes for a remote host"))
    env = envinfo.collect({"type": "openai",
                           "base_url": "https://api.example.invalid/v1"})
    assert "runtime" not in env["backend"]


def test_amd_gpu_is_enumerated(monkeypatch):
    """With Radeon on ROCm/Vulkan, nvidia-smi alone leaves the GPU row empty."""
    def _sh(cmd, timeout=None):
        if cmd[0] == "rocm-smi" and "--showproductname" in cmd:
            return ("device,Card Series,Card Model,Card Vendor\n"
                    "card0,Radeon 8060S Graphics,0x150e,AMD")
        return None

    monkeypatch.setattr(envinfo, "_sh", _sh)
    assert envinfo._gpu_amd() == [
        {"name": "Radeon 8060S Graphics", "vendor": "amd"}]


def test_amd_gpu_falls_back_to_lspci(monkeypatch):
    def _sh(cmd, timeout=None):
        if cmd[0] == "lspci":
            return ('c5:00.0 "VGA compatible controller" "AMD/ATI" '
                    '"Radeon 8060S" -r10 "AMD" "Device 150e"')
        return None

    monkeypatch.setattr(envinfo, "_sh", _sh)
    gpus = envinfo._gpu_amd()
    assert gpus and gpus[0]["vendor"] == "amd" and "Radeon" in gpus[0]["name"]


def test_report_shows_compute_backend_and_mismatch():
    env = {"execution": "local", "host": {}, "backend": {
        "kind": "llama.cpp", "runtime": {
            "compute": "llama.cpp/Vulkan", "source": "config",
            "detected": "CUDA", "mismatch": True,
            "binary": "/home/u/llama.cpp/build-cuda/bin/llama-server"}}}
    md = "\n".join(_env_section(env))
    assert "Compute backend" in md
    assert "llama.cpp/Vulkan (from config)" in md
    assert "Mismatch: the detected value is **CUDA**" in md
    assert "build-cuda/bin/llama-server" in md


def test_report_shows_detected_backend_with_evidence():
    env = {"execution": "local", "host": {}, "backend": {
        "kind": "llama.cpp", "runtime": {
            "compute": "ROCm", "source": "detected",
            "evidence": ["libamdhip64.so.6", "librocblas.so.4"]}}}
    md = "\n".join(_env_section(env))
    assert "ROCm (auto-detected) — libamdhip64.so.6, librocblas.so.4" in md
    assert "Mismatch" not in md


# ────────── Inference setup from the launch arguments (--device / -ngl / …) ──────────
#
# llama.cpp /props returns neither the GPU offload amount (-ngl) nor the device
# used (--device). The launch arguments carry all of it, so read them for
# CUDA, ROCm and Vulkan alike.

# llama-server launch arguments from a real machine (ROCm build)
_ROCM_ARGV = [
    "/home/u/llama.cpp/build-rocm/bin/llama-server",
    "-m", "/mnt/data/models/Qwythos-9B-v2-GGUF/Qwythos-9B-v2-MTP-Q8_0.gguf",
    "--port", "8085", "--device", "ROCm0", "--spec-type", "draft-mtp",
    "-ngl", "0", "--ctx-size", "32768", "--threads", "16", "--mlock",
]
_HOST_GPUS = [
    {"name": "NVIDIA GeForce RTX 3090", "vendor": "nvidia"},
    {"name": "NVIDIA GeForce RTX 5090", "vendor": "nvidia"},
    {"name": "AMD Radeon 8060S Graphics", "vendor": "amd"},
]


def test_parse_server_args_reads_real_launch_line():
    a = envinfo.parse_server_args(_ROCM_ARGV)
    assert a["device"] == "ROCm0"
    assert a["n_gpu_layers"] == 0        # not a single layer on the GPU
    assert a["n_ctx"] == 32768
    assert a["threads"] == 16
    assert a["spec_type"] == "draft-mtp"
    assert a["mlock"] is True


def test_parse_server_args_short_and_long_forms():
    a = envinfo.parse_server_args(
        ["llama-server", "-c", "4096", "--n-gpu-layers", "99",
         "-ts", "0,1", "-mg", "1", "-fa"])
    assert a["n_ctx"] == 4096 and a["n_gpu_layers"] == 99
    assert a["tensor_split"] == "0,1" and a["main_gpu"] == 1
    assert a["flash_attn"] is True


def test_launch_command_redacts_secrets():
    """results.json may be shared, so always redact secrets in the launch arguments."""
    cmd = envinfo._redact_argv(
        ["llama-server", "--api-key", "sk-TOPSECRET", "--port", "8085",
         "--api-key-file=/etc/TOPSECRET"])
    assert "TOPSECRET" not in cmd
    assert "--api-key ***" in cmd and "--port 8085" in cmd


# ── `llama-server --list-devices` output from a real machine (NucBox EVO-X2), verbatim ──
#
# ★ The CUDA default order is FASTEST_FIRST, so CUDA0 is the RTX 5090.
#   nvidia-smi lists by PCI bus order, where GPU0 is the RTX 3090 → indexing gives the reverse.
# ★ Even for the same GPU, a different build gives a different ID and number:
#     RTX 3090       → CUDA1 / Vulkan0
#     RTX 5090       → CUDA0 / Vulkan1
#     Radeon 8060S   → ROCm0 / Vulkan2 (even the name is spelled differently)
_LIST_DEVICES_CUDA = (
    "Available devices:\n"
    "  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31136 MiB free)\n"
    "  CUDA1: NVIDIA GeForce RTX 3090 (24123 MiB, 16522 MiB free)\n"
)
_LIST_DEVICES_ROCM = (
    "Available devices:\n"
    "  ROCm0: AMD Radeon 8060S Graphics (98304 MiB, 15342 MiB free)\n"
)
_LIST_DEVICES_VULKAN = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 3090 (24822 MiB, 17008 MiB free)\n"
    "  Vulkan1: NVIDIA GeForce RTX 5090 (32607 MiB, 31641 MiB free)\n"
    "  Vulkan2: Radeon 8060S Graphics (RADV GFX1151) (114164 MiB, 113573 MiB free)\n"
)


def test_list_devices_parses_nested_parens_in_name(monkeypatch):
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_VULKAN)
    devs = envinfo.list_devices("/x/llama-server")
    assert [d["id"] for d in devs] == ["Vulkan0", "Vulkan1", "Vulkan2"]
    # do not treat the heading line "Available devices:" as a device
    assert len(devs) == 3
    # do not swallow the parentheses inside the name; cut only the trailing memory parentheses
    assert devs[2]["name"] == "Radeon 8060S Graphics (RADV GFX1151)"
    assert devs[2]["vram_total_gb"] == 111.5
    assert devs[2]["vram_free_gb"] == 110.9


def test_same_gpu_has_different_id_per_build(monkeypatch):
    """The RTX 3090 is CUDA1 in CUDA but Vulkan0 in Vulkan. Numbers do not carry across builds."""
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_CUDA)
    cuda = envinfo.resolve_device("CUDA1", [], binary="/x")["device_name"]
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_VULKAN)
    vulkan = envinfo.resolve_device("Vulkan0", [], binary="/x")["device_name"]
    assert cuda == vulkan == "NVIDIA GeForce RTX 3090"


@pytest.mark.parametrize("dev,expected", [
    ("CUDA0", "NVIDIA GeForce RTX 5090"),
    ("CUDA1", "NVIDIA GeForce RTX 3090"),
])
def test_cuda_device_order_is_not_nvidia_smi_order(monkeypatch, dev, expected):
    """CUDA0 = 5090. Do not confuse it with nvidia-smi GPU0 (=3090) (a regression hit on a real machine).

    --list-devices is authoritative. Indexing a per-vendor list gives the reverse.
    """
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_CUDA)
    r = envinfo.resolve_device(dev, _HOST_GPUS, binary="/x/llama-server")
    assert r["device_name"] == expected
    assert r["device_name_source"] == "--list-devices"
    assert "device_name_uncertain" not in r


def test_resolve_device_handles_multiple_ids(monkeypatch):
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_CUDA)
    r = envinfo.resolve_device("CUDA0,CUDA1", _HOST_GPUS,
                               binary="/x/llama-server")
    assert r["device_name"] == "NVIDIA GeForce RTX 5090 + NVIDIA GeForce RTX 3090"
    assert [d["id"] for d in r["devices"]] == ["CUDA0", "CUDA1"]


def test_rocm_and_vulkan_resolve_through_the_same_path(monkeypatch):
    """ROCm / Vulkan also resolve through --list-devices (no per-backend branch)."""
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_VULKAN)
    r = envinfo.resolve_device("Vulkan2", [], binary="/x/llama-server")
    assert r["device_name"] == "Radeon 8060S Graphics (RADV GFX1151)"
    assert r["device_name_source"] == "--list-devices"
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: _LIST_DEVICES_ROCM)
    r = envinfo.resolve_device("ROCm0", [], binary="/x/llama-server")
    # the same physical GPU is spelled differently per build (ROCm adds a leading "AMD ")
    assert r["device_name"] == "AMD Radeon 8060S Graphics"


def test_fallback_guess_is_marked_uncertain():
    """Index-based guessing must always set uncertain when --list-devices is unavailable."""
    r = envinfo.resolve_device("CUDA0", _HOST_GPUS)   # no binary
    assert r["device_name_uncertain"] is True


@pytest.mark.parametrize("dev,expected", [
    ("ROCm0", "AMD Radeon 8060S Graphics"),
    ("CUDA0", "NVIDIA GeForce RTX 3090"),
    ("CUDA1", "NVIDIA GeForce RTX 5090"),
])
def test_resolve_device_falls_back_to_vendor_index(dev, expected):
    assert envinfo.resolve_device(dev, _HOST_GPUS)["device_name"] == expected


def test_resolve_device_vulkan_uses_vulkaninfo(monkeypatch):
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: (
        "GPU0:\n\tdeviceName = NVIDIA GeForce RTX 3090\n"
        "GPU1:\n\tdeviceName = NVIDIA GeForce RTX 5090\n"
        "GPU2:\n\tdeviceName = AMD Radeon 8060S Graphics (RADV GFX1151)\n"))
    r = envinfo.resolve_device("Vulkan2", _HOST_GPUS)
    assert r["device_name"] == "AMD Radeon 8060S Graphics (RADV GFX1151)"
    assert "device_name_uncertain" not in r


def test_resolve_device_vulkan_falls_back_with_uncertainty(monkeypatch):
    monkeypatch.setattr(envinfo, "_sh", lambda *a, **k: None)  # no vulkaninfo
    r = envinfo.resolve_device("Vulkan2", _HOST_GPUS)
    assert r["device_name"] == "AMD Radeon 8060S Graphics"
    assert r["device_name_uncertain"] is True


def test_resolve_device_unparsable_is_kept_as_is():
    assert envinfo.resolve_device("none", [])["device"] == "none"
    assert envinfo.resolve_device("", []) == {}


def _fake_local_server(monkeypatch, argv, maps, env_vars=None):
    monkeypatch.setattr(envinfo, "find_server_pid", lambda port: 4242)
    monkeypatch.setattr(envinfo, "_proc_argv", lambda pid: argv)
    monkeypatch.setattr(envinfo, "_proc_maps", lambda pid: maps)
    monkeypatch.setattr(envinfo, "_proc_exe", lambda pid: argv[0])
    monkeypatch.setattr(envinfo, "_proc_visible_env", lambda pid: env_vars or {})
    monkeypatch.setattr(envinfo, "collect_host", lambda: {"gpu": _HOST_GPUS})
    monkeypatch.setattr(envinfo, "collect_backend",
                        lambda cfg, sm=None: {"kind": "llama.cpp"})


def test_collect_records_device_and_ngl(monkeypatch):
    _fake_local_server(monkeypatch, _ROCM_ARGV, _MAPS["rocm"],
                       {"HIP_VISIBLE_DEVICES": "0"})
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {})
    launch = envinfo.collect({"type": "openai",
                              "base_url": "http://localhost:8085/v1"}
                             )["backend"]["launch"]
    assert launch["device"] == "ROCm0"
    assert launch["device_name"] == "AMD Radeon 8060S Graphics"
    assert launch["n_gpu_layers"] == 0
    assert launch["visible_devices_env"] == {"HIP_VISIBLE_DEVICES": "0"}
    assert "llama-server" in launch["command"]


def test_nvidia_attribution_suppressed_for_non_cuda_runtime(monkeypatch):
    """Do not misread the inference GPU when a Vulkan build shows up in nvidia-smi with a few tens of MiB.

    Prevents the regression seen on a real machine: "RTX 3090 0.0GB" while running on Radeon.
    """
    _fake_local_server(monkeypatch, ["llama-server", "--device", "Vulkan2"],
                       _MAPS["vulkan"])
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {
        "inference": {"pid": 1, "process": "llama-server", "multi_gpu": False,
                      "vram_total_gb": 0.0,
                      "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090",
                                "vram_gb": 0.0}]},
        "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090"}]})
    backend = envinfo.collect({"type": "openai",
                               "base_url": "http://localhost:8085/v1"})["backend"]
    assert "inference" not in backend["gpu_usage"]
    assert "Vulkan" in backend["gpu_usage"]["note"]
    # the device from the launch arguments remains instead
    assert backend["launch"]["device"] == "Vulkan2"


def test_context_only_shard_is_not_counted_as_split(monkeypatch):
    """Even with --device CUDA0, an unselected GPU carries a few hundred MiB of context.

    Real machine: 7650MiB on the 5090 / 256MiB on the 3090. Reporting that as
    "sharded over 2 cards" would be wrong, so a very small share is marked as context.
    """
    def _sh(cmd, timeout=None):
        if cmd[0] != "nvidia-smi":
            return None
        j = " ".join(cmd)
        if "--query-compute-apps" in j:
            return ("GPU-aaaa, 3684862, /x/build-cuda/bin/llama-server, 256\n"
                    "GPU-bbbb, 3684862, /x/build-cuda/bin/llama-server, 7650")
        if "--query-gpu" in j:
            return ("GPU-aaaa, 0, NVIDIA GeForce RTX 3090, 280, 24576\n"
                    "GPU-bbbb, 1, NVIDIA GeForce RTX 5090, 7661, 32607")
        return None

    monkeypatch.setattr(envinfo, "_sh", _sh)
    inf = envinfo.collect_gpu_usage()["inference"]
    assert inf["multi_gpu"] is False          # not a sharded load
    assert inf["gpus"][0]["context_only"] is True   # 3090 = 0.2GB
    assert "context_only" not in inf["gpus"][1]     # 5090 = 7.5GB


def test_cuda_runtime_keeps_nvidia_attribution(monkeypatch):
    _fake_local_server(monkeypatch, ["llama-server", "-ngl", "99"],
                       _MAPS["cuda"])
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {
        "inference": {"pid": 1, "multi_gpu": False, "vram_total_gb": 6.6,
                      "gpus": [{"index": 1, "name": "RTX 5090",
                                "vram_gb": 6.6}]}})
    usage = envinfo.collect({"type": "openai",
                             "base_url": "http://localhost:8085/v1"}
                            )["backend"]["gpu_usage"]
    assert usage["inference"]["gpus"][0]["name"] == "RTX 5090"


def test_summary_shows_device_and_flags_ngl_zero():
    env = {"execution": "local", "host": {"gpu": _HOST_GPUS}, "backend": {
        "launch": {"device": "ROCm0", "device_name": "AMD Radeon 8060S Graphics",
                   "n_gpu_layers": 0}}}
    s = envinfo.format_summary(env)
    assert "AMD Radeon 8060S Graphics" in s
    assert "-ngl 0 (GPU unused)" in s


def test_report_flags_ngl_zero_as_cpu_execution():
    env = {"execution": "local", "host": {}, "backend": {
        "kind": "llama.cpp",
        "runtime": {"compute": "ROCm", "source": "detected"},
        "launch": {"device": "ROCm0", "device_name": "AMD Radeon 8060S Graphics",
                   "n_gpu_layers": 0, "threads": 16, "n_ctx": 32768,
                   "command": "llama-server --device ROCm0 -ngl 0"}}}
    md = "\n".join(_env_section(env))
    assert "Device used" in md and "ROCm0 — AMD Radeon 8060S Graphics" in md
    assert "no layer is on the GPU" in md
    assert "CPU execution in effect" in md
    assert "Launch command" in md


def test_report_ngl_positive_has_no_cpu_warning():
    env = {"execution": "local", "host": {}, "backend": {
        "kind": "llama.cpp",
        "launch": {"device": "Vulkan2", "n_gpu_layers": 99}}}
    md = "\n".join(_env_section(env))
    assert "`-ngl 99`" in md
    assert "CPU execution in effect" not in md


# ────────────────────────── Report output ──────────────────────────


def test_report_section_local_shows_backend_config(server_url):
    env = envinfo.collect(
        {"type": "ollama", "base_url": server_url, "model": "qwen2.5-coder:32b"}
    )
    md = "\n".join(_env_section(env))
    assert "## 🖥 Execution environment" in md
    assert "Q4_K_M" in md
    assert "GPU offload ratio" in md
    assert "measuring client" not in md  # no warning for local execution


def test_report_section_remote_warns_specs_are_client_side():
    env = envinfo.collect({"type": "cli", "preset": "claude"})
    md = "\n".join(_env_section(env))
    assert "measuring client" in md
    assert "do not compare its tok/s" in md


def test_report_shows_multi_gpu_split_and_warns():
    env = {"execution": "local", "host": {"gpu": [
        {"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0,
         "compute_capability": "8.6", "driver": "610.43.02"},
        {"name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8,
         "compute_capability": "12.0", "driver": "610.43.02"}]},
        "backend": {"kind": "llama.cpp", "gpu_usage": {"inference": {
            "process": "llama-server", "multi_gpu": True,
            "vram_total_gb": 12.6, "gpus": [
                {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 6.0},
                {"index": 1, "name": "NVIDIA GeForce RTX 5090",
                 "vram_gb": 6.6}]}}}}
    md = "\n".join(_env_section(env))
    assert "GPU0 NVIDIA GeForce RTX 3090 6.0GB" in md
    assert "GPU1 NVIDIA GeForce RTX 5090 6.6GB" in md
    assert "12.6GB in total, sharded over 2 cards" in md
    assert "slower card" in md            # warning: not comparable with a single GPU
    assert "CC 8.6" in md and "CC 12.0" in md


def test_report_single_gpu_has_no_split_warning():
    env = {"execution": "local", "host": {}, "backend": {
        "kind": "llama.cpp", "gpu_usage": {"inference": {
            "process": "llama-server", "multi_gpu": False,
            "vram_total_gb": 6.6,
            "gpus": [{"index": 1, "name": "RTX 5090", "vram_gb": 6.6}]}}}}
    md = "\n".join(_env_section(env))
    assert "GPU1 RTX 5090 6.6GB" in md
    assert "sharded over multiple GPUs" not in md
    assert "slower card" not in md


def test_report_section_empty_env_is_omitted():
    assert _env_section({}) == []


def test_partial_offload_is_flagged():
    """Show a warning mark when the model does not fit in the GPU (the main cause of low tok/s)."""
    env = {"execution": "local", "host": {}, "backend": {
        "kind": "ollama", "gpu_offload_ratio": 0.62, "vram_resident_gb": 13.0}}
    md = "\n".join(_env_section(env))
    assert "⚠️ 62%" in md


# ────────────────────── Embedding in results.json ──────────────────────


def test_save_run_embeds_environment(tmp_path):
    from llmbench.runner import RunResult, save_run

    run = RunResult(model="m", issue_lang="en")
    run.environment = envinfo.collect({"type": "mock"})
    json_path, md_path = save_run(run, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["environment"]["execution"] == "mock"
    assert "## 🖥 Execution environment" in md_path.read_text(encoding="utf-8")


def test_summary_carries_speed_metrics(tmp_path):
    """External tools that read only summary (CodeRouter etc.) must see the speed."""
    from llmbench.runner import RunResult, TaskResult, save_run

    run = RunResult(model="m", issue_lang="en")
    run.results = [
        TaskResult(task_id="t1", difficulty="easy", title="a",
                   tokens_per_sec=30.0, latency_sec=2.0),
        TaskResult(task_id="t2", difficulty="easy", title="b",
                   tokens_per_sec=40.0, latency_sec=4.0),
        # do not count tasks with no measured speed in the denominator
        TaskResult(task_id="t3", difficulty="easy", title="c"),
    ]
    payload = json.loads(
        save_run(run, tmp_path)[0].read_text(encoding="utf-8"))
    assert payload["summary"]["tokens_per_sec"] == 35.0
    assert payload["summary"]["avg_latency_ms"] == 3000


def test_summary_omits_speed_metrics_when_unavailable(tmp_path):
    from llmbench.runner import RunResult, TaskResult, save_run

    run = RunResult(model="m", issue_lang="en")
    run.results = [TaskResult(task_id="t1", difficulty="easy", title="a")]
    summary = json.loads(
        save_run(run, tmp_path)[0].read_text(encoding="utf-8"))["summary"]
    assert "tokens_per_sec" not in summary
    assert "avg_latency_ms" not in summary


def test_save_run_without_environment_stays_backward_compatible(tmp_path):
    from llmbench.runner import RunResult, save_run

    json_path, md_path = save_run(RunResult(model="m", issue_lang="en"), tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "environment" not in payload
    assert "## 🖥 Execution environment" not in md_path.read_text(encoding="utf-8")


# ─────────────── compare: hardware comparison mode ───────────────
#
# When the same model is measured on different hardware, "the environments differ, so tok/s is not comparable" is backwards.
# The model is fixed and the hardware is what is compared, so tok/s is the main subject.


def _hw_run(device, compute, tps, ngl=99, n_ctx=16384, model="Ornith-9B-Q6_K"):
    return {
        "model": model, "issue_lang": "en",
        "summary": {"resolved_rate": 1.0, "avg_quality_resolved": 88.6,
                    "avg_combined": 94.3, "runs": 1, "usability": {},
                    "tokens_per_sec": tps},
        "results": [{"task_id": "t001", "difficulty": "easy", "title": "a",
                     "combined": 94.3, "tokens_per_sec": tps}],
        "environment": {
            "execution": "local",
            "host": {"cpu": "AMD RYZEN AI MAX+ 395", "ram_gb": 31.0,
                     "gpu": [{"name": "NVIDIA GeForce RTX 3090",
                              "vendor": "nvidia"}]},
            "backend": {"kind": "llama.cpp", "quantization": "Q6_K",
                        "runtime": {"compute": compute},
                        "launch": {"device_name": device, "n_gpu_layers": ngl,
                                   "n_ctx": n_ctx, "parallel": 1}},
        },
    }


# four real runs (same model, matched with -ngl 99 and n_ctx 16384)
_HW_RUNS = [
    _hw_run("NVIDIA GeForce RTX 5090", "CUDA", 149.5),
    _hw_run("NVIDIA GeForce RTX 3090", "CUDA", 63.7),
    _hw_run("AMD Radeon 8060S Graphics", "ROCm", 26.5),
    _hw_run("Radeon 8060S Graphics (RADV GFX1151)", "Vulkan", 23.7),
]


def test_hardware_comparison_is_detected_and_ranked_by_speed():
    from llmbench.compare import render_comparison

    md = render_comparison(_HW_RUNS)
    assert "# 🆚 Hardware Comparison Report" in md
    assert "## 🖥 Ranking (tok/s descending)" in md
    # no "not comparable" warning (that is the whole point of this mode)
    assert "Measurement environments do not match" not in md
    assert "Inference conditions (quantization / -ngl / n_ctx / parallel) match in all environments" in md
    # ★ No Combined ranking. All environments tie, so the ranking is meaningless,
    #   and the fast environments line up at the bottom and mislead (confirmed on a real report)
    assert "Ranking (Descending by Avg Combined)" not in md
    # rows come in tok/s descending order
    order = [md.index(d) for d in ("RTX 5090", "RTX 3090",
                                   "AMD Radeon 8060S Graphics",
                                   "RADV GFX1151")]
    assert order == sorted(order)
    assert "149.5" in md


def test_hardware_comparison_flags_mismatched_conditions():
    """When -ngl 0 (CPU execution) is mixed in, do not let a speed gap read as a hardware gap."""
    from llmbench.compare import render_comparison

    runs = [_HW_RUNS[0],
            _hw_run("AMD Radeon 8060S Graphics", "ROCm", 27.5, ngl=0,
                    n_ctx=32768)]
    md = render_comparison(runs)
    assert "Inference conditions do not match" in md
    assert "-ngl" in md and "n_ctx" in md
    assert "does not use the GPU" in md


def test_hardware_mode_labels_rows_by_device_not_model():
    """Every row carries the same model name, so tell the table columns apart by device name."""
    from llmbench.compare import render_comparison

    md = render_comparison(_HW_RUNS)
    matrix = md[md.index("## Per-Task Combined Matrix"):]
    assert "NVIDIA GeForce RTX 5090 (CUDA)" in matrix
    assert "Radeon 8060S Graphics (RADV GFX1151) (Vulkan)" in matrix


def test_duplicate_device_labels_get_numbered():
    """Even when the same device × backend is measured twice, the matrix columns stay distinguishable."""
    from llmbench.compare import render_comparison

    runs = [_HW_RUNS[0],
            _hw_run("NVIDIA GeForce RTX 5090", "CUDA", 118.2),
            _HW_RUNS[2]]
    md = render_comparison(runs)
    assert "NVIDIA GeForce RTX 5090 (CUDA) #1" in md
    assert "NVIDIA GeForce RTX 5090 (CUDA) #2" in md
    # do not number a label that is not duplicated
    assert "AMD Radeon 8060S Graphics (ROCm) #" not in md


def test_empty_results_are_excluded_and_reported():
    """A results file with 0 tasks only creates a column, so exclude it and say so."""
    from llmbench.compare import render_comparison

    empty = {"model": "Ornith-9B-Q6_K", "issue_lang": "en",
             "summary": {"resolved_rate": 0.0, "avg_combined": 0.0,
                         "n_tasks": 0, "runs": 1, "usability": {}},
             "results": [], "_path": "20260731_dead_results.json"}
    md = render_comparison([*_HW_RUNS, empty])
    assert "Excluded (0 results)" in md
    assert "20260731_dead_results.json" in md
    # the empty run does not mix into the columns or the ranking
    assert md.count("| 0.0% |") == 0
    assert "environments: 4" in md


def test_model_comparison_mode_is_unchanged_when_models_differ():
    from llmbench.compare import render_comparison

    runs = [_HW_RUNS[0], _hw_run("NVIDIA GeForce RTX 5090", "CUDA", 90.0,
                                 model="other-model")]
    md = render_comparison(runs)
    assert "# 🆚 Model Comparison Report" in md
    assert "Hardware Comparison" not in md


def test_compare_uses_summary_tokens_per_sec():
    """Use the speed in summary when present (independent of re-aggregating results[])."""
    from llmbench.compare import _run_tps

    assert _run_tps({"summary": {"tokens_per_sec": 42.0}, "results": []}) == 42.0
    assert _run_tps({"summary": {},
                     "results": [{"tokens_per_sec": 10.0},
                                 {"tokens_per_sec": 20.0}]}) == 15.0


def test_compare_warns_on_mixed_environments():
    from llmbench.compare import render_comparison

    def _run(model, env):
        d = {"model": model, "issue_lang": "en",
             "summary": {"resolved_rate": 0.5, "avg_quality_resolved": 50,
                         "avg_combined": 50, "runs": 1, "usability": {}},
             "results": [{"task_id": "t1", "difficulty": "easy", "title": "a",
                          "combined": 50, "tokens_per_sec": 30}]}
        if env:
            d["environment"] = env
        return d

    local = {"execution": "local", "host": {"cpu": "M3 Max", "ram_gb": 128.0},
             "backend": {"kind": "ollama", "quantization": "Q4_K_M"}}
    remote = {"execution": "remote-api", "host": {"cpu": "M3 Max"},
              "backend": {"kind": "openai-compat-api", "model": "x"}}
    md = render_comparison([_run("a", local), _run("b", remote)])
    assert "Measurement environments do not match" in md
    # the same environment is shown as comparable
    md2 = render_comparison([_run("a", local), _run("b", dict(local))])
    assert "measured in the same environment" in md2
    # detect a missing environment record (old results) too
    md3 = render_comparison([_run("a", local), _run("b", None)])
    assert "no recorded execution environment" in md3


# ── Strata family (Strata / Maya / Abyss) detection ──────────────────────────────


def test_find_server_pid_does_not_fall_back_to_another_port(monkeypatch):
    """With a port given and no match, return None (do not return ollama on another port)."""
    monkeypatch.setattr(envinfo, "_listening_pids", lambda port: [])
    monkeypatch.setattr(envinfo, "_proc_cmdlines", lambda: [
        (1464, "/usr/local/bin/ollama serve"),
    ])
    assert envinfo.find_server_pid(8082) is None
    assert envinfo.find_server_pid(None) == 1464


def test_find_server_pid_prefers_the_listening_process(monkeypatch):
    """Also catch python server.py (not matched by name) from the port in LISTEN."""
    monkeypatch.setattr(envinfo, "_listening_pids", lambda port: [777] if port == 8082 else [])
    monkeypatch.setattr(envinfo, "_proc_cmdlines", lambda: [
        (1464, "/usr/local/bin/ollama serve"),
    ])
    assert envinfo.find_server_pid(8082) == 777


def _fake_strata_tree(tmp_path, folder, origin, version, model="UD-Q2_K_XL/pack-dsv4"):
    root = tmp_path / folder
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "server.py").write_text("")
    if origin is not None:
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text(
            f'[core]\n\tbare = false\n[remote "origin"]\n\turl = {origin}\n'
            '\tfetch = +refs/heads/*:refs/remotes/origin/*\n')
        (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (root / ".git" / "refs" / "heads").mkdir(parents=True)
        (root / ".git" / "refs" / "heads" / "main").write_text("2b68887abcdef\n")
    if version is not None:
        (root / "VERSION").write_text(version + "\n")
    cfg = root / "run.json"
    cfg.write_text(json.dumps({
        "exe": str(root / "build" / "strata"),
        "args": ["--glm-pack", f"/data/{model}", "--max-context", "65536"],
        "model_name": "deepseek-v4-flash",
    }))
    return root, cfg


@pytest.mark.parametrize("folder,origin,version,family", [
    ("Abyss", "git@github.com:zephel01/Abyss.git", "1.1.0", "Abyss"),
    ("Maya", "https://github.com/mw00/project-maya.git", "1.0.24", "Maya"),
    ("Strata", "https://github.com/Niko1221/Strata.git", None, "Strata"),
    ("Maya-DeepSeek", None, None, "Maya"),          # fall back to the folder name without git
])
def test_strata_family_from_server_folder(monkeypatch, tmp_path, folder, origin, version, family):
    root, cfg = _fake_strata_tree(tmp_path, folder, origin, version)
    monkeypatch.setattr(envinfo, "_proc_argv", lambda pid: [
        str(root / ".venv/bin/python"), "serve/server.py", "--engine", "strata",
        "--config", str(cfg), "--port", "8082"])
    monkeypatch.setattr(envinfo, "_proc_cwd", lambda pid: str(root))
    monkeypatch.setattr(envinfo, "_child_pids", lambda pid: [4242])
    monkeypatch.setattr(envinfo, "_proc_exe",
                        lambda pid: str(root / "build" / "strata") if pid == 4242 else None)
    fam = envinfo.strata_family(1000)
    assert fam["family"] == family
    assert fam["root"] == str(root)
    assert fam["engine_pid"] == 4242
    assert fam["model_path"] == "/data/UD-Q2_K_XL/pack-dsv4"
    assert fam["n_ctx"] == 65536
    assert fam.get("version") == version
    if origin is not None:
        assert fam["commit"] == "2b68887"
        assert envinfo._family_label(fam).startswith(f"{family} ")


def test_strata_family_ignores_other_servers(monkeypatch):
    monkeypatch.setattr(envinfo, "_proc_argv", lambda pid: [
        "/opt/llama.cpp/build/bin/llama-server", "-m", "x.gguf", "--port", "8082"])
    assert envinfo.strata_family(1000) == {}
    assert envinfo.strata_family(None) == {}


def test_collect_labels_strata_family_and_uses_the_engine_process(monkeypatch):
    """kind is Abyss, the model is the pack, and the compute backend is the engine (child) process's.

    Do not treat "Strata/CUDA" (config) and the detected "CUDA" as a mismatch.
    """
    fam = {"family": "Abyss", "version": "1.1.0", "commit": "2b68887",
           "model_path": "/data/UD-Q2_K_XL/pack-dsv4", "n_ctx": 65536,
           "engine_pid": 4242}
    seen = {}

    def runtime(pid):
        seen["pid"] = pid
        return {"compute": "CUDA", "evidence": ["libcudart.so.13"]}

    monkeypatch.setattr(envinfo, "find_server_pid", lambda port: 1000)
    monkeypatch.setattr(envinfo, "strata_family", lambda pid: dict(fam))
    monkeypatch.setattr(envinfo, "detect_runtime", runtime)
    monkeypatch.setattr(envinfo, "collect_launch", lambda pid, gpus=None: {})
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {
        "inference": {"pid": 4242, "process": "strata", "vram_total_gb": 30.6,
                      "gpus": [], "uncertain": True}})
    monkeypatch.setattr(envinfo, "_backend_openai", lambda url, model: {
        "kind": "llama.cpp", "base_url": url, "model_path": "pack/full",
        "model_file": {"path": "pack/full"}})
    cfg = {"type": "openai", "base_url": "http://localhost:8082/v1",
           "runtime": "Strata/CUDA"}
    b = envinfo.collect(cfg, "deepseek-v4-flash")["backend"]
    assert b["kind"] == "Abyss"
    assert b["server_version"] == "Abyss 1.1.0 (2b68887)"
    assert b["model_path"] == "/data/UD-Q2_K_XL/pack-dsv4" and "model_file" not in b
    assert b["quantization"] == "Q2_K_XL"
    assert seen["pid"] == 4242
    assert b["runtime"]["compute"] == "Strata/CUDA" and not b["runtime"].get("mismatch")
    assert "uncertain" not in b["gpu_usage"]["inference"]
    assert "note" not in b["gpu_usage"]          # it is CUDA, so keep the GPU attribution


# ── Detection through /health ─────────────────────────────────────────────

_STRATA_HEALTH = {"status": "ok", "max_context": 131072, "model": "qwen3.8-flash-next-iq3_s",
                  "images": False, "api_key": False, "loaded": True, "service": "strata"}
_MAYA_HEALTH = {"status": "ok", "max_context": 65536, "model": "glm-5.3-flash", "images": False,
                "api_key": False, "defaults": {"temperature": 1.0}, "version": "1.0.24",
                "mcp_tools": False}
_ABYSS_HEALTH = {**_MAYA_HEALTH, "model": "deepseek-v4-flash", "version": "1.1.1",
                 "service": "abyss"}


@pytest.mark.parametrize("health,family,version", [
    (_STRATA_HEALTH, "Strata", None),
    (_ABYSS_HEALTH, "Abyss", "1.1.1"),
    (_MAYA_HEALTH, "Maya", "1.0.24"),          # no service + version/mcp_tools = the Maya shape
    ({"status": "ok"}, None, None),            # llama.cpp /health
    (None, None, None),
])
def test_health_family(health, family, version):
    fam = envinfo.health_family(health)
    assert fam.get("family") == family
    assert fam.get("version") == version


class _HealthHandler(_Handler):
    health: dict = {}

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            b = json.dumps(self.health).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        super().do_GET()


@pytest.mark.parametrize("health,kind", [
    (_ABYSS_HEALTH, "Abyss"), (_MAYA_HEALTH, "Maya"), (_STRATA_HEALTH, "Strata"),
    ({"status": "ok"}, "llama.cpp"),
])
def test_backend_kind_from_health(health, kind):
    handler = type("H", (_HealthHandler,), {"health": health})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = envinfo.collect_backend(
            {"type": "openai", "base_url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "auto"},
            served_model="m")
    finally:
        srv.shutdown()
    assert b["kind"] == kind
    if health.get("version"):
        assert b["server_version"] == f"{kind} {health['version']}"


def test_folder_family_corrects_a_maya_guess_from_old_abyss(monkeypatch):
    """The /health of Abyss v1.1.0 and older has the Maya shape: trust the folder-side detection (Abyss)."""
    monkeypatch.setattr(envinfo, "find_server_pid", lambda port: 1000)
    monkeypatch.setattr(envinfo, "strata_family",
                        lambda pid: {"family": "Abyss", "version": "1.1.0", "engine_pid": 4242})
    monkeypatch.setattr(envinfo, "detect_runtime", lambda pid: {"compute": "CUDA"})
    monkeypatch.setattr(envinfo, "collect_launch", lambda pid, gpus=None: {})
    monkeypatch.setattr(envinfo, "collect_gpu_usage", lambda: {})
    monkeypatch.setattr(envinfo, "_backend_openai",
                        lambda url, model: {"kind": "Maya", "base_url": url})
    b = envinfo.collect({"type": "openai", "base_url": "http://localhost:8082/v1"})["backend"]
    assert b["kind"] == "Abyss"
    # the kind reported by /health wins over the folder name
    monkeypatch.setattr(envinfo, "strata_family",
                        lambda pid: {"family": "Maya", "engine_pid": 4242})
    monkeypatch.setattr(envinfo, "_backend_openai",
                        lambda url, model: {"kind": "Abyss", "base_url": url})
    b = envinfo.collect({"type": "openai", "base_url": "http://localhost:8082/v1"})["backend"]
    assert b["kind"] == "Abyss" and b["strata_family"]["folder_family"] == "Maya"

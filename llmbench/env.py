"""Metadata collection for the execution environment (hardware + backend setup).

Why this is needed:
    You cannot compare the ``tokens_per_sec`` in results.json without knowing the machine,
    the quantization, and how much of the model sat on the GPU. Even on the same GPU it
    changes by several times with quantization, GPU offload ratio and context length, so
    specs alone are not enough: the backend setup has to be recorded as well.

Approach:
    - Everything is best-effort: fields that cannot be collected become None / are
      omitted, and **exceptions never escape** (a failure here never stops the bench).
    - No new dependencies: standard library + requests, which is already required.
    - Never include secrets such as api_key.
    - Host specs do not affect throughput for cloud API models, so
      ``execution`` states explicitly that this is not local inference.

Output schema (the "environment" key of results.json):
    {
      "execution": "local" | "remote-api" | "subscription-cli" | "mock",
      "note": "…human-readable note…",
      "host": {"os", "arch", "python", "cpu", "cpu_cores", "ram_gb", "gpu": [...]},
      "backend": {"kind", "base_url", "model", "quantization", ...}
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

__all__ = [
    "EXEC_LABEL", "collect", "collect_host", "collect_backend",
    "collect_gpu_usage", "detect_runtime", "find_server_pid", "format_summary",
    "health_family", "model_file_info", "sampler_defaults", "strata_family",
]

# Human-readable label for execution (shared by report / compare)
EXEC_LABEL = {
    "local": "🖥 Local inference",
    "remote-api": "☁️ Cloud API",
    "subscription-cli": "☁️ Subscription CLI (inference runs vendor-side)",
    "mock": "🧪 Mock",
    "unknown": "❔ Unknown",
}

_CMD_TIMEOUT = 6.0
_HTTP_TIMEOUT = 3.0

# Host names treated as local inference (the host part of base_url)
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


def _sh(cmd: list[str], timeout: float = _CMD_TIMEOUT) -> str | None:
    """Run a command and return stdout. None on failure (never raises)."""
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _sysctl(key: str) -> str | None:
    return _sh(["sysctl", "-n", key])


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _gb(nbytes) -> float | None:
    n = _int(nbytes)
    return round(n / (1024 ** 3), 1) if n else None


def _get_json(url: str, timeout: float = _HTTP_TIMEOUT):
    """GET and return JSON. None on failure (works without requests installed)."""
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code >= 400:
            return None
        return resp.json()
    except Exception:
        return None


def _port_of(base_url: str) -> int | None:
    try:
        from urllib.parse import urlparse

        return urlparse(base_url).port
    except Exception:
        return None


def _is_local_url(base_url: str) -> bool:
    try:
        from urllib.parse import urlparse

        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    return host in _LOCAL_HOSTS or host.endswith(".local")


# ─────────────────────────── Host (hardware) ───────────────────────────


def _gpu_macos() -> list[dict]:
    """Read GPU info from system_profiler (Apple Silicon and external GPUs alike)."""
    raw = _sh(["system_profiler", "-json", "SPDisplaysDataType"], timeout=15.0)
    if not raw:
        return []
    try:
        items = json.loads(raw).get("SPDisplaysDataType") or []
    except Exception:
        return []
    gpus = []
    for it in items:
        g: dict = {"name": it.get("sppci_model") or it.get("_name")}
        # Apple Silicon: "sppci_cores" holds the GPU core count (e.g. "40")
        cores = _int(it.get("sppci_cores"))
        if cores:
            g["cores"] = cores
        # Dedicated VRAM (discrete GPUs only; Apple Silicon has unified memory, so none)
        vram = it.get("spdisplays_vram") or it.get("spdisplays_vram_shared")
        if vram:
            g["vram"] = str(vram)
        # Metal support generation. system_profiler returns the internal key name
        # as-is ("spdisplays_metal4"), so normalise it to the readable "Metal 4".
        mtl = it.get("spdisplays_mtlgpufamilysupport")
        if mtl:
            m = re.search(r"metal[_ ]?(\d+)", str(mtl), re.I)
            g["metal"] = f"Metal {m.group(1)}" if m else str(mtl)
        gpus.append({k: v for k, v in g.items() if v})
    return gpus


def _gpu_nvidia() -> list[dict]:
    """Read GPU info from nvidia-smi (NVIDIA machines only)."""
    out = _sh([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        g: dict = {"name": parts[0], "vendor": "nvidia"}
        if len(parts) > 1 and _int(parts[1]):
            g["vram_gb"] = round(_int(parts[1]) / 1024, 1)  # MiB → GiB
        if len(parts) > 2 and parts[2]:
            g["driver"] = parts[2]
        if len(parts) > 3 and parts[3]:
            g["compute_capability"] = parts[3]
        gpus.append(g)
    return gpus


# Process names treated as inference servers (used to tell which GPU ran)
_INFERENCE_PROCS = (
    "llama-server", "llama_server", "llama-cli", "ollama", "vllm", "sglang",
    "text-generation", "tabbyapi", "exllama", "koboldcpp", "python",
)


def _nvidia_gpu_memory() -> dict[str, dict]:
    """GPU UUID -> {index, name, vram_used_gb, vram_total_gb} (nvidia-smi)."""
    out = _sh([
        "nvidia-smi",
        "--query-gpu=uuid,index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    gpus: dict[str, dict] = {}
    for line in (out or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5 or not p[0]:
            continue
        gpus[p[0]] = {
            "index": _int(p[1]),
            "name": p[2],
            "vram_used_gb": round(_int(p[3]) / 1024, 1) if _int(p[3]) else 0.0,
            "vram_total_gb": round(_int(p[4]) / 1024, 1) if _int(p[4]) else None,
        }
    return gpus


def collect_gpu_usage() -> dict:
    """Decide which GPU ran the inference from an nvidia-smi snapshot.

    On a multi-GPU machine tok/s means nothing unless you know whether it was
    measured on the 3090 or on the 5090. llama.cpp keeps no GPU offload info in
    ``/props``, so the VRAM a process occupies is the practical proxy for it.

    Called once after the run (model still loaded). Returns {} when unavailable.
    """
    gpus = _nvidia_gpu_memory()
    if not gpus:
        return {}
    out = _sh([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    procs = []
    for line in (out or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4 or p[0] not in gpus:
            continue
        used = _int(p[3])
        g = gpus[p[0]]
        procs.append({
            "gpu_index": g["index"],
            "gpu_name": g["name"],
            "pid": _int(p[1]),
            "process": str(p[2]).split("/")[-1],
            "vram_gb": round(used / 1024, 1) if used else 0.0,
        })
    usage: dict = {}
    if procs:
        usage["processes"] = procs
        # The same PID on several GPUs = the model is split-loaded by tensor split.
        # Record how it was split, not just which GPU ran the inference.
        infer = [p for p in procs
                 if any(k in p["process"].lower() for k in _INFERENCE_PROCS)]
        by_pid: dict[int | None, list[dict]] = {}
        for p in (infer or procs):
            by_pid.setdefault(p["pid"], []).append(p)
        # The PID with the largest total VRAM is taken as the inference process
        pid, shards = max(
            by_pid.items(), key=lambda kv: sum(s["vram_gb"] for s in kv[1])
        )
        shards = sorted(shards, key=lambda s: s["gpu_index"] or 0)
        total = sum(s["vram_gb"] for s in shards)
        # GPUs that were not selected also carry a few hundred MiB of CUDA/HIP context
        # (real machine: --device CUDA0, yet 256MiB landed on the 3090). Counting that
        # as a split load would be a false alarm, so flag tiny shares as context only.
        floor = max(0.5, total * 0.05)
        shard_gpus = [
            {"index": s["gpu_index"], "name": s["gpu_name"],
             "vram_gb": s["vram_gb"],
             **({"context_only": True} if s["vram_gb"] < floor else {})}
            for s in shards
        ]
        inference = {
            "pid": pid,
            "process": shards[0]["process"],
            "vram_total_gb": round(total, 1),
            "multi_gpu": sum(
                1 for g in shard_gpus if not g.get("context_only")) > 1,
            "gpus": shard_gpus,
        }
        if not infer:
            # The name matches no known inference server = it may serve another purpose
            inference["uncertain"] = True
        usage["inference"] = inference
    # Usage of every GPU (a clue even when the inference process is unknown)
    usage["gpus"] = [
        {"index": g["index"], "name": g["name"],
         "vram_used_gb": g["vram_used_gb"], "vram_total_gb": g["vram_total_gb"]}
        for g in sorted(gpus.values(), key=lambda g: g["index"] or 0)
    ]
    return usage


# ─────────── Compute backend (CUDA / ROCm / Vulkan / Metal) detection ───────────
#
# llama.cpp `/props` returns only build_info (e.g. "b10157-c6292cfb8");
# **the API never tells which backend the build used**. Yet tok/s changes a lot
# between CUDA / ROCm / Vulkan, so leaving this blank makes results incomparable.
# So infer it from the shared libraries the inference process has loaded.
# The order matters: a CUDA/ROCm/SYCL build that indirectly loads libvulkan
# must not be read as Vulkan, so check the dedicated runtimes first.
_RUNTIME_LIBS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CUDA", ("libggml-cuda", "libcudart", "libcublas")),
    ("ROCm", ("libggml-hip", "libamdhip64", "librocblas", "libhipblas")),
    ("SYCL", ("libggml-sycl", "libsycl", "libmkl_sycl")),
    ("Vulkan", ("libggml-vulkan", "libvulkan")),
    ("Metal", ("libggml-metal",)),
    ("BLAS", ("libggml-blas", "libopenblas", "libmkl_rt")),
)
# Inference server process names (matched by executable name)
_SERVER_NAMES = ("llama-server", "llama_server", "ollama", "vllm", "sglang",
                 "koboldcpp", "tabbyapi")


def _proc_cmdlines() -> list[tuple[int, str]]:
    """List of (pid, cmdline). Taken from /proc on Linux, from ps on macOS."""
    out: list[tuple[int, str]] = []
    if sys.platform.startswith("linux"):
        try:
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    with open(f"/proc/{name}/cmdline", "rb") as f:
                        cmd = f.read().replace(b"\x00", b" ").decode(
                            "utf-8", "replace"
                        ).strip()
                except Exception:
                    continue
                if cmd:
                    out.append((int(name), cmd))
        except Exception:
            return []
        return out
    # macOS / others
    ps = _sh(["ps", "-Ao", "pid=,args="])
    for line in (ps or "").splitlines():
        line = line.strip()
        pid, _, cmd = line.partition(" ")
        if pid.isdigit() and cmd:
            out.append((int(pid), cmd.strip()))
    return out


def _listening_pids(port: int) -> list[int]:
    """PIDs LISTENing on that port (from /proc on Linux; empty on other systems).

    It does not rely on process names, so it also catches Strata / Maya / Abyss,
    served by python server.py. Only fds of my own (same user) processes are readable.
    """
    if not sys.platform.startswith("linux") or not port:
        return []
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii", errors="ignore") as f:
                next(f, None)
                for line in f:
                    parts = line.split()
                    if len(parts) < 10 or parts[3] != "0A":   # 0A = LISTEN
                        continue
                    if int(parts[1].rsplit(":", 1)[-1], 16) == port:
                        inodes.add(parts[9])
        except Exception:
            continue
    if not inodes:
        return []
    want = {f"socket:[{i}]" for i in inodes}
    pids: list[int] = []
    try:
        names = os.listdir("/proc")
    except Exception:
        return []
    for name in names:
        if not name.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{name}/fd"):
                try:
                    if os.readlink(f"/proc/{name}/fd/{fd}") in want:
                        pids.append(int(name))
                        break
                except Exception:
                    continue
        except Exception:
            continue
    return sorted(pids)


def find_server_pid(port: int | None = None) -> int | None:
    """Find the PID of the inference server running locally.

    It assumes builds get swapped on the same port, so **a port match wins**
    (straight after you stop the CUDA build and start the ROCm one it takes the live one).
    It first looks at the process LISTENing on that port (the process name is ignored).
    When a port is given and nothing matches, **never return a server on another port**
    (it used to return the first candidate: a measurement of Abyss on 8082 recorded the
    start command of the ollama on 11434).
    """
    if port:
        listening = _listening_pids(port)
        if listening:
            return listening[0]
    cands = [(pid, cmd) for pid, cmd in _proc_cmdlines()
             if any(n in cmd for n in _SERVER_NAMES)]
    if not cands:
        return None
    if port:
        for pid, cmd in cands:
            if f"--port {port}" in cmd or f"--port={port}" in cmd or \
                    f":{port}" in cmd:
                return pid
        return None
    return cands[0][0]


# ── Detect the Strata family (Strata / Maya / Abyss) ──────────────────────────
#
# All three serve an OpenAI-compatible API from python serve/server.py and start
# the engine (build/strata) as a child process. /props answers in llama.cpp
# compatible form and model_path is only "pack/full", so HTTP alone cannot tell
# them from llama.cpp (Maya and Abyss share the same /health shape). So identify
# them from the server.py folder: git origin, VERSION, and the --config JSON.
_STRATA_FAMILY = (            # (word found in origin / folder name, display name)
    ("abyss", "Abyss"),
    ("project-maya", "Maya"),
    ("maya", "Maya"),
    ("strata", "Strata"),
)


def _proc_ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="ignore") as f:
            stat = f.read()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except Exception:
        return None


def _proc_cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return ""


def _child_pids(pid: int) -> list[int]:
    out: list[int] = []
    try:
        names = os.listdir("/proc")
    except Exception:
        return out
    for name in names:
        if name.isdigit() and _proc_ppid(int(name)) == pid:
            out.append(int(name))
    return sorted(out)


def _git_origin(root: str) -> str | None:
    """The origin URL in root/.git/config (no git command is used)."""
    try:
        with open(os.path.join(root, ".git", "config"), encoding="utf-8",
                  errors="ignore") as f:
            text = f.read()
    except Exception:
        return None
    m = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*(\S+)', text, re.DOTALL)
    return m.group(1) if m else None


def _git_head(root: str) -> str | None:
    """The short HEAD commit of root. packed-refs is checked as well."""
    try:
        with open(os.path.join(root, ".git", "HEAD"), encoding="ascii",
                  errors="ignore") as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = os.path.join(root, ".git", ref)
            if os.path.exists(path):
                with open(path, encoding="ascii", errors="ignore") as f:
                    return f.read().strip()[:7]
            with open(os.path.join(root, ".git", "packed-refs"), encoding="ascii",
                      errors="ignore") as f:
                for line in f:
                    if line.rstrip().endswith(" " + ref):
                        return line.split()[0][:7]
            return None
        return head[:7] or None
    except Exception:
        return None


def strata_family(pid: int | None) -> dict:
    """If pid is a Strata-family server.py: {family, version, root, engine_pid, ...}.

    {} otherwise. Never raises.
    """
    if not pid:
        return {}
    argv = _proc_argv(pid)
    script = next((a for a in argv if a.replace("\\", "/").endswith("serve/server.py")),
                  None)
    if not script:
        return {}
    cwd = _proc_cwd(pid)
    script = os.path.normpath(os.path.join(cwd, script))
    root = os.path.dirname(os.path.dirname(script))
    out: dict = {"root": root}
    origin = _git_origin(root)
    if origin:
        out["origin"] = origin
    hay = f"{origin or ''} {os.path.basename(root)}".lower()
    for key, label in _STRATA_FAMILY:
        if key in hay:
            out["family"] = label
            break
    else:
        out["family"] = "Strata-family"
    try:
        with open(os.path.join(root, "VERSION"), encoding="utf-8") as f:
            ver = f.read().strip()
        if ver:
            out["version"] = ver
    except Exception:
        pass
    head = _git_head(root)
    if head:
        out["commit"] = head
    # --config JSON: engine executable, pack (= model) and context length
    cfg_path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
        elif a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
    if cfg_path:
        cfg_path = os.path.normpath(os.path.join(cwd, cfg_path))
        out["config"] = cfg_path
        try:
            with open(cfg_path, encoding="utf-8") as f:
                rc = json.load(f)
        except Exception:
            rc = {}
        if isinstance(rc, dict):
            args = [str(a) for a in rc.get("args") or []]
            for flag in ("--glm-pack", "--pack", "--model", "-m"):
                if flag in args and args.index(flag) + 1 < len(args):
                    out["model_path"] = args[args.index(flag) + 1]
                    break
            if "--max-context" in args and args.index("--max-context") + 1 < len(args):
                n = _int(args[args.index("--max-context") + 1])
                if n:
                    out["n_ctx"] = n
            if rc.get("exe"):
                out["engine_exe"] = str(rc["exe"])
            if rc.get("model_name"):
                out["model_name"] = str(rc["model_name"])
    # Engine (child process). The libraries it loads show the compute backend
    exe_base = os.path.basename(out.get("engine_exe") or "strata")
    for c in _child_pids(pid):
        if os.path.basename(_proc_exe(c) or "") == exe_base:
            out["engine_pid"] = c
            break
    return out


def _family_label(fam: dict) -> str:
    """A display string such as 'Abyss 1.1.0 (2b68887)'."""
    s = str(fam.get("family") or "Strata-family")
    if fam.get("version"):
        s += f" {fam['version']}"
    if fam.get("commit"):
        s += f" ({fam['commit']})"
    return s


def _proc_exe(pid: int) -> str | None:
    """Real path of the executable that /proc/<pid>/exe points to (None if absent)."""
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except Exception:
        return None


def _proc_maps(pid: int) -> str | None:
    """Contents of /proc/<pid>/maps (no root needed for my own process, else None)."""
    try:
        with open(f"/proc/{pid}/maps", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def detect_runtime(pid: int | None) -> dict:
    """Detect the compute backend from the libraries the inference process loaded.

    ``/proc/<pid>/maps`` is readable without root for a process that I own.
    It works for builds that split ggml into per-backend .so files (libggml-cuda.so)
    and for statically linked builds (where libcudart / libamdhip64 show directly).
    """
    if not pid:
        return {}
    info: dict = {}
    exe = _proc_exe(pid)
    if exe:
        info["binary"] = exe
    libs: set[str] = set()
    for line in (_proc_maps(pid) or "").splitlines():
        path = line.rstrip("\n").split(" ", 5)[-1].strip()
        if path.startswith("/") and ".so" in path:
            libs.add(path.rsplit("/", 1)[-1])
    if libs:
        hits = []
        for label, keys in _RUNTIME_LIBS:
            matched = sorted({lib for lib in libs
                              for k in keys if lib.startswith(k)})
            if matched:
                hits.append((label, matched))
        if hits:
            info["compute"] = hits[0][0]
            info["evidence"] = hits[0][1]
            if len(hits) > 1:
                info["also_loaded"] = [h[0] for h in hits[1:]]
        else:
            info["compute"] = "CPU"
    # Even when the libraries are unreadable, the build directory name is a strong clue
    binpath = (info.get("binary") or "").lower()
    for label, key in (("CUDA", "cuda"), ("ROCm", "rocm"), ("ROCm", "hip"),
                       ("Vulkan", "vulkan"), ("SYCL", "sycl"),
                       ("Metal", "metal")):
        if key in binpath:
            info.setdefault("compute", label)
            info["binary_hint"] = label
            break
    return info


# ── Read the inference setup from the start-up arguments ──────────────────────
#
# llama.cpp /props returns only n_ctx and build_info: **neither the GPU offload
# (-ngl) nor the device in use (--device) can be read from it**. The start-up
# arguments list it all, and CUDA / ROCm / Vulkan read the same way, so this is
_ARG_VALUE_KEYS = {
    "-ngl": "n_gpu_layers", "--n-gpu-layers": "n_gpu_layers",
    "--gpu-layers": "n_gpu_layers",
    "-c": "n_ctx", "--ctx-size": "n_ctx",
    "-t": "threads", "--threads": "threads",
    "-ts": "tensor_split", "--tensor-split": "tensor_split",
    "-mg": "main_gpu", "--main-gpu": "main_gpu",
    "-sm": "split_mode", "--split-mode": "split_mode",
    "-dev": "device", "--device": "device",
    "-np": "parallel", "--parallel": "parallel",
    "-b": "batch_size", "--batch-size": "batch_size",
    "-ub": "ubatch_size", "--ubatch-size": "ubatch_size",
    "--spec-type": "spec_type",
    "-md": "draft_model", "--model-draft": "draft_model",
    # Sampling settings. They drive **reproducibility of the result**, not tok/s.
    # A llama-server without --seed runs with seed=-1 (random every time), so to say
    # "measured under the same conditions" these values have to be recorded.
    "--temp": "temp", "--temperature": "temp",
    "--top-p": "top_p", "--top-k": "top_k", "--min-p": "min_p",
    "--seed": "seed",
    # KV cache quantization. It affects quality and speed, but /props does not show it.
    "-ctk": "cache_type_k", "--cache-type-k": "cache_type_k",
    "-ctv": "cache_type_v", "--cache-type-v": "cache_type_v",
}
_ARG_FLAGS = {
    "--mlock": "mlock", "--no-mmap": "no_mmap",
    "-fa": "flash_attn", "--flash-attn": "flash_attn",
    "-cb": "cont_batching", "--cont-batching": "cont_batching",
}
_INT_ARGS = ("n_gpu_layers", "n_ctx", "threads", "parallel", "main_gpu",
             "batch_size", "ubatch_size", "top_k", "seed")
# Secrets that can appear in start-up args (results.json gets shared, so redact)
_SECRET_ARGS = ("key", "token", "secret", "password", "passwd")
# Env vars that limit visible devices (with --device the real mapping shifts)
_VISIBLE_ENV = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL")


def parse_server_args(argv: list[str]) -> dict:
    """Extract the inference setup from llama-server argv (secrets are redacted)."""
    out: dict = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _ARG_FLAGS:
            out[_ARG_FLAGS[tok]] = True
            i += 1
            continue
        key = _ARG_VALUE_KEYS.get(tok)
        if key and i + 1 < len(argv):
            out[key] = argv[i + 1]
            i += 2
            continue
        i += 1
    for k in _INT_ARGS:
        if k in out and _int(out[k]) is not None:
            out[k] = _int(out[k])
    return out


def _redact_argv(argv: list[str]) -> str:
    """Keep the start-up command on one line for reproduction; --api-key values redacted."""
    safe: list[str] = []
    redact_next = False
    for tok in argv:
        if redact_next:
            safe.append("***")
            redact_next = False
            continue
        low = tok.lower()
        if low.startswith("-") and any(s in low for s in _SECRET_ARGS):
            if "=" in tok:
                safe.append(tok.split("=", 1)[0] + "=***")
                continue
            safe.append(tok)
            redact_next = True
            continue
        safe.append(tok)
    return " ".join(safe)


def _proc_argv(pid: int) -> list[str]:
    """Split /proc/<pid>/cmdline on NUL, so paths containing spaces stay intact."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except Exception:
        return []
    return [a for a in raw.decode("utf-8", "replace").split("\x00") if a]


def _proc_visible_env(pid: int) -> dict:
    """Collect only the visible-device env vars (other env vars can hold secrets)."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except Exception:
        return {}
    env = {}
    for item in raw.decode("utf-8", "replace").split("\x00"):
        k, _, v = item.partition("=")
        if k in _VISIBLE_ENV and v:
            env[k] = v
    return env


def _vulkan_device_names() -> list[str]:
    """Device names in vulkaninfo --summary order (used to resolve Vulkan<N>)."""
    out = _sh(["vulkaninfo", "--summary"], timeout=20.0) or ""
    return [m.group(1).strip()
            for m in re.finditer(r"deviceName\s*=\s*(.+)", out)]


# One line of llama.cpp --list-devices. The form is the same for every ggml backend:
#   "  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31610 MiB free)"
# Names can contain parentheses ("AMD Radeon Graphics (RADV GFX1151)"), so the
# memory parentheses are anchored at the end of the line and the name is matched lazily.
_LIST_DEVICES_RE = re.compile(
    r"^\s*(\S+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$"
)


def list_devices(binary: str) -> list[dict]:
    """Map device ID -> name with ``<llama-server> --list-devices``.

    **This is the only correct source.** The CUDA default device order is
    ``FASTEST_FIRST``, not the nvidia-smi PCI bus order (real machine: nvidia-smi
    GPU0=RTX 3090 while CUDA0=RTX 5090), so indexing a per-vendor list flips it.
    ROCm / Vulkan use per-build namespaces too, so asking the executable is surest.
    """
    out = _sh([binary, "--list-devices"], timeout=15.0)
    devices = []
    for line in (out or "").splitlines():
        m = _LIST_DEVICES_RE.match(line)
        if not m:
            continue
        devices.append({
            "id": m.group(1),
            "name": m.group(2),
            "vram_total_gb": round(int(m.group(3)) / 1024, 1),
            "vram_free_gb": round(int(m.group(4)) / 1024, 1),
        })
    return devices


def resolve_device(
    dev: str,
    host_gpus: list[dict],
    binary: str | None = None,
    devices: list[dict] | None = None,
) -> dict:
    """Resolve ``ROCm0`` / ``Vulkan2`` / ``CUDA0,CUDA1`` to real GPU names.

    The ``--list-devices`` result (``devices``, taken from ``binary`` if missing) is
    the source of truth. CUDA / ROCm / Vulkan all resolve through the same path, so
    no per-backend branch is needed. Only when it is missing, estimate with the index
    into a per-vendor list and set ``device_name_uncertain`` (the CUDA default order
    is FASTEST_FIRST, which differs from the nvidia-smi PCI order, so it can be wrong).
    """
    dev = (dev or "").strip()
    if not dev:
        return {}
    ids = [d.strip() for d in dev.split(",") if d.strip()]
    info: dict = {"device": dev}

    if devices is None:
        devices = list_devices(binary) if binary else []
    table = {d["id"]: d for d in devices}
    resolved: list[dict] = []
    uncertain = False
    for did in ids:
        if did in table:
            resolved.append({"id": did, "name": table[did]["name"],
                             "vram_total_gb": table[did]["vram_total_gb"]})
            continue
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", did)
        if not m:
            continue
        kind, idx = m.group(1).lower(), int(m.group(2))
        name = None
        if kind == "vulkan":
            names = _vulkan_device_names()
            name = names[idx] if idx < len(names) else None
        if name is None:
            # Fallback: index into the per-vendor list. CUDA is in FASTEST_FIRST order,
            # so it can differ from a PCI-ordered list -> always mark the name uncertain.
            vendor = "nvidia" if kind == "cuda" else (
                "amd" if kind in ("rocm", "hip") else None)
            pool = ([g for g in host_gpus if g.get("vendor") == vendor]
                    if vendor else host_gpus)
            name = pool[idx].get("name") if idx < len(pool) else None
            uncertain = name is not None
        if name:
            resolved.append({"id": did, "name": name})
    if resolved:
        info["devices"] = resolved
        info["device_name"] = " + ".join(d["name"] for d in resolved)
        if uncertain:
            info["device_name_uncertain"] = True
        elif table:
            info["device_name_source"] = "--list-devices"
    return info


def collect_launch(pid: int | None, host_gpus: list[dict] | None = None) -> dict:
    """Collect the inference server start-up setup (device, -ngl, n_ctx, ...)."""
    if not pid:
        return {}
    argv = _proc_argv(pid)
    if not argv:
        return {}
    launch = parse_server_args(argv)
    launch["command"] = _redact_argv(argv)
    # Ask the executable that was started for the device list - it is authoritative
    # (CUDA0 is not necessarily nvidia-smi GPU0, and ID namespaces differ per build).
    # CUDA / ROCm / Vulkan share this path, so no per-backend branch is needed.
    devices = list_devices(_proc_exe(pid) or argv[0])
    if devices:
        launch["available_devices"] = devices
    dev = launch.pop("device", None)
    if dev:
        launch.update(resolve_device(str(dev), host_gpus or [],
                                     devices=devices))
    env = _proc_visible_env(pid)
    if env:
        launch["visible_devices_env"] = env
    return launch


def _gpu_amd() -> list[dict]:
    """Enumerate AMD GPUs (needed when ROCm / Vulkan use an iGPU or Radeon).

    Looking only at nvidia-smi leaves the GPU field empty for a run that uses Radeon
    through ROCm or Vulkan. Use rocm-smi when it exists, otherwise use lspci.
    """
    gpus: list[dict] = []
    csv = _sh(["rocm-smi", "--showproductname", "--csv"], timeout=15.0)
    if csv:
        lines = [ln for ln in csv.splitlines() if ln.strip()]
        if len(lines) >= 2:
            head = [h.strip().lower() for h in lines[0].split(",")]
            for ln in lines[1:]:
                cells = [c.strip() for c in ln.split(",")]
                row = dict(zip(head, cells))
                name = (row.get("card series") or row.get("card model")
                        or row.get("card sku") or "")
                if name and not name.lower().startswith("n/a"):
                    gpus.append({"name": name, "vendor": "amd"})
    if gpus:
        mem = _sh(["rocm-smi", "--showmeminfo", "vram", "--csv"], timeout=15.0)
        for ln in (mem or "").splitlines()[1:]:
            cells = [c.strip() for c in ln.split(",")]
            tot = next((_int(c) for c in cells[1:] if _int(c)), None)
            idx = len([g for g in gpus if "vram_gb" in g])
            if tot and idx < len(gpus):
                gpus[idx]["vram_gb"] = round(tot / (1024 ** 3), 1)
        return gpus
    # Fallback for environments without rocm-smi
    out = _sh(["lspci", "-mm"], timeout=10.0) or ""
    for line in out.splitlines():
        if not any(k in line for k in ("VGA compatible", "Display controller",
                                       "3D controller")):
            continue
        if "AMD" not in line and "ATI" not in line:
            continue
        # The -mm order is class, vendor, device, subvendor, subdevice.
        # The device name (fields[2]) is the product name, so use that.
        fields = re.findall(r'"([^"]*)"', line)
        if len(fields) >= 3 and fields[2]:
            gpus.append({"name": fields[2], "vendor": "amd"})
    return gpus


def _cuda_version() -> str | None:
    out = _sh(["nvidia-smi", "--query", "--display=COMPUTE"]) or ""
    m = re.search(r"CUDA Version\s*:\s*([\d.]+)", out)
    if m:
        return m.group(1)
    out2 = _sh(["nvcc", "--version"]) or ""
    m2 = re.search(r"release ([\d.]+)", out2)
    return m2.group(1) if m2 else None


def _host_macos() -> dict:
    h: dict = {
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "cpu_cores": _int(_sysctl("hw.ncpu")),
        "ram_gb": _gb(_sysctl("hw.memsize")),
    }
    # On Apple Silicon the P/E core split is useful when comparing machines
    p = _int(_sysctl("hw.perflevel0.physicalcpu"))
    e = _int(_sysctl("hw.perflevel1.physicalcpu"))
    if p:
        h["cpu_cores_perf"] = p
    if e:
        h["cpu_cores_eff"] = e
    h["gpu"] = _gpu_macos()
    h["unified_memory"] = platform.machine() == "arm64"
    return h


def _host_linux() -> dict:
    h: dict = {"cpu": None, "cpu_cores": None, "ram_gb": None}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if m:
            h["cpu"] = m.group(1).strip()
        h["cpu_cores"] = text.count("\nprocessor") + (
            1 if text.startswith("processor") else 0
        ) or None
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="ignore") as f:
            m = re.search(r"^MemTotal:\s*(\d+) kB", f.read(), re.M)
        if m:
            h["ram_gb"] = round(int(m.group(1)) / (1024 ** 2), 1)
    except Exception:
        pass
    # Watching only NVIDIA leaves the GPU field empty for runs that use Radeon
    # (iGPU included) through ROCm/Vulkan. Enumerate the AMD side as well.
    h["gpu"] = _gpu_nvidia() + _gpu_amd()
    cuda = _cuda_version()
    if cuda:
        h["cuda"] = cuda
    return h


def collect_host() -> dict:
    """Return host hardware/OS info. Even on failure the basic keys are filled in."""
    host: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        if sys.platform == "darwin":
            host["os"] = f"macOS {platform.mac_ver()[0] or platform.release()}"
            host.update(_host_macos())
        elif sys.platform.startswith("linux"):
            host.update(_host_linux())
        else:
            host["cpu"] = platform.processor() or None
            host["gpu"] = _gpu_nvidia()
    except Exception as e:  # a collection failure must not stop the benchmark
        host["collect_error"] = f"{type(e).__name__}: {e}"
    return {k: v for k, v in host.items() if v not in (None, [], "")}


# ─────────────────────── Inference backend (measured setup) ───────────────────────


def _backend_ollama(base_url: str, model: str | None) -> dict:
    """Read the loaded model's real setup from Ollama /api/ps and /api/show.

    size_vram / size in /api/ps give the GPU offload ratio, which drives tok/s.
    """
    b: dict = {"kind": "ollama", "base_url": base_url}
    url = base_url.rstrip("/")
    ps = _get_json(f"{url}/api/ps") or {}
    loaded = ps.get("models") or []
    entry = None
    if model:
        entry = next((m for m in loaded if m.get("name") == model
                      or m.get("model") == model), None)
    entry = entry or (loaded[0] if loaded else None)
    if entry:
        b["model"] = entry.get("name") or entry.get("model")
        size = _int(entry.get("size"))
        vram = _int(entry.get("size_vram"))
        if size:
            b["weights_gb"] = round(size / (1024 ** 3), 1)
        if vram is not None and size:
            b["vram_resident_gb"] = round(vram / (1024 ** 3), 1)
            b["gpu_offload_ratio"] = round(vram / size, 3)
        det = entry.get("details") or {}
        for src, dst in (("quantization_level", "quantization"),
                         ("parameter_size", "parameter_size"),
                         ("family", "family")):
            if det.get(src):
                b[dst] = det[src]
        ctx = _int(entry.get("context_length"))
        if ctx:
            b["n_ctx"] = ctx
    elif model:
        b["model"] = model
        b["note"] = (
            "No loaded model (/api/ps is empty) — the model is not loaded, "
            "or every generation failed"
        )
    ver = _get_json(f"{url}/api/version") or {}
    if ver.get("version"):
        b["server_version"] = f"ollama {ver['version']}"
    return b


# ── Model file fingerprint / sampler defaults ─────────────────────────────

# Hashing the whole file is impractical for a 20GB-class gguf (tens of seconds+).
# Reading 1MB from the head and 1MB from the tail catches a re-quantized, re-downloaded
# or swapped build almost every time, because the header and tensor tail change.
_FINGERPRINT_CHUNK = 1024 * 1024

_SAMPLER_KEYS = (
    "temperature", "temp", "top_p", "top_k", "min_p", "typical_p",
    "seed", "repeat_penalty", "presence_penalty", "frequency_penalty",
    "dynatemp_range", "dynatemp_exponent", "mirostat", "n_predict",
)


def model_file_info(path) -> dict | None:
    """Identification data for a model file (size / mtime / head+tail hash).

    Kept so a later reader can tell that the content behind the same file name changed.
    When it cannot be read (remote run, no permission, file deleted) no exception is
    raised; a dict filled with None is returned (the path alone is worth recording).
    """
    if not path:
        return None
    info: dict = {
        "path": str(path),
        "size_bytes": None,
        "mtime": None,
        "sha256_head_tail": None,
    }
    try:
        st = os.stat(path)
        info["size_bytes"] = st.st_size
        info["mtime"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)
        )
    except OSError:
        return info
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(_FINGERPRINT_CHUNK))
            if st.st_size > _FINGERPRINT_CHUNK:
                f.seek(max(0, st.st_size - _FINGERPRINT_CHUNK))
                h.update(f.read(_FINGERPRINT_CHUNK))
        # Record how many bytes were hashed, so the step can be repeated the same way
        info["sha256_head_tail"] = h.hexdigest()
        info["fingerprint_chunk_bytes"] = _FINGERPRINT_CHUNK
    except OSError:
        info["sha256_head_tail"] = None
    return info


def sampler_defaults(props: dict | None) -> dict:
    """Collect sampler defaults from llama.cpp ``/props`` default_generation_settings.

    Depending on the llama.cpp version the values sit
    directly under ``default_generation_settings``, or inside its ``params``,
    so both are checked. An empty dict when nothing is available.
    """
    if not isinstance(props, dict):
        return {}
    gen = props.get("default_generation_settings") or {}
    if not isinstance(gen, dict):
        return {}
    sources = [gen]
    params = gen.get("params")
    if isinstance(params, dict):
        sources.append(params)
    out: dict = {}
    for src in sources:
        for key in _SAMPLER_KEYS:
            if key in src and src[key] is not None:
                out[key] = src[key]
    # Normalise the temp / temperature spelling to temperature
    if "temp" in out:
        out.setdefault("temperature", out.pop("temp"))
    return out


def _quant_of(path) -> str | None:
    """The quantization name contained in a path (gguf file name / pack folder)."""
    m = re.search(r"(IQ\d[A-Z_]*|Q\d[A-Z0-9_]*|F16|BF16|F32)", str(path))
    return m.group(1) if m else None


def health_family(health) -> dict:
    """Tell a Strata-family server from its /health answer ({family, version} / {}).

    - Strata: ``"service": "strata"``
    - Abyss (from v1.1.1): ``"service": "abyss"`` and ``version``
    - Maya: no service field, but returning ``version`` and ``mcp_tools`` is Maya's shape
      (Abyss v1.1.0 and earlier match too, so process-side detection covers that)
    """
    if not isinstance(health, dict):
        return {}
    service = str(health.get("service") or "").lower()
    out: dict = {}
    if service in ("strata", "abyss", "maya"):
        out["family"] = {"strata": "Strata", "abyss": "Abyss", "maya": "Maya"}[service]
    elif not service and "version" in health and "mcp_tools" in health:
        out["family"] = "Maya"
        out["guessed"] = True
    else:
        return {}
    if health.get("version"):
        out["version"] = str(health["version"])
    return out


def _backend_openai(base_url: str, model: str | None) -> dict:
    """OpenAI-compatible server. For a local one (llama.cpp / LM Studio ...) read /props too."""
    local = _is_local_url(base_url)
    b: dict = {
        "kind": "llama.cpp/openai-compat" if local else "openai-compat-api",
        "base_url": base_url,
    }
    if model:
        b["model"] = model
    if not local:
        return b
    # llama.cpp server: /props carries n_ctx, the model path and the build info
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    props = _get_json(f"{root}/props")
    if not isinstance(props, dict):
        return b
    b["kind"] = "llama.cpp"
    # The Strata family names itself on /health (llama.cpp-compatible /props cannot)
    fam = health_family(_get_json(f"{root}/health"))
    if fam:
        b["kind"] = fam["family"]
        if fam.get("version"):
            b["server_version"] = f"{fam['family']} {fam['version']}"
    gen = props.get("default_generation_settings") or {}
    n_ctx = _int(gen.get("n_ctx") or props.get("n_ctx"))
    if n_ctx:
        b["n_ctx"] = n_ctx
    slots = _int(props.get("total_slots"))
    if slots:
        b["parallel_slots"] = slots
    # Sampler defaults (temperature / top_p / top_k / min_p / seed ...).
    # Every parameter the client did not send is **generated with these defaults**.
    # It used to read only n_ctx, so "what it was measured with" was not recorded.
    defaults = sampler_defaults(props)
    if defaults:
        b["sampler_defaults"] = defaults
    path = props.get("model_path") or gen.get("model") or props.get("model_path")
    if path:
        b["model_path"] = str(path)
        # The gguf file name carries the quantization (e.g. ...-Q4_K_M.gguf)
        q = _quant_of(path)
        if q:
            b["quantization"] = q
        # Keep a fingerprint, in case a file with the same name is swapped and re-measured
        info = model_file_info(path)
        if info:
            b["model_file"] = info
    for key in ("build_info", "chat_template_name"):
        if props.get(key):
            b[key] = props[key]
    return b


def collect_backend(cfg: dict | None, served_model: str | None = None) -> dict:
    """Collect the backend setup from the model config (models.<name> in config).

    cfg is expected to be the return value of resolve_model(). api_key is not read.
    """
    cfg = cfg or {}
    ctype = (cfg.get("type") or "openai").lower()
    model = served_model or cfg.get("model") or None
    if model and str(model).lower() == "auto":
        model = served_model or None
    base_url = cfg.get("base_url") or ""
    try:
        if ctype == "ollama":
            return _backend_ollama(base_url or "http://localhost:11434", model)
        if ctype == "openai":
            return _backend_openai(base_url, model)
        if ctype == "cli":
            b = {"kind": f"cli:{cfg.get('preset', 'custom')}"}
            if model:
                b["model"] = model
            return b
        if ctype == "mock":
            return {"kind": "mock"}
        b = {"kind": ctype}
        if base_url:
            b["base_url"] = base_url
        if model:
            b["model"] = model
        return b
    except Exception as e:
        return {"kind": ctype, "collect_error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────── Summary ───────────────────────────────


def _execution_kind(cfg: dict) -> tuple[str, str]:
    """Return (execution, note). The note goes into the report to prevent misreading."""
    ctype = (cfg.get("type") or "openai").lower()
    if ctype == "mock":
        return "mock", "Mock run (excluded from performance measurement)"
    if ctype == "cli":
        return (
            "subscription-cli",
            "Through a subscription CLI. Inference runs on the vendor side, so "
            "the host specs below do not affect generation speed",
        )
    if ctype == "ollama":
        return "local", "Local inference (this host's GPU/memory set the generation speed)"
    if _is_local_url(cfg.get("base_url") or ""):
        return "local", "Local inference (this host's GPU/memory set the generation speed)"
    return (
        "remote-api",
        "Cloud API inference. Inference runs on vendor-side hardware, so the "
        "host specs below do not affect generation speed (this is the measuring client)",
    )


def collect(cfg: dict | None = None, served_model: str | None = None) -> dict:
    """Return all execution environment metadata in one dict (never raises)."""
    cfg = cfg or {}
    try:
        execution, note = _execution_kind(cfg)
        host = collect_host()
        backend = collect_backend(cfg, served_model)
        if execution == "local":
            # Collect the setup starting from the server PID. A port match wins, so a setup
            # that swaps builds on the same port yields the one running right now.
            pid = find_server_pid(_port_of(cfg.get("base_url") or ""))
            usage = collect_gpu_usage()
            pid = pid or ((usage.get("inference") or {}).get("pid")
                          if usage else None)

            # Compute backend (CUDA / ROCm / Vulkan / ...).
            # A runtime: entry in config is authoritative and detection only backs it up
            # (swapping builds on the same port makes people forget to edit it, so when the
            #  detected value disagrees, keep both so the mismatch stays visible).
            # Strata / Maya / Abyss: read the kind, version and pack from the server.py folder,
            # and detect the compute backend in the child engine process (build/strata)
            # (python server.py itself never loads CUDA, so it looks like "CPU")
            fam = strata_family(pid)
            if fam:
                # The name given on /health is authoritative (folder-name guessing only supports it).
                # For a guessed Maya (no service field), prefer the folder-side detection if present
                hk = backend.get("kind")
                if hk in ("Strata", "Abyss", "Maya") and not (
                        hk == "Maya" and fam.get("family") not in (None, "Maya", "Strata-family")):
                    if fam.get("family") not in (None, hk, "Strata-family"):
                        fam["folder_family"] = fam["family"]
                    fam["family"] = hk
                backend["kind"] = fam.get("family") or "Strata-family"
                backend["server_version"] = _family_label(fam)
                backend["strata_family"] = fam
                if fam.get("model_path"):
                    backend["model_path"] = fam["model_path"]
                    backend.pop("model_file", None)   # fingerprint of the "pack/full" from /props
                    q = _quant_of(fam["model_path"])
                    if q:
                        backend["quantization"] = q
                if fam.get("n_ctx"):
                    backend.setdefault("n_ctx", fam["n_ctx"])
            runtime = detect_runtime((fam or {}).get("engine_pid") or pid)
            declared = cfg.get("runtime")
            if declared:
                detected = runtime.get("compute")
                runtime["compute"] = str(declared)
                runtime["source"] = "config"
                # "Strata/CUDA" and the detected value "CUDA" count as a match
                if detected and detected.lower() not in str(declared).lower():
                    runtime["detected"] = detected
                    runtime["mismatch"] = True
            elif runtime.get("compute"):
                runtime["source"] = "detected"
            if runtime:
                backend["runtime"] = runtime

            # Start-up arguments (--device / -ngl / --ctx-size ...). In llama.cpp /props
            # carries no GPU offload info, so -ngl is the effective offload amount.
            launch = collect_launch(pid, host.get("gpu") or [])
            if launch:
                backend["launch"] = launch

            if usage:
                # A ROCm/Vulkan build can also show up in nvidia-smi compute-apps with a few
                # tens of MiB; reading that as the inference GPU reports "RTX 3090" for a run
                # that is on Radeon. Do not accept the association for anything but CUDA.
                compute = (runtime.get("compute") or "").lower()
                if compute and "cuda" not in compute:
                    usage.pop("inference", None)
                    usage["note"] = (
                        f"Compute backend is {runtime.get('compute')}, so the nvidia-smi based "
                        "inference GPU detection is not used"
                    )
                inf = usage.get("inference") or {}
                if fam and inf.get("pid") and inf.get("pid") == fam.get("engine_pid"):
                    inf.pop("uncertain", None)   # confirmed as the Strata-family engine (child of server.py)
                backend["gpu_usage"] = usage
        return {
            "execution": execution,
            "note": note,
            "host": host,
            "backend": backend,
        }
    except Exception as e:
        return {"execution": "unknown",
                "collect_error": f"{type(e).__name__}: {e}"}


def format_summary(env: dict) -> str:
    """One-line summary (for the run log), e.g. 'Apple M3 Max / 128GB / Q4_K_M / GPU 100%'."""
    if not env:
        return ""
    host = env.get("host") or {}
    backend = env.get("backend") or {}
    parts: list[str] = []
    # For remote inference host specs are unrelated to speed -> omit them (no misreading)
    if env.get("execution") not in ("local", None, ""):
        for key in ("kind", "model"):
            if backend.get(key):
                parts.append(str(backend[key]))
        parts.append("Inference runs remotely")
        return " / ".join(parts)
    # The device given in the start-up arguments is the surest source (works for ROCm/Vulkan)
    launch = backend.get("launch") or {}
    inf = (backend.get("gpu_usage") or {}).get("inference") or {}
    if launch.get("device"):
        parts.append(launch.get("device_name") or launch["device"])
        ngl = launch.get("n_gpu_layers")
        if ngl == 0:
            parts.append("⚠️ -ngl 0 (GPU unused)")
        elif ngl is not None:
            parts.append(f"-ngl {ngl}")
    elif inf.get("gpus"):
        parts.append(" + ".join(
            f"{g.get('name', 'GPU')} {g.get('vram_gb', 0)}GB"
            for g in inf["gpus"]
        ) + (f" (split over {inf.get('vram_total_gb')}GB total)"
             if inf.get("multi_gpu") else ""))
    elif host.get("gpu"):
        g = (host.get("gpu") or [])[0]
        name = g.get("name") or "GPU"
        if g.get("cores"):
            name += f" {g['cores']}core"
        if g.get("vram_gb"):
            name += f" {g['vram_gb']}GB"
        parts.append(name)
    elif host.get("cpu"):
        parts.append(host["cpu"])
    if host.get("ram_gb"):
        parts.append(f"RAM {host['ram_gb']}GB")
    rt = backend.get("runtime") or {}
    if rt.get("compute"):
        parts.append(str(rt["compute"]))
    if backend.get("quantization"):
        parts.append(str(backend["quantization"]))
    ratio = backend.get("gpu_offload_ratio")
    if ratio is not None:
        parts.append(f"GPU offload {ratio * 100:.0f}%")
    if backend.get("n_ctx"):
        parts.append(f"n_ctx {backend['n_ctx']}")
    return " / ".join(parts)

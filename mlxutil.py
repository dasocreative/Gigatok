"""
mlxutil.py — version-tolerant shims over MLX / mlx-lm, plus small helpers.

Every MLX API whose *location* changed between releases is resolved here at
import time by probing, never by assuming. `apicheck.py` prints what resolved.

Why this file exists: MLX moved the memory APIs from `mx.metal.*` to top-level
`mx.*` (ml-explore/mlx PR #1982). On 0.32.x the docs list
mlx.core.{get_active_memory,get_peak_memory,reset_peak_memory,get_cache_memory,
set_memory_limit,set_cache_limit,set_wired_limit,clear_cache} and mlx.core.metal
retains only {is_available,device_info,start_capture,stop_capture}. Older builds
have them under mx.metal. We accept either.
"""

from __future__ import annotations

import platform
import re
import subprocess
from typing import Any, Iterable, List, Optional

import mlx.core as mx

# --------------------------------------------------------------------------
# API resolution
# --------------------------------------------------------------------------


def _resolve(*names: str):
    """Return the first attribute found among 'a.b.c' dotted names, else None."""
    for dotted in names:
        obj: Any = mx
        ok = True
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and callable(obj):
            return obj
    return None


_get_peak = _resolve("get_peak_memory", "metal.get_peak_memory")
_reset_peak = _resolve("reset_peak_memory", "metal.reset_peak_memory")
_get_active = _resolve("get_active_memory", "metal.get_active_memory")
_get_cache = _resolve("get_cache_memory", "metal.get_cache_memory")
_clear_cache = _resolve("clear_cache", "metal.clear_cache")
_set_wired = _resolve("set_wired_limit", "metal.set_wired_limit")
_set_memlimit = _resolve("set_memory_limit", "metal.set_memory_limit")
# mx.device_info first: mx.metal.device_info still works on 0.32.2 but emits a
# DeprecationWarning on every call.
_device_info = _resolve("device_info", "metal.device_info")
_synchronize = _resolve("synchronize")

RESOLVED = {
    "get_peak_memory": _get_peak,
    "reset_peak_memory": _reset_peak,
    "get_active_memory": _get_active,
    "get_cache_memory": _get_cache,
    "clear_cache": _clear_cache,
    "set_wired_limit": _set_wired,
    "set_memory_limit": _set_memlimit,
    "device_info": _device_info,
    "synchronize": _synchronize,
}


def peak_memory() -> Optional[int]:
    return int(_get_peak()) if _get_peak else None


def reset_peak_memory() -> None:
    if _reset_peak:
        _reset_peak()


def active_memory() -> Optional[int]:
    return int(_get_active()) if _get_active else None


def cache_memory() -> Optional[int]:
    return int(_get_cache()) if _get_cache else None


def clear_cache() -> None:
    if _clear_cache:
        _clear_cache()


def device_info() -> dict:
    if _device_info is None:
        return {}
    try:
        return dict(_device_info())
    except Exception:
        return {}


def max_working_set_bytes() -> Optional[int]:
    """Metal's recommended max working set (the real GPU-addressable ceiling)."""
    di = device_info()
    for k in (
        "max_recommended_working_set_size",
        "max_recommended_working_set",
        "recommended_max_working_set_size",
    ):
        if k in di:
            try:
                return int(di[k])
            except Exception:
                pass
    return None


def barrier(*arrays: Any) -> None:
    """
    THE eval barrier. MLX is lazy: building a graph costs microseconds and
    measures nothing. Every timed region must be closed with this.

    Pass the arrays whose *values* the next step genuinely depends on. Passing
    fewer arrays than you depend on under-measures; passing logits during
    prefill over-measures (it forces an lm_head matmul you are about to throw
    away). See roofline.prefill() for how that distinction is used.
    """
    flat = [a for a in flat_arrays(arrays)]
    if flat:
        mx.eval(*flat)
    elif _synchronize:
        _synchronize()


def flat_arrays(obj: Any) -> Iterable[mx.array]:
    """Yield every mx.array leaf inside arbitrarily nested tuples/lists/dicts."""
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from flat_arrays(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from flat_arrays(v)


# --------------------------------------------------------------------------
# Host / system facts
# --------------------------------------------------------------------------


def _sysctl(key: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def wired_limit_mb() -> Optional[int]:
    """iogpu.wired_limit_mb. 0 means 'system default' (not 'zero MB')."""
    v = _sysctl("iogpu.wired_limit_mb")
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


# --------------------------------------------------------------------------
# model_type compatibility
# --------------------------------------------------------------------------

# mlx-lm dispatches on config["model_type"] by importing
# mlx_lm.models.<model_type>, after passing it through MODEL_REMAPPING.
# `gemma4_unified` has no module and no remap entry (checked against mlx-lm
# 0.31.3 and main), so mlx-lm refuses the Gemma 4 12B Unified checkpoint even
# though it ships a complete implementation of that architecture.
#
# gemma4.py IS the text-only wrapper for it: ModelArgs consumes only
# text_config + vocab_size, Model builds gemma4_text.Model from text_config,
# and sanitize() explicitly drops vision_tower / audio_tower / embed_vision /
# embed_audio weights. gemma4_text.ModelArgs already carries every Unified
# field — global_head_dim, num_global_key_value_heads, num_kv_shared_layers,
# attention_k_eq_v, embed_tokens_per_layer, layer_types, partial_rotary_factor.
#
# So the remap below routes the Unified checkpoint into machinery that was
# built for it. It is still OUR patch, not upstream behaviour: it is announced
# loudly on every load and recorded in runs.jsonl, and roofline.py runs a
# generation sanity check afterwards. If a future mlx-lm ships a real
# gemma4_unified module, this becomes a no-op — we never overwrite an existing
# entry.
KNOWN_MODEL_REMAPS = {
    "gemma4_unified": "gemma4",
}


def patch_model_remapping(extra: Optional[dict] = None) -> List[tuple]:
    """Add missing model_type -> module remaps. Never overrides an existing one."""
    try:
        import mlx_lm.utils as MU
    except Exception:
        return []
    table = getattr(MU, "MODEL_REMAPPING", None)
    if not isinstance(table, dict):
        return []
    applied = []
    for k, v in {**KNOWN_MODEL_REMAPS, **(extra or {})}.items():
        if k in table:
            continue
        try:
            __import__(f"mlx_lm.models.{k}")
            continue          # upstream module exists; leave it alone
        except Exception:
            pass
        table[k] = v
        applied.append((k, v))
    return applied


# gemma4.py's sanitize() drops the multimodal towers by prefix, but it only
# knows the names used by the ENCODER-BASED Gemma 4 variants:
#   vision_tower, multi_modal_projector, audio_tower, embed_audio, embed_vision
# The 12B Unified model is encoder-FREE — it projects raw image patches and
# audio through lightweight linear layers named `vision_embedder.*` (and
# `audio_embedder.*`), which that tuple does not match. The weights therefore
# survive sanitize and load_weights rejects them as "not in model".
#
# Dropping them is exactly right for this project: the brief is text-path only,
# and no vision or audio projection should ever enter a timed region.
EXTRA_DROP_PREFIXES = {
    "gemma4": (
        "vision_embedder", "audio_embedder", "video_embedder",
        "vision_projector", "audio_projector", "mm_embedder",
    ),
}


def patch_multimodal_weight_drop(extra: Optional[dict] = None) -> dict:
    """Wrap Model.sanitize for the listed modules to drop extra non-text prefixes."""
    import importlib

    applied: dict = {}
    table = {**EXTRA_DROP_PREFIXES, **(extra or {})}
    for modname, prefixes in table.items():
        try:
            mod = importlib.import_module(f"mlx_lm.models.{modname}")
        except Exception:
            continue
        Model = getattr(mod, "Model", None)
        if Model is None or getattr(Model, "_mlxbench_drop_patched", False):
            continue
        orig = getattr(Model, "sanitize", None)

        def _make(orig_fn, pfx):
            def sanitize(self, weights):
                kept, dropped = {}, []
                for k, v in weights.items():
                    # match gemma4.sanitize semantics: test after the optional
                    # "model." prefix, so both spellings are caught
                    if k.removeprefix("model.").startswith(pfx):
                        dropped.append(k)
                        continue
                    kept[k] = v
                if dropped:
                    groups = sorted({d.removeprefix("model.").split(".")[0] for d in dropped})
                    print(f"  [compat] dropped {len(dropped)} non-text weight(s) "
                          f"from {groups} — text path only, as intended")
                return orig_fn(self, kept) if orig_fn else kept

            return sanitize

        Model.sanitize = _make(orig, prefixes)
        Model._mlxbench_drop_patched = True
        applied[modname] = prefixes
    return applied


PAGE_SIZE = 16384  # Apple Silicon


def vm_stat() -> dict:
    """
    Parse vm_stat. The fields that matter for benchmark validity:

      swapouts / swapins        pages moved to and from the swap file. ANY
                                increase during a timed run invalidates it —
                                you measured the SSD, not the GPU.
      compressor pages          macOS compresses memory BEFORE it swaps. A
                                growing compressor means the machine is already
                                over-committed; decode slows and nothing in the
                                MLX numbers hints at why.
      pages free                headroom before either of the above starts.

    On a 16 GB machine running a 6.7 GB model, a browser is enough to cross
    both thresholds, so this is recorded on every run.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return {}
    except Exception:
        return {}
    d: dict = {}
    for line in out.stdout.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if not v.isdigit():
            continue
        key = k.strip().lower().replace(" ", "_").replace('"', "")
        d[key] = int(v)
    return {
        "free_bytes": d.get("pages_free", 0) * PAGE_SIZE,
        "inactive_bytes": d.get("pages_inactive", 0) * PAGE_SIZE,
        "wired_bytes": d.get("pages_wired_down", 0) * PAGE_SIZE,
        "compressor_bytes": d.get("pages_occupied_by_compressor", 0) * PAGE_SIZE,
        "swapins": d.get("swapins", 0),
        "swapouts": d.get("swapouts", 0),
    }


def memory_pressure_pct() -> Optional[int]:
    """kern.memorystatus_level: roughly percent of memory still free.
    Below ~20 macOS starts reclaiming aggressively."""
    v = _sysctl("kern.memorystatus_level")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def top_processes(n: int = 8, min_mb: int = 200) -> List[dict]:
    """
    Biggest resident processes, so a contaminated run names its contaminator.

    The benchmark's OWN process is marked `self: True` and must not be counted
    as competition — it is holding the model weights on purpose.
    """
    import os

    me = os.getpid()
    try:
        me_group = os.getpgid(me)
    except Exception:
        me_group = None
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,pgid=,rss=,comm="], capture_output=True, text=True, timeout=8
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []
    agg: dict = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        pid, pgid, rss, comm = int(parts[0]), parts[1], int(parts[2]), parts[3]
        rss_mb = rss / 1024.0
        mine = (pid == me) or (me_group is not None and pgid.isdigit()
                               and int(pgid) == me_group)
        name = comm.rsplit("/", 1)[-1]
        for app in ("Google Chrome", "Claude", "Safari", "Code", "Docker", "Slack", "firefox"):
            if app.lower() in comm.lower():
                name = app
                break
        if mine:
            name = f"{name} (this benchmark)"
        cur = agg.setdefault(name, {"rss_mb": 0.0, "self": mine})
        cur["rss_mb"] += rss_mb
        cur["self"] = cur["self"] or mine
    rows = [{"name": k, "rss_mb": round(v["rss_mb"]), "self": v["self"]}
            for k, v in agg.items() if v["rss_mb"] >= min_mb]
    rows.sort(key=lambda r: -r["rss_mb"])
    return rows[:n]


def power_info() -> dict:
    """
    Power source, charge, and Low Power Mode.

    Apple Silicon is designed to deliver the same performance on battery as on
    AC — unlike Intel Macs, where the gap was large. So being on battery is NOT
    on its own a reason to refuse to measure. What genuinely throttles is:

      * Low Power Mode, which caps CPU/GPU frequency outright
      * very low charge, where the SoC limits peak power draw

    The real hazard is MIXING states across runs you later compare, so this is
    recorded on every run and the dashboard can separate on it.
    """
    src, pct = None, None
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=5).stdout
        if "AC Power" in out:
            src = "AC"
        elif "Battery Power" in out:
            src = "Battery"
        m = re.search(r"(\d+)%", out)
        if m:
            pct = int(m.group(1))
    except Exception:
        pass
    lpm = None
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"lowpowermode\s+(\d)", out)
        if m:
            lpm = bool(int(m.group(1)))
    except Exception:
        pass
    return {"power_source": src, "battery_percent": pct, "low_power_mode": lpm}


def contention_report() -> dict:
    vs = vm_stat()
    return {
        "vm": vs,
        "memory_pressure_pct": memory_pressure_pct(),
        "top_processes": top_processes(),
    }


def _dist_version(dist: str):
    """Installed version from package metadata, with a module-attr fallback."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(dist)
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        m = __import__(dist.replace("-", "_"))
        return getattr(m, "__version__", None)
    except Exception:
        return None


def host_info() -> dict:
    di = device_info()
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_cores": _sysctl("hw.ncpu"),
        "perf_cores": _sysctl("hw.perflevel0.logicalcpu"),
        "eff_cores": _sysctl("hw.perflevel1.logicalcpu"),
        "gpu_architecture": di.get("architecture"),
        "physical_memory_bytes": int(mem) if mem and mem.isdigit() else None,
        "metal_device_memory_bytes": di.get("memory_size"),
        "max_recommended_working_set_bytes": max_working_set_bytes(),
        "iogpu_wired_limit_mb": wired_limit_mb(),
        "macos": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "python": platform.python_version(),
        # NOT getattr(mx, "__version__"): the mlx package exposes no top-level
        # __version__, so that probe returned None on every run and every row in
        # runs.jsonl has recorded a null mlx version. The entire project rests on
        # "these numbers were measured on mlx 0.32.2" and the field that proves it
        # was silently empty. Read the installed distribution metadata instead.
        "mlx_version": _dist_version("mlx"),
        "mlx_lm_version": _mlx_lm_version(),
    }


def _mlx_lm_version() -> Optional[str]:
    try:
        import mlx_lm

        v = getattr(mlx_lm, "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("mlx-lm")
    except Exception:
        return None


# --------------------------------------------------------------------------
# powermetrics (energy). Always started BEFORE and stopped AFTER the timed
# region, in its own process, so sampling never runs inside a timed loop.
# --------------------------------------------------------------------------


class PowerSampler:
    """
    Requires sudo. Authenticate once in the same terminal first:

        sudo -v

    powermetrics itself costs a little CPU, so energy runs are tagged
    energy_pass=true in runs.jsonl and the dashboard excludes them from
    throughput charts by default. Never take a throughput median from an
    energy pass.
    """

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self.ok = False

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                [
                    "sudo", "-n", "powermetrics",
                    "--samplers", "cpu_power,gpu_power",
                    "-i", str(self.interval_ms),
                    "--format", "text",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.ok = True
        except Exception:
            self.proc = None
            self.ok = False
        return self

    def __exit__(self, *exc):
        self.text = ""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.text = self.proc.stdout.read() if self.proc.stdout else ""
            except Exception:
                self.text = ""
        return False

    def summary(self, duration_s: float) -> dict:
        if not getattr(self, "text", ""):
            return {"energy_sampled": False}
        gpu = [float(m) for m in re.findall(r"GPU Power:\s+([\d.]+)\s*mW", self.text)]
        cpu = [float(m) for m in re.findall(r"CPU Power:\s+([\d.]+)\s*mW", self.text)]
        pkg = [float(m) for m in re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\):\s+([\d.]+)\s*mW", self.text)]
        if not gpu and not cpu:
            return {"energy_sampled": False}

        def avg(x):
            return sum(x) / len(x) if x else None

        g, c, p = avg(gpu), avg(cpu), avg(pkg)
        total_mw = p if p is not None else ((g or 0.0) + (c or 0.0))
        return {
            "energy_sampled": True,
            "gpu_power_mw_avg": g,
            "cpu_power_mw_avg": c,
            "combined_power_mw_avg": p,
            "energy_joules": total_mw / 1000.0 * duration_s,
            "power_samples": len(gpu) or len(cpu),
        }


def summarize_quantization(q):
    """
    Collapse a quantization config to something worth storing per run.

    Sensitivity-aware conversions (OptiQ and friends) emit one entry PER LAYER.
    On gemma-4-e4b-it-OptiQ-4bit that is 348 entries — 28 KB of every 31 KB
    run record, 89% of runs.jsonl, identical on every row.

    Keep the global settings, a histogram of per-layer bit widths (which is the
    actually interesting part of a mixed-precision build), and the per-layer
    embedding entry, because whether PLE is 4-bit or 8-bit decides whether the
    checkpoint is usable at all.
    """
    if not isinstance(q, dict):
        return q
    out = {k: q[k] for k in ("group_size", "bits", "mode") if k in q}
    per = {k: v for k, v in q.items() if isinstance(v, dict)}
    if not per:
        return q
    hist: dict = {}
    for k, v in per.items():
        key = f"{v.get('bits')}bit_g{v.get('group_size')}"
        hist[key] = hist.get(key, 0) + 1
    out["per_layer_entries"] = len(per)
    out["bit_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    for k, v in per.items():
        if "embed_tokens_per_layer" in k:
            out["ple_quant"] = v
            break
    return out


GB = 1024 ** 3
MB = 1024 ** 2


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    if n >= GB:
        return f"{n / GB:.3f} GiB"
    if n >= MB:
        return f"{n / MB:.1f} MiB"
    return f"{n / 1024:.1f} KiB"

#!/usr/bin/env python3
"""
bench.py — Recipe 0 measurement harness (mlx-lm baseline).

Everything reported here is measured. Nothing is estimated except the two
quantities explicitly labelled as projections (KV headroom before the run, and
the mean KV read used for effective GB/s, which is integrated over the actual
generated range rather than sampled at one context).

MLX IS LAZY. This is the whole methodological problem with naive benchmarks:

    t0 = time.perf_counter()
    out = generate(...)            # builds a graph, executes almost nothing
    t1 = time.perf_counter()       # you just timed graph construction

Every timed region in this file is closed by mlxutil.barrier() -> mx.eval(),
which blocks until the Metal command buffers complete. Where each barrier sits
and why is documented at the point of use.

Two sync modes, because they measure different things and both are legitimate:

  --sync per-token  one barrier per decoded token. Gives a true inter-token
                    latency distribution (p50/p95). Costs one CPU<->GPU
                    round trip per token, so throughput reads slightly low.
  --sync pipelined  mx.async_eval, matching mlx-lm's own decode loop: the next
                    token's graph is built while the current one runs, so CPU
                    graph-building hides behind GPU execution. Best throughput,
                    but per-token timestamps are sync points, not pure latency.

Report both. If they differ by more than a few percent on a 10-core M3, the
gap is Metal launch/graph overhead that is NOT hidden — which is itself the
finding, and it is what mx.compile and command-buffer batching go after later.

Two engines:
  --engine manual   our own loop. Exact barrier control. Default.
  --engine stream   mlx_lm.stream_generate. Cross-check against stock.
If these two disagree by more than ~3%, trust neither until you know why.

Output: one JSON object per run appended to runs.jsonl (schema_version 1).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics as stats
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

import mlxutil as U
import roofline as RF

SCHEMA_VERSION = 1

# Deterministic filler. Tokenized once, then sliced to an exact token count so
# "2048 prompt tokens" means exactly 2048, not "about 2048".
FILLER = (
    "Memory bandwidth is the binding constraint for single-stream autoregressive "
    "decoding on unified-memory systems. Each generated token requires reading the "
    "full set of active weights from DRAM, plus the key and value tensors for every "
    "attending layer. Arithmetic intensity is therefore near one operation per byte, "
    "and the achievable token rate is bounded above by effective bandwidth divided by "
    "bytes touched per token. Kernel fusion reduces intermediate traffic; quantization "
    "reduces weight traffic; windowed attention bounds cache traffic; speculative "
    "decoding amortizes weight traffic across several accepted tokens at once. "
)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def _encode(tokenizer, text: str) -> List[int]:
    """mlx-lm wraps the HF tokenizer; not every wrapper forwards kwargs."""
    for attempt in (
        lambda: tokenizer.encode(text, add_special_tokens=False),
        lambda: tokenizer.encode(text),
        lambda: tokenizer(text)["input_ids"],
    ):
        try:
            out = attempt()
            if out:
                return list(out)
        except TypeError:
            continue
        except Exception:
            continue
    raise RuntimeError("could not tokenize with this tokenizer")


def build_prompt_ids(tokenizer, n_tokens: int) -> List[int]:
    """
    Exactly n_tokens ids, deterministic, no chat template — but WITH <bos>.

    The chat template is deliberately bypassed: it adds a variable number of
    control tokens, so "2048 tokens" would mean different things across models
    and break cross-run comparability. Template cost is measured separately by
    --measure-template.

    BOS is NOT bypassed. Gemma is trained with a leading <bos>; feeding it a
    bare continuation makes it degenerate into repetition. That does not change
    decode throughput (timing is shape-driven, not content-driven), but it does
    make output_sha256 a hash of broken text, which is useless as the
    losslessness baseline every speculative recipe must match.
    """
    base = _encode(tokenizer, FILLER)
    if not base:
        raise RuntimeError("tokenizer produced no tokens for the filler text")
    b = RF.bos_id(tokenizer)
    prefix = [b] if b is not None else []
    if len(prefix) > n_tokens:
        prefix = prefix[:n_tokens]
    need = n_tokens - len(prefix)
    reps = (need // len(base)) + 2
    ids = prefix + (base * reps)[:need]
    if len(ids) != n_tokens:
        raise RuntimeError(f"prompt build failed: {len(ids)} != {n_tokens}")
    return ids


def dump_prompts(tokenizer, lengths: List[int], outdir: Path) -> Dict[int, str]:
    """Write the exact prompt text for each length so the stock mlx_lm.generate
    command benchmarks the identical input."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}
    for n in lengths:
        ids = build_prompt_ids(tokenizer, n)
        text = tokenizer.decode(ids)
        p = outdir / f"p{n}.txt"
        p.write_text(text)
        recheck = len(_encode(tokenizer, text))
        written[n] = str(p)
        note = "" if recheck == n else f"  (re-tokenizes to {recheck}; decode/encode is not round-trip exact)"
        print(f"  {p}  target={n}{note}")
    return written


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


_GEN_STREAM = None


def gen_stream():
    """
    A dedicated MLX stream for generation, matching mlx-lm.

    mlx-lm runs generate_step inside `with mx.stream(generation_stream)` on a
    stream created once at import. Our manual engine originally used the
    default stream, and stock came out 1-4% faster on decode — a dedicated
    stream is the most likely reason, so match it and remove the confound.

    Created lazily on first use, which must be the main thread: cache arrays
    take stream affinity from where they are allocated, and mlx-lm 0.31.3 has
    an open bug where a stream created on one thread is unusable from another
    (the mlx_lm.server crash on sliding-window models).
    """
    global _GEN_STREAM
    if _GEN_STREAM is None:
        _GEN_STREAM = mx.new_stream(mx.default_device())
    return _GEN_STREAM


def make_sampler(temp: float):
    if temp <= 0.0:
        return lambda logits: mx.argmax(logits, axis=-1)
    inv = 1.0 / temp
    return lambda logits: mx.random.categorical(logits * inv)


# --------------------------------------------------------------------------
# one measured run
# --------------------------------------------------------------------------


def run_once(
    model,
    tokenizer,
    prompt_tokens: int,
    gen_tokens: int,
    sampler,
    prefill_step: int,
    sync: str,
    measure_template: bool,
) -> dict:
    from mlx_lm.models.cache import make_prompt_cache

    U.clear_cache()
    gc.collect()
    U.reset_peak_memory()
    mem_before = U.active_memory()
    # System-level contention snapshot. macOS compresses then swaps under
    # pressure; either one silently turns a bandwidth measurement into a
    # measurement of the memory subsystem giving up.
    vm0 = U.vm_stat()
    contention0 = U.contention_report()

    # ---- tokenization, timed on its own -------------------------------
    text = None
    t_tok0 = time.perf_counter()
    ids_list = build_prompt_ids(tokenizer, prompt_tokens)
    t_tok1 = time.perf_counter()
    tokenize_ms = (t_tok1 - t_tok0) * 1e3

    template_ms = None
    if measure_template:
        try:
            t0 = time.perf_counter()
            tokenizer.apply_chat_template(
                [{"role": "user", "content": FILLER}], add_generation_prompt=True
            )
            template_ms = (time.perf_counter() - t0) * 1e3
        except Exception:
            template_ms = None

    ids = mx.array([ids_list])
    # Everything from cache creation through the last decoded token runs on the
    # dedicated stream, so the cache arrays and the ops that touch them share
    # stream affinity.
    stream_ctx = mx.stream(gen_stream())
    stream_ctx.__enter__()
    cache = make_prompt_cache(model)

    # ---- prefill ------------------------------------------------------
    # BARRIER: inside RF.prefill, on cache state per chunk (not logits), then
    # a single-token forward for the final logits. See roofline.prefill.
    U.barrier(ids)
    t_pf0 = time.perf_counter()
    logits = RF.prefill(model, ids, cache, step=prefill_step, tail_logits=True)
    y = sampler(logits)
    U.barrier(y, RF.cache_arrays(cache))   # first token is now a real value
    t_pf1 = time.perf_counter()

    prefill_s = t_pf1 - t_pf0
    ttft_excl_tokenize_ms = prefill_s * 1e3
    ttft_incl_tokenize_ms = (t_pf1 - t_tok0) * 1e3

    # ---- decode -------------------------------------------------------
    # CPU time alongside wall time. This is the decisive discriminator for
    # where the ~9 ms/token shortfall lives:
    #   cpu_fraction near 1.0  -> the Python/MLX graph-construction thread is
    #                             the bottleneck; the GPU is waiting on the CPU.
    #                             Sensitive to P-core vs E-core placement, which
    #                             is a plausible source of bimodal run times.
    #   cpu_fraction near 0.2  -> the CPU issues work and blocks; the cost is on
    #                             the GPU side (per-kernel launch), and only
    #                             fusion reduces it.
    # process_time() counts all threads in the process and excludes any sleep.
    itl: List[float] = []
    out_tokens: List[int] = [int(y.item())]
    cpu0 = time.process_time()
    thr0 = time.thread_time()

    if sync == "per-token":
        for _ in range(gen_tokens - 1):
            t0 = time.perf_counter()
            logits = model(y.reshape(1, 1), cache=cache)[:, -1, :]
            y = sampler(logits)
            U.barrier(y)                  # one hard barrier per token
            itl.append(time.perf_counter() - t0)
            out_tokens.append(int(y.item()))
    else:
        # pipelined — mirrors mlx-lm's decode loop exactly: token n+1's graph is
        # queued with async_eval BEFORE token n is synced, so CPU graph building
        # overlaps GPU execution. Exactly one token stays in flight.
        pending = None
        prev = y                          # already a value: the prefill barrier ran
        t_prev = time.perf_counter()
        for _ in range(gen_tokens - 1):
            logits = model(prev.reshape(1, 1), cache=cache)[:, -1, :]
            cur = sampler(logits)
            mx.async_eval(cur)            # queue, do not block
            if pending is not None:
                out_tokens.append(int(pending.item()))   # <- the real sync point
                t_now = time.perf_counter()
                itl.append(t_now - t_prev)
                t_prev = t_now
            pending = cur
            prev = cur
        if pending is not None:
            out_tokens.append(int(pending.item()))
            t_now = time.perf_counter()
            itl.append(t_now - t_prev)

    decode_cpu_s = time.process_time() - cpu0
    decode_thread_s = time.thread_time() - thr0
    decode_s = sum(itl)
    n_decoded = len(itl) + 1

    stream_ctx.__exit__(None, None, None)

    peak = U.peak_memory()
    mem_after = U.active_memory()
    kv_arrays = RF.cache_arrays(cache)
    kv_raw = sum(a.nbytes for a in kv_arrays)
    seen, kv_dedup = set(), 0
    for a in kv_arrays:
        if id(a) not in seen:
            seen.add(id(a))
            kv_dedup += a.nbytes

    # Chronological order matters as much as the distribution: a run that
    # starts at 43 ms/token and ends at 52 is throttling DURING the run, which
    # no percentile over the whole run can show.
    itl_chrono = [t * 1e3 for t in itl]
    itl_ms = sorted(itl_chrono)
    q = max(1, len(itl_chrono) // 4)
    first_q = stats.fmean(itl_chrono[:q]) if itl_chrono else None
    last_q = stats.fmean(itl_chrono[-q:]) if itl_chrono else None
    intra_decay = (100.0 * (last_q - first_q) / first_q) if (first_q and last_q) else None

    def pct(p: float) -> Optional[float]:
        if not itl_ms:
            return None
        k = min(len(itl_ms) - 1, int(round(p * (len(itl_ms) - 1))))
        return itl_ms[k]

    del cache, logits, y
    U.clear_cache()

    vm1 = U.vm_stat()
    swapouts_delta = (vm1.get("swapouts", 0) - vm0.get("swapouts", 0)) if vm0 and vm1 else None
    swapins_delta = (vm1.get("swapins", 0) - vm0.get("swapins", 0)) if vm0 and vm1 else None
    comp_delta = ((vm1.get("compressor_bytes", 0) - vm0.get("compressor_bytes", 0))
                  if vm0 and vm1 else None)
    # A run is invalid if the OS paged during it. Compression growth over
    # ~256 MiB means the machine was already over-committed and the number is
    # suspect even without a single swapout.
    invalid = bool(swapouts_delta) or bool(comp_delta and comp_delta > 256 * U.MB)

    return {
        "vm_free_bytes_before": vm0.get("free_bytes"),
        "vm_compressor_bytes_before": vm0.get("compressor_bytes"),
        "vm_compressor_bytes_after": vm1.get("compressor_bytes"),
        "vm_compressor_delta_bytes": comp_delta,
        "vm_swapouts_delta": swapouts_delta,
        "vm_swapins_delta": swapins_delta,
        "memory_pressure_pct": contention0.get("memory_pressure_pct"),
        "top_processes": contention0.get("top_processes"),
        "host_contended": invalid,
        "tokenize_ms": tokenize_ms,
        "chat_template_ms": template_ms,
        "ttft_ms_excl_tokenize": ttft_excl_tokenize_ms,
        "ttft_ms_incl_tokenize": ttft_incl_tokenize_ms,
        "prefill_s": prefill_s,
        "prefill_tok_s": prompt_tokens / prefill_s if prefill_s > 0 else None,
        "decode_s": decode_s,
        "decode_cpu_s": decode_cpu_s,
        "decode_thread_s": decode_thread_s,
        "cpu_fraction": (decode_cpu_s / decode_s) if decode_s > 0 else None,
        "cpu_ms_per_token": (decode_cpu_s * 1e3 / len(itl)) if itl else None,
        "decode_tokens": n_decoded,
        "decode_tok_s": (len(itl) / decode_s) if decode_s > 0 else None,
        "ms_per_token_median": stats.median(itl_ms) if itl_ms else None,
        "ms_per_token_mean": stats.fmean(itl_ms) if itl_ms else None,
        "itl_p50_ms": pct(0.50),
        "itl_p90_ms": pct(0.90),
        "itl_p95_ms": pct(0.95),
        "itl_p99_ms": pct(0.99),
        "itl_min_ms": itl_ms[0] if itl_ms else None,
        "itl_max_ms": itl_ms[-1] if itl_ms else None,
        "itl_first_quarter_ms": first_q,
        "itl_last_quarter_ms": last_q,
        "itl_intra_run_decay_pct": intra_decay,
        "peak_memory_bytes": peak,
        "active_memory_before_bytes": mem_before,
        "active_memory_after_bytes": mem_after,
        "kv_measured_raw_bytes": kv_raw,
        "kv_measured_dedup_bytes": kv_dedup,
        "output_sha256": hashlib.sha256(
            ",".join(str(t) for t in out_tokens).encode()
        ).hexdigest(),
        "output_first_16_tokens": out_tokens[:16],
        "output_token_count": len(out_tokens),
    }


def run_once_stream(model, tokenizer, prompt_tokens: int, gen_tokens: int, temp: float) -> dict:
    """Cross-check path: stock mlx_lm.stream_generate. Fields come from
    GenerationResponse (text, token, prompt_tps, generation_tps, peak_memory...)."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

    U.clear_cache()
    U.reset_peak_memory()
    vm0 = U.vm_stat()
    contention0 = U.contention_report()
    ids = build_prompt_ids(tokenizer, prompt_tokens)

    t0 = time.perf_counter()
    first_t = None
    toks: List[int] = []
    marks: List[float] = []
    last: Any = None
    for resp in stream_generate(
        model, tokenizer, mx.array(ids), max_tokens=gen_tokens,
        sampler=mlx_make_sampler(temp=temp),
    ):
        now = time.perf_counter()
        if first_t is None:
            first_t = now
        else:
            marks.append(now)
        toks.append(resp.token)
        last = resp
    t1 = time.perf_counter()

    itl_ms = sorted(
        (marks[i] - (marks[i - 1] if i else first_t)) * 1e3 for i in range(len(marks))
    )
    vm1 = U.vm_stat()
    swapouts_delta = (vm1.get("swapouts", 0) - vm0.get("swapouts", 0)) if vm0 and vm1 else None
    comp_delta = ((vm1.get("compressor_bytes", 0) - vm0.get("compressor_bytes", 0))
                  if vm0 and vm1 else None)
    return {
        "vm_free_bytes_before": vm0.get("free_bytes"),
        "vm_compressor_delta_bytes": comp_delta,
        "vm_swapouts_delta": swapouts_delta,
        "memory_pressure_pct": contention0.get("memory_pressure_pct"),
        "top_processes": contention0.get("top_processes"),
        "host_contended": bool(swapouts_delta) or bool(comp_delta and comp_delta > 256 * U.MB),
        "ttft_ms_excl_tokenize": (first_t - t0) * 1e3 if first_t else None,
        "ttft_ms_incl_tokenize": (first_t - t0) * 1e3 if first_t else None,
        "prefill_tok_s": getattr(last, "prompt_tps", None),
        "decode_tok_s": getattr(last, "generation_tps", None),
        "decode_tokens": len(toks),
        "decode_s": (t1 - first_t) if first_t else None,
        "ms_per_token_median": stats.median(itl_ms) if itl_ms else None,
        "itl_p50_ms": stats.median(itl_ms) if itl_ms else None,
        "itl_p95_ms": itl_ms[min(len(itl_ms) - 1, int(0.95 * (len(itl_ms) - 1)))] if itl_ms else None,
        "peak_memory_bytes": (int(getattr(last, "peak_memory", 0) * U.GB) or U.peak_memory()),
        "output_sha256": hashlib.sha256(",".join(str(t) for t in toks).encode()).hexdigest(),
        "output_first_16_tokens": toks[:16],
        "output_token_count": len(toks),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Recipe 0 measurement harness")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", default="128,2048,8192")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="Discarded runs before timing.")
    ap.add_argument("--cooldown", type=float, default=20.0,
                    help="Seconds idle between runs. Base M3 is thermally limited "
                         "and has no High Power Mode; do not set this to 0.")
    ap.add_argument("--engine", choices=["manual", "stream"], default="manual")
    ap.add_argument("--sync", choices=["per-token", "pipelined"], default="per-token")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill-step", type=int, default=512)
    ap.add_argument("--achievable-gbs", type=float, default=None,
                    help="From bwprobe.py. Without it, %%-of-roofline is null.")
    ap.add_argument("--activation-slack-gb", type=float, default=0.8)
    ap.add_argument("--out", default="runs.jsonl")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--energy", action="store_true",
                    help="Sample powermetrics (needs `sudo -v` first). Tagged "
                         "energy_pass=true and excluded from throughput medians.")
    ap.add_argument("--measure-template", action="store_true")
    ap.add_argument("--dump-prompts", default=None, metavar="DIR")
    ap.add_argument("--kv-from", default=None, metavar="roofline.json",
                    help="Reuse a KV profile from `roofline.py --json` instead of "
                         "re-running the three probe prefills.")
    ap.add_argument("--print-plan", action="store_true",
                    help="Show the memory projection and exit without running.")
    ap.add_argument("--remap-model-type", action="append", metavar="FROM=TO",
                    help="Route an unimplemented config model_type to an existing "
                         "mlx_lm.models module. Repeatable.")
    ap.add_argument("--allow-contended", action="store_true",
                    help="Run even when other apps leave too little headroom. "
                         "Contended runs are tagged and excluded from medians.")
    ap.add_argument("--allow-oversubscribe", action="store_true",
                    help="Run even when projected memory exceeds headroom. "
                         "macOS will swap and your numbers will be garbage.")
    args = ap.parse_args()

    lengths = [int(x) for x in args.prompt_tokens.split(",") if x.strip()]
    mx.random.seed(args.seed)

    host = U.host_info()
    cfg, local = RF.find_config(args.model)
    rev = RF.model_revision(local)

    print("Loading model ...")
    remap = {}
    for pair in (args.remap_model_type or []):
        if "=" in pair:
            k, v = pair.split("=", 1)
            remap[k.strip()] = v.strip()
    model, tokenizer = RF.load_model(args.model, remap)
    remap_applied = sorted(set(list(U.KNOWN_MODEL_REMAPS.items()) + list(remap.items())))
    U.reset_peak_memory()
    weights_resident = U.active_memory()
    pb = RF.param_breakdown(model, cfg)

    if args.dump_prompts:
        print(f"\nWriting exact prompts to {args.dump_prompts}/ :")
        dump_prompts(tokenizer, lengths, Path(args.dump_prompts))

    if args.kv_from:
        print(f"\nReusing KV profile from {args.kv_from}")
        kv = RF.kvmodel_from_json(args.kv_from)
    else:
        print("\nProfiling KV growth from the model's real cache objects ...")
        kv = RF.profile_kv(model, tokenizer, (64, 1024, 2176), args.prefill_step)

    # ---- memory guard --------------------------------------------------
    mws = host["max_recommended_working_set_bytes"]
    budget = (mws * 0.92) if mws else None
    slack = args.activation_slack_gb * U.GB
    print("\n" + "=" * 78)
    print("MEMORY PLAN (projection — the guard that stops silent swapping)")
    print("=" * 78)
    print(f"  metal max working set     : {U.fmt_bytes(mws)}")
    print(f"  usable budget (92%)       : {U.fmt_bytes(budget)}")
    print(f"  weights resident          : {U.fmt_bytes(weights_resident)}")
    print(f"  activation slack reserved : {U.fmt_bytes(slack)}")
    headroom = (budget - (weights_resident or 0) - slack) if budget else None
    print(f"  KV headroom               : {U.fmt_bytes(headroom)}")
    if headroom:
        print(f"  max context that fits     : {kv.max_context_within(headroom)} tokens")

    blocked: List[int] = []
    print(f"\n  {'prompt':>8}{'ctx@end':>10}{'KV resident':>14}{'projected total':>18}  verdict")
    for n in lengths:
        ctx = n + args.gen_tokens
        res = kv.resident_bytes(ctx)
        total = (weights_resident or 0) + res + slack
        ok = budget is None or total <= budget
        if not ok:
            blocked.append(n)
        print(f"  {n:>8}{ctx:>10}{U.fmt_bytes(res):>14}{U.fmt_bytes(total):>18}  "
              f"{'OK' if ok else 'EXCEEDS BUDGET'}")

    if blocked and not args.allow_oversubscribe:
        biggest = kv.max_context_within(headroom) - args.gen_tokens if headroom else 0
        print("\n" + "!" * 78)
        print(f"ABORT: prompt lengths {blocked} do not fit.")
        print(f"Largest prompt that fits with {args.gen_tokens} generated tokens: "
              f"{max(0, biggest)} tokens.")
        print("Running anyway would page to swap and silently destroy the measurement.")
        print("Override with --allow-oversubscribe only if you intend to measure swap.")
        print("!" * 78)
        return 3
    # ---- host contention preflight ------------------------------------
    # 16 GB machine, ~6.7 GB of weights. A browser and an Electron app can put
    # the system into memory compression before MLX ever allocates its KV, and
    # a compressed/swapping run reports a bandwidth number that is really the
    # memory subsystem failing. Check BEFORE spending 25 minutes.
    pw = U.power_info()
    if pw.get("low_power_mode"):
        print("\n" + "!" * 78)
        print("ABORT: Low Power Mode is ON. It caps CPU and GPU frequency, so every")
        print("number would be a measurement of that cap. Turn it off in")
        print("System Settings > Battery > Low Power Mode, then re-run.")
        print("!" * 78)
        return 5

    cr = U.contention_report()
    vm = cr["vm"]
    # The weights are ALREADY loaded by this process — asking for headroom to
    # cover them again double-counts and rejects machines that are fine. What
    # still has to be allocated is KV plus activations.
    kv_need = kv.resident_bytes(max(lengths) + args.gen_tokens)
    need_additional = kv_need + slack
    avail = (vm.get("free_bytes", 0) or 0) + (vm.get("inactive_bytes", 0) or 0)
    pressure = cr["memory_pressure_pct"]
    print("\n" + "=" * 78)
    print("HOST CONTENTION PREFLIGHT")
    print("=" * 78)
    print(f"  power                     : {pw.get('power_source')} "
          f"{pw.get('battery_percent')}%  lowPowerMode={pw.get('low_power_mode')}")
    if pw.get("power_source") == "Battery":
        print("    (fine on Apple Silicon — but tagged, so don't mix with AC runs)")
    print(f"  weights already resident  : {U.fmt_bytes(weights_resident)}  (loaded, not re-requested)")
    print(f"  still to allocate         : {U.fmt_bytes(need_additional)}  "
          f"(KV {U.fmt_bytes(kv_need)} + activations {U.fmt_bytes(slack)})")
    print(f"  free + inactive now       : {U.fmt_bytes(avail)}")
    print(f"  memory pressure level     : {pressure}  (higher is better; "
          f"under ~20 macOS is reclaiming hard)")
    print(f"  compressor pool           : {U.fmt_bytes(vm.get('compressor_bytes'))}  "
          f"(historical total, not necessarily current pressure)")
    if cr["top_processes"]:
        print("  biggest resident processes:")
        for p in cr["top_processes"]:
            tag = "  <- this process, holding the weights" if p.get("self") else ""
            print(f"      {p['name']:<34}{p['rss_mb']:>7} MB{tag}")
    others = sum(p["rss_mb"] for p in cr["top_processes"] if not p.get("self"))
    print(f"  other apps total          : {others} MB")
    # Two independent gates. Margin is 1.25x of the ADDITIONAL allocation.
    tight = (avail < need_additional * 1.25) or (pressure is not None and pressure < 20)
    if tight:
        print(f"\n  !! Not enough headroom for the {U.fmt_bytes(need_additional)} still to be")
        print("     allocated. macOS will compress or swap during the run, and the")
        print("     resulting tok/s will measure that, not MLX.")
        print("     Quit Chrome (each window is hundreds of MB) and any Electron app")
        print("     you are not reading this from, then re-run.")
        if not args.allow_contended:
            print("\n  Aborting. Override with --allow-contended if you want the data anyway;")
            print("  contended runs are tagged host_contended and excluded from medians.")
            print("=" * 78)
            return 4
        print("  --allow-contended set: continuing, runs will be tagged.")
    else:
        print("  verdict: enough headroom to start.")
    print("  Per-run swapout/compressor deltas are recorded either way, so a run that")
    print("  goes bad mid-flight is flagged rather than silently averaged in.")
    print("=" * 78)

    if args.print_plan:
        return 0

    # ---- runs ----------------------------------------------------------
    outp = Path(args.out)
    ts_start = datetime.now(timezone.utc).isoformat()
    print(f"\nAppending to {outp.resolve()}")

    for n in lengths:
        print(f"\n=== prompt={n} gen={args.gen_tokens} engine={args.engine} "
              f"sync={args.sync} runs={args.runs} (+{args.warmup} warmup) ===")
        series: List[dict] = []
        for i in range(args.warmup + args.runs):
            warm = i < args.warmup
            if i > 0:
                time.sleep(args.cooldown)
            t_wall = time.time()

            sampler = make_sampler(args.temp)
            if args.energy and not warm:
                with U.PowerSampler() as ps:
                    t0 = time.perf_counter()
                    r = (run_once(model, tokenizer, n, args.gen_tokens, sampler,
                                  args.prefill_step, args.sync, args.measure_template)
                         if args.engine == "manual"
                         else run_once_stream(model, tokenizer, n, args.gen_tokens, args.temp))
                    dur = time.perf_counter() - t0
                r.update(ps.summary(dur))
            else:
                r = (run_once(model, tokenizer, n, args.gen_tokens, sampler,
                              args.prefill_step, args.sync, args.measure_template)
                     if args.engine == "manual"
                     else run_once_stream(model, tokenizer, n, args.gen_tokens, args.temp))
                r["energy_sampled"] = False

            tag = "warmup" if warm else f"run {i - args.warmup + 1}/{args.runs}"
            mark = ""
            if r.get("host_contended"):
                mark = f"  <-- CONTENDED (swapouts {r.get('vm_swapouts_delta')}, " \
                       f"compressor {U.fmt_bytes(r.get('vm_compressor_delta_bytes'))})"
            print(f"  {tag:<12} ttft={(r.get('ttft_ms_excl_tokenize') or 0):8.1f} ms   "
                  f"prefill={(r.get('prefill_tok_s') or 0):7.1f} tok/s   "
                  f"decode={(r.get('decode_tok_s') or 0):6.2f} tok/s   "
                  f"peak={U.fmt_bytes(r.get('peak_memory_bytes'))}{mark}")
            if warm:
                continue

            kv_mean_p = kv.mean_read_bytes(n, args.gen_tokens, "pessimistic")
            kv_mean_o = kv.mean_read_bytes(n, args.gen_tokens, "optimistic")
            ctx_mean_kv = kv_mean_p
            bytes_per_token = pb.active_bytes_per_token + kv_mean_p
            bytes_per_token_opt = pb.active_bytes_per_token + kv_mean_o
            spt = (r.get("ms_per_token_median") or 0) / 1e3
            eff_gbs = (bytes_per_token / spt / 1e9) if spt > 0 else None
            eff_gbs_opt = (bytes_per_token_opt / spt / 1e9) if spt > 0 else None
            pct_roof = (100.0 * eff_gbs / args.achievable_gbs) if (eff_gbs and args.achievable_gbs) else None
            pct_roof_opt = (100.0 * eff_gbs_opt / args.achievable_gbs) if (eff_gbs_opt and args.achievable_gbs) else None

            rec = {
                "schema_version": SCHEMA_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "session_started_utc": ts_start,
                "wall_clock_epoch": t_wall,
                "tag": args.tag,
                "run_index": i - args.warmup + 1,
                "runs_total": args.runs,
                # identity
                "chip": host["chip"],
                "gpu_architecture": host["gpu_architecture"],
                "cpu_cores": host["cpu_cores"],
                "physical_memory_bytes": host["physical_memory_bytes"],
                "max_recommended_working_set_bytes": host["max_recommended_working_set_bytes"],
                "iogpu_wired_limit_mb": host["iogpu_wired_limit_mb"],
                "macos": host["macos"],
                "power_source": pw.get("power_source"),
                "battery_percent": pw.get("battery_percent"),
                "low_power_mode": pw.get("low_power_mode"),
                "mlx_version": host["mlx_version"],
                "mlx_lm_version": host["mlx_lm_version"],
                "python": host["python"],
                # model
                "model": args.model,
                "model_revision": rev,
                "model_type": (cfg or {}).get("model_type"),
                "model_type_remaps_active": [f"{k}->{v}" for k, v in remap_applied],
                "quantization": U.summarize_quantization(
                    (cfg or {}).get("quantization") or (cfg or {}).get("quantization_config")),
                "quant_bits": ((cfg or {}).get("quantization") or {}).get("bits"),
                "quant_group_size": ((cfg or {}).get("quantization") or {}).get("group_size"),
                "tied_embeddings": pb.tied_embeddings,
                # config of the run
                "engine": args.engine,
                "sync_mode": args.sync,
                "prefill_mode": "tail1",
                "prefill_step": args.prefill_step,
                "prompt_tokens": n,
                "gen_tokens": args.gen_tokens,
                "temperature": args.temp,
                "seed": args.seed,
                "chat_template_applied": False,
                "eos_honoured": False,
                # bytes model
                "active_weight_bytes_per_token": pb.active_bytes_per_token,
                "total_param_bytes": pb.total_bytes,
                "non_text_param_bytes": pb.non_text_bytes,
                "gather_only_param_bytes": pb.gather_only_bytes,
                "dense_param_bytes": pb.dense_bytes,
                "kv_read_bytes_per_token_mean": ctx_mean_kv,
                "kv_read_bytes_per_token_mean_optimistic": kv_mean_o,
                "kv_read_bytes_at_end": kv.read_bytes_per_token(n + args.gen_tokens),
                "kv_resident_bytes_at_end": kv.resident_bytes(n + args.gen_tokens),
                "kv_layers": kv.n_layers,
                "kv_caches": len(kv.profiles),
                "bytes_per_token": bytes_per_token,
                "bytes_per_token_optimistic": bytes_per_token_opt,
                # results
                "effective_gbs": eff_gbs,
                "effective_gbs_optimistic": eff_gbs_opt,
                "achievable_gbs": args.achievable_gbs,
                "pct_of_roofline": pct_roof,
                "pct_of_roofline_optimistic": pct_roof_opt,
                "energy_pass": bool(args.energy),
            }
            rec.update(r)
            series.append(rec)
            with outp.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

        if series:
            clean = [s for s in series if not s.get("host_contended")]
            n_bad = len(series) - len(clean)
            if n_bad:
                print(f"  !! {n_bad}/{len(series)} run(s) contended — the OS compressed or "
                      f"swapped mid-run. Those measure macOS, not MLX.")
                procs = series[0].get("top_processes") or []
                if procs:
                    print("     biggest resident processes at run start: "
                          + ", ".join(f"{p['name']} {p['rss_mb']}MB" for p in procs[:5]))
                if clean:
                    print(f"     median below uses the {len(clean)} clean run(s) only.")
                else:
                    print("     NO clean runs — treat this whole cell as invalid.")
            d = [s["decode_tok_s"] for s in (clean or series) if s.get("decode_tok_s")]
            if d:
                med = stats.median(d)
                drift = 100.0 * (d[0] - d[-1]) / d[0] if d[0] else 0.0
                monotonic = all(d[i] >= d[i + 1] for i in range(len(d) - 1))
                suspect = drift > 5.0 and monotonic
                print(f"  --> median decode {med:.2f} tok/s   spread "
                      f"{min(d):.2f}-{max(d):.2f}   first->last drift {drift:+.1f}%"
                      f"{'   THERMAL THROTTLING SUSPECTED' if suspect else ''}")

                # Bimodality. A monotonic-decline test misses the far more
                # common Apple Silicon pattern: the SoC sits in one of two
                # power states, so runs cluster into a fast group and a slow
                # group with nothing in between. Averaging across the two
                # produces a median that describes neither.
                ds = sorted(d)
                bimodal = False
                if len(ds) >= 4:
                    gaps = [(ds[i + 1] - ds[i], i) for i in range(len(ds) - 1)]
                    gap, idx = max(gaps)
                    if ds[0] > 0 and gap / ds[0] > 0.05:
                        lo, hi = ds[:idx + 1], ds[idx + 1:]
                        bimodal = True
                        print(f"      !! BIMODAL: {len(lo)} run(s) at "
                              f"{stats.fmean(lo):.2f} and {len(hi)} at {stats.fmean(hi):.2f} tok/s "
                              f"({100 * gap / ds[0]:.1f}% apart, nothing between).")
                        print("         Two SoC power states, not noise. The median describes")
                        print("         neither. Decide which state is the steady one before")
                        print("         quoting a number.")
                # Intra-run throttling: does a single run slow down as it goes?
                dec = [s.get("itl_intra_run_decay_pct") for s in (clean or series)
                       if s.get("itl_intra_run_decay_pct") is not None]
                if dec:
                    md = stats.median(dec)
                    note = "  <- slows DURING each run: thermal, not noise" if md > 4 else ""
                    print(f"      intra-run ITL decay (last quarter vs first): {md:+.1f}%{note}")
                cf = [s.get("cpu_fraction") for s in (clean or series)
                      if s.get("cpu_fraction") is not None]
                cm = [s.get("cpu_ms_per_token") for s in (clean or series)
                      if s.get("cpu_ms_per_token") is not None]
                if cf:
                    mcf, mcm = stats.median(cf), (stats.median(cm) if cm else 0.0)
                    print(f"      CPU time during decode: {100 * mcf:.0f}% of wall "
                          f"({mcm:.1f} ms/token of {stats.median([s['ms_per_token_median'] for s in (clean or series)]):.1f})")
                    if mcf > 0.7:
                        print("         -> CPU-BOUND: graph construction is the bottleneck, not")
                        print("            memory. mx.compile / fewer ops are the lever, and run")
                        print("            times will be sensitive to P-core vs E-core placement.")
                    elif mcf < 0.35:
                        print("         -> GPU-BOUND: the CPU issues and blocks. The shortfall is")
                        print("            per-kernel launch on the GPU side; only fusion helps.")
                eg = [s["effective_gbs"] for s in (clean or series) if s.get("effective_gbs")]
                if eg:
                    med_gbs = stats.median(eg)
                    print(f"      effective {med_gbs:.1f} GB/s", end="")
                    if args.achievable_gbs:
                        pct = 100 * med_gbs / args.achievable_gbs
                        print(f"   = {pct:.1f}% of roofline")
                        if pct > 105:
                            print("      !! OVER 100% OF ROOFLINE — this is impossible, so the")
                            print("         DENOMINATOR is wrong, not the kernel. Re-run bwprobe.py")
                            print("         with GEMV shapes large enough to be DRAM-resident; small")
                            print("         matrices measure dispatch overhead, not bandwidth.")
                            print(f"         Your measured {med_gbs:.1f} GB/s is a lower bound on the"
                                  f" real achievable figure.")
                    else:
                        print("   (pass --achievable-gbs for %-of-roofline)")
                shas = {s["output_sha256"] for s in series}
                if args.temp == 0.0 and len(shas) > 1:
                    print("      !! greedy output differs between runs — "
                          "non-determinism, investigate before trusting anything")
                # persist the drift verdict onto the records already written
                with outp.open("a") as f:
                    f.write(json.dumps({
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "series_summary",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "chip": host["chip"], "model": args.model, "tag": args.tag,
                        "prompt_tokens": n, "gen_tokens": args.gen_tokens,
                        "engine": args.engine, "sync_mode": args.sync,
                        "decode_tok_s_median": med,
                        "decode_tok_s_min": min(d), "decode_tok_s_max": max(d),
                        "thermal_drift_pct": drift,
                        "thermal_suspect": suspect,
                        "bimodal": bimodal,
                        "intra_run_decay_pct_median": (stats.median(dec) if dec else None),
                        "greedy_deterministic": len(shas) == 1,
                        "runs_total_recorded": len(series),
                        "runs_contended": n_bad,
                        "median_from_clean_runs_only": bool(n_bad and clean),
                        "cell_valid": bool(clean),
                        "min_memory_pressure_pct": min(
                            [s.get("memory_pressure_pct") for s in series
                             if s.get("memory_pressure_pct") is not None] or [None]
                        ) if any(s.get("memory_pressure_pct") is not None for s in series) else None,
                    }, default=str) + "\n")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

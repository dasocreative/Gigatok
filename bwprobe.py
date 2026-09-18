#!/usr/bin/env python3
"""
bwprobe.py — what bandwidth can this machine's kernels ACTUALLY hit.

The spec sheet number (M3 base: 100 GB/s over a 128-bit LPDDR5-6400 bus) is a
bus figure. No real kernel reaches it. Every roofline in this project uses the
number this script measures, never the spec number.

Four probes, in increasing order of relevance to batch-1 decode:

  read   sum over a buffer                       N bytes moved
  copy   elementwise add into a new buffer       2N bytes (read + write)
  triad  a*s + b                                 3N bytes
  gemv   matvec at the model's real shapes       the only probe whose access
                                                 pattern matches decode

The size sweep matters: on M-series the system level cache will absorb small
buffers and report absurd bandwidth. Only the plateau at sizes well past the
SLC is the DRAM number. The sweep makes that visible instead of letting you
pick a flattering size.

GEMV is the probe to quote for a decode roofline. A [N, K] weight matrix read
once against a [K] vector is exactly what every projection in a decode step
does, and it is where MLX's quantized kernels either do or do not reach the
DRAM ceiling. Dense fp16 and 4-bit quantized are both measured, because a 4-bit
GEMV moves 1/4 the bytes but runs a different kernel with different efficiency.

Usage:
    python bwprobe.py
    python bwprobe.py --max-mb 1024 --reps 20
    python bwprobe.py --gemv-shape 3840x15360 --group-size 64 --bits 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import mlx.core as mx

import mlxutil as U


def _time(fn: Callable[[], object], reps: int, warmup: int = 3) -> List[float]:
    for _ in range(warmup):
        U.barrier(fn())
    ts: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        U.barrier(out)              # blocks until the command buffer completes
        ts.append(time.perf_counter() - t0)
    return ts


def _report(name: str, size_bytes: int, bytes_moved: int, ts: List[float]) -> dict:
    ts_sorted = sorted(ts)
    best = ts_sorted[0]
    med = ts_sorted[len(ts_sorted) // 2]
    return {
        "probe": name,
        "buffer_bytes": size_bytes,
        "bytes_moved_per_rep": bytes_moved,
        "best_s": best,
        "median_s": med,
        "gbs_best": bytes_moved / best / 1e9,
        "gbs_median": bytes_moved / med / 1e9,
    }


def sweep(max_mb: int, reps: int, dtype=mx.float32) -> List[dict]:
    itemsize = mx.zeros(1, dtype=dtype).nbytes
    sizes_mb = [4, 16, 32, 64, 128, 256, 512, 1024, 2048]
    sizes_mb = [s for s in sizes_mb if s <= max_mb]
    rows: List[dict] = []
    one = mx.array(1.0, dtype=dtype)
    scal = mx.array(1.0000001, dtype=dtype)

    for mb in sizes_mb:
        n = (mb * U.MB) // itemsize
        U.clear_cache()
        x = mx.random.uniform(shape=(n,), dtype=dtype)
        U.barrier(x)
        nb = x.nbytes

        rows.append(_report("read", nb, nb, _time(lambda: mx.sum(x), reps)))
        rows.append(_report("copy", nb, 2 * nb, _time(lambda: mx.add(x, one), reps)))

        y = mx.random.uniform(shape=(n,), dtype=dtype)
        U.barrier(y)
        rows.append(_report("triad", nb, 3 * nb, _time(lambda: mx.add(mx.multiply(x, scal), y), reps)))

        del x, y
        U.clear_cache()
    return rows


def fit_overhead_and_bw(points: List[dict], min_bytes: int) -> dict:
    """
    Separate fixed per-dispatch cost from streaming bandwidth.

    A single timed op costs   t = c + bytes / BW
    where c is Metal dispatch + command-buffer submit + the eval barrier's
    CPU<->GPU round trip. On a small matrix c DOMINATES, and reporting
    bytes/t as "bandwidth" then reports launch overhead wearing a bandwidth
    costume — which is how you end up with a roofline denominator so low that
    real decode appears to run at 250% of it.

    Least-squares over points large enough to be DRAM-resident gives both.
    c is not noise to be discarded: it is the launch-overhead budget that
    mx.compile, command-buffer batching and indirect command buffers go after,
    and the per-step tax any speculative scheme pays gamma times.
    """
    pts = [(p["bytes_moved_per_rep"], p["best_s"]) for p in points
           if p["bytes_moved_per_rep"] >= min_bytes]
    if len(pts) < 2:
        return {"ok": False, "reason": f"need >=2 points over {min_bytes/MB:.0f} MiB"}
    n = len(pts)
    sb = sum(b for b, _ in pts)
    stt = sum(t for _, t in pts)
    sbb = sum(b * b for b, _ in pts)
    sbt = sum(b * t for b, t in pts)
    den = n * sbb - sb * sb
    if den == 0:
        return {"ok": False, "reason": "degenerate fit"}
    m = (n * sbt - sb * stt) / den            # seconds per byte
    c = (stt - m * sb) / n                    # seconds of fixed cost
    if m <= 0:
        return {"ok": False, "reason": "non-physical slope"}
    # R^2 so we can tell whether the linear model actually holds
    tbar = stt / n
    ss_res = sum((t - (c + m * b)) ** 2 for b, t in pts)
    ss_tot = sum((t - tbar) ** 2 for _, t in pts) or 1e-30
    return {
        "ok": True,
        "stream_gbs": 1.0 / m / 1e9,
        "dispatch_overhead_ms": c * 1e3,
        "r2": 1.0 - ss_res / ss_tot,
        "points_used": n,
        "min_bytes": min_bytes,
    }


def gemv(rows: int, cols: int, reps: int, bits: Optional[int], group_size: int) -> dict:
    """
    Matvec: W[rows, cols] @ v[cols]. Bytes moved ~= W.nbytes (+ scales/biases).

    This is the shape decode actually runs. On a 12B model the big ones are
    qkv/o (3840 x 3840-ish), mlp gate/up/down (3840 x 15360), and the vocab
    head (262144 x 3840) — that last one alone is ~8% of a 4-bit 12B model's
    bytes and is read on every single token.
    """
    U.clear_cache()
    v = mx.random.uniform(shape=(1, cols), dtype=mx.float16)
    if bits is None:
        W = mx.random.uniform(shape=(rows, cols), dtype=mx.float16)
        U.barrier(W, v)
        moved = W.nbytes
        fn = lambda: mx.matmul(v, W.T)
        label = f"gemv_fp16_{rows}x{cols}"
    else:
        Wf = mx.random.uniform(shape=(rows, cols), dtype=mx.float16)
        Wq, scales, biases = mx.quantize(Wf, group_size=group_size, bits=bits)
        U.barrier(Wq, scales, biases, v)
        moved = Wq.nbytes + scales.nbytes + biases.nbytes
        del Wf
        fn = lambda: mx.quantized_matmul(
            v, Wq, scales, biases, True, group_size, bits
        )
        label = f"gemv_q{bits}g{group_size}_{rows}x{cols}"
    ts = _time(fn, reps)
    out = _report(label, moved, moved, ts)
    out["effective_bits_per_weight"] = moved * 8 / (rows * cols)
    U.clear_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure achievable memory bandwidth")
    ap.add_argument("--max-mb", type=int, default=1024,
                    help="Largest sweep buffer. Keep well under your working set.")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--gemv-cols", type=int, default=3840,
                    help="K dimension = model hidden size. 3840 for Gemma-4-12B.")
    ap.add_argument("--gemv-rows", default="1024,4096,16384,65536,262144",
                    help="N dimensions to sweep. Must span well past the SLC (~8 MB "
                         "on M3 base) or the fit measures cache, not DRAM.")
    ap.add_argument("--fit-min-mb", type=int, default=32,
                    help="Only points at least this large enter the bandwidth fit.")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--json", default="bwprobe.json")
    args = ap.parse_args()

    host = U.host_info()
    mws = host["max_recommended_working_set_bytes"]
    print(f"chip                    : {host['chip']}")
    print(f"gpu arch                : {host['gpu_architecture']}")
    print(f"physical memory         : {U.fmt_bytes(host['physical_memory_bytes'])}")
    print(f"max working set (Metal) : {U.fmt_bytes(mws)}")
    print(f"iogpu.wired_limit_mb    : {host['iogpu_wired_limit_mb']} (0 = system default)")
    print(f"mlx                     : {host['mlx_version']}")
    print()

    if mws and args.max_mb * U.MB > mws * 0.35:
        cap = int(mws * 0.35 / U.MB)
        print(f"note: capping --max-mb {args.max_mb} -> {cap} to stay clear of the working set\n")
        args.max_mb = cap

    rows = sweep(args.max_mb, args.reps)
    print(f"{'probe':<8}{'buffer':>12}{'GB/s best':>12}{'GB/s med':>12}")
    for r in rows:
        print(f"{r['probe']:<8}{U.fmt_bytes(r['buffer_bytes']):>12}"
              f"{r['gbs_best']:>12.1f}{r['gbs_median']:>12.1f}")

    print()
    gem: List[dict] = []
    cc = args.gemv_cols
    for rr in (int(x) for x in args.gemv_rows.split(",")):
        for b in (None, args.bits):
            try:
                gem.append(gemv(rr, cc, args.reps, b, args.group_size))
            except Exception as e:
                print(f"  gemv {rr}x{cc} bits={b} failed: {type(e).__name__}: {e}")
    fit_min = args.fit_min_mb * U.MB
    print(f"{'gemv probe':<32}{'bytes':>12}{'GB/s best':>11}{'us/call':>10}  note")
    for r in gem:
        note = "launch/cache-bound - excluded from fit" if r["buffer_bytes"] < fit_min else ""
        print(f"{r['probe']:<32}{U.fmt_bytes(r['buffer_bytes']):>12}"
              f"{r['gbs_best']:>11.1f}{r['best_s'] * 1e6:>10.0f}  {note}")

    big = [r for r in rows if r["buffer_bytes"] >= 256 * U.MB]
    plateau_read = max((r["gbs_best"] for r in big if r["probe"] == "read"), default=0.0)
    plateau_stream = max((r["gbs_best"] for r in big if r["probe"] in ("copy", "triad")), default=0.0)
    q_gemv = [r for r in gem if r["probe"].startswith("gemv_q")]
    f_gemv = [r for r in gem if r["probe"].startswith("gemv_fp")]
    plateau_gemv = max((r["gbs_best"] for r in q_gemv
                        if r["buffer_bytes"] >= fit_min), default=0.0)

    fit_q = fit_overhead_and_bw(q_gemv, fit_min)
    fit_f = fit_overhead_and_bw(f_gemv, fit_min)

    print()
    print("=" * 76)
    print("  DISPATCH OVERHEAD vs STREAMING BANDWIDTH    t = c + bytes / BW")
    for name, f in ((f"q{args.bits} GEMV", fit_q), ("fp16 GEMV", fit_f)):
        if f.get("ok"):
            print(f"    {name:<10} BW {f['stream_gbs']:6.1f} GB/s    "
                  f"fixed cost {f['dispatch_overhead_ms'] * 1e3:6.0f} us    "
                  f"R2 {f['r2']:.4f}  (n={f['points_used']})")
        else:
            print(f"    {name:<10} fit unavailable: {f.get('reason')}")
    print("=" * 76)
    print(f"  achievable read     (>=256 MiB) : {plateau_read:6.1f} GB/s")
    print(f"  achievable stream   (copy/triad): {plateau_stream:6.1f} GB/s")
    print(f"  largest q{args.bits} GEMV single call   : {plateau_gemv:6.1f} GB/s")

    # The denominator is the streaming bandwidth of the kernel decode actually
    # uses, with launch overhead removed. Falls back to the largest single call,
    # then to the read plateau.
    denom, src = 0.0, ""
    if fit_q.get("ok") and fit_q.get("r2", 0) > 0.90:
        denom, src = fit_q["stream_gbs"], f"q{args.bits} GEMV fit, launch cost removed"
    if denom <= 0:
        denom, src = plateau_gemv, "largest q GEMV call"
    if denom <= 0:
        denom, src = plateau_read, "read plateau"
    # A GEMV fit that beats the pure-read plateau is over-extrapolation, not a
    # faster kernel: reads are the cheapest possible access pattern. With only
    # 3-4 fit points a small timing error at the largest shape swings the slope.
    if denom > plateau_read * 1.01 and plateau_read > 0:
        print(f"\n  !! fit ({denom:.1f}) exceeds the measured read plateau "
              f"({plateau_read:.1f}). That is not physical —")
        print("     the fit is over-extrapolating. Clamping to the read plateau.")
        denom, src = plateau_read, "read plateau (GEMV fit was over-extrapolated)"
    print("=" * 76)
    print(f"  ROOFLINE DENOMINATOR            : {denom:6.1f} GB/s   ({src})")
    print("=" * 76)
    print(f"    --achievable-gbs {denom:.1f}")
    print()
    print("  Sanity rule: measured effective GB/s during decode must land BELOW")
    print("  this number. If bench.py reports over 100% of roofline the")
    print("  denominator is wrong — almost always because the GEMV shapes were")
    print("  small enough that dispatch overhead dominated the timing.")

    Path(args.json).write_text(json.dumps(
        {"host": host, "sweep": rows, "gemv": gem,
         "fit_quantized": fit_q, "fit_fp16": fit_f,
         "achievable_read_gbs": plateau_read,
         "achievable_stream_gbs": plateau_stream,
         "largest_gemv_call_gbs": plateau_gemv,
         "achievable_gemv_gbs": denom,
         "denominator_source": src}, indent=2, default=str))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

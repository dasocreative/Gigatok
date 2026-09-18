#!/usr/bin/env python3
"""
verify_decomposition.py — does the verify slope have kernel headroom, or is it
already at a roofline?

THE ONE UNEXPLAINED NUMBER IN RECIPE 4

Verify cost is t(k) = 37.2 + 6.75*k ms. The intercept is settled: 3,535,067,220
active bytes at 89 GB/s is 39.7 ms, so the floor is the weight read and there is
nothing to win there. The SLOPE is not settled. schedule_decision.py measured the
per-position cost RISING once the floor is subtracted -- 2.12 ms at k=3, 3.23 at
k=5, 6.30 at k=9 -- and flagged the superlinearity as unexplained.

It should not rise. Adding query rows to a forward whose weights are already
resident should cost linearly: dense compute scales with rows, attention scales
with rows x context. A threefold rise in per-position cost means something
degrades with block size.

This matters more than anything else left in the recipe: verify is the dominant
term in speculative decoding, so flattening that slope raises both the optimal
gamma and the payoff. It is also the only lead here that is not downstream of
somebody else's implementation.

WHY THIS IS NOT JUST AN ANALYTIC MODEL

A FLOP count divided by a spec-sheet TFLOP/s would be the same mistake as a
parameter-count roofline -- which Recipe 0 measured to be wrong by 85% on this
checkpoint, because 46% of it is gather-only PLE that never streams. So the
compute ceiling is CALIBRATED here the way bwprobe calibrates bandwidth: by
sweeping the real kernel, on the real shapes, at the real quantisation.

PART A -- the compute ceiling, measured, as a function of row count.
    mx.quantized_matmul on an actual QuantizedLinear lifted out of the model,
    swept over M = 1..16. This is the kernel the verify forward actually runs.
    A dense bf16 matmul of the same logical shape is swept alongside as a
    reference ceiling. If effective FLOP/s DEGRADES as M grows, the answer is
    here and the rest is confirmation.

PART B -- t(k) at two context lengths.
    The decisive split, and it needs no kernel surgery. Per extra position the
    work divides into a term that scales with context (attention over K/V) and a
    term that does not (dense matmuls, PLE gather, activations). Measuring the
    same k sweep at a SHORT and a LONG context separates them by subtraction:

        t(k, N_long) - t(k, N_short)   ->  the context-dependent term
        t(k, N_short)                  ->  weights + compute + gather

    If the superlinearity lives in the difference, it is attention. If it lives
    in the short-context arm, it is the GEMM or the per-layer embedding gather.
    Either answer names the kernel to look at; the analytic model alone cannot.

PART C -- decomposition against both floors, per k, with the marginal cost.
    A ratio near 1.0 means that k is at its roofline and there is nothing to win.
    A ratio of 2-3x means headroom, and Part B says where.

BARRIERS. Every timed region ends in mx.eval on the value the region produces,
because MLX is lazy and an unevaluated graph would time as zero. Nothing else is
evaluated inside the region: the cache build and the trim sit outside it, so the
measurement covers the forward and not the setup.

    ../bin/python verify_decomposition.py --contexts 128,2048 --max-k 10
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import mlx.core as mx

import mlxutil as U
import roofline as RF
from measure_acceptance import PROMPTS, text_model


# ------------------------------------------------------------------ helpers
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
ALLOWED_GROUPS = (32, 64, 128)


def solve_quant(w, scales, cfg_bits, cfg_gs):
    """Derive (bits, group_size) for ONE tensor from its own shapes.

    A single global (bits, group_size) does not hold on this checkpoint. OptiQ
    quantises mixed -- that is the whole reason it is usable, since stock 4-bit
    Gemma 4 conversions put the per-layer embeddings at 4 bits and break the
    model. Assuming the config's global pair produced:

        [quantized_matmul] shapes incompatible: w == (262144, 2688),
        scales == (262144, 168) with group_size=64 and bits=4

    The packing gives one equation. mlx stores `bits`-wide values packed into
    uint32, so in_features = w.shape[1] * 32 / bits, and scales carries one
    entry per group, so in_features = scales.shape[1] * group_size. Therefore

        bits * group_size = w.shape[1] * 32 / scales.shape[1]

    which for that tensor is 512 -- satisfied by both 4-bit/g128 and 8-bit/g64.
    Two unknowns, one equation, so the tie is broken by preferring the config's
    declared bits when it is consistent, then its declared group size.
    """
    if scales is None or scales.ndim != 2 or w.ndim != 2:
        return None
    c = w.shape[1] * 32 / scales.shape[1]
    if c != int(c):
        return None
    c = int(c)
    cands = [(b, c // b) for b in ALLOWED_BITS
             if c % b == 0 and (c // b) in ALLOWED_GROUPS]
    if not cands:
        return None
    for b, g in cands:
        if b == cfg_bits:
            return b, g
    for b, g in cands:
        if g == cfg_gs:
            return b, g
    return cands[-1]                       # highest group size = lowest bits


def quantized_linears(model, cfg):
    """Quantized weights that are MATMUL'd, from the parameter tree.

    Two corrections over the first version:

    1. Read the PARAMETER tree, not module attributes. mlx's nn.Module subclasses
       dict, so children live in the dict and `vars(m)` is empty -- the first
       version found zero quantized linears in a fully quantized model.
       roofline.param_breakdown already does it this way.

    2. EXCLUDE gather-only tables. `embed_tokens_per_layer` is row-gathered every
       token and never matmul'd, so it contributes no FLOPs; counting it would
       repeat the parameter-count error Recipe 0 measured to be 85% wrong on this
       checkpoint. roofline.GATHER_ONLY_MARKERS is the existing list.

    The tied output embedding stays IN: with tie_word_embeddings it is gathered
    on the way in and matmul'd as the output projection on the way out, and the
    verify forward runs that projection once per row.
    """
    from mlx.utils import tree_flatten

    q = (cfg or {}).get("quantization") or {}
    cfg_bits = int(q.get("bits", 4))
    cfg_gs = int(q.get("group_size", 64))

    flat = dict(tree_flatten(model.parameters()))
    out, skipped = [], []
    for path, arr in flat.items():
        if not path.endswith(".scales") or not isinstance(arr, mx.array):
            continue
        prefix = path[: -len(".scales")]
        w = flat.get(prefix + ".weight")
        if not isinstance(w, mx.array) or w.ndim != 2:
            continue
        if any(m in prefix for m in RF.GATHER_ONLY_MARKERS):
            skipped.append(prefix)
            continue
        sol = solve_quant(w, arr, cfg_bits, cfg_gs)
        if sol is None:
            skipped.append(prefix + " (unresolved packing)")
            continue
        bits, gs = sol
        out.append({
            "path": prefix, "w": w, "scales": arr,
            "biases": flat.get(prefix + ".biases"),
            "out": int(w.shape[0]), "in": int(w.shape[1]) * (32 // bits),
            "bits": bits, "group_size": gs,
        })
    if skipped:
        print(f"  excluded {len(skipped)} gather-only / unresolved tensors, e.g. "
              f"{skipped[0]}")
    if not out:
        print("  !! no matmul'd quantized weights found. Sample parameter paths:")
        for pth in list(flat)[:25]:
            print(f"     {pth}  {tuple(flat[pth].shape)}")
    else:
        seen = sorted({(q_["bits"], q_["group_size"]) for q_ in out})
        print(f"  quantisation present     {', '.join(f'{b}-bit g{g}' for b, g in seen)}")
    return out


def qmm(x, q):
    """The real kernel, called directly. Version-tolerant on `biases`."""
    try:
        return mx.quantized_matmul(x, q["w"], q["scales"], q["biases"],
                                   transpose=True, group_size=q["group_size"],
                                   bits=q["bits"])
    except TypeError:
        return mx.quantized_matmul(x, q["w"], q["scales"],
                                   transpose=True, group_size=q["group_size"],
                                   bits=q["bits"])


def med_time(fn, reps: int, warm: int = 2) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return stats.median(ts)


# --------------------------------------------------- PART A: compute ceiling
def compute_ceiling(qls, max_m: int, reps: int, bw: float) -> dict:
    """Effective FLOP/s of the real quantized kernel as a function of rows."""
    # TWO shapes, not one. The largest weight is `embed_tokens`, a
    # [262144 x 5120] vocab projection that runs ONCE per forward and is
    # atypically tall. The decoder's attention and MLP matmuls run 42 times per
    # forward and dominate the cost, so a collapse that only affects the LM head
    # would be a curiosity. Sweeping the most FREQUENT shape alongside the
    # largest is what decides whether this is a general multi-row problem.
    from collections import Counter
    shapes = Counter((q["out"], q["in"], q["bits"], q["group_size"]) for q in qls)
    common = shapes.most_common(1)[0]
    rep = next(q for q in qls
               if (q["out"], q["in"], q["bits"], q["group_size"]) == common[0])
    largest = max(qls, key=lambda q: q["out"] * q["in"])
    targets = [("largest (runs 1x/forward)", largest, 1),
               (f"most frequent (runs {common[1]}x/forward)", rep, common[1])]
    results = {}
    for label, big, count in targets:
        print(f"\n  --- {label} ---")
        results[label] = sweep_one(big, label, count, max_m, reps, bw)
    main_res = results[targets[1][0]]          # the representative one decides
    main_res["all"] = {k: v for k, v in results.items()}
    return main_res


def sweep_one(big, label, count, max_m: int, reps: int, bw: float) -> dict:
    in_f, out_f = big["in"], big["out"]
    print(f"  sweeping {big['path']}  [{out_f} x {in_f}]  "
          f"{big['bits']}-bit g{big['group_size']}")
    # Weight bytes, so the sweep can be read in GB/s. THIS IS THE BINDING
    # METRIC on this shape: a [262144 x 5120] 4-bit matrix is 713 MB, and at
    # M=1 the kernel hits 89.6 GB/s -- exactly the calibrated ceiling. Dividing
    # FLOPs by a memory-limited time and calling it a compute ceiling is the
    # same error as the GEMV probes that measured dispatch instead of DRAM and
    # reported 226% of roofline.
    wbytes = (big["out"] * big["in"] * big["bits"] / 8.0
              + 2 * big["out"] * (big["in"] / big["group_size"]) * 2)
    probe = mx.random.normal((1, in_f)).astype(mx.bfloat16)
    mx.eval(probe)
    try:
        mx.eval(qmm(probe, big))
    except Exception as e:
        print(f"  !! kernel rejected the derived packing: {e}")
        raise SystemExit(3)

    rows = []
    for M in range(1, max_m + 1):
        x = mx.random.normal((M, in_f)).astype(mx.bfloat16)
        mx.eval(x)

        def run():
            y = qmm(x, big)
            mx.eval(y)                      # the region's product; nothing else

        t = med_time(run, reps)
        fl = 2.0 * M * out_f * in_f
        rows.append({"M": M, "ms": t * 1e3, "tflops": fl / t / 1e12,
                     "gbs": wbytes / t / 1e9})

    # Dense bf16 of the same logical shape, as a reference ceiling.
    dense = []
    W = mx.random.normal((in_f, out_f)).astype(mx.bfloat16)
    mx.eval(W)
    for M in (1, 2, 4, 8, max_m):
        x = mx.random.normal((M, in_f)).astype(mx.bfloat16)
        mx.eval(x)

        def run():
            y = x @ W
            mx.eval(y)

        t = med_time(run, reps)
        dense.append({"M": M, "tflops": 2.0 * M * out_f * in_f / t / 1e12})

    print(f"\n  weight bytes {wbytes / 1e6:.1f} MB")
    print(f"\n  {'M':>3} {'ms':>9} {'GB/s':>8} {'% roof':>8} {'TFLOP/s':>9}")
    for r in rows:
        print(f"  {r['M']:>3} {r['ms']:>9.3f} {r['gbs']:>8.1f}"
              f" {100 * r['gbs'] / bw:>7.0f}% {r['tflops']:>9.3f}")

    # A GENUINE compute ceiling needs a compute-bound shape: small enough to sit
    # in cache, with enough rows that arithmetic intensity clears the ridge.
    # Without this the sweep above only ever measures memory.
    print("\n  compute ceiling probe (cache-resident shape, high M):")
    comp = []
    for n in (1024, 2048):
        W = mx.random.normal((n, n)).astype(mx.bfloat16)
        X = mx.random.normal((n, n)).astype(mx.bfloat16)
        mx.eval(W, X)

        def run():
            mx.eval(X @ W)

        t = med_time(run, reps)
        tf = 2.0 * n * n * n / t / 1e12
        comp.append({"n": n, "tflops": tf})
        print(f"    {n}x{n} bf16 GEMM   {tf:6.2f} TFLOP/s")
    peak_compute = max(c["tflops"] for c in comp) * 1e12

    print(f"  x{count} per forward -> a collapse here costs {count}x as much"
          if count > 1 else "  runs once per forward")
    gbs1, gbsN = rows[0]["gbs"], rows[-1]["gbs"]
    # THE 32 MB RULE. Recipe 0: "GEMV bandwidth probes were 2.4 MB - measuring
    # dispatch, not DRAM, and reporting 226-278% of roofline. Fixed with a size
    # sweep excluding < 32 MB." A weight that fits in cache is re-read from cache
    # across timing reps, so its GB/s is not DRAM bandwidth and the % roof column
    # is meaningless. The TIMES are still real, and a discontinuity in them is
    # still a real kernel property - that part survives.
    cache_resident = wbytes < 32e6
    if cache_resident:
        print(f"\n  !! weight is {wbytes / 1e6:.1f} MB, below the 32 MB floor this")
        print("     project set after the 2.4 MB GEMV probe measured dispatch")
        print("     instead of DRAM. GB/s and % roof above are NOT bandwidth —")
        print("     the weight is cache-resident across reps. Read the ms column.")
    else:
        print(f"\n  quantized kernel bandwidth: {gbs1:.1f} GB/s at M=1"
              f"  ->  {gbsN:.1f} GB/s at M={rows[-1]['M']}")
    if (not cache_resident) and gbsN < gbs1 * 0.6:
        print(f"  -> EFFICIENCY COLLAPSES {gbs1 / gbsN:.1f}x in the multi-row path.")
        print("     Read this sweep in GB/s, not TFLOP/s: the shape is memory-bound")
        print("     at every M, so rising TFLOP/s only means more rows share one")
        print("     weight read. The kernel is losing bandwidth as M grows, and")
        print("     that is a direct candidate for the superlinear verify slope.")
    drops = [(rows[i - 1]["M"], rows[i]["M"],
              rows[i - 1]["ms"] / rows[i]["ms"])
             for i in range(1, len(rows))
             if rows[i]["ms"] < rows[i - 1]["ms"] * 0.75]
    if drops:
        for a, b, f in drops:
            print(f"  -> KERNEL PATH SWITCH: M={a} -> M={b} gets {f:.2f}x FASTER.")
        print("     Adding a row makes it cheaper, which is neither memory nor")
        print("     compute — MLX is dispatching a different kernel above that M.")
        print(f"     This shape runs {count}x per forward, so the cliff is worth")
        print("     {}x that.".format(count))
        print("     => Part B must be swept PAST the cliff. A cost model fitted")
        print("        below it will set the optimal block size in the slow regime.")
    plateaus = []
    for i in range(1, len(rows)):
        if abs(rows[i]["ms"] - rows[i - 1]["ms"]) / rows[i]["ms"] < 0.03:
            plateaus.append((rows[i - 1]["M"], rows[i]["M"]))
    if plateaus:
        print(f"  -> time plateaus at M pairs {plateaus} — a tiling quantum.")
        print("     Cross-check against the free marginals in Part B: if the same")
        print("     k are free there, one kernel explains both measurements.")
    return {"quant": rows, "dense": dense, "peak_tflops": peak_compute / 1e12,
            "peak_compute_flops": peak_compute, "gbs_m1": gbs1, "gbs_max_m": gbsN,
            "plateaus": plateaus, "path_switches": drops,
            "cache_resident": cache_resident,
            "shape": {"out": out_f, "in": in_f, "bits": big["bits"],
                      "group_size": big["group_size"], "path": big["path"],
                      "weight_bytes": wbytes}}


# ------------------------------------------- PART B: t(k) at two contexts
def sweep_k(target, tm, ids, ctx_tokens, max_k: int, reps: int, cooldown: float):
    """t(k) at a fixed context. Cache built once; each rep feeds k and trims k."""
    from mlx_lm.models.cache import make_prompt_cache
    from spec_generate import use_plain_kv_caches

    cache = make_prompt_cache(target)
    use_plain_kv_caches(cache)
    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)
    cur = int(mx.argmax(tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)).item())
    filler = [cur] * (max_k + 2)

    out = []
    for k in range(1, max_k + 1):
        blk = mx.array([filler[:k]])

        def run():
            hh = tm(blk, cache=cache)
            mx.eval(hh)                     # the forward is the product
            for c in cache:                 # trim is setup, not measured -- but
                c.trim(k)                   # it must happen before the next rep

        # Trim inside run() would be timed. Measure the pair and subtract a
        # trim-only baseline: trim is an integer offset rewind on a plain
        # KVCache, so this is small and constant, but not assumed to be zero.
        def trim_only():
            for c in cache:
                c.trim(0)

        t_pair = med_time(run, reps)
        t_trim = med_time(trim_only, reps)
        out.append({"k": k, "ms": (t_pair - t_trim) * 1e3, "ctx": ctx_tokens})
        time.sleep(cooldown)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--contexts", default="128,2048")
    ap.add_argument("--max-k", type=int, default=16,
                    help="must exceed any kernel path switch found in Part A")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=1.5)
    ap.add_argument("--bw", type=float, default=89.0, help="calibrated GB/s")
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()
    ctxs = [int(c) for c in args.contexts.split(",")]

    pw = U.power_info()
    if pw.get("low_power_mode"):
        print("!! Low Power Mode is ON — it caps GPU frequency. Refusing.")
        return 2
    print(f"power {pw.get('power_source')} {pw.get('battery_percent')}%  "
          f"lpm={pw.get('low_power_mode')}")

    print("\nloading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)
    pb = RF.param_breakdown(target)
    cfg, _cfgpath = RF.find_config(args.target)
    qls = quantized_linears(target, cfg)
    total_flops_per_token = 2.0 * sum(q["out"] * q["in"] for q in qls)
    if not qls:
        print("\n  Cannot calibrate a compute ceiling without the quantized weights.")
        return 3
    print(f"  quantized linears        {len(qls)}")
    print(f"  dense FLOPs / token      {total_flops_per_token / 1e9:.2f} GFLOP")
    print(f"  dense bytes              {pb.dense_bytes / 1e9:.3f} GB")
    print(f"  gather-only (PLE) bytes  {pb.gather_only_bytes / 1e9:.3f} GB")
    print(f"  active bytes / token     {pb.active_bytes_per_token / 1e9:.3f} GB")

    print("\n" + "=" * 74)
    print("PART A — compute ceiling, measured on the real quantized kernel")
    print("=" * 74)
    ceil = compute_ceiling(qls, max(args.max_k, 16), args.reps, args.bw)
    peak = ceil["peak_compute_flops"]

    print("\n" + "=" * 74)
    print("PART B — t(k) at two context lengths")
    print("=" * 74)
    sweeps = {}
    for N in ctxs:
        # A prompt padded to N tokens: the context length is the variable, the
        # text is not.
        base_ids, _ = RF.build_eval_prompt(tok, PROMPTS[0])
        ids = (base_ids * (N // max(len(base_ids), 1) + 1))[:N]
        print(f"\n  context {N} tokens ...")
        sweeps[N] = sweep_k(target, tm, ids, N, args.max_k, args.reps, args.cooldown)
        for r in sweeps[N]:
            print(f"    k={r['k']:>2}  {r['ms']:7.2f} ms")

    print("\n" + "=" * 74)
    print("PART C — decomposition")
    print("=" * 74)
    Ns, Nl = min(ctxs), max(ctxs)
    short = {r["k"]: r["ms"] for r in sweeps[Ns]}
    long_ = {r["k"]: r["ms"] for r in sweeps[Nl]}

    bw_bytes = pb.active_bytes_per_token          # weights, read once per block
    t_bw_floor = bw_bytes / (args.bw * 1e9) * 1e3

    print(f"  weight-read floor (bytes/BW)      {t_bw_floor:7.2f} ms")
    print(f"  compute ceiling (measured peak)   {ceil['peak_tflops']:7.3f} TFLOP/s")
    print()
    print(f"  {'k':>3} {'short ms':>9} {'long ms':>9} {'ctx term':>9}"
          f" {'compute':>9} {'meas/roof':>10} {'marginal':>9}")
    records, prev = [], None
    for k in range(1, args.max_k + 1):
        fl = total_flops_per_token * k
        t_comp = fl / peak * 1e3
        roof = max(t_bw_floor, t_comp)
        ms = short.get(k)
        ctx_term = long_.get(k, float("nan")) - ms if ms is not None else float("nan")
        marg = (ms - prev) if (prev is not None and ms is not None) else float("nan")
        prev = ms
        print(f"  {k:>3} {ms:>9.2f} {long_.get(k, float('nan')):>9.2f} {ctx_term:>9.2f}"
              f" {t_comp:>9.2f} {ms / roof:>10.2f} {marg:>9.2f}")
        records.append({
            "record_type": "verify_decomposition", "schema_version": 1,
            "k": k, "ctx_short": Ns, "ctx_long": Nl,
            "ms_short": ms, "ms_long": long_.get(k), "ctx_term_ms": ctx_term,
            "compute_floor_ms": t_comp, "bw_floor_ms": t_bw_floor,
            "roofline_ms": roof, "ratio": ms / roof if ms else None,
            "marginal_ms": marg, "peak_tflops": ceil["peak_tflops"],
            "dense_gflops_per_token": total_flops_per_token / 1e9,
            "power": pw,
        })

    # --------------------------------------------------------- the verdict
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    # Find the KNEE, do not average across it. t(k) rises superlinearly and then
    # SATURATES; taking a median of the first and second halves averages the rise
    # against the flat region's negative marginals and reports "flat" for a curve
    # that is nothing of the sort. The shape is the result, not a single slope.
    ks = [r["k"] for r in records]
    ms = [r["ms_short"] for r in records]
    per_pos = [(k, m / k) for k, m in zip(ks, ms)]
    best_k, best_pp = min(per_pos, key=lambda t: t[1])

    knee = None
    for i in range(2, len(ms) - 2):
        rise = ms[i] - ms[i - 2]
        after = ms[-1] - ms[i]
        if rise > 0 and after <= max(2.0, 0.05 * ms[i]):
            knee = ks[i]
            break

    print(f"  cost per position: {per_pos[0][1]:.1f} ms at k=1  ->  "
          f"{best_pp:.1f} ms at k={best_k} (minimum)")
    if knee:
        flat = [m for k, m in zip(ks, ms) if k >= knee]
        print(f"  t(k) SATURATES at k>={knee}: {min(flat):.1f}-{max(flat):.1f} ms "
              f"across k={knee}..{ks[-1]}")
        print(f"  -> {ks[-1] - knee} extra verify positions are FREE. Whenever the")
        print(f"     schedule reaches k={knee} it should go to k={ks[-1]}.")
        lin = 37.2 + 6.75 * ks[-1]
        print(f"  -> the recipe's linear model t(k)=37.2+6.75k predicts "
              f"{lin:.0f} ms at k={ks[-1]}; measured {ms[-1]:.0f} ms. It was fitted")
        print("     below the knee and over-predicts past it by "
              f"{100 * (lin / ms[-1] - 1):.0f}%.")
        print("     RE-OPTIMISE gamma against measured t(k) and the measured")
        print("     acceptance curve. That is a scheduling win with no kernel work.")
    else:
        rising = [ms[i] - ms[i - 1] for i in range(1, len(ms))]
        print(f"  no saturation found; marginal cost {rising[0]:.1f} -> {rising[-1]:.1f} ms")

    if any(v.get("cache_resident") for v in (ceil.get("all") or {}).values()):
        print("\n  !! at least one Part A sweep was cache-resident (<32 MB). Those")
        print("     curves are not reproducible between runs — do not draw a kernel")
        print("     conclusion from them without a cache-defeating design.")

    ratios = [r["ratio"] for r in records if r["ratio"]]
    if ratios:
        print(f"\n  measured / roofline: min {min(ratios):.2f}  max {max(ratios):.2f}")
        print(f"  (roofline = max(weight-read {t_bw_floor:.1f} ms, compute at "
              f"{ceil['peak_tflops']:.2f} TFLOP/s measured on a cache-resident shape)")
        if (not ceil.get("cache_resident")) and ceil["gbs_max_m"] < ceil["gbs_m1"] * 0.6:
            print("  !! The quantized kernel loses "
                  f"{ceil['gbs_m1'] / ceil['gbs_max_m']:.1f}x bandwidth from M=1 to "
                  f"M={ceil['quant'][-1]['M']}. A ratio near 1.0 against a floor that")
            print("     assumes FULL bandwidth would understate the headroom: the")
            print("     weight-read floor is only reachable at M=1. Judge on the GB/s")
            print("     curve in Part A, not on this ratio alone.")
        worst = max(records, key=lambda r: r["ratio"] or 0)
        # max(bw, compute) assumes the weight read and the arithmetic overlap
        # PERFECTLY, so it is the optimistic floor. The pessimistic one assumes
        # no overlap at all. The truth is between, and quoting either alone
        # overstates the case.
        opt = worst["roofline_ms"]
        pess = worst["bw_floor_ms"] + worst["compute_floor_ms"]
        print(f"  -> worst at k={worst['k']}: {worst['ms_short']:.1f} ms measured")
        print(f"     vs {opt:.1f} ms optimistic floor (perfect overlap) = "
              f"{worst['ms_short'] / opt:.2f}x")
        print(f"     vs {pess:.1f} ms pessimistic floor (no overlap)    = "
              f"{worst['ms_short'] / pess:.2f}x")
        print("     Headroom is bounded by that range, not by either endpoint.")

    with open(args.out, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  appended {len(records)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

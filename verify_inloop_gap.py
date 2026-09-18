#!/usr/bin/env python3
"""
verify_inloop_gap.py — why does the verify cost 52 ms inside the loop when the
same k=4 forward costs 44 ms on its own?

THE GAP THIS EXISTS TO EXPLAIN

verify_decomposition measured t(k=4) = 44.09 ms at ctx 128, ratio 1.110 against the
39.72 ms bandwidth floor -- i.e. a 4-row verify is close to its roofline. Its
context term is small at the length this loop actually runs (10.80 ms for
128 -> 2048, so ~0.3 ms for 128 -> 186).

But spec_decode's own barrier timer says the in-loop verify waits 52.1 ms. That is
~7.7 ms per cycle -- ~13 % of the cycle -- that is neither row count nor context.
Phase 0 wrongly booked it as "batching 4 tokens is less bandwidth-efficient";
the isolated sweep says it is not. Something the LOOP does, and the isolated
forward does not, is the cause.

WHAT THE LOOP DOES THAT THE ISOLATED FORWARD DOES NOT

  B  the KV-capture wrapper on DecoderLayer.__call__, which stashes a reference to
     every layer's K/V on every forward and keeps 42 of them alive
  C  use_plain_kv_caches: 20 RotatingKVCache -> plain KVCache, so sliding layers
     retain the whole context instead of a 512 ring
  D  collect_shared_kv + slice_shared_kv every cycle, which allocates fresh sliced
     K/V copies for the drafter and holds them across the cycle

Each is added CUMULATIVELY to the same measurement, so the difference between
consecutive conditions is that ingredient's cost. Anything left over after D is
genuinely unexplained and should be reported as such rather than attributed.

METHOD. Context is held FIXED: after every timed verify the cache is rewound by k,
so iteration i+1 runs at exactly the context iteration i did. Median of --iters
after --warmup discarded. Every timed region ends in mx.eval on the value it
produced, because MLX is lazy and an unevaluated graph times as zero.

    ../bin/python verify_inloop_gap.py --ctx 186 --k 4
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import mlx.core as mx

import mlxutil as U
import roofline as RF
from measure_acceptance import collect_shared_kv, enable_kv_capture, text_model
from spec_generate import rewind, slice_shared_kv, use_plain_kv_caches

BW, ACTIVE_BYTES = 89e9, 3.535e9


def prefill(target, tm, ids, plain: bool):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(target)
    if plain:
        use_plain_kv_caches(cache)
    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)
    return cache


def time_verify(target, tm, cache, k, iters, warmup, keep_shared_kv):
    """Median wall time of one k-row verify, context held fixed by rewinding."""
    xv = mx.array([[2] * k])          # ids are gather indices; values do not time
    held = None
    ts = []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        hv = tm(xv, cache=cache)
        preds = mx.argmax(tm.embed_tokens.as_linear(hv), axis=-1)[0]
        mx.eval(preds)
        dt = time.perf_counter() - t0
        if keep_shared_kv:
            # exactly what the loop holds across a cycle
            held = slice_shared_kv(collect_shared_kv(target), 1)
            mx.eval([v for kv in held.values() for v in kv])
        rewind(cache, k)              # restore the context for the next iteration
        if i >= warmup:
            ts.append(dt * 1e3)
    return stats.median(ts), held


def time_split(target, tm, cache, k, iters, warmup):
    """Forward-only vs forward+LM-head, alternated at the SAME context.

    verify_decomposition times `mx.eval(hh)` on the forward alone. spec_decode's
    verify barrier also pays `tm.embed_tokens.as_linear(hv)` -- the LM head over a
    262144 vocab, which OptiQ keeps at 8 bits. If that is the missing term, the
    project's t(k) curve understates the cost the loop actually pays every cycle,
    and every prediction built on it (the k=4 optimum, 17.0 ms/token, 58.8 tok/s)
    inherits the error. Alternating the two within one iteration keeps thermal and
    cache state common to both.
    """
    xv = mx.array([[2] * k])
    fwd, full = [], []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        hv = tm(xv, cache=cache)
        mx.eval(hv)                                     # forward only
        d1 = time.perf_counter() - t0
        rewind(cache, k)

        t0 = time.perf_counter()
        hv = tm(xv, cache=cache)
        pr = mx.argmax(tm.embed_tokens.as_linear(hv), axis=-1)[0]
        mx.eval(pr)                                     # forward + LM head
        d2 = time.perf_counter() - t0
        rewind(cache, k)

        if i >= warmup:
            fwd.append(d1 * 1e3)
            full.append(d2 * 1e3)
    return stats.median(fwd), stats.median(full)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--ctx", type=int, default=186, help="context the loop actually runs at")
    ap.add_argument("--k", type=int, default=4, help="verify rows = gamma+1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    print("loading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)

    ids = [2] + [1000 + (i % 5000) for i in range(args.ctx - 1)]
    floor = ACTIVE_BYTES / BW * 1e3

    rows = []

    def run(label, plain, capture, keep):
        U.clear_cache()
        mx.eval(mx.zeros(1))
        cache = prefill(target, tm, ids, plain)
        ms, _ = time_verify(target, tm, cache, args.k, args.iters, args.warmup, keep)
        gb = ACTIVE_BYTES / (ms / 1e3) / 1e9
        rows.append({"condition": label, "ms": ms, "gb_s": gb,
                     "pct_of_ceiling": 100 * gb / (BW / 1e9),
                     "plain_kv": plain, "kv_capture": capture, "shared_kv_held": keep})
        print(f"  {label:<46} {ms:7.2f} ms   {gb:5.1f} GB/s  {100*gb/(BW/1e9):3.0f}% of ceiling")
        return ms

    print(f"\ncontext {args.ctx}, k={args.k}, median of {args.iters} "
          f"(+{args.warmup} warmup), bandwidth floor {floor:.2f} ms\n")

    a = run("A  isolated forward (no wrapper, rotating KV)", False, False, False)
    enable_kv_capture()               # global and permanent; every later row has it
    b = run("B  + KV-capture wrapper on all 42 layers", False, True, False)
    c = run("C  + plain KV caches (sliding layers keep all)", True, True, False)
    d = run("D  + shared_kv collected and sliced each cycle", True, True, True)

    print("\n" + "=" * 74)
    print("ATTRIBUTION  (each line is that ingredient's marginal cost)")
    print("=" * 74)
    print(f"  KV-capture wrapper        {b - a:+7.2f} ms")
    print(f"  plain KV caches           {c - b:+7.2f} ms")
    print(f"  shared_kv collect+slice   {d - c:+7.2f} ms")
    print(f"  ---------------------------------")
    print(f"  total loop ingredients    {d - a:+7.2f} ms")
    print(f"  isolated forward           {a:7.2f} ms")
    print(f"  loop-equivalent forward    {d:7.2f} ms")

    # ------------------------------------------------ PART 2: forward vs LM head
    print("\n" + "=" * 74)
    print("FORWARD vs LM HEAD  (verify_decomposition times only the forward)")
    print("=" * 74)
    print(f"  {'ctx':>5} {'forward':>9} {'+LM head':>9} {'head cost':>10}")
    for ctx in (128, args.ctx):
        ids_c = [2] + [1000 + (i % 5000) for i in range(ctx - 1)]
        U.clear_cache()
        mx.eval(mx.zeros(1))
        cache = prefill(target, tm, ids_c, True)
        fwd, full = time_split(target, tm, cache, args.k, args.iters, args.warmup)
        print(f"  {ctx:>5} {fwd:9.2f} {full:9.2f} {full - fwd:10.2f} ms")
        rows.append({"condition": f"split_ctx{ctx}", "ms": full, "forward_ms": fwd,
                     "lm_head_ms": full - fwd, "ctx_measured": ctx,
                     "gb_s": ACTIVE_BYTES / (full / 1e3) / 1e9,
                     "pct_of_ceiling": 100 * (ACTIVE_BYTES / (full / 1e3)) / BW})

    with open(args.out, "a") as f:
        for r in rows:
            r.update({"record_type": "verify_inloop_gap", "schema_version": 1,
                      "ctx": args.ctx, "k": args.k, "iters": args.iters,
                      "bw_floor_ms": floor, "power": U.power_info()})
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  appended {len(rows)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

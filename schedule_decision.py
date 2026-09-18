#!/usr/bin/env python3
"""
schedule_decision.py — re-measures the cost model that confidence_calibration.py
got wrong, then re-decides the confidence schedule on numbers that respect the
roofline.

WHAT WENT WRONG THE FIRST TIME

confidence_calibration.py fitted verify_ms = 17.15 + 8.641*k from three in-loop
points at gamma = 2, 4, 8. A verify forward must stream the active weights once,
and roofline.json says that is 3,535,067,220 bytes against an achievable
89.0 GB/s:

    floor = 3.535e9 / 89e9 = 39.7 ms

An intercept of 17.15 ms implies 206 GB/s on an 89 GB/s machine. It is not a
noisy estimate, it is impossible, and everything derived from it is void:
q* = 0.465, the simulated tok/s, and the headline 1.635x.

Three separate faults, worth keeping straight because they have different fixes:

  1. THE FIT. Three in-loop points, one pass each, no cooldown between gammas
     on a fanless M3 Air, against a curve that is visibly not a line
     (residuals +2.99, -4.49, +1.49 -- structured, not random). The measured
     per-position cost rises: 2.12 ms at k=3, 3.23 at k=5, 6.30 at k=9 once the
     39.7 ms floor is subtracted. Forcing a line through a convex curve drags
     the intercept below the floor. Fixed here by measuring t(k) ISOLATED at
     fixed context, k = 1..10, medians of 5 with cooldown, and by checking the
     result against the roofline instead of trusting the regression.
     (Not MoE: gemma4_text sets enable_moe_block=False for this model, the MoE
     block is the 26B. The superlinearity is still unexplained -- this script
     measures the shape rather than assuming one.)

  2. THE BASELINE. The 1.635x compared the schedule against fixed gamma=8, the
     WORST of the three measured configurations. Straight from the measured
     numbers, (mean_acc + 1) / (draft + verify): gamma=2 -> 46.9 tok/s,
     gamma=4 -> 50.4, gamma=8 -> 32.3. gamma=4 is best, and it cross-checks
     against the recipe's measured 49.27 tok/s at gamma=3. A schedule has to
     beat ~50 tok/s, not 33.

  3. THE RULE. An extra draft token only materialises into an emitted token if
     EVERY earlier draft in the block was also accepted, so the break-even is
     on the cumulative product q = prod(p_j), not on the current step's
     confidence. simulate() thresholded the per-step value, which never stops a
     confident-but-deep draft the cumulative rule would kill. At p=0.9 per step
     the cumulative is 0.66 by step 4 and 0.43 by step 8.

And the marginal cost is not a constant, so q* is not a scalar. Measured
between adjacent gammas: 5.52 ms per extra draft near gamma=2, 10.79 ms near
gamma=8. At 19.84 ms per emitted token that is q* = 0.28 rising to 0.54. A
rising marginal cost is exactly what creates an interior optimum at gamma=4.

WHAT SURVIVES

The calibration itself. P(accept) climbs 0.333 -> 0.974 across confidence
buckets (spread 0.641) over 311 leading-run positions, and its n-weighted mean
of 0.714 independently reproduces the measured mean-accepted of 2.29 at
gamma=8. That is a per-token statistic, not a timing one, so the broken cost
model never touched it. The drafter's confidence does predict acceptance. Only
the sizing was wrong.

    ../bin/python schedule_decision.py --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import time
from pathlib import Path

import mlx.core as mx

import roofline as RF
from measure_acceptance import (
    PROMPTS, collect_shared_kv, enable_kv_capture, text_model,
)
from spec_generate import rewind, slice_shared_kv, use_plain_kv_caches
from confidence_calibration import draft_with_conf


NEG_INF = float("-inf")


def new_cache(target):
    from mlx_lm.models.cache import make_prompt_cache
    c = make_prompt_cache(target)
    use_plain_kv_caches(c)
    return c


def warm_cache(target, tm, ids, n):
    """Prefill + greedy-decode n tokens, returning (cache, decoded_ids).

    The cost curves are measured against a REALISTIC context, not an empty one:
    at 26 prompt tokens plus ~100 decoded the verify forward sees the same KV
    volume the real loop does.
    """
    cache = new_cache(target)
    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)
    out = []
    cur = mx.argmax(tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1))
    mx.eval(cur)
    out.append(int(cur.item()))
    for _ in range(n - 1):
        h = tm(cur.reshape(1, 1), cache=cache)
        cur = mx.argmax(tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1))
        mx.eval(cur)
        out.append(int(cur.item()))
    return cache, out


def verify_curve(target, tm, cache, toks, ks, reps, cool):
    """Median isolated verify time for each k, at fixed context.

    The timed region is exactly one forward plus the argmax that consumes it,
    terminated by mx.eval(preds) -- nothing lazy escapes. The int() conversions
    that would otherwise sync are hoisted out of the region, which is one of the
    things the in-loop measurement got wrong. trim(k) restores the context after
    each rep so every rep sees an identical cache.
    """
    out = {}
    for k in ks:
        blk = [int(t) for t in toks[:k]]
        xv = mx.array([blk])
        mx.eval(xv)                       # build the input OUTSIDE the timed region
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            hv = tm(xv, cache=cache)
            preds = mx.argmax(tm.embed_tokens.as_linear(hv), axis=-1)
            mx.eval(preds)
            samples.append((time.perf_counter() - t0) * 1e3)
            for c in cache:
                c.trim(k)
        out[k] = stats.median(samples)
        time.sleep(cool)                  # fanless M3 Air: cool between points
    return out


def draft_curve(target, tm, drafter, cache, ids, gs, reps, cool):
    """Median isolated draft time for each gamma, from a fixed cache state."""
    h = tm(mx.array([[int(ids[-1])]]), cache=cache)
    mx.eval(h)
    for c in cache:
        c.trim(1)
    shared_kv = collect_shared_kv(target)
    pos = cache[0].offset
    cur = mx.array(int(ids[-1]))
    out = {}
    for g in gs:
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            draft_with_conf(drafter, tm, h[:, -1:, :], cur, shared_kv, g, pos)
            samples.append((time.perf_counter() - t0) * 1e3)
        out[g] = stats.median(samples)
        time.sleep(cool)
    return out


def run_cycles(target, tm, drafter, ids, max_tokens, gamma):
    """spec_decode's loop, recording per-cycle confidences and accept counts."""
    cache = new_cache(target)
    hidden = tm(mx.array([ids]), cache=cache)
    mx.eval(hidden)
    shared_kv = collect_shared_kv(target)
    cur = mx.argmax(tm.embed_tokens.as_linear(hidden[:, -1:]).reshape(-1))
    mx.eval(cur)
    out = [int(cur.item())]
    pos = len(ids)
    cycles = []
    while len(out) < max_tokens:
        g = min(gamma, max_tokens - len(out))
        drafts, probs, _ = draft_with_conf(
            drafter, tm, hidden[:, -1:, :], cur, shared_kv, g, pos)
        xv = mx.array([[int(cur.item())] + drafts])
        hv = tm(xv, cache=cache)
        preds = mx.argmax(tm.embed_tokens.as_linear(hv), axis=-1)[0]
        mx.eval(preds)
        pl = preds.tolist()
        n = 0
        for i in range(g):
            if pl[i] != drafts[i]:
                break
            n += 1
        cycles.append({"g": g, "n": n, "probs": probs})
        out.extend(drafts[:n])
        out.append(pl[n])
        rewind(cache, g - n)
        hidden = hv[:, n: n + 1, :]
        cur = mx.array(pl[n])
        pos += n + 1
        shared_kv = slice_shared_kv(collect_shared_kv(target), g - n)
    return cycles


def cycle_ms(d, dc, vc):
    """Measured cost of a cycle that drafts d tokens: draft(d) + verify(d+1)."""
    return dc[d] + vc[d + 1]


def sim_cumulative(cycles, dc, vc, ms_tok, gmax, scale=1.0):
    """Stop drafting when the CUMULATIVE accept probability stops paying.

    After drafting token i (0-based) the block holds d = i+1 drafts. Drafting
    one more costs marginal = cycle(d+1) - cycle(d), measured, and yields one
    extra emitted token with probability q = prod(p_0..p_i). Continue iff
    q * ms_tok > marginal. `scale` multiplies q to let the sweep correct for the
    drafter being over- or under-confident without re-deriving the rule.
    """
    toks = ms = 0.0
    hist = {}
    for c in cycles:
        q = 1.0
        d = c["g"]
        for i in range(c["g"]):
            q *= c["probs"][i]
            nd = i + 1
            if nd >= gmax:
                d = nd
                break
            marginal = cycle_ms(nd + 1, dc, vc) - cycle_ms(nd, dc, vc)
            if q * scale * ms_tok <= marginal:
                d = nd
                break
        acc = min(c["n"], d)
        toks += acc + 1
        ms += cycle_ms(d, dc, vc)
        hist[d] = hist.get(d, 0) + 1
    return (toks / ms * 1e3 if ms else 0.0), hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--gmax", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cool", type=float, default=0.4)
    ap.add_argument("--warm", type=int, default=100)
    args = ap.parse_args()

    from mlx_lm.models import gemma4_assistant

    print("loading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)
    enable_kv_capture()
    print("loading drafter ...")
    cfg, path = RF.find_config(args.assistant)
    drafter = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(cfg))
    w = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        w.update(mx.load(str(f)))
    drafter.load_weights(list(w.items()), strict=True)
    mx.eval(drafter.parameters())

    try:
        rj = json.load(open("roofline.json"))
        active = rj["table"]["active_weight_bytes_per_token"]
        gbs = rj["table"]["achievable_gbs_used"]
    except Exception:
        active, gbs = 3_535_067_220, 89.0
    floor = active / (gbs * 1e9) * 1e3
    print(f"\n  roofline floor for ONE verify forward: {floor:.2f} ms"
          f"  ({active/1e9:.3f} GB at {gbs:.1f} GB/s)")

    ids, _ = RF.build_eval_prompt(tok, PROMPTS[0])
    print(f"  warming a realistic cache ({len(ids)} + {args.warm} tokens) ...")
    cache, dec = warm_cache(target, tm, ids, args.warm)

    ks = list(range(1, args.gmax + 2))
    print(f"  measuring verify t(k) isolated, k=1..{ks[-1]}, "
          f"median of {args.reps}, {args.cool}s cooldown ...")
    vc = verify_curve(target, tm, cache, dec, ks, args.reps, args.cool)
    gs = list(range(1, args.gmax + 1))
    print(f"  measuring draft d(g) isolated, g=1..{gs[-1]} ...")
    dc = draft_curve(target, tm, drafter, cache, dec, gs, args.reps, args.cool)

    print("\n" + "=" * 78)
    print("MEASURED COST CURVES (isolated, fixed context, medians)")
    print("=" * 78)
    print(f"  {'k':>3}{'verify ms':>11}{'implied GB/s':>14}{'per-pos ms':>12}"
          f"{'marginal':>10}")
    bad = []
    prev = None
    for k in ks:
        t = vc[k]
        imp = active / (t / 1e3) / 1e9
        pp = (t - floor) / k
        mg = "" if prev is None else f"{t - prev:>10.2f}"
        flag = ""
        if imp > gbs * 1.02:
            bad.append(k)
            flag = "  <<< ABOVE ROOFLINE"
        print(f"  {k:>3}{t:>11.2f}{imp:>14.1f}{pp:>12.2f}{mg}{flag}")
        prev = t
    if bad:
        print(f"\n  !! k={bad} imply more bandwidth than the machine has. Either the")
        print("     timing region is not capturing the whole forward, or the")
        print("     roofline's active-byte figure is wrong for a k-row forward.")
        print("     Resolve this BEFORE using these numbers for a decision.")
    else:
        print("\n  every point respects the roofline.")

    print(f"\n  {'g':>3}{'draft ms':>11}{'per-step':>11}")
    for g in gs:
        print(f"  {g:>3}{dc[g]:>11.2f}{dc[g]/g:>11.3f}")

    # ------------------------------------------------- fixed-gamma reference
    print("\n" + "=" * 78)
    print("FIXED GAMMA, from the measured curves and measured acceptance")
    print("=" * 78)
    allc = {}
    for g in (2, 4, args.gmax):
        cyc = []
        for pi, prompt in enumerate(PROMPTS[: args.prompts]):
            pid, _ = RF.build_eval_prompt(tok, prompt)
            cyc.extend(run_cycles(target, tm, drafter, pid, args.tokens, g))
        allc[g] = cyc
        mean_acc = stats.fmean(c["n"] for c in cyc)
        tps = (mean_acc + 1) / cycle_ms(g, dc, vc) * 1e3
        print(f"  gamma {g:>2}: mean acc {mean_acc:.2f}  "
              f"cycle {cycle_ms(g, dc, vc):6.2f} ms  -> {tps:6.2f} tok/s")
    best_g = max(allc, key=lambda g: (stats.fmean(c["n"] for c in allc[g]) + 1)
                 / cycle_ms(g, dc, vc))
    best_acc = stats.fmean(c["n"] for c in allc[best_g])
    best_tps = (best_acc + 1) / cycle_ms(best_g, dc, vc) * 1e3
    ms_tok = cycle_ms(best_g, dc, vc) / (best_acc + 1)
    print(f"\n  BEST FIXED gamma = {best_g} at {best_tps:.2f} tok/s "
          f"({ms_tok:.2f} ms/token)  <- the baseline to beat")
    print(f"  recipe's measured spec_generate result: 49.27 tok/s at gamma=3")

    # ------------------------------------------------------- the schedule
    cyc = allc[args.gmax]
    print("\n" + "=" * 78)
    print("SCHEDULE (cumulative rule, measured marginal cost) — SIMULATED")
    print("=" * 78)
    print("  rule: draft another token iff  prod(p) * scale * ms_tok > marginal(d)")
    print(f"  {'scale':>7}{'pred tok/s':>12}{'vs best fixed':>15}"
          f"{'mean drafted':>14}")
    rows = []
    m = ms_tok
    for _ in range(6):                      # ms_tok depends on the schedule it prices
        tps, _ = sim_cumulative(cyc, dc, vc, m, args.gmax)
        m = 1e3 / tps if tps else m
    for scale in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0):
        tps, hist = sim_cumulative(cyc, dc, vc, m, args.gmax, scale)
        md = sum(k * v for k, v in hist.items()) / max(1, sum(hist.values()))
        rows.append((scale, tps, hist))
        print(f"  {scale:>7.2f}{tps:>12.2f}{tps/best_tps:>15.3f}{md:>14.2f}")
    bs, bt, bh = max(rows, key=lambda r: r[1])
    gain = bt / best_tps
    print(f"\n  best scale {bs:.2f} -> {bt:.2f} tok/s  = {gain:.3f}x over fixed "
          f"gamma={best_g}")
    print(f"  drafted histogram: {dict(sorted(bh.items()))}")

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    if bad:
        print("  BLOCKED — the cost curve violates the roofline at "
              f"k={bad}. Fix the")
        print("  measurement before deciding anything; the last cost model was")
        print("  wrong in exactly this way and produced a 1.635x that was really 1.09x.")
    elif gain < 1.05:
        print(f"  DROP the confidence schedule. {gain:.3f}x over the best fixed gamma")
        print("  does not justify the implementation risk, and the calibration was")
        print("  never the problem -- the drafter's confidence is genuinely")
        print("  predictive (spread 0.641). The win simply is not there once the")
        print("  cycle's fixed cost is priced correctly.")
        print("  -> go to mx.compile on the draft step "
              f"({dc[best_g]:.2f} ms/cycle at gamma={best_g}).")
    else:
        print(f"  BUILD it. {gain:.3f}x over the best fixed gamma, on cost curves")
        print("  that respect the roofline. Implement in spec_generate.py as a")
        print("  per-cycle dynamic gamma and benchmark against greedy_baseline on")
        print("  these same prompts before claiming anything.")
    print("\n  Still a simulation: drafting fewer tokens changes which tokens each")
    print("  cycle emits and therefore the whole trajectory. No speedup is claimed")
    print("  until the schedule is implemented and measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

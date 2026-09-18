#!/usr/bin/env python3
"""
control_variables.py — the two variables nobody controlled for.

WHY THIS RUNS BEFORE shape_stability.py

shape_stability asks "does the argmax depend on the SHAPE of the matmul?" and
the decision rule attached to it is:

    unstable positions found, 121/82 among them -> loop is correct
    nothing unstable                            -> bookkeeping is wrong

That rule has a hole in it, because shape is not the only thing that differs
between the two runs being compared. Reading spec_generate.py against the
installed mlx-lm:

    greedy_baseline()   cache = make_prompt_cache(target)
                        -> RotatingKVCache on the 20 sliding layers

    spec_decode()       cache = make_prompt_cache(target)
                        use_plain_kv_caches(cache)
                        -> KVCache everywhere

So the "divergence" is measured across TWO changed variables: the batch shape
of the verify forward AND the cache class. If the cache swap alone moves a
logit by a ULP, then shape_stability can honestly report "nothing unstable"
while the real cause sits in the control it never varied — and the decision
rule would send you hunting a bookkeeping bug that does not exist.

Note that shape_stability.greedy_reference() DOES call use_plain_kv_caches,
so its reference sequence is not necessarily the same sequence `base` that
the divergences at 121 and 82 were found against. Asking "is 121 unstable?"
is only meaningful once we know both runs agree on what token 121 is.

CONTROL 1 — CACHE LAYOUT
    greedy decode with RotatingKVCache  vs  greedy decode with plain KVCache
    Same shapes everywhere (1 row at a time, both runs). The only variable is
    the cache class. Agreement here is what makes shape_stability's reference
    trustworthy; disagreement at 121/82 answers the whole question outright.

CONTROL 2 — FINAL LOGIT SOFTCAP
    mlx_lm.models.gemma4_text.Model.__call__ ends with

        out = logit_softcap(30.0, out)          # tanh(x/30) * 30

    Every script in this project scores `tm.embed_tokens.as_linear(h)` and
    stops there, so none of them apply it. That is self-consistent across
    baseline and speculative, so the losslessness comparison is still a fair
    one -- but it means the ULP arithmetic in shape_stability's verdict is
    computed at the WRONG SCALE relative to the path mlx_lm.generate takes,
    and the tie threshold derived from it inherits that error.

    The compression is not negligible. At a top logit of ~28:

        tanh(28/30)*30            = 21.95        (scale drops by 1.28x)
        d/dx tanh(x/30)*30 |_28   = 0.46         (gaps shrink by 2.2x)
        observed margin 6.37e-03  = 0.178 absolute -> 0.083 after the cap
        bf16 ULP at 21.95         = 2^(4-7)      = 0.125

    0.083 < 0.125. A margin that is ~1.4 ULP uncapped is SUB-ULP capped: the
    two candidates collapse onto the same bfloat16 value and argmax falls back
    to lowest index. The production path therefore has a WIDER tie band than
    these scripts measure, not a narrower one.

    This control measures that directly instead of trusting the arithmetic:
    how many positions change their argmax when the softcap is applied.

Everything here is untimed, so the mx.eval barriers exist only to force the
graph before a host-side .item()/.tolist() read. There is no timed region to
protect and no barrier placement to justify against a measurement.

    ../bin/python control_variables.py --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import math

import mlx.core as mx

import roofline as RF
from measure_acceptance import PROMPTS, text_model
from spec_generate import use_plain_kv_caches


NEG_INF = float("-inf")


def top2(lg: mx.array):
    """(best_id, best, second, rel_margin, ulps) with sort semantics, no sort.

    spec_generate._top2_margin uses mx.sort and reads v[-2], so an EXACT tie
    gives a margin of exactly 0. Reproduced here with three reductions instead
    of an O(V log V) sort over a 262k vocabulary -- taking max, counting how
    many entries hit it, and only masking when the maximum is unique.
    """
    best_a = mx.max(lg)
    ties_a = mx.sum(lg == best_a)
    bid_a = mx.argmax(lg)
    mx.eval(best_a, ties_a, bid_a)
    best, ties, bid = float(best_a.item()), int(ties_a.item()), int(bid_a.item())
    if ties > 1:
        second = best
    else:
        second_a = mx.max(mx.where(lg == best_a, mx.array(NEG_INF, lg.dtype), lg))
        mx.eval(second_a)
        second = float(second_a.item())
    scale = max(abs(best), 1e-6)
    rel = (best - second) / scale
    ulp = 2.0 ** (math.floor(math.log2(scale)) - 7)
    return bid, best, second, rel, (best - second) / ulp, ulp


def greedy(target, tm, ids, n, plain: bool, softcap):
    """Greedy decode one row at a time. Returns per-position records.

    `plain` selects the cache class -- that is the only thing control 1 varies.
    `softcap` is the model's final_logit_softcapping value (or None); when set,
    each position is ALSO scored through tanh(x/c)*c so control 2 can see
    whether the cap moves the argmax. The capped logits never feed the loop:
    generation follows the uncapped argmax so that the token sequence stays
    identical to what greedy_baseline() produces.
    """
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.gemma4_text import logit_softcap

    cache = make_prompt_cache(target)
    n_plain = use_plain_kv_caches(cache) if plain else 0

    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)

    recs = []
    cur = None
    for step in range(n):
        if step:
            h = tm(cur.reshape(1, 1), cache=cache)
        lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
        mx.eval(lg)

        bid, best, second, rel, ulps, ulp = top2(lg)

        cap_id, cap_rel, cap_ulps, cap_ulp = None, None, None, None
        if softcap is not None:
            cl = logit_softcap(softcap, lg)
            mx.eval(cl)
            cap_id, cb, cs, cap_rel, cap_ulps, cap_ulp = top2(cl)

        recs.append({
            "id": bid, "best": best, "rel": rel, "ulps": ulps, "ulp": ulp,
            "cap_id": cap_id, "cap_rel": cap_rel,
            "cap_ulps": cap_ulps, "cap_ulp": cap_ulp,
        })
        cur = mx.array(bid)

    return recs, n_plain, cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    # the two positions the recipe is trying to explain
    ap.add_argument("--watch", default="0:121,1:82")
    args = ap.parse_args()

    watch = {}
    for part in args.watch.split(","):
        if ":" in part:
            a, b = part.split(":")
            watch.setdefault(int(a), []).append(int(b))

    print("loading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)

    inner = getattr(target, "language_model", target)
    margs = getattr(inner, "args", None)
    softcap = getattr(inner, "final_logit_softcapping", None)

    # ------------------------------------------------------- ground truth
    from mlx_lm.models.cache import make_prompt_cache
    probe = make_prompt_cache(target)
    types = {}
    for c in probe:
        types[type(c).__name__] = types.get(type(c).__name__, 0) + 1
    ltypes = {}
    for l in tm.layers:
        k = getattr(l, "layer_type", "?")
        ltypes[k] = ltypes.get(k, 0) + 1

    print("\n" + "=" * 78)
    print("CONFIG AS INSTALLED (not assumed)")
    print("=" * 78)
    print(f"  sliding_window                   {getattr(tm, 'window_size', '?')}")
    print(f"  sliding_window_pattern           {getattr(tm, 'sliding_window_pattern', '?')}")
    print(f"  layers                           {len(tm.layers)}  {ltypes}")
    print(f"  caches from make_prompt_cache    {len(probe)}  {types}")
    print(f"  kv-shared layers (no cache)      {len(tm.layers) - len(probe)}")
    print(f"  final_logit_softcapping          {softcap}")
    if margs is not None:
        print(f"  vocab_size                       {getattr(margs, 'vocab_size', '?')}")
    del probe

    any_cache_divergence = False
    any_softcap_flip = False

    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, how = RF.build_eval_prompt(tok, prompt)
        print("\n" + "=" * 78)
        print(f"PROMPT {pi}  ({len(ids)} prompt tokens, {how})")
        print(f"  {prompt}")
        print("=" * 78)

        total = len(ids) + args.tokens
        win = getattr(tm, "window_size", None)
        if isinstance(win, int):
            print(f"  context reaches {total} positions; sliding_window = {win}"
                  f"  -> rotating cache {'WRAPS' if total > win else 'never wraps'}")
            if total > win:
                print("  !! the rotating cache wraps inside this run. Control 1 is")
                print("     then testing ring bookkeeping as well as layout.")

        print("\n  decoding with RotatingKVCache (greedy_baseline's cache) ...")
        rot, _, _ = greedy(target, tm, ids, args.tokens, plain=False, softcap=softcap)
        print("  decoding with plain KVCache (spec_decode's cache) ...")
        pln, n_plain, _ = greedy(target, tm, ids, args.tokens, plain=True, softcap=softcap)
        print(f"  ({n_plain} rotating caches replaced)")

        # ------------------------------------------- CONTROL 1: cache layout
        rt = [r["id"] for r in rot]
        pt = [r["id"] for r in pln]
        diff = [j for j in range(args.tokens) if rt[j] != pt[j]]

        print("\n  --- CONTROL 1: cache layout (shapes identical, 1 row both runs) ---")
        if not diff:
            print("    IDENTICAL over all "
                  f"{args.tokens} positions. The cache class is not a variable;")
            print("    shape_stability's plain-cache reference is the same sequence")
            print("    greedy_baseline produces, so its position indices are valid.")
        else:
            any_cache_divergence = True
            j = diff[0]
            print(f"    DIVERGES at {len(diff)} position(s); first at {j}")
            print(f"    rotating -> {rt[j]} {tok.decode([rt[j]])!r}"
                  f"   plain -> {pt[j]} {tok.decode([pt[j]])!r}")
            print(f"    margin there (rotating): {rot[j]['rel']:.3e}"
                  f"  ({rot[j]['ulps']:.1f} ULP)")
            print(f"    all divergent positions: {diff[:16]}")
            print("    -> The CACHE CLASS alone flips the argmax. The baseline and")
            print("       the speculative run were never scoring the same function,")
            print("       and shape is not the variable to be testing.")

        for w in watch.get(pi, []):
            if w < args.tokens:
                r = rot[w]
                print(f"\n    watched position {w}:")
                print(f"      token          {r['id']} {tok.decode([r['id']])!r}")
                print(f"      margin         {r['rel']:.3e}  ({r['ulps']:.2f} uncapped ULP)")
                if r["cap_rel"] is not None:
                    print(f"      after softcap  {r['cap_rel']:.3e}"
                          f"  ({r['cap_ulps']:.2f} capped ULP)")
                    if r["cap_ulps"] < 1.0:
                        print("      -> SUB-ULP once capped: indistinguishable in bf16")
                print(f"      cache control  "
                      f"{'AGREES' if rt[w] == pt[w] else 'DIVERGES <<<<'}")

        # ------------------------------------------------ CONTROL 2: softcap
        if softcap is not None:
            flips = [j for j in range(args.tokens) if rot[j]["cap_id"] != rot[j]["id"]]
            sub = [j for j in range(args.tokens) if rot[j]["cap_ulps"] < 1.0]
            print("\n  --- CONTROL 2: final logit softcap (skipped by every script) ---")
            print(f"    positions where the cap moves the argmax   {len(flips)}"
                  + (f"  at {flips[:12]}" if flips else ""))
            print(f"    positions sub-ULP AFTER the cap            {len(sub)}"
                  f"  ({100 * len(sub) / args.tokens:.1f}%)")
            print(f"    positions sub-ULP BEFORE the cap           "
                  f"{sum(1 for r in rot if r['ulps'] < 1.0)}")
            if flips:
                any_softcap_flip = True
                print("    -> The cap is monotonic in exact arithmetic but NOT in")
                print("       bfloat16. Any tie threshold has to be measured on the")
                print("       capped logits, which is the path mlx_lm.generate takes.")

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    print("WHAT TO DO WITH THIS")
    print("=" * 78)
    if any_cache_divergence:
        print("  Control 1 FAILED. Fix greedy_baseline() to call use_plain_kv_caches()")
        print("  so baseline and speculative score the same function, re-run")
        print("  spec_generate.py, and see whether the divergences survive at all.")
        print("  Do not interpret shape_stability until they do.")
    else:
        print("  Control 1 passed: cache class is not a variable. shape_stability's")
        print("  reference is sound and its position indices line up with `base`.")
    if any_softcap_flip:
        print("  Control 2 FAILED. Score through logit_softcap before deriving any")
        print("  tie threshold; the uncapped ULP counts understate the tie band.")
    elif softcap is not None:
        print("  Control 2 passed on this sample, but note the capped ULP counts")
        print("  above -- set the threshold from those, not the uncapped ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

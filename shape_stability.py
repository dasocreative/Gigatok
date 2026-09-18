#!/usr/bin/env python3
"""
shape_stability.py — does the TARGET MODEL agree with itself?

WHY THIS EXISTS

Two positions still diverge between the greedy baseline and the speculative
loop. Four hypotheses have already been killed by measurement: the centroid
shortlist, the eval barriers, batched-verify equivalence, and the rotating
cache. The fifth candidate is that nothing is wrong with the loop at all —
that the target model's own greedy output depends on the SHAPE of the tensor
it is scored in.

That is not a hand-wave. debug_verify measured it: scoring one hidden state
row through mx.quantized_matmul gives logits that differ from scoring four
rows at once by up to 1.25e-01 in absolute terms. At a logit magnitude of ~28
that is exactly one unit in the last place of bfloat16 — the two kernels agree
to the last representable bit and no further. Meanwhile the two divergences
sit at relative top-2 margins of 6.37e-03 and 1.55e-02, which at that same
magnitude is between one and four ULPs.

So the noise and the margins are the same size. debug_verify's verdict said
"noise floor BELOW the margins" and therefore "real bug", but that test was
too crude in two ways:

  1. It measured a ONE-SIDED difference (max over the vocabulary of |delta|)
     and compared it to a TWO-SIDED quantity. Flipping an argmax needs the
     top-1 logit to fall and the top-2 to rise; each can move by a ULP.
  2. It sampled four positions at one generation depth. The failures are at
     depth 121 and 82, across 46 cycles. A worst case over 4 samples is not a
     worst case over 200.

THE EXPERIMENT

Rather than argue about the size of the noise, remove the speculative loop
entirely and ask the model directly:

    decode greedily one token at a time      -> reference token sequence
    then feed those SAME tokens back through the model in blocks of N,
    taking the argmax at every position

Teacher forcing means every position is scored on identical context. The only
thing that changes is how many rows the matmul had. If the model predicts a
different token at position 121 when scored four rows at a time than when
scored one at a time, then the "divergence" the speculative loop reports is
the model disagreeing with itself, and no amount of correct bookkeeping will
remove it.

There is no drafter here, no accept logic, no cache trimming, no rollback.
Nothing that could be blamed. Just: is greedy decoding shape-invariant?

WHAT IT PRODUCES

The set of positions whose argmax depends on batch shape, each with its
baseline top-2 margin expressed both relatively and in bfloat16 ULPs. The
largest margin among the unstable positions is the empirically measured tie
threshold — the number spec_generate.py currently guesses at 1e-4.

    ../bin/python shape_stability.py --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import math
import statistics as stats

import mlx.core as mx

import roofline as RF
from measure_acceptance import PROMPTS, text_model
from spec_generate import use_plain_kv_caches


# --------------------------------------------------------------- margins
def top2(logits: mx.array):
    """Return (best_id, best, second, relative_margin, margin_in_bf16_ULPs).

    The ULP count is the number that makes the result interpretable. bfloat16
    carries 8 significand bits, so a value with exponent e is quantised to
    steps of 2**(e-7). A margin of 1 ULP means the two candidate tokens are
    adjacent representable numbers — the model has not distinguished them at
    all, it has merely rounded one of them up.
    """
    v = mx.sort(logits.reshape(-1))
    best = float(v[-1].item())
    second = float(v[-2].item())
    bid = int(mx.argmax(logits.reshape(-1)).item())
    scale = max(abs(best), 1e-6)
    rel = (best - second) / scale
    e = math.floor(math.log2(scale))
    ulp = 2.0 ** (e - 7)
    return bid, best, second, rel, (best - second) / ulp


# --------------------------------------------------- reference: 1 row at a time
def greedy_reference(target, tm, ids, n):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    use_plain_kv_caches(cache)
    h = tm(mx.array([ids]), cache=cache)
    toks, margins, ulps = [], [], []
    lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
    mx.eval(lg)
    bid, _, _, rel, u = top2(lg)
    toks.append(bid)
    margins.append(rel)
    ulps.append(u)
    for _ in range(n - 1):
        h = tm(mx.array([[toks[-1]]]), cache=cache)
        lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
        mx.eval(lg)
        bid, _, _, rel, u = top2(lg)
        toks.append(bid)
        margins.append(rel)
        ulps.append(u)
    return toks, margins, ulps


# ------------------------------------------------ replay: N rows at a time
def teacher_force(target, tm, ids, toks, block):
    """
    Feed `toks` back through the model in blocks of `block` rows and return the
    argmax at every position.

    preds[j] is what the model predicts for toks[j] having been fed
    toks[0:j] — identical context to the reference run at that position. Any
    difference is attributable to the matmul shape alone.

    Teacher forcing, not free running: after a disagreement we keep feeding the
    REFERENCE tokens, so every position is an independent test rather than the
    start of a diverging continuation.
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    use_plain_kv_caches(cache)
    h = tm(mx.array([ids]), cache=cache)
    lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
    p0 = mx.argmax(lg)
    mx.eval(p0)
    preds = [int(p0.item())]

    i = 0
    while len(preds) < len(toks):
        blk = toks[i: i + block]
        if not blk:
            break
        hh = tm(mx.array([blk]), cache=cache)
        lgs = tm.embed_tokens.as_linear(hh)[0]        # [len(blk), V]
        p = mx.argmax(lgs, axis=-1)
        mx.eval(p)
        preds.extend(int(v) for v in p.tolist())
        i += len(blk)
    return preds[: len(toks)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--blocks", default="2,3,4,8")
    args = ap.parse_args()

    blocks = [int(b) for b in args.blocks.split(",") if b.strip()]

    print("loading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)

    unstable_margins, unstable_ulps = [], []
    stable_margins = []
    all_margins = []
    rows = []

    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        print(f"\nprompt {pi}: {prompt}")
        print("  decoding reference (1 row at a time) ...")
        toks, margins, ulps = greedy_reference(target, tm, ids, args.tokens)
        all_margins.extend(margins)

        disagree = {}   # position -> [block sizes that disagree]
        for b in blocks:
            preds = teacher_force(target, tm, ids, toks, b)
            bad = [j for j in range(len(toks)) if preds[j] != toks[j]]
            for j in bad:
                disagree.setdefault(j, []).append(b)
            print(f"    block {b:>2}: {len(bad):>3} mismatches"
                  + (f"  at {bad[:8]}" if bad else ""))

        for j in range(len(toks)):
            if j in disagree:
                unstable_margins.append(margins[j])
                unstable_ulps.append(ulps[j])
                rows.append((pi, j, toks[j], margins[j], ulps[j], disagree[j]))
            else:
                stable_margins.append(margins[j])

    # ------------------------------------------------------------ report
    print("\n" + "=" * 78)
    print("POSITIONS WHOSE ARGMAX DEPENDS ON BATCH SHAPE")
    print("=" * 78)
    if not rows:
        print("  none. Greedy decoding is shape-invariant over this sample, so the")
        print("  speculative divergences are NOT numerical and the loop has a bug.")
    else:
        print(f"  {'prompt':>7}{'pos':>6}{'token':>9}{'rel margin':>13}"
              f"{'ULPs':>8}   blocks")
        for pi, j, t, rel, u, bs in sorted(rows, key=lambda r: (r[0], r[1])):
            print(f"  {pi:>7}{j:>6}{t:>9}{rel:>13.2e}{u:>8.1f}   "
                  f"{','.join(str(b) for b in bs)}")

    print("\n" + "=" * 78)
    print("MARGIN SEPARATION")
    print("=" * 78)
    print(f"  positions tested                 {len(all_margins)}")
    print(f"  shape-unstable                   {len(unstable_margins)}"
          f"  ({100 * len(unstable_margins) / max(len(all_margins), 1):.1f}%)")
    if unstable_margins:
        print(f"  largest margin, UNSTABLE         {max(unstable_margins):.2e}"
              f"  ({max(unstable_ulps):.1f} ULP)")
        print(f"  median margin, unstable          {stats.median(unstable_margins):.2e}")
    if stable_margins:
        print(f"  smallest margin, STABLE          {min(stable_margins):.2e}")
        print(f"  median margin, all positions     {stats.median(all_margins):.2e}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    obs = [6.37e-03, 1.55e-02]
    if unstable_margins:
        thr = max(unstable_margins)
        covered = [m for m in obs if m <= thr]
        print(f"  Greedy decoding on this model is NOT shape-invariant.")
        print(f"  {len(unstable_margins)} of {len(all_margins)} positions flip their argmax purely")
        print(f"  because the logits were computed in a matmul of a different shape.")
        print(f"\n  Every unstable position has a top-2 margin at or below "
              f"{thr:.2e}")
        print(f"  ({max(unstable_ulps):.1f} bfloat16 ULP). Above that the model is decisive.")
        print(f"\n  speculative divergences observed at margins "
              f"{obs[0]:.2e} and {obs[1]:.2e}")
        if len(covered) == len(obs):
            print("  -> BOTH fall inside the measured instability band. They are the")
            print("     model disagreeing with itself, not the loop being wrong.")
            print(f"  -> set the tie threshold in spec_generate.py to {thr * 1.5:.1e}")
            print("     (measured, with 1.5x headroom) instead of the guessed 1e-4.")
        else:
            print(f"  -> {len(obs) - len(covered)} of them sit ABOVE the band. That part is")
            print("     still unexplained; the loop needs more work.")
    else:
        print("  Shape-invariant over this sample. The divergences are real.")
    print("\n  NOTE: margins are pre-softcap. final_logit_softcapping is monotonic")
    print("  so it cannot change an argmax, but it does compress margins — the")
    print("  ULP counts here are the honest, uncompressed ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

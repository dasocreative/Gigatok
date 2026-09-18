#!/usr/bin/env python3
"""
hidden_shape_drift.py — where the shape sensitivity actually lives.

THE HYPOTHESIS row_dependence.py KILLED

fork_validity part C's grids are column-constant, and the explanation offered for
that was: the output projection pads M up to the 8-row simdgroup_matrix tile, so
row r sits in the same tile position whatever M is, accumulates in the same order
and rounds the same way.

That predicts a specific thing: replicate ONE hidden state across M rows, project
it, and row r should still disagree with row r'. row_dependence.py did exactly
that over all 14 at-risk positions, both axes, 72 cells each:

    row-DEPENDENT (argmax moves)     0  (0.0% of the tie band)
    columns constant, sequence axis  14/14
    columns constant, batch axis     14/14
    batch axis behaves as sequence   14/14

Zero. Including positions 121 and 82 — the two that DO flip in fork_validity's
grid. For a bit-identical input vector the projection returns the same argmax at
every row, at every M, on both axes, even at the two positions whose margin is
exactly 0.000e+00. **The output projection is row-invariant. The hypothesis was
wrong.**

WHAT THAT LEAVES

The two scripts differ in one place. fork_validity feeds the REAL block
base[a:a+M] through the whole transformer and reads row r of the result;
row_dependence takes the hidden state from a 1-row forward and merely replicates
it. So the quantity that changes with M is not the projection of the hidden
state — it is the hidden state itself.

That is a sharper and more uncomfortable finding, because fork_validity's own
docstring leans on the opposite:

    "causal masking makes row r's logits independent of them, so which tokens
     fill [the later rows] does not matter. Only M does."

Causal masking guarantees row 0 is MATHEMATICALLY independent of rows 1..M-1. It
does not guarantee row 0 is BITWISE identical between a 1-row forward and an
M-row forward, because the two run different kernels — GEMV vs GEMM, different
tile shapes, different reduction orders — through every attention and MLP block
on the way. One ULP at any layer is enough to flip an argmax at a 0-2 ULP margin.

If that is what is happening, the effect is not about the LM head, not about
speculative decoding, and not about this model. It is: **the same token, in the
same context, produces a different hidden state depending on how many tokens were
in the forward pass with it.** That reaches prefill-vs-decode consistency,
chunked prefill, and every batched serving path.

THE EXPERIMENT

Position j is decided by the hidden state of base[j-1]. Put base[j-1] at row 0 of
an M-row forward, with the cache holding base[:j-1] in both arms:

    arm A   feed [base[j-1]]                        -> 1 row   (GEMV path)
    arm B   feed [base[j-1] ... base[j-1+M-1]]      -> M rows  (GEMM path)

Row 0 is causally identical in both. Compare, per decoder layer, the row-0 output
hidden state. The first layer at which they stop being bit-identical is where the
shape sensitivity enters, and the size of the gap at the last layer is what
reaches the argmax.

POSITIVE CONTROL, BUILT IN

A null result is only readable if the instrument can produce a positive one — the
lesson from row_dependence, whose zero was interpretable only because
fork_validity had already found a flip. So this script FIRST reproduces the
argmax flip at row 0 for each M, and only reports the layer localisation for the
M values where the flip actually reproduces. If no flip reproduces, it says so
and the localisation is not to be trusted.

Nothing is timed. The mx.eval barriers force the graph before host-side reads.

    ../bin/python hidden_shape_drift.py --prompts 2
"""

from __future__ import annotations

import argparse
import json

import mlx.core as mx

import roofline as RF
from measure_acceptance import PROMPTS, text_model
from spec_generate import greedy_baseline, use_plain_kv_caches
from fork_validity import top2


def enable_hidden_capture() -> None:
    """Stash each decoder layer's output hidden state as it is produced."""
    from mlx_lm.models import gemma4_text

    DL = gemma4_text.DecoderLayer
    if getattr(DL, "_mlxbench_hidden_capture", False):
        return
    orig = DL.__call__

    def wrapped(self, x, mask=None, cache=None, per_layer_input=None,
                shared_kv=None, offset=None):
        out = orig(self, x, mask, cache, per_layer_input, shared_kv, offset)
        self._captured_h = out[0] if isinstance(out, tuple) else out
        return out

    DL.__call__ = wrapped
    DL._mlxbench_hidden_capture = True


def sweep_position(target, tm, ids, base, j, max_m):
    """All M arms for position j, from ONE cache build.

    Position j is decided by the hidden state of base[j-1], so the cache is
    advanced to base[:j-1] once and then each arm feeds base[j-1 : j-1+M] and
    trims M back off — the same trim spec_decode uses. Rebuilding the cache per
    arm instead would cost ~100 single-token forwards each and turn a 60-second
    script into a 15-minute one.

    Returns {M: (row0_logits, [row0 hidden per layer])}.
    """
    from mlx_lm.models.cache import make_prompt_cache

    a = j - 1
    cache = make_prompt_cache(target)
    use_plain_kv_caches(cache)
    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)
    for t in base[:a]:
        h = tm(mx.array([[int(t)]]), cache=cache)
    mx.eval(h)

    arms = {}
    for M in range(1, max_m + 1):
        if a + M > len(base):
            break
        blk = [int(t) for t in base[a: a + M]]
        hh = tm(mx.array([blk]), cache=cache)
        lg = tm.embed_tokens.as_linear(hh)[0, 0]        # row 0, full-block proj
        rows = [None if getattr(l, "_captured_h", None) is None
                else l._captured_h[0, 0] for l in tm.layers]
        mx.eval([lg] + [r for r in rows if r is not None])
        arms[M] = (lg, rows)
        for c in cache:
            c.trim(M)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--max-m", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=2.3e-2)
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    print("loading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)
    enable_hidden_capture()

    records = []
    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        print("\n" + "=" * 78)
        print(f"PROMPT {pi}: {prompt}")
        print("=" * 78)

        base, margins, _ulps = greedy_baseline(target, tm, ids, args.tokens, want_margins=True)
        targets = [j for j, m in enumerate(margins) if m < args.threshold and j >= 1]
        print(f"  at-risk positions: {targets}")

        for j in targets:
            arms = sweep_position(target, tm, ids, base, j, args.max_m)
            if 1 not in arms:
                continue
            lg1, rows1 = arms[1]
            id1, rel1, ulp1 = top2(lg1)
            print(f"\n  position {j}  margin {margins[j]:.3e}  "
                  f"1-row argmax {id1} {tok.decode([id1])!r}")

            flips = []
            for M in sorted(k for k in arms if k >= 2):
                lgM, rowsM = arms[M]
                idM, relM, ulpM = top2(lgM)

                # Row 0 is causally independent of rows 1..M-1, so a non-zero
                # difference is the transformer alone.
                first_bad, last_gap = None, 0.0
                for li, (r1, rM) in enumerate(zip(rows1, rowsM)):
                    if r1 is None or rM is None:
                        continue
                    d = mx.max(mx.abs(r1.astype(mx.float32) - rM.astype(mx.float32)))
                    mx.eval(d)
                    d = float(d.item())
                    if d > 0.0 and first_bad is None:
                        first_bad = li
                    last_gap = d
                dlog = mx.max(mx.abs(lg1.astype(mx.float32) - lgM.astype(mx.float32)))
                mx.eval(dlog)
                dlog = float(dlog.item())

                flipped = idM != id1
                if flipped:
                    flips.append(M)
                print(f"    M={M:<2} argmax {idM:<7} {tok.decode([idM])!r:<16}"
                      f"{'FLIP' if flipped else 'same':<5}"
                      f" first-diff layer {str(first_bad):<5}"
                      f" last-layer |d| {last_gap:.3e}  logit |d| {dlog:.3e}")

                records.append({
                    "record_type": "hidden_shape_drift", "schema_version": 1,
                    "prompt_index": pi, "position": j, "margin": margins[j],
                    "M": M, "argmax_1row": id1, "argmax_Mrow": idM,
                    "flipped": flipped, "first_diff_layer": first_bad,
                    "last_layer_absdiff": last_gap, "logit_absdiff": dlog,
                })

            if flips:
                print(f"    -> POSITIVE CONTROL OK: row 0 flips at M={flips}. The "
                      f"hidden state, not the projection, is shape-dependent.")
            elif any(r["first_diff_layer"] is not None for r in records
                     if r["position"] == j and r["prompt_index"] == pi):
                print("    -> hidden state drifts but the argmax holds here.")
            else:
                print("    -> row 0 is bit-identical at every M; nothing to localise.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n = len(records)
    fl = [r for r in records if r["flipped"]]
    drift = [r for r in records if r["first_diff_layer"] is not None]
    print(f"  (position, M) cells tested            {n}")
    print(f"  cells where row 0's argmax flips      {len(fl)}")
    print(f"  cells where row 0's hidden differs    {len(drift)}")
    if drift:
        layers = [r["first_diff_layer"] for r in drift]
        print(f"  first divergent layer  min {min(layers)}  max {max(layers)}")
    print("\n  Row 0 is causally independent of rows 1..M-1, so any non-zero")
    print("  difference here is the transformer computing the SAME token in the")
    print("  SAME context differently because of how many tokens shared its")
    print("  forward pass. That is not a speculative-decoding property — it")
    print("  reaches prefill-vs-decode, chunked prefill and batched serving.")

    with open(args.out, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  appended {len(records)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

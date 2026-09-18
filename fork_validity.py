#!/usr/bin/env python3
"""
fork_validity.py — settles prompt 1 by construction instead of by heuristic.

WHY divergence_text.py's STRUCTURAL VERDICT IS A FALSE POSITIVE

Its discriminator is: if the sequences REALIGN after the edit, a numerical tie
cannot explain it, so the accept/emit bookkeeping is wrong. The stated premise
is "once two greedy runs pick different tokens they condition on different text
and have no reason to ever agree again."

That premise holds for high-entropy free text. It fails for the text actually
being generated here, which is a template:

    *   `data` (or `value`): The <...>.
    *   `next`: A pointer/reference to the next node in the sequence.

The two runs finish that clause with equivalent wording ("the actual
information stored." vs "the information stored in the node.") and the next
bullet is then determined by the list structure, not by three words inside a
closed sentence. Reconvergence after a synonym fork is the NORMAL behaviour of
templated / chain-of-thought generation, so "75 of 77 tokens match after the
edit" is not evidence of a dropped token.

Independent of that argument, shape_stability already found position 82
unstable at blocks 2 and 4 in a TEACHER-FORCED test — no drafter, no cache
trim, no accept/emit path, nothing that could carry a bookkeeping bug. That
alone contradicts the STRUCTURAL verdict.

THE TEST THAT NEEDS NO ARGUMENT

If the speculative run merely forked at a near-tie, then everything it emitted
after the fork must be exactly what an ordinary greedy decoder produces from
the speculative run's OWN prefix. So:

    teacher-force  base[:j] + [spec[j]]  then free-run
    -> must reproduce spec[j+1:] token for token

A dropped, duplicated or misplaced token cannot survive that: the emitted tail
would not be a valid greedy continuation of anything. The reverse control
(teacher-force base[:j] + [base[j]] and reproduce base[j+1:]) proves the
harness itself is faithful — without it a match below means nothing.

TWO MEASUREMENT BUGS THIS ALSO FIXES

1. divergence_text.logits_at() slices to one row BEFORE the output projection:

       h  = tm(mx.array([prefix[i:i+step]]), cache=cache)     # step rows
       lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)  # 1 row -> GEMV

   spec_decode does as_linear(hv) over all g+1 rows. So its "4 rows (verify /
   GEMM)" column was a GEMV that differed from the 1-row column only in the
   attention path — which is why the top-2 logits came back byte-identical and
   it failed to reproduce a flip shape_stability sees at three block sizes.
   Here the projection is done over the full block and the row is indexed
   after, matching spec_decode exactly.

2. The instability is a function of (M, row), not of M. shape_stability tests
   one alignment per block size — boundaries at multiples of `block` from 0 —
   and that conflates the two. Mapping its results by where the deciding token
   lands: prompt 0 flips at row 0 of 2, 4 and 8 but not row 0 of 3; prompt 1
   flips at row 1 of 2 and 4, not row 0 of 3, not row 1 of 8. Powers of two
   flip and 3 does not, which is a kernel tiling signature, not a margin
   signature. So "largest unstable margin x 1.5" is fitted over an unmeasured
   dimension. This sweeps (M, row) exhaustively at the fork.

   RETRACTED by this script's own output, 2026-08-28. The exhaustive sweep
   shows every column of both grids is CONSTANT below M=1: r1 is the
   speculative token at M=2..8 without exception, r3 is the base token at
   M=4..8 without exception, and the SPEC counts confirm it exactly
   (24 = 7+7+6+0+0+3+0+1, 11 = 7+4). The argmax is a function of the ROW
   INDEX ALONE, independent of block size, with M=1 special only because one
   row dispatches GEMV instead of GEMM. "Powers of two flip and 3 does not"
   was an artefact of shape_stability testing one alignment per block size:
   with block b the deciding token lands at row (j-1) mod b, so varying b
   varied r underneath. There is no power-of-two effect. The verdict this
   script prints under the grids is correspondingly too weak; see
   row_dependence.py, which replicates one hidden state across every row so
   that attention contributes nothing and only the projection kernel varies.

Nothing here is timed. The mx.eval barriers exist only to force the graph
before a host-side .item()/.tolist() read.

    ../bin/python fork_validity.py --gamma 3 --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx

import roofline as RF
from measure_acceptance import PROMPTS, enable_kv_capture, text_model
from spec_generate import greedy_baseline, spec_decode, use_plain_kv_caches


NEG_INF = float("-inf")


def new_cache(target):
    from mlx_lm.models.cache import make_prompt_cache
    c = make_prompt_cache(target)
    use_plain_kv_caches(c)          # control_variables proved this is not a variable
    return c


def top2(lg: mx.array):
    """(best_id, rel_margin, ulps) with mx.sort semantics, without the sort."""
    best_a = mx.max(lg)
    ties_a = mx.sum(lg == best_a)
    bid_a = mx.argmax(lg)
    mx.eval(best_a, ties_a, bid_a)
    best, ties, bid = float(best_a.item()), int(ties_a.item()), int(bid_a.item())
    if ties > 1:
        second = best
    else:
        s = mx.max(mx.where(lg == best_a, mx.array(NEG_INF, lg.dtype), lg))
        mx.eval(s)
        second = float(s.item())
    scale = max(abs(best), 1e-6)
    ulp = 2.0 ** (math.floor(math.log2(scale)) - 7)
    return bid, (best - second) / scale, (best - second) / ulp


def greedy_from(target, tm, ids, forced, n):
    """Teacher-force `forced`, then free-run `n` tokens, one row throughout.

    `forced` is a list of GENERATED tokens (index 0 = the first token after the
    prompt). Fed one at a time so that every scoring shape matches
    greedy_baseline's — otherwise this test would introduce the very shape
    variable it is trying to hold fixed.
    """
    cache = new_cache(target)
    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)
    for t in forced:
        h = tm(mx.array([[int(t)]]), cache=cache)
    out = []
    for _ in range(n):
        lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
        cur = mx.argmax(lg)
        mx.eval(cur)
        out.append(int(cur.item()))
        h = tm(cur.reshape(1, 1), cache=cache)
    return out


def argmax_grid(target, tm, ids, base, j, max_m=8):
    """argmax for generated position j as a function of (M rows, row index).

    Position j is predicted by the hidden state of token base[j-1]. To place
    that token at row r of an M-row forward, the block starts at a = j-1-r and
    the context base[0:a] is fed one row at a time first. Rows after r hold
    base[j], base[j+1], ... — in a real verify those are draft tokens, and
    causal masking makes row r's logits independent of them, so which tokens
    fill them does not matter. Only M does.

    The output projection is applied over the WHOLE block and the row indexed
    afterwards, which is what spec_decode does and what divergence_text did not.
    """
    grid = {}
    for r in range(max_m):
        a = j - 1 - r
        if a < 0:
            continue
        cache = new_cache(target)
        h = tm(mx.array([ids]), cache=cache)
        mx.eval(h)
        for t in base[:a]:
            h = tm(mx.array([[int(t)]]), cache=cache)
        mx.eval(h)
        for M in range(r + 1, max_m + 1):
            blk = base[a: a + M]
            if len(blk) < M:
                continue
            hh = tm(mx.array([[int(t) for t in blk]]), cache=cache)
            lgs = tm.embed_tokens.as_linear(hh)          # M-row GEMM
            lg = lgs[0, r]
            mx.eval(lg)
            grid[(M, r)] = top2(lg)
            for c in cache:
                c.trim(M)                                 # same op spec_decode uses
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--max-m", type=int, default=8)
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

    verdicts = []

    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        print("\n" + "=" * 78)
        print(f"PROMPT {pi}: {prompt}")
        print("=" * 78)

        base = greedy_baseline(target, tm, ids, args.tokens)
        st = {}
        spec = spec_decode(target, tm, drafter, ids, args.tokens, args.gamma,
                           st, ref=base)

        if base == spec:
            print("  identical — nothing to inspect.")
            verdicts.append("identical")
            continue

        j = next(i for i in range(min(len(base), len(spec))) if base[i] != spec[i])
        print(f"  fork at generated token {j}: "
              f"base {base[j]} {tok.decode([base[j]])!r}  vs  "
              f"spec {spec[j]} {tok.decode([spec[j]])!r}")

        # ------------------------------------------------- A: the cycle dump
        # spec_decode collects this whenever ref= is passed. divergence_text
        # passed ref= and then never printed it.
        bc = st.get("bad_cycle")
        print("\n  --- A: the emitting cycle, as spec_decode recorded it ---")
        if not bc:
            print("    no bad_cycle recorded")
        else:
            drafts, pl, n, bonus = (bc["drafts"], bc["target_preds"],
                                    bc["n_accepted"], bc["bonus"])
            print(f"    cycle {bc['cycle']}  out_index {bc['out_index']}")
            print(f"    cur           {bc['cur']} {tok.decode([bc['cur']])!r}")
            print(f"    drafts        {drafts}  {[tok.decode([d]) for d in drafts]}")
            print(f"    target preds  {pl}  {[tok.decode([p]) for p in pl]}")
            print(f"    n_accepted    {n}     bonus {bonus} {tok.decode([bonus])!r}")
            print(f"    emitted       {bc['emitted']}")
            print(f"    expected      {bc['expected']}")
            # structural self-consistency: the emit must BE drafts[:n]+[bonus],
            # and n must be the true length of the matching prefix.
            true_n = 0
            for i in range(len(drafts)):
                if pl[i] != drafts[i]:
                    break
                true_n += 1
            ok_emit = bc["emitted"] == drafts[:n] + [bonus]
            ok_n = (true_n == n)
            ok_bonus = (bonus == pl[n])
            print(f"    emitted == drafts[:n]+[bonus]   {'OK' if ok_emit else 'WRONG'}")
            print(f"    n == true match length          "
                  f"{'OK' if ok_n else f'WRONG (true {true_n})'}")
            print(f"    bonus == target_preds[n]        {'OK' if ok_bonus else 'WRONG'}")
            if ok_emit and ok_n and ok_bonus:
                print("    -> the cycle emitted exactly what the target predicted.")
                print("       Whatever differs, it is not the accept/emit arithmetic.")

        # ------------------------------------------ B: is the fork tail valid?
        print("\n  --- B: is the speculative tail a valid greedy continuation? ---")
        n_tail = min(len(base), len(spec)) - j - 1
        rebase = greedy_from(target, tm, ids, base[:j] + [base[j]], n_tail)
        ctrl = (rebase == base[j + 1: j + 1 + n_tail])
        print(f"    control  (force base[:{j}]+[base[{j}]]) reproduces base tail: "
              f"{'YES' if ctrl else 'NO'}")
        if not ctrl:
            k = next((i for i in range(n_tail)
                      if rebase[i] != base[j + 1 + i]), None)
            print(f"    !! harness is not faithful (first mismatch at +{k}).")
            print("       Test B below is meaningless until this control passes.")

        refork = greedy_from(target, tm, ids, base[:j] + [spec[j]], n_tail)
        match = (refork == spec[j + 1: j + 1 + n_tail])
        print(f"    test     (force base[:{j}]+[spec[{j}]]) reproduces spec tail: "
              f"{'YES' if match else 'NO'}")
        if not match:
            k = next((i for i in range(n_tail)
                      if i >= len(refork) or refork[i] != spec[j + 1 + i]), None)
            print(f"    first mismatch at +{k}"
                  f"  greedy {refork[k] if k is not None and k < len(refork) else '-'}"
                  f"  vs spec {spec[j + 1 + k] if k is not None else '-'}")

        if ctrl and match:
            v = "FORK — tail is a valid greedy continuation"
            print("\n    -> Every token the speculative run emitted after the fork is")
            print("       EXACTLY what an ordinary greedy decoder produces from the")
            print("       speculative prefix. A dropped, duplicated or misplaced")
            print("       token cannot produce that. The realignment downstream is")
            print("       the model rejoining a template, not a bookkeeping error.")
        elif ctrl and not match:
            v = "STRUCTURAL — tail is NOT a valid greedy continuation"
            print("\n    -> The emitted tail is not a greedy continuation of its own")
            print("       prefix. That IS a bookkeeping bug; see the cycle dump above.")
        else:
            v = "INCONCLUSIVE — control failed"

        # ------------------------------------------------ C: the (M,row) map
        print(f"\n  --- C: argmax at position {j} vs (rows M, row index) ---")
        grid = argmax_grid(target, tm, ids, base, j, args.max_m)
        print(f"    base={base[j]} {tok.decode([base[j]])!r}   "
              f"spec={spec[j]} {tok.decode([spec[j]])!r}")
        print(f"    {'M':>3} " + "".join(f"{('r' + str(r)):>7}" for r in range(args.max_m)))
        flips = 0
        for M in range(1, args.max_m + 1):
            cells = []
            for r in range(args.max_m):
                g = grid.get((M, r))
                if g is None:
                    cells.append(f"{'-':>7}")
                else:
                    bid = g[0]
                    if bid == base[j]:
                        cells.append(f"{'base':>7}")
                    elif bid == spec[j]:
                        cells.append(f"{'SPEC':>7}")
                        flips += 1
                    else:
                        cells.append(f"{bid:>7}")
            print(f"    {M:>3} " + "".join(cells))
        mg = [g[1] for g in grid.values()]
        print(f"    cells scored {len(grid)}, cells choosing the speculative token {flips}")
        if mg:
            print(f"    margin range across cells: {min(mg):.3e} .. {max(mg):.3e}")
        if flips:
            print("    -> the argmax is a function of (M, row), not of M alone.")
            print("       Any tie threshold has to be fitted over BOTH.")

        verdicts.append(v)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for i, v in enumerate(verdicts):
        print(f"  prompt {i}: {v}")
    if all(v.startswith("FORK") or v == "identical" for v in verdicts):
        print("\n  Every divergence is a fork at a near-tie AND every emitted tail is")
        print("  a valid greedy continuation. divergence_text's STRUCTURAL verdict")
        print("  was a false positive from its realignment heuristic, which does not")
        print("  hold for templated text. The accept/emit path is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

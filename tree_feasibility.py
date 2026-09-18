#!/usr/bin/env python3
"""
tree_feasibility.py — Phase 2 gate: can a token TREE beat the chain, against the
cost curve we actually measured?

WHY A GATE BEFORE ANY TREE CODE

CLAUDE.md already records one killed "tree-width recommendation that ignored the
cost curve". Phase 1 was gated the same way -- confidence_calibration measured the
precondition (is the drafter's confidence informative?) before a scheduler existed,
and that gate is the reason Phase 1's negative cost one run instead of a week. This
does the same for width.

A tree pays only if the extra positions it asks the target to score buy more
accepted tokens than they cost. Both halves are measurable HERE, with no tree:

  COVERAGE  P_W[d] = P(the target's actual token at depth d is inside the drafter's
            top-W shortlist | the whole prefix up to d is correct).
            Measured by TEACHER FORCING: the drafter is fed the target's own tokens,
            not its own guesses, so depth-d numbers are conditioned on a correct
            prefix. That conditioning is the whole point -- a tree only ever extends
            branches that are still alive, and confidence_calibration already showed
            that positions past the first rejection are scored against a context the
            target would never have produced and must be excluded.

  COST      t_full(k) for k nodes, from the measured sweep in runs.jsonl plus the
            LM head the sweep omits (verify_gap_diagnosis: the sweep times the
            forward only, ~7.3 ms short).

THE ARITHMETIC, FOR A SHAPE (W_1..W_D)

    nodes    k        = sum_d  prod_{i<=d} W_i        every branch must be scored
    accepted E[acc]   = sum_d  prod_{i<=d} P_{W_i}[i]
    ms/token          = (t_full(k) + draft_ms*k + host_ms) / (E[acc] + 1)

BIASED IN THE TREE'S FAVOUR ON PURPOSE. The sweep's forward-only t(k) runs ~4 ms
BELOW what this machine measures directly at k=4 (44.09 vs 48.42), and drafting a
tree node is charged the same 1.338 ms as a chain step even though a branching
drafter must redo work. If a tree still loses under assumptions that flatter it,
the negative is safe.

    ../bin/python tree_feasibility.py --depth 6 --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import mlx.core as mx

import mlxutil as U
import roofline as RF
from measure_acceptance import (
    PROMPTS, collect_shared_kv, enable_kv_capture, text_model,
)

WIDTHS = (1, 2, 4, 8, 16)
LM_HEAD_MS = 7.3          # verify_gap_diagnosis: the sweep omits this
DRAFT_MS_PER_NODE = 1.338  # confidence_calibration's fitted slope
HOST_MS = 4.6              # phase0_finding: graph build + tail


def load_tk():
    """Full verify cost per k: the measured forward-only sweep + the LM head."""
    rows = [json.loads(l) for l in open("runs.jsonl") if l.strip().startswith("{")]
    vd = [r for r in rows if r.get("record_type") == "verify_decomposition"]
    by = {}
    for r in vd:
        by[r["k"]] = r["ms_short"]          # last write wins; sweeps are consistent
    return {k: v + LM_HEAD_MS for k, v in by.items()}


def t_full(tk, k):
    """Cost of scoring k nodes. Beyond the measured range, hold the saturated value:
    the sweep is FLAT k=11..16 (111-116 ms), so extrapolating linearly would invent a
    penalty the machine does not charge -- again biased toward the tree."""
    if k in tk:
        return tk[k]
    ks = sorted(tk)
    if k < ks[0]:
        return tk[ks[0]]
    if k > ks[-1]:
        return tk[ks[-1]]
    lo = max(x for x in ks if x <= k)
    hi = min(x for x in ks if x >= k)
    if lo == hi:
        return tk[lo]
    f = (k - lo) / (hi - lo)
    return tk[lo] * (1 - f) + tk[hi] * f


def rank_of(target_id, cand, logits):
    """Rank of the target's token inside the drafter's shortlist, or None if absent."""
    order = mx.argsort(-logits.reshape(-1))
    ids = cand.reshape(-1)[order]
    hit = (ids == target_id)
    if not bool(mx.any(hit).item()):
        return None
    return int(mx.argmax(hit).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--depth", type=int, default=6, help="max tree depth to characterise")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--chain-ms-per-token", type=float, default=22.6,
                    help="measured chain operating point to beat (42.46 tok/s at 2.587 emitted)")
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    from mlx_lm.models import gemma4_assistant
    from mlx_lm.models.cache import make_prompt_cache

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
    scale = getattr(tm, "embed_scale", 1.0)

    # hits[d][W] = number of positions where the target's depth-d token was inside top-W
    hits = {d: {W: 0 for W in WIDTHS} for d in range(1, args.depth + 1)}
    seen = {d: 0 for d in range(1, args.depth + 1)}
    shortlist_n = None

    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        print(f"\n--- prompt {pi + 1} ---")

        # PASS 1: the target's own greedy continuation. This is the ground truth every
        # depth-d statistic is conditioned on.
        cache = make_prompt_cache(target)
        h = tm(mx.array([ids]), cache=cache)
        cur = mx.argmax(tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1))
        mx.eval(cur)
        ref = [int(cur.item())]
        for _ in range(args.tokens - 1):
            h = tm(cur.reshape(1, 1), cache=cache)
            cur = mx.argmax(tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1))
            mx.eval(cur)
            ref.append(int(cur.item()))
        print(f"  reference continuation: {len(ref)} tokens")

        # PASS 2: replay, and at each position run the drafter TEACHER-FORCED down the
        # reference path, recording where the target's own next token sits in the
        # drafter's shortlist at each depth.
        cache = make_prompt_cache(target)
        hidden = tm(mx.array([ids]), cache=cache)
        mx.eval(hidden)
        shared_kv = collect_shared_kv(target)
        pos = len(ids)

        for t in range(len(ref) - args.depth):
            h_back = hidden[:, -1:, :]
            tok_in = mx.array(ref[t - 1]) if t > 0 else mx.array(ref[0])
            # the drafter's first query conditions on the token at index pos
            tok_in = mx.array(ref[t])
            for d in range(1, args.depth + 1):
                emb = tm.embed_tokens(tok_in.reshape(1, 1)) * scale
                nxt, cand, logits = drafter(emb, h_back, shared_kv, offset=pos + d - 1)
                if shortlist_n is None:
                    shortlist_n = int(cand.reshape(-1).shape[0])
                    print(f"  drafter shortlist size: {shortlist_n}")
                want = ref[t + d]
                r = rank_of(want, cand, logits)
                seen[d] += 1
                if r is not None:
                    for W in WIDTHS:
                        if r < W:
                            hits[d][W] += 1
                # TEACHER FORCING: feed the target's token, not the drafter's guess
                tok_in = mx.array(want)
                h_back = nxt

            # advance the target by one real token
            hidden = tm(mx.array([[ref[t]]]), cache=cache)
            mx.eval(hidden)
            shared_kv = collect_shared_kv(target)
            pos += 1

    # ------------------------------------------------------------------ coverage
    P = {d: {W: (hits[d][W] / seen[d] if seen[d] else 0.0) for W in WIDTHS}
         for d in hits}
    print("\n" + "=" * 74)
    print("COVERAGE  P_W[d] = P(target's token in drafter's top-W | prefix correct)")
    print("=" * 74)
    print("  depth      n  " + "".join(f"   W={W:<5}" for W in WIDTHS))
    for d in sorted(P):
        print(f"  {d:>5} {seen[d]:>6}  " + "".join(f"  {P[d][W]:6.3f} " for W in WIDTHS))

    # ------------------------------------------------------------------ economics
    tk = load_tk()
    print("\n" + "=" * 74)
    print("SHAPE ECONOMICS  (t(k) from the measured sweep + LM head; biased FOR the tree)")
    print("=" * 74)
    print(f"  chain operating point to beat: {args.chain_ms_per_token:.1f} ms/token "
          f"= {1000/args.chain_ms_per_token:.1f} tok/s\n")

    shapes = []
    for D in range(1, args.depth + 1):
        shapes.append(tuple([1] * D))                       # the chain
    for W in (2, 4, 8):
        for D in range(2, args.depth + 1):
            shapes.append(tuple([W] + [1] * (D - 1)))        # widen at the root only
    for D in range(2, 5):
        shapes.append(tuple([2] * D))                        # full binary

    print(f"  {'shape':<22} {'k':>4} {'E[acc]':>7} {'t_full':>8} {'draft':>7} "
          f"{'ms/tok':>8} {'tok/s':>7}  verdict")
    best = best_chain = best_wide = None
    for sh in shapes:
        k = sum(int(mx.prod(mx.array(sh[:d + 1])).item()) for d in range(len(sh)))
        acc = 0.0
        run = 1.0
        for d, W in enumerate(sh, start=1):
            if d not in P:
                run = 0.0
                break
            run *= P[d][W] if W in P[d] else 0.0
            acc += run
        if k > 64:
            continue
        cost = t_full(tk, k) + DRAFT_MS_PER_NODE * k + HOST_MS
        mspt = cost / (acc + 1.0)
        tps = 1000.0 / mspt
        good = mspt < args.chain_ms_per_token
        # Track chains and widened shapes SEPARATELY. The question Phase 2 asks is not
        # "does the best shape beat the current operating point" -- a deeper CHAIN can do
        # that, and reporting it as a tree win would be exactly the "tree-width
        # recommendation that ignored the cost curve" CLAUDE.md already has on file.
        # The question is whether WIDTH pays at all.
        if max(sh) == 1:
            if best_chain is None or mspt < best_chain[1]:
                best_chain = (sh, mspt, tps, k, acc)
        else:
            if best_wide is None or mspt < best_wide[1]:
                best_wide = (sh, mspt, tps, k, acc)
        if best is None or mspt < best[1]:
            best = (sh, mspt, tps, k, acc)
        print(f"  {str(sh):<22} {k:>4} {acc:>7.3f} {t_full(tk,k):>8.1f} "
              f"{DRAFT_MS_PER_NODE*k:>7.1f} {mspt:>8.2f} {tps:>7.1f}  "
              f"{'BEATS CHAIN' if good else ''}")

    print(f"\n  best CHAIN   {str(best_chain[0]):<20} {best_chain[1]:6.2f} ms/token "
          f"= {best_chain[2]:5.1f} tok/s  (k={best_chain[3]}, E[acc]={best_chain[4]:.3f})")
    print(f"  best WIDENED {str(best_wide[0]):<20} {best_wide[1]:6.2f} ms/token "
          f"= {best_wide[2]:5.1f} tok/s  (k={best_wide[3]}, E[acc]={best_wide[4]:.3f})")
    pen = 100 * (best_wide[1] / best_chain[1] - 1)
    print(f"  width costs {pen:+.0f} % on ms/token")
    wins = best_wide[1] < best_chain[1]
    verdict = ("GO — width pays: the best widened shape beats the best chain"
               if wins else
               f"NO-GO — width never pays. The best widened shape {best_wide[0]} is "
               f"{pen:+.0f} % WORSE than the best chain {best_chain[0]}, even with the cost "
               f"model biased in the tree's favour. Coverage does rise with W "
               f"(P_1=%.3f -> P_8=%.3f at depth 1) but k rises faster than t(k) forgives."
               % (P[1][1], P[1][8]))
    print(f"\n  PHASE 2 VERDICT: {verdict}")

    with open(args.out, "a") as f:
        f.write(json.dumps({
            "record_type": "tree_feasibility", "schema_version": 1,
            "depth": args.depth, "widths": list(WIDTHS), "shortlist_n": shortlist_n,
            "coverage": {str(d): P[d] for d in P}, "n_positions": seen,
            "chain_ms_per_token": args.chain_ms_per_token,
            "best_shape": list(best[0]), "best_ms_per_token": best[1],
            "best_tok_s": best[2], "best_k": best[3], "best_expected_accepted": best[4],
            "best_chain": {"shape": list(best_chain[0]), "ms_per_token": best_chain[1],
                           "tok_s": best_chain[2], "k": best_chain[3],
                           "expected_accepted": best_chain[4]},
            "best_widened": {"shape": list(best_wide[0]), "ms_per_token": best_wide[1],
                             "tok_s": best_wide[2], "k": best_wide[3],
                             "expected_accepted": best_wide[4]},
            "width_penalty_pct": pen, "width_pays": wins,
            "verdict": verdict,
            "cost_model": {"lm_head_ms": LM_HEAD_MS, "draft_ms_per_node": DRAFT_MS_PER_NODE,
                           "host_ms": HOST_MS, "biased_toward_tree": True},
        }, default=str) + "\n")
    print(f"\n  appended tree_feasibility to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

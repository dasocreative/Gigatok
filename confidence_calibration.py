#!/usr/bin/env python3
"""
confidence_calibration.py — Recipe 4 step 1: does the drafter know when it is
about to be rejected?

THE OPPORTUNITY, RESTATED AS ARITHMETIC

30-33 % of cycles accept zero tokens and pay the full verify. Verify cost is
t(k) = 37.2 + 6.75*k ms for k = gamma+1 positions, so at gamma=3 roughly 20 ms
per cycle is spent scoring draft tokens that were never going to be accepted.
A DSpark-style schedule stops drafting early when the drafter is unsure,
trading draft tokens (cheap) against verify positions (6.75 ms each).

Whether that works depends entirely on one thing nobody has measured here: is
the drafter's own confidence CALIBRATED against the target's acceptance? If a
low-confidence draft is accepted just as often as a high-confidence one, the
signal is worthless and the schedule cannot beat a fixed gamma. This script
measures that before any scheduling code is written.

THE BREAK-EVEN, FROM MEASURED COSTS

Adding one more draft token to a block costs

    b_draft (ms per extra draft step) + b_verify (ms per extra verify position)

and yields one extra emitted token with probability q = P(this token and every
earlier one in the block are accepted). An emitted token is worth the current
ms/token of the speculative run. So draft another token iff

    q  >  (b_draft + b_verify) / ms_per_token        =: q*

Both slopes are fitted here from runs at gamma = 2, 4, 8 rather than reused
from the recipe's earlier t(k) fit, so the numbers come from the same machine
state as the calibration.

WHAT IS AND IS NOT MEASURED

Acceptance is a PREFIX property: draft i is accepted only if drafts 0..i-1 were
too. Past the first rejection the target is scoring a context built from tokens
it rejected, so those positions say what the target would have predicted given
a context it would never have produced. They are recorded but kept out of the
calibration curve, which covers only i <= n_accepted -- the region a real
schedule operates in.

The threshold sweep at the end is a SIMULATION over recorded cycles, not a
measured speedup. Drafting fewer tokens changes which tokens are emitted per
cycle and therefore the whole downstream trajectory, so it is a first-order
sizing estimate. It is also exact only up to the (M, row) instability
fork_validity measured: pl[i] is causally independent of later rows, but at a
<= 2 ULP margin the argmax can still move. Treat the predicted tok/s as a
go/no-go number, not as a result. Nothing is claimed as a speedup until the
schedule is implemented and benchmarked against the same baseline.

    ../bin/python confidence_calibration.py --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import statistics as stats
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx

import roofline as RF
from measure_acceptance import (
    PROMPTS, collect_shared_kv, enable_kv_capture, text_model,
)
from spec_generate import rewind, slice_shared_kv, use_plain_kv_caches


NEG_INF = float("-inf")


def draft_with_conf(drafter, tm, hidden, last_token, shared_kv, gamma, pos):
    """measure_acceptance.draft(), plus per-step confidence.

    Same single-barrier discipline: token ids and confidences stay as device
    arrays through all gamma steps so the block is one lazy graph, and there is
    exactly ONE mx.eval at the end. Adding a barrier per step would impose the
    same gamma CPU<->GPU serialisations that cost this loop 3x -> 1.4x before,
    and would corrupt the draft timing this script fits a slope to.

    Returns (ids, top1_probs, logit_gaps). Both signals are over the drafter's
    OWN shortlist -- that is what a real schedule would have available at draft
    time, with no target forward to consult.
    """
    scale = getattr(tm, "embed_scale", 1.0)
    toks, probs, gaps = [], [], []
    tok = last_token
    h_back = hidden
    for i in range(gamma):
        emb = tm.embed_tokens(tok.reshape(1, 1)) * scale
        nxt, cand, logits = drafter(emb, h_back, shared_kv, offset=pos + i)
        lg = logits.reshape(-1)
        j = mx.argmax(lg)
        tid = cand.reshape(-1)[j]
        p = mx.softmax(lg.astype(mx.float32), axis=-1)
        best = mx.max(lg)
        second = mx.max(mx.where(lg == best, mx.array(NEG_INF, lg.dtype), lg))
        toks.append(tid)
        probs.append(mx.max(p))
        gaps.append((best - second).astype(mx.float32))
        tok = tid
        h_back = nxt
    mx.eval(toks, probs, gaps)          # ONE barrier for the whole block
    return ([int(t.item()) for t in toks],
            [float(p.item()) for p in probs],
            [float(g.item()) for g in gaps])


def run_cycles(target, tm, drafter, ids, max_tokens, gamma):
    """spec_decode's loop, instrumented. Returns (cycles, timing, emitted)."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    use_plain_kv_caches(cache)

    hidden = tm(mx.array([ids]), cache=cache)
    mx.eval(hidden)
    shared_kv = collect_shared_kv(target)
    cur = mx.argmax(tm.embed_tokens.as_linear(hidden[:, -1:]).reshape(-1))
    mx.eval(cur)

    out = [int(cur.item())]
    pos = len(ids)
    cycles = []
    t_draft = t_verify = 0.0

    while len(out) < max_tokens:
        g = min(gamma, max_tokens - len(out))

        # Both timed regions end on the barrier that materialises their result:
        # draft_with_conf's single mx.eval, and mx.eval(preds) below. Nothing
        # lazy escapes either region, so the two numbers sum to the cycle.
        w0 = time.perf_counter()
        drafts, probs, gaps = draft_with_conf(
            drafter, tm, hidden[:, -1:, :], cur, shared_kv, g, pos)
        t_draft += time.perf_counter() - w0

        w1 = time.perf_counter()
        xv = mx.array([[int(cur.item())] + drafts])
        hv = tm(xv, cache=cache)
        preds = mx.argmax(tm.embed_tokens.as_linear(hv), axis=-1)[0]
        mx.eval(preds)
        pl = preds.tolist()
        t_verify += time.perf_counter() - w1

        n = 0
        for i in range(g):
            if pl[i] != drafts[i]:
                break
            n += 1
        bonus = pl[n]

        cycles.append({
            "g": g, "n": n, "probs": probs, "gaps": gaps,
            "matched": [pl[i] == drafts[i] for i in range(g)],
        })

        out.extend(drafts[:n])
        out.append(bonus)
        rewind(cache, g - n)
        hidden = hv[:, n: n + 1, :]
        cur = mx.array(bonus)
        pos += n + 1
        shared_kv = slice_shared_kv(collect_shared_kv(target), g - n)

    c = max(1, len(cycles))
    return cycles, {
        "draft_ms": t_draft / c * 1e3,
        "verify_ms": t_verify / c * 1e3,
        "cycles": len(cycles),
        "tokens": len(out),
        "total_s": None,
    }, out[:max_tokens]


def linfit(xs, ys):
    """Least squares y = a + b x. Returns (a, b)."""
    n = len(xs)
    mx_ = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx_) ** 2 for x in xs)
    b = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return my - b * mx_, b


def calibration_table(cycles, key, edges):
    """P(accept | signal in bucket), over the leading-run region only.

    Draft i is a real acceptance opportunity only if drafts 0..i-1 were all
    accepted; that is i <= n. Positions past the first rejection are scored on
    a context the target rejected, so they are counted separately rather than
    folded into a curve a scheduler would rely on.
    """
    live = [[0, 0] for _ in range(len(edges) - 1)]
    dead = [0, 0]
    for c in cycles:
        for i in range(c["g"]):
            v = c[key][i]
            hit = c["matched"][i]
            if i <= c["n"]:
                for b in range(len(edges) - 1):
                    if edges[b] <= v < edges[b + 1]:
                        live[b][0] += 1
                        live[b][1] += int(hit)
                        break
            else:
                dead[0] += 1
                dead[1] += int(hit)
    return live, dead


def simulate(cycles, thr, a_d, b_d, a_v, b_v):
    """Predicted tok/s if drafting stopped when top-1 prob drops below thr."""
    toks = ms = 0.0
    drafted_hist = Counter()
    for c in cycles:
        d = c["g"]
        for i in range(c["g"]):
            if c["probs"][i] < thr:
                d = i + 1                      # this token is drafted, then stop
                break
        acc = min(c["n"], d)
        toks += acc + 1
        ms += (a_d + b_d * d) + (a_v + b_v * (d + 1))
        drafted_hist[d] += 1
    return (toks / ms * 1e3 if ms else 0.0), toks, ms, drafted_hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--gammas", default="2,4,8")
    args = ap.parse_args()

    gammas = [int(g) for g in args.gammas.split(",") if g.strip()]

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

    # warmup, discarded: fanless M3 Air, first pass pays graph construction
    ids0, _ = RF.build_eval_prompt(tok, PROMPTS[0])
    run_cycles(target, tm, drafter, ids0, 24, 4)

    per_gamma = {}
    pooled = {g: [] for g in gammas}

    for g in gammas:
        d_ms, v_ms, allc = [], [], []
        for pi, prompt in enumerate(PROMPTS[: args.prompts]):
            ids, _ = RF.build_eval_prompt(tok, prompt)
            cycles, t, _ = run_cycles(target, tm, drafter, ids, args.tokens, g)
            d_ms.append(t["draft_ms"])
            v_ms.append(t["verify_ms"])
            allc.extend(cycles)
        per_gamma[g] = {
            "draft_ms": stats.median(d_ms),
            "verify_ms": stats.median(v_ms),
            "cycles": len(allc),
            "mean_acc": stats.fmean(c["n"] for c in allc),
            "zero_frac": sum(1 for c in allc if c["n"] == 0) / len(allc),
        }
        pooled[g] = allc
        pg = per_gamma[g]
        print(f"  gamma {g}: draft {pg['draft_ms']:6.2f} ms  "
              f"verify {pg['verify_ms']:6.2f} ms  "
              f"mean acc {pg['mean_acc']:.2f}  zero {100*pg['zero_frac']:.1f}%")

    # ------------------------------------------------------------ cost model
    a_d, b_d = linfit(gammas, [per_gamma[g]["draft_ms"] for g in gammas])
    a_v, b_v = linfit([g + 1 for g in gammas],
                      [per_gamma[g]["verify_ms"] for g in gammas])

    print("\n" + "=" * 78)
    print("COST MODEL (fitted on this run, not reused from the recipe)")
    print("=" * 78)
    print(f"  draft   ms = {a_d:7.2f} + {b_d:6.3f} * gamma")
    print(f"  verify  ms = {a_v:7.2f} + {b_v:6.3f} * k        (k = gamma+1)")
    print(f"  recipe's earlier verify fit: 37.2 + 6.75 * k")

    best_g = max(gammas, key=lambda g: (per_gamma[g]["mean_acc"] + 1) /
                 ((a_d + b_d * g) + (a_v + b_v * (g + 1))))
    pg = per_gamma[best_g]
    ms_tok = ((a_d + b_d * best_g) + (a_v + b_v * (best_g + 1))) / (pg["mean_acc"] + 1)
    q_star = (b_d + b_v) / ms_tok
    print(f"\n  best fixed gamma here      {best_g}")
    print(f"  ms per emitted token       {ms_tok:.2f}")
    print(f"  marginal cost of +1 draft  {b_d + b_v:.2f} ms")
    print(f"  BREAK-EVEN q*              {q_star:.3f}")
    print("  -> draft another token only where P(accept) exceeds that.")

    # ------------------------------------------------------- calibration
    gmax = max(gammas)
    cyc = pooled[gmax]
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    live, dead = calibration_table(cyc, "probs", edges)

    print("\n" + "=" * 78)
    print(f"CALIBRATION at gamma={gmax}: P(target accepts | drafter top-1 prob)")
    print("=" * 78)
    print(f"  {'bucket':>16}{'n':>8}{'accepted':>10}{'P(acc)':>9}   vs q*")
    for b in range(len(edges) - 1):
        n, h = live[b]
        if not n:
            continue
        p = h / n
        print(f"  [{edges[b]:.2f},{edges[b+1]:.2f}){n:>8}{h:>10}{p:>9.3f}"
              f"   {'DRAFT' if p > q_star else 'stop '}")
    if dead[0]:
        print(f"  (past first rejection: {dead[0]} positions, "
              f"{dead[1]/dead[0]:.3f} would have matched — excluded)")

    ns = [live[b][0] for b in range(len(edges) - 1)]
    ps = [live[b][1] / live[b][0] for b in range(len(edges) - 1) if live[b][0]]
    if len(ps) >= 2 and max(ps) - min(ps) < 0.10:
        print("\n  !! P(accept) is FLAT across confidence buckets (spread "
              f"{max(ps)-min(ps):.3f}).")
        print("     The drafter's confidence does not predict acceptance, so a")
        print("     confidence schedule cannot beat a fixed gamma. Do not build")
        print("     it -- go to mx.compile on the draft step instead.")
    else:
        print(f"\n  P(accept) spread across buckets: "
              f"{max(ps)-min(ps):.3f} — the signal separates.")

    # ---------------------------------------------------------- simulation
    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP (SIMULATED on recorded cycles — not a measurement)")
    print("=" * 78)
    base_tps, base_toks, base_ms, _ = simulate(cyc, 0.0, a_d, b_d, a_v, b_v)
    print(f"  {'thr':>6}{'pred tok/s':>12}{'vs fixed':>10}{'mean drafted':>14}"
          f"   drafted histogram")
    rows = []
    for thr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        tps, toks, ms, hist = simulate(cyc, thr, a_d, b_d, a_v, b_v)
        md = sum(k * v for k, v in hist.items()) / max(1, sum(hist.values()))
        rows.append((thr, tps))
        print(f"  {thr:>6.2f}{tps:>12.2f}{tps/base_tps:>10.3f}{md:>14.2f}"
              f"   {dict(sorted(hist.items()))}")
    best_thr, best_tps = max(rows, key=lambda r: r[1])
    print(f"\n  best simulated threshold {best_thr:.2f} -> {best_tps:.2f} tok/s "
          f"({best_tps/base_tps:.3f}x over fixed gamma={gmax} in the same model)")
    print("  This is a sizing estimate. Drafting fewer tokens changes which")
    print("  tokens each cycle emits and therefore the whole trajectory, so the")
    print("  only honest number comes from implementing the schedule and running")
    print("  it against greedy_baseline on these same prompts.")
    if best_tps / base_tps < 1.05:
        print("\n  < 5 % predicted. Not worth the implementation risk; the next")
        print("  item (mx.compile on the draft step) targets a measured "
              f"{per_gamma[gmax]['draft_ms']:.1f} ms/cycle instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
spec_generate.py — Recipe 4, step 4: lossless speculative decoding with the
Gemma 4 MTP drafter, and the losslessness check that proves it.

    ../bin/python spec_generate.py --gamma 8 --tokens 128 --verify-lossless

===================== THE SLIDING-WINDOW ROLLBACK TRAP =====================

Speculative decoding writes gamma+1 tokens into the KV cache and then must
discard the rejected tail. mlx-lm exposes `trim_prompt_cache` for this, but:

    RotatingKVCache.is_trimmable()  ->  self.offset < self.max_size

Past the sliding window (512 here) a rotating cache refuses to be trimmed, and
this model has 20 of them against 4 global ones. The reason is real: writing
gamma+1 entries into a full ring EVICTS the gamma+1 oldest entries in the
window. Rewinding the offset cannot bring them back — they are gone. The window
would silently lose up to gamma-n genuine tokens on every rejection, the target
would then see a different context than an ordinary run, and the output would
diverge. That is a losslessness failure, not a quality nuance.

The fix here is to give each rotating cache `gamma` extra slots. Attention is
unaffected: gemma4_text builds its own mask with `window_size=sliding_window`,
so the model still attends over exactly 512 positions no matter how large the
buffer is. The slack only absorbs the speculative overwrite, so the entries a
rejection needs back were never evicted in the first place.

With that slack in place we call `c.trim()` per cache directly rather than
`trim_prompt_cache`, whose `is_trimmable` gate would still refuse.

`--verify-lossless` is what settles whether this reasoning holds: it generates
the same prompt greedily with and without speculation and compares token ids.
If the sliding-window argument above is wrong, that check fails loudly.

================================ THE LOOP ================================

    draft gamma tokens with the drafter        ~0.3 ms each, target untouched
    verify [cur, d0..d_{gamma-1}] in ONE target forward
    accept the longest prefix the target agrees with
    the target's own next token comes free from the same forward
    trim the rejected tail
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as stats
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx

import mlxutil as U
import roofline as RF
from measure_acceptance import (
    PROMPTS, collect_shared_kv, draft, enable_kv_capture, text_model,
)


def use_plain_kv_caches(cache) -> int:
    """
    Replace every RotatingKVCache with a plain KVCache.

    WHY, after two failed attempts at making the ring trim correctly:

    RotatingKVCache switches between an in-place update and a concat update
    depending on the sequence length, and `trim` rewinds both `offset` and
    `_idx` on assumptions that hold for the in-place path. A speculative verify
    feeds L = gamma+1 tokens with a non-empty cache, then trims — a combination
    ordinary decoding never produces. The desync does not show up immediately;
    it accumulates. That is exactly what we saw: cycles 1-26 perfect, cycle 27
    wrong, and a different failure cycle at every gamma.

    A plain KVCache has one integer of state and `trim` is `offset -= n`. There
    is nothing to desynchronise.

    This does not change what the model attends to. gemma4_text builds its mask
    with `create_attention_mask(h, c, window_size=self.window_size)`, so the
    sliding window is enforced by the MASK regardless of cache type. The cache
    only decides what is retained, and retaining more is harmless.

    Cost is memory: sliding layers hold the full context instead of 512
    entries — ~40 KB per token across the 20 sliding layers, so 346 MB at 8K
    against the 4 GiB Recipe 0 measured as headroom.
    """
    from mlx_lm.models.cache import KVCache

    n = 0
    for i, c in enumerate(cache):
        if getattr(c, "max_size", None) is not None:
            cache[i] = KVCache()
            n += 1
    return n


def add_rollback_slack(cache, capacity: int) -> int:
    """
    Stop the rotating caches from ever rotating, by giving them capacity for
    the whole run.

    THE INSIGHT: the sliding window is enforced by the ATTENTION MASK, not by
    the cache size. gemma4_text builds its mask with
    `create_attention_mask(h, c, window_size=self.window_size)`, so the model
    attends over exactly `sliding_window` positions however many the cache
    holds. A rotating cache that never wraps is behaviourally a plain growing
    cache — and a plain growing cache trims correctly.

    An earlier version added only `gamma` slots. That was not enough: it left
    the caches rotating, and a multi-token verify (L = gamma+1) followed by a
    trim does not round-trip through the ring bookkeeping. The symptom was a
    DROPPED token mid-sequence — the target's predictions stayed right while
    the cache state drifted.

    The cost is memory: sliding layers now hold the full context instead of
    512 entries. For 20 layers at 2 KV heads x 256 dims that is ~40 KB per
    token — 346 MB at 8K context, against the 4 GiB of headroom Recipe 0
    measured. Cheap, and it buys exact rollback.
    """
    n = 0
    for c in cache:
        if getattr(c, "max_size", None) is not None:
            c.max_size = capacity
            n += 1
    return n


def draft_scheduled(drafter, tm, hidden, last_token, shared_kv, gamma_max, pos, thr):
    """PHASE 1. Draft up to gamma_max tokens, stopping as soon as the drafter's own
    top-1 probability falls below `thr`.

    WHY THIS IS THE LEVER, IN THE PROJECT'S OWN NUMBERS

    The full verify costs 23.34 + 8.434*k ms (fitted in confidence_calibration on
    this machine, and independently confirmed: 57.1 ms at k=4 against a measured
    56.0 ms). A draft step costs 1.338 ms. A verify SLOT is therefore ~6.3x a draft
    step, so the cycle is priced by how many positions we ask the target to score,
    not by how many we ask the drafter to produce.

    confidence_calibration measured that the drafter knows when it is about to be
    rejected: P(accept) runs 0.333 in the [0.00,0.50) confidence bucket to 0.974 in
    [0.99,1.00), a spread of 0.641, against a break-even q* of 0.422. Below ~0.70
    a drafted token is not worth the 8.434 ms slot it would occupy.

    WHY A BARRIER PER STEP, AFTER P0.1/P0.3 WORKED TO REMOVE THEM

    Deciding whether to draft another token requires this token's confidence on the
    host, so the single-barrier discipline cannot survive early stopping. That is a
    deliberate trade, priced from the same fit: stopping early skips whole drafter
    steps (1.338 ms each) AND the verify slots they would have claimed (8.434 ms
    each), while a barrier costs the ~1 % that P0.1 measured for gamma+2 of them.
    Drafting all gamma_max lazily and truncating afterwards would keep one barrier
    but pay every drafter step regardless -- ~6 ms/cycle worse at gamma_max=8.

    The first token is always drafted: stopping at zero degenerates to plain decode
    (k=1) and forfeits the free bonus token the verify produces anyway. That also
    matches the simulated histograms, whose minimum drafted count is 1.

    Losslessness is untouched by construction. This changes only WHICH tokens are
    offered to the target; the target still verifies every one of them and only
    tokens it produced itself are ever emitted.
    """
    scale = getattr(tm, "embed_scale", 1.0)
    toks, confs = [], []
    tok, h_back = last_token, hidden
    for i in range(gamma_max):
        emb = tm.embed_tokens(tok.reshape(1, 1)) * scale
        nxt, cand, logits = drafter(emb, h_back, shared_kv, offset=pos + i)
        lg = logits.reshape(-1)
        j = mx.argmax(lg)
        toks.append(cand.reshape(-1)[j])
        # Same signal confidence_calibration fitted its curve on: top-1 probability
        # over the drafter's OWN shortlist, which is all a schedule has at draft time.
        confs.append(mx.max(mx.softmax(lg.astype(mx.float32), axis=-1)))
        tok, h_back = toks[-1], nxt
    # ONE barrier for the whole block, then truncate on the host. An earlier version
    # evaluated per step so it could stop the drafter early; that saved drafter steps
    # (1.338 ms) but cost ~3.7 CPU<->GPU barriers per cycle, and measured SLOWER than
    # fixed gamma=3 even on cycles where it both accepted more and used fewer verify
    # slots. Verify slots are the expensive term; buy them back without giving up the
    # single-barrier discipline P0.1/P0.3 established.
    ids = mx.stack(toks)
    cf = mx.stack(confs)
    mx.eval(ids, cf)
    cl = cf.tolist()
    g = 1
    while g < gamma_max and cl[g - 1] >= thr:
        g += 1
    return ids[:g]
    return mx.stack(toks)


def slice_shared_kv(shared_kv: dict, drop: int) -> dict:
    """
    Trim the captured K/V by `drop` positions to match the trimmed cache.

    Without this the drafter conditions on tokens the target just REJECTED.
    That cannot break losslessness — the target re-verifies everything — but it
    does depress acceptance, which is why the in-loop mean (1.78-2.12) came in
    below the 2.15 measured offline.

    Safe only because the caches no longer rotate, so keys are in natural
    position order and the tail is the newest entries.
    """
    if drop <= 0:
        return shared_kv
    out = {}
    for k, (keys, values) in shared_kv.items():
        out[k] = (keys[..., : keys.shape[-2] - drop, :],
                  values[..., : values.shape[-2] - drop, :])
    return out


def rewind(cache, n_tokens: int) -> None:
    """Discard the last n_tokens from every cache, bypassing is_trimmable."""
    if n_tokens <= 0:
        return
    for c in cache:
        try:
            c.trim(n_tokens)
        except Exception as e:
            raise RuntimeError(
                f"cache {type(c).__name__} could not be rewound by {n_tokens}: {e}"
            ) from e


def _top2_margin(logits: mx.array):
    """(relative gap, gap in bf16 ULPs) between the best and second-best token.

    RETURN BOTH, AND THRESHOLD ON THE ULP COUNT. A relative margin is
    scale-dependent in a way that makes a fixed relative threshold ragged:

        rel = ulps * ulp/scale,  and  ulp/scale = 2^(floor(log2 s) - 7)/s
                                                 in (2^-8, 2^-7]

    so ulp/scale varies by a factor of 2 depending on where in its binade the
    top logit sits. control_variables measured a top logit of ~39.25 at prompt 0
    position 121 (ULP 0.25) against the ~28.5 (ULP 0.125) assumed elsewhere in
    this project -- a whole binade apart, on the same run. A single relative
    threshold of 2.3e-2 therefore means anywhere from 2.9 to 5.9 ULP depending
    on the position, which is a 2x-ragged band masquerading as one number.

    The ULP count is the invariant. control_variables also showed it survives
    final_logit_softcapping exactly (1.00 -> 1.00 and 2.00 -> 2.00 at the two
    divergent positions), because the cap shrank the gap and dropped the scale
    a binade together. Relative margins do NOT survive it (6.369e-03 ->
    4.831e-03). One more reason to publish the ULP number.

    A near-zero margin means greedy decoding is balanced on a knife edge at
    that position, and ANY difference in floating-point reduction order can
    flip it. That matters here: verification scores gamma+1 positions in one
    matmul while the baseline scores one at a time. Different shapes, different
    accumulation order, occasionally a different argmax on a near-tie. That is
    numerical, not a logic bug, and no amount of cache correctness removes it.
    """
    v = mx.sort(logits.reshape(-1))
    best, second = float(v[-1].item()), float(v[-2].item())
    scale = max(abs(best), 1e-6)
    ulp = 2.0 ** (math.floor(math.log2(scale)) - 7)
    return (best - second) / scale, (best - second) / ulp


def greedy_baseline(target, tm, ids, max_tokens, want_margins: bool = False):
    """Ordinary autoregressive greedy decode — the reference to match."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    h = tm(mx.array([ids]), cache=cache)
    lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
    cur = mx.argmax(lg)
    mx.eval(cur)
    out, margins, ulps = [int(cur.item())], [], []
    if want_margins:
        m, u = _top2_margin(lg)
        margins.append(m)
        ulps.append(u)
    for _ in range(max_tokens - 1):
        h = tm(cur.reshape(1, 1), cache=cache)
        lg = tm.embed_tokens.as_linear(h[:, -1:]).reshape(-1)
        cur = mx.argmax(lg)
        mx.eval(cur)
        out.append(int(cur.item()))
        if want_margins:
            m, u = _top2_margin(lg)
            margins.append(m)
            ulps.append(u)
    return (out, margins, ulps) if want_margins else out


def spec_decode(target, tm, drafter, ids, max_tokens, gamma, stats_out=None, ref=None,
                conf_threshold: float = 0.0):
    """
    `ref` is an optional baseline token list. When given, every cycle is checked
    against it in lockstep and the FIRST divergent cycle is dumped with full
    state. Comparing only the final strings tells you a run went wrong; this
    tells you which cycle, with what drafts, and what the target actually said.
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    widened = use_plain_kv_caches(cache)
    if stats_out is not None:
        stats_out["rotating_caches_widened"] = widened
        stats_out["untrimmable"] = [
            type(c).__name__ for c in cache if not c.is_trimmable()
        ]

    hidden = tm(mx.array([ids]), cache=cache)
    mx.eval(hidden)
    shared_kv = collect_shared_kv(target)
    cur = mx.argmax(tm.embed_tokens.as_linear(hidden[:, -1:]).reshape(-1))
    mx.eval(cur)

    out = [int(cur.item())]
    pos = len(ids)
    accepted_hist = Counter()
    drafted_hist = Counter()
    n_cycles = 0
    # Wall AND cpu for each half of the cycle. The profiler says the pieces
    # cost 51.9 ms/cycle while the loop takes ~110, so the gap is in building
    # the graphs, not running them. cpu ~= wall on a phase means the GPU is
    # waiting on Python — the same finding Recipe 0 made for plain decode, and
    # the case for mx.compile on the draft step.
    t_draft = t_verify = c_draft = c_verify = 0.0
    # PHASE 0 PROFILING. The verify window covers two very different things:
    # Python building 42 layers of graph, and the GPU actually running it. They
    # need separate numbers, because the fix for one is not the fix for the other.
    # t_build ends at the barrier, t_wait is the barrier. t_tail is the per-cycle
    # Python tax (rewind + the shared-kv walk) that neither timer was counting.
    t_build = t_wait = t_tail = 0.0

    while len(out) < max_tokens:
        gmax = min(gamma, max_tokens - len(out))
        w0, p0 = time.perf_counter(), time.process_time()
        if conf_threshold > 0.0:
            drafts_dev = draft_scheduled(drafter, tm, hidden[:, -1:, :], cur,
                                         shared_kv, gmax, pos, conf_threshold)
        else:
            drafts_dev = draft(drafter, tm, hidden[:, -1:, :], cur, shared_kv, gmax, pos)
        # gamma is now a CEILING, not a constant: the schedule picks g per cycle.
        g = int(drafts_dev.shape[0])
        t_draft += time.perf_counter() - w0
        c_draft += time.process_time() - p0
        wv, pv = time.perf_counter(), time.process_time()

        # ONE target forward over [cur, d0 .. d_{g-1}]. Position i predicts the
        # token that follows xv[i], so preds[i] is the target's verdict on
        # drafts[i]. This is where the speedup comes from: the weights are read
        # once for g+1 positions.
        #
        # P0.1: built on device. This was `mx.array([[int(cur.item())] + drafts])`,
        # which read gamma+1 values back from the GPU only to send the same values
        # straight back. `cur` is the worst of them: it is `mx.array(bonus)` built
        # from a Python int this loop already has, so `int(cur.item())` was a
        # round-trip to recover a number that never left the host.
        # astype(int32) reproduces the dtype the Python-list constructor inferred,
        # so the embedding gather sees byte-identical indices.
        xv = mx.concatenate([cur.reshape(1), drafts_dev]).astype(mx.int32).reshape(1, g + 1)
        hv = tm(xv, cache=cache)
        preds = mx.argmax(tm.embed_tokens.as_linear(hv), axis=-1)[0]

        # ONE host read per cycle, down from gamma+2. The accept compare genuinely
        # needs host values for both preds and the drafts, so they are concatenated
        # and fetched in a single device->host sync rather than two.
        combo = mx.concatenate([preds.astype(mx.int32), drafts_dev.astype(mx.int32)])
        wb = time.perf_counter()      # graph is built; nothing has run yet
        mx.eval(combo)                # the barrier: this is where the GPU time lands
        t_build += wb - wv
        t_wait += time.perf_counter() - wb
        cl = combo.tolist()
        pl, drafts = cl[: g + 1], cl[g + 1:]

        t_verify += time.perf_counter() - wv
        c_verify += time.process_time() - pv

        n = 0
        for i in range(g):
            if pl[i] != drafts[i]:
                break
            n += 1
        bonus = pl[n]

        before = len(out)
        out.extend(drafts[:n])
        out.append(bonus)
        accepted_hist[n] += 1
        drafted_hist[g] += 1
        n_cycles += 1

        if ref is not None and stats_out is not None and "bad_cycle" not in stats_out:
            got = out[before:]
            want = ref[before:before + len(got)]
            if got != want:
                stats_out["bad_cycle"] = {
                    "cycle": n_cycles, "out_index": before,
                    "cur": int(xv[0, 0].item()), "drafts": drafts,
                    "target_preds": pl, "n_accepted": n, "bonus": bonus,
                    "emitted": got, "expected": want,
                    "ref_context": ref[max(0, before - 3):before + len(got) + 2],
                }

        # The cache holds g+1 tokens; only n+1 were accepted.
        wt = time.perf_counter()
        rewind(cache, g - n)

        hidden = hv[:, n : n + 1, :]
        cur = mx.array(bonus)
        pos += n + 1
        # Captured during the verify forward, so it covers the rejected tail
        # too — trim it to match the cache.
        shared_kv = slice_shared_kv(collect_shared_kv(target), g - n)
        t_tail += time.perf_counter() - wt

    if stats_out is not None:
        c = max(1, n_cycles)
        stats_out["cycles"] = n_cycles
        stats_out["accepted_hist"] = dict(sorted(accepted_hist.items()))
        stats_out["drafted_hist"] = dict(sorted(drafted_hist.items()))
        stats_out["mean_drafted"] = sum(k * v for k, v in drafted_hist.items()) / c
        stats_out["conf_threshold"] = conf_threshold
        tot = sum(k * v for k, v in accepted_hist.items())
        stats_out["mean_accepted"] = tot / c
        stats_out.update({
            "draft_ms_per_cycle": t_draft / c * 1e3,
            "draft_cpu_ms_per_cycle": c_draft / c * 1e3,
            "verify_ms_per_cycle": t_verify / c * 1e3,
            "verify_cpu_ms_per_cycle": c_verify / c * 1e3,
            "verify_build_ms_per_cycle": t_build / c * 1e3,
            "verify_wait_ms_per_cycle": t_wait / c * 1e3,
            "tail_ms_per_cycle": t_tail / c * 1e3,
            "draft_cpu_frac": c_draft / t_draft if t_draft else None,
            "verify_cpu_frac": c_verify / t_verify if t_verify else None,
        })
    return out[:max_tokens]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--verify-lossless", action="store_true")
    ap.add_argument("--out", default="runs.jsonl")
    ap.add_argument("--tag", default="mtp-spec")
    # MEASURED, not guessed. shape_stability.py: greedy decoding on this model
    # is not shape-invariant -- 2 of 320 positions flip their argmax purely from
    # the shape of the matmul, at margins 6.37e-03 (1.0 bf16 ULP) and 1.55e-02
    # (2.0 ULP). fork_validity.py part C: at those positions the margin ITSELF
    # ranges 0.000e+00 .. 1.550e-02 depending on (rows, row index), so a
    # threshold on the 1-row margin needs headroom over that spread. 1.55e-02
    # x 1.5 = 2.3e-02. Replaces a guessed 1e-4, which was ~150x too tight and
    # classified both real ties as logic bugs.
    ap.add_argument("--tie-threshold", type=float, default=2.3e-2,
                    help="legacy relative threshold, reported but no longer decisive")
    # THE DECISIVE THRESHOLD, in bf16 ULPs rather than relative margin.
    # control_variables measured the two divergent positions at exactly 1.00 and
    # 2.00 ULP, and showed the ULP count survives final_logit_softcapping while
    # the relative margin does not (6.369e-03 -> 4.831e-03 at the same position,
    # 1.00 ULP -> 1.00 ULP). 3.0 keeps the same 1.5x headroom over the largest
    # observed flip (2.00) that 2.3e-2 was chosen to give, without the 2x raggedness
    # a fixed relative threshold inherits from where the top logit sits in its binade.
    ap.add_argument("--tie-ulps", type=float, default=3.0)
    # PHASE 1. 0.0 keeps the fixed-gamma behaviour exactly, so every earlier number
    # in runs.jsonl remains reproducible from this file. Above 0 it turns --gamma
    # into a CEILING and lets the drafter's own confidence pick the block length.
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="stop drafting when top-1 prob < this (0 = fixed gamma). "
                         "confidence_calibration measured break-even at 0.422 and "
                         "the best simulated threshold at 0.90")
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

    results = []
    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        print(f"\n--- prompt {pi + 1} ({len(ids)} tokens) ---")

        # With --verify-lossless the baseline is generated anyway, so produce it
        # FIRST and hand it to the speculative run for lockstep checking.
        ref = margins = None
        t_base = None
        if args.verify_lossless:
            U.clear_cache()
            t0 = time.perf_counter()
            ref, margins, ulps = greedy_baseline(target, tm, ids, args.tokens, want_margins=True)
            t_base = time.perf_counter() - t0

        st = {}
        U.clear_cache()
        mx.eval(mx.zeros(1))
        t0 = time.perf_counter()
        spec = spec_decode(target, tm, drafter, ids, args.tokens, args.gamma, st, ref=ref,
                           conf_threshold=args.conf_threshold)
        t_spec = time.perf_counter() - t0
        if pi == 0:
            print(f"  replaced {st['rotating_caches_widened']} RotatingKVCache with "
                  f"plain KVCache (mask still enforces the window)")
            print(f"  untrimmable caches after that: {st['untrimmable'] or 'none'}")
        print(f"  speculative : {args.tokens / t_spec:6.2f} tok/s  "
              f"({t_spec:.2f}s, {st['cycles']} cycles, "
              f"mean accepted {st['mean_accepted']:.2f}"
              + (f", mean drafted {st['mean_drafted']:.2f} of <={args.gamma}"
                 if args.conf_threshold > 0 else "") + ")")
        print(f"    per cycle : draft {st['draft_ms_per_cycle']:6.1f} ms "
              f"(cpu {st['draft_cpu_ms_per_cycle']:5.1f} = {100 * st['draft_cpu_frac']:3.0f}%)"
              f"   verify {st['verify_ms_per_cycle']:6.1f} ms "
              f"(cpu {st['verify_cpu_ms_per_cycle']:5.1f} = {100 * st['verify_cpu_frac']:3.0f}%)")
        print(f"    verify    : build {st['verify_build_ms_per_cycle']:6.1f} ms "
              f"(python graph)   wait {st['verify_wait_ms_per_cycle']:6.1f} ms (gpu barrier)"
              f"   tail {st['tail_ms_per_cycle']:5.1f} ms (rewind+kv walk)")
        if st["draft_cpu_frac"] and st["draft_cpu_frac"] > 0.7:
            print("      -> drafting is CPU-BOUND: the GPU waits while Python builds")
            print(f"         {args.gamma} x 4 layers of graph. mx.compile is the lever.")

        row = {
            "tag": args.tag, "prompt_index": pi, "prompt_tokens": len(ids),
            "gen_tokens": args.tokens, "gamma": args.gamma,
            "spec_tok_s": args.tokens / t_spec,
            "cycles": st["cycles"], "mean_accepted": st["mean_accepted"],
            "accepted_hist": st["accepted_hist"],
            "drafted_hist": st.get("drafted_hist"),
            "mean_drafted": st.get("mean_drafted"),
            "conf_threshold": args.conf_threshold,
            "spec_sha256": hashlib.sha256(",".join(map(str, spec)).encode()).hexdigest(),
        }

        if args.verify_lossless:
            base = ref
            same = base == spec
            if "bad_cycle" in st:
                bc = st["bad_cycle"]
                print(f"\n  >>> FIRST BAD CYCLE {bc['cycle']} at out index {bc['out_index']}")
                print(f"      cur (fed as xv[0])   {bc['cur']}")
                print(f"      drafts               {bc['drafts']}")
                print(f"      target preds         {bc['target_preds']}")
                print(f"      n_accepted           {bc['n_accepted']}   bonus {bc['bonus']}")
                print(f"      emitted              {bc['emitted']}")
                print(f"      expected             {bc['expected']}")
                print(f"      baseline context     {bc['ref_context']}")
                em, ex = bc["emitted"], bc["expected"]
                if len(em) == len(ex) and em and ex and em[0] == ex[0]:
                    print("      -> emitted[0] matches; divergence is later in the block")
                elif ex and em and ex[0] in em:
                    print("      -> expected[0] appears later in emitted: a token was SKIPPED")
                print()
            first_diff = next((i for i, (a, b) in enumerate(zip(base, spec)) if a != b), None)
            print(f"  baseline    : {args.tokens / t_base:6.2f} tok/s  ({t_base:.2f}s)")
            print(f"  SPEEDUP     : {t_base / t_spec:6.2f}x")
            print(f"  LOSSLESS    : {'YES — token-for-token identical' if same else 'NO'}")
            row_margin = None
            if not same:
                print(f"     first divergence at token {first_diff} of {args.tokens}")
                print(f"     baseline {base[max(0, first_diff - 3):first_diff + 3]}")
                print(f"     spec     {spec[max(0, first_diff - 3):first_diff + 3]}")
                # A drop shows up as: the baseline token at the divergence
                # reappears nowhere, and the baseline SHIFTED BY ONE matches the
                # spec from here on.
                #
                # The old test asked that for a FIXED 12-token window and
                # reported a bool. That throws away the only interesting number.
                # A tie between "emit X now" and "skip X" produces a fork whose
                # edit shape IS a deletion and whose sequences realign for a few
                # tokens before drifting apart again — which reads as
                # "SUBSTITUTION" under a 12-token all-or-nothing test and as
                # "DROP" under a 2-token one. Measure the run length instead and
                # let it speak: 0 is a clean fork, a short run is a
                # skip-vs-emit tie, a run to the end of the sequence is a real
                # dropped token.
                shift_run = 0
                while (first_diff + shift_run < len(spec)
                       and first_diff + 1 + shift_run < len(base)
                       and spec[first_diff + shift_run] == base[first_diff + 1 + shift_run]):
                    shift_run += 1
                tail = len(spec) - first_diff
                dropped = shift_run >= 12 and shift_run >= tail - 2
                row_margin = margins[first_diff] if first_diff < len(margins) else None
                row_ulps = ulps[first_diff] if first_diff < len(ulps) else None
                med = stats.median(margins)
                if dropped:
                    shape = "looks like a real DROP — confirm with fork_validity.py"
                elif shift_run:
                    shape = (f"SKIP-VS-EMIT TIE: spec matches base shifted +1 for "
                             f"{shift_run} token(s), then drifts")
                else:
                    shape = "clean FORK — no realignment"
                print(f"     shape        : {shape}")
                if shift_run:
                    print(f"                    base[{first_diff}]={base[first_diff]} does not appear in spec at "
                          f"this position; spec[{first_diff}]={spec[first_diff]} == base[{first_diff + 1}].")
                    print("                    The top-2 at this position are the token and its")
                    print("                    successor, so the tie-break reads as a deletion.")
                    print("                    fork_validity.py part B is what settles it: is the")
                    print("                    emitted tail a valid greedy continuation of ITS OWN")
                    print("                    prefix? A dropped token cannot survive that test.")
                if row_margin is not None:
                    print(f"     top-2 margin : {row_margin:.2e}  (median over the run {med:.2e})")
                    print(f"     in bf16 ULPs : {row_ulps:.2f}   <- this is the decisive number")
                    if row_ulps < args.tie_ulps:
                        print("     -> NEAR-TIE. Verification scores gamma+1 positions in one")
                        print("        matmul; the baseline scores one at a time. Different")
                        print("        accumulation order flips a coin-flip argmax. This is")
                        print("        numerical, not a correctness bug — the sampled")
                        print("        distribution is still exact.")
                    else:
                        print("     -> ABOVE the measured instability band. Run")
                        print("        fork_validity.py before calling it a logic bug:")
                        print("        the realignment heuristic gives false positives on")
                        print("        templated text, and the only sound test is whether")
                        print("        the emitted tail is a valid greedy continuation.")
            row.update({
                "baseline_tok_s": args.tokens / t_base,
                "speedup": t_base / t_spec,
                "lossless": same,
                "first_divergence": first_diff,
                "shift_run": shift_run,
                "edit_shape": ("drop" if dropped else
                               "skip_vs_emit_tie" if shift_run else "fork"),
                "divergence_top2_margin": row_margin,
                "divergence_top2_ulps": row_ulps,
                "median_top2_margin": stats.median(margins) if margins else None,
                "n_margins": len(margins),
                "n_margins_under_tie": sum(1 for u in ulps if u < args.tie_ulps),
                "n_margins_under_rel_tie": sum(1 for m in margins
                                               if m < args.tie_threshold),
                "median_top2_ulps": stats.median(ulps) if ulps else None,
                "tie_threshold": args.tie_threshold,
                "tie_ulps": args.tie_ulps,
                "baseline_sha256": hashlib.sha256(
                    ",".join(map(str, base)).encode()).hexdigest(),
            })
        results.append(row)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    sp = [r["spec_tok_s"] for r in results]
    print(f"  speculative      {stats.median(sp):.2f} tok/s (median of {len(sp)})")
    if args.verify_lossless:
        bs = [r["baseline_tok_s"] for r in results]
        exact = all(r["lossless"] for r in results)
        # A divergence at a position where the top-2 logits are tied is not a
        # correctness failure. Speculative decoding is exact by construction —
        # only tokens the target itself produced are accepted. Bitwise-identical
        # greedy output is a STRONGER property that batched verification cannot
        # promise: scoring gamma+1 positions in one matmul accumulates in a
        # different order than scoring one at a time, so an exact tie can break
        # either way. Judge on margin, not on equality.
        TIE = args.tie_ulps

        def _margin(r):
            # NOT `r.get(k) or 1.0`: a margin of exactly 0.0 is the strongest
            # possible evidence of a tie, and `0.0 or 1.0` yields 1.0 because
            # zero is falsy. That inversion made a perfect tie report as a real
            # divergence.
            m = r.get("divergence_top2_ulps")
            return 1e9 if m is None else float(m)

        ties = [r for r in results if not r["lossless"] and _margin(r) < TIE]
        real = [r for r in results if not r["lossless"] and _margin(r) >= TIE]
        print(f"  baseline         {stats.median(bs):.2f} tok/s")
        print(f"  SPEEDUP          {stats.median(sp) / stats.median(bs):.2f}x")
        if exact:
            print("  LOSSLESS         EXACT — every prompt token-for-token identical")
        elif not real:
            print(f"  LOSSLESS         PASS — {len(ties)} divergence(s), all inside the")
            print(f"                   measured instability band ({TIE:.1f} bf16 ULP).")
            print("                   Distribution is exact; only the tie-break differs.")
        else:
            # THIS BRANCH WAS UNREACHABLE. It used to hang off `if nm:` below,
            # so it fired only when NO margins had been recorded — i.e. never,
            # since --verify-lossless always records them. The consequence was
            # that a run with a real divergence printed no verdict line at all:
            # not EXACT, not PASS, not FAILED, just straight on to the tie-band
            # line. The single outcome the headline depends on detecting was the
            # one outcome that printed nothing.
            print(f"  LOSSLESS         FAILED — {len(real)} divergence(s) with a real margin")
            for r in real:
                print(f"                   prompt {r['prompt_index']} at token "
                      f"{r['first_divergence']}, margin "
                      f"{r['divergence_top2_margin']:.2e}")
            print("                   Do not call these bugs on this line alone —")
            print("                   run fork_validity.py. The sound test is whether")
            print("                   the emitted tail is a valid greedy continuation")
            print("                   of its own prefix, not whether the margin is small.")

        # How much the threshold actually waives. A tie threshold turns the
        # losslessness CHECK into a losslessness WAIVER, so the fraction of
        # positions it covers is part of the claim, not a footnote. This is the
        # R in "lossless except where the margin is under N, at rate R", so it
        # prints on every verified run, pass or fail.
        nm = sum(r.get("n_margins") or 0 for r in results)
        nu = sum(r.get("n_margins_under_tie") or 0 for r in results)
        if nm:
            print(f"  tie band         {nu} of {nm} baseline positions "
                  f"({100 * nu / nm:.2f}%) sit below {TIE:.1f} ULP")
            print("                   — that is the fraction of tokens the criterion")
            print("                     does not pin down. Keep it small or the claim")
            print("                     is weak.")
    ma = [r["mean_accepted"] for r in results]
    print(f"  mean accepted    {stats.fmean(ma):.2f} of gamma={args.gamma}")

    with open(args.out, "a") as f:
        for r in results:
            r["record_type"] = "spec_decode"
            r["schema_version"] = 1
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  appended {len(results)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

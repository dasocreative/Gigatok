#!/usr/bin/env python3
"""
measure_acceptance.py — Recipe 4, step 3: how many drafted tokens does the
target actually accept?

Everything about MTP's payoff reduces to this one number, and nothing predicts
it. Drafting costs ~0.29 ms against a 40.9 ms target step, so the speedup is
essentially the mean accepted length. At acceptance 2.0 you get ~1.9x; at 3.0,
~2.9x. This measures it directly, before any decode loop exists.

WHAT IT DOES

For each position it runs the target once (which it would do anyway), captures
the two things the drafter needs, drafts gamma tokens, then continues the
target greedily and checks how many of the drafts the target would have
produced itself. That is exactly the acceptance test a lossless speculative
loop performs — computed here offline so it can be measured without building
the loop first.

THE TWO THINGS THE DRAFTER NEEDS

  last_hidden_state   the target's post-norm hidden state. Already returned by
                      Gemma4TextModel.__call__ — no patching needed.
  shared_kv_states    the target's K/V for the LAST layer of each layer_type.
                      The drafter has no K/V projections of its own.

The second is a local variable inside the target's forward, so we capture it by
wrapping DecoderLayer.__call__ to stash what it already returns. Iterating
layers in order means the last layer of each type naturally wins, which is
exactly the transformers contract. No extra memory: these are the same arrays
the cache already holds.

    ../bin/python measure_acceptance.py --gamma 4 --tokens 128
"""

from __future__ import annotations

import argparse
import statistics as stats
from collections import Counter

import mlx.core as mx

import mlxutil as U
import roofline as RF

PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that reverses a linked list.",
    "Summarise the causes of the French Revolution.",
    "What is the difference between a mutex and a semaphore?",
]


# ---------------------------------------------------------------- KV capture
def enable_kv_capture() -> bool:
    """Wrap DecoderLayer.__call__ so each layer stashes the K/V it returns."""
    from mlx_lm.models import gemma4_text

    DL = gemma4_text.DecoderLayer
    if getattr(DL, "_mlxbench_kv_capture", False):
        return True
    orig = DL.__call__

    def wrapped(self, x, mask=None, cache=None, per_layer_input=None,
                shared_kv=None, offset=None):
        h, kvs, off = orig(self, x, mask, cache, per_layer_input, shared_kv, offset)
        # Store the (keys, values) tuple UNWRAPPED. The layer consumes
        # `shared_kv` as exactly that pair; nesting it with the offset makes
        # scaled_dot_product_attention receive a tuple where it wants an array.
        self._captured_kv = kvs
        self._captured_offset = off
        return h, kvs, off

    DL.__call__ = wrapped
    DL._mlxbench_kv_capture = True
    return True


def text_model(target):
    inner = getattr(target, "language_model", target)
    return getattr(inner, "model", inner)


def collect_shared_kv(target) -> dict:
    """Last layer of each layer_type wins, matching the transformers contract."""
    tm = text_model(target)
    out = {}
    for layer in tm.layers:
        kv = getattr(layer, "_captured_kv", None)
        if kv is not None and kv[0] is not None:
            out[getattr(layer, "layer_type", "full_attention")] = kv
    return out


def describe_shared_kv(shared_kv: dict) -> None:
    for k, kv in sorted(shared_kv.items()):
        keys, values = kv
        print(f"    {k:<20}keys {tuple(keys.shape)}  values {tuple(values.shape)}")


# ---------------------------------------------------------------- drafting
def draft(drafter, tm, hidden, last_token, shared_kv, gamma, pos):
    """
    Run the drafter gamma steps. Returns the drafted token ids as ONE mx.array
    of shape (gamma,), still on device.

    WHY A DEVICE ARRAY AND NOT A LIST OF PYTHON ints (P0.1)

    This used to end `return [int(t.item()) for t in toks]` — gamma separate
    device->host reads — and its caller then did `mx.array([[int(cur.item())] +
    drafts])`, one more read plus a host->device rebuild of a tensor whose
    values had never needed to leave the device. That is gamma+1 synchronising
    round-trips per cycle to construct something the GPU already held.

    Returning the stacked array lets spec_decode build the verify input with
    mx.concatenate and read the host ONCE per cycle, for the accept compare
    that genuinely needs host values.

    The single barrier is unchanged: mx.eval on the stacked array is the same
    one barrier `mx.eval(toks)` was, so the draft/verify timing split stays
    meaningful. Values and their order are untouched, so this cannot move
    numerics — verified anyway.

    Callers that want host ints call .tolist() on the result.

    `pos` is the number of tokens the target has consumed. The drafter's first
    query sits at that absolute position: it conditions on the token the target
    just produced (which lives at index `pos`, not yet in the cache) plus the
    hidden state at index pos-1. Each further step advances one position.

    The recurrence carries new information through post_projection, not through
    K/V: the target's cache has no entry for a token that has only been drafted,
    so every step attends over the same target K/V and mask=None stays correct.

    ONE eval for the WHOLE draft block, not one per step. The token id stays an
    mx.array and indexes the next embedding lookup on-device, so all gamma
    steps build a single lazy graph.

    This is the difference between a 3x speedup and a 1.4x one. Calling
    mx.eval() per draft step imposes gamma hard CPU<->GPU barriers per cycle —
    the exact serialisation Recipe 0 measured as costing 18% at ONE barrier per
    token. At gamma=8 that penalty applies eight times per cycle, against a
    draft step whose actual memory traffic is only 0.29 ms.
    """
    scale = getattr(tm, "embed_scale", 1.0)
    toks = []
    tok = last_token
    h_back = hidden
    for i in range(gamma):
        emb = tm.embed_tokens(tok.reshape(1, 1)) * scale
        nxt, cand, logits = drafter(emb, h_back, shared_kv, offset=pos + i)
        j = mx.argmax(logits.reshape(-1))
        tid = cand.reshape(-1)[j]          # stays on device — no .item() here
        toks.append(tid)
        tok = tid
        h_back = nxt
    dtok = mx.stack(toks)                  # (gamma,) — one array, still on device
    # P0.3: NO BARRIER HERE. Returned lazily.
    #
    # The drafted ids are only ever consumed as GATHER INDICES -- spec_decode
    # concatenates them into xv and the target embeds them on device. Nothing
    # host-side needs their values before the verify barrier, so evaluating here
    # bought nothing and cost a full CPU<->GPU round trip per cycle: the GPU went
    # idle while Python then built 42 layers of verify graph.
    #
    # Left lazy, draft and verify fuse into ONE graph with ONE barrier, and the
    # verify graph construction overlaps the drafter's GPU work.
    #
    # Callers needing host values (.tolist(), or measure_acceptance's own loop)
    # force evaluation themselves; that is still exactly one barrier.
    return dtok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=128, help="Target tokens per prompt.")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    args = ap.parse_args()

    from mlx_lm.models import gemma4_assistant
    from mlx_lm.models.cache import make_prompt_cache
    from pathlib import Path

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
    all_acc, per_pos = [], Counter()

    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        cache = make_prompt_cache(target)
        x = mx.array([ids])

        # One prefill gives everything: the text model returns the post-norm
        # hidden state (what the drafter conditions on), advances the cache, and
        # triggers the K/V capture. Logits come from the tied embedding.
        # final_logit_softcapping is skipped deliberately — it is monotonic, so
        # argmax is identical and we avoid a second pass through the full model.
        hidden = tm(x, cache=cache)
        mx.eval(hidden)
        shared_kv = collect_shared_kv(target)
        if not shared_kv:
            print("  FAIL — captured no shared K/V. Is this a gemma4_text target?")
            return 3
        if pi == 0:
            print(f"  shared_kv layer types: {sorted(shared_kv)}")
            describe_shared_kv(shared_kv)

        cur = mx.argmax(tm.embed_tokens.as_linear(hidden[:, -1:]).reshape(-1))
        mx.eval(cur)
        pos = len(ids)          # tokens the target has consumed

        accepted_here = []
        generated = 0
        while generated < args.tokens:
            h_last = hidden[:, -1:, :]
            # This offline tool compares against the target one token at a time on
            # the host, so it wants host ints. It is not the hot loop — spec_decode
            # is — so the one read here costs nothing worth saving.
            drafts = draft(drafter, tm, h_last, cur, shared_kv, args.gamma, pos).tolist()

            # Walk the target forward one token at a time, stopping at the FIRST
            # mismatch. A real speculative loop advances by accepted+1, not by
            # gamma — advancing by gamma makes different gamma settings sample
            # different parts of the sequence, which is what produced the wild
            # per-prompt spread at gamma=12 (1.90 to 4.43) against a tight
            # 2.62-2.84 at gamma=8. Stopping early is also strictly cheaper:
            # no target steps are spent past the first rejection.
            n, t_cur, t_hidden = 0, cur, hidden
            for i in range(args.gamma):
                hh = tm(t_cur.reshape(1, 1), cache=cache)
                lg = tm.embed_tokens.as_linear(hh[:, -1:]).reshape(-1)
                nxt_tok = mx.argmax(lg)
                mx.eval(nxt_tok)
                pos += 1
                t_hidden = hh
                t_cur = nxt_tok
                if int(nxt_tok.item()) != drafts[i]:
                    break          # rejected here; this token is the bonus
                n += 1

            accepted_here.append(n)
            per_pos[n] += 1
            all_acc.append(n)
            generated += n + 1     # accepted drafts plus the free bonus token
            cur, hidden = t_cur, t_hidden
            shared_kv = collect_shared_kv(target)

        m = stats.fmean(accepted_here) if accepted_here else 0
        print(f"  prompt {pi + 1}: mean accepted {m:.2f} / {args.gamma}")

    print("\n" + "=" * 72)
    print("ACCEPTANCE")
    print("=" * 72)
    if not all_acc:
        print("  no data")
        return 1
    mean_acc = stats.fmean(all_acc)
    print(f"  gamma                {args.gamma}")
    print(f"  cycles measured      {len(all_acc)}")
    print(f"  mean accepted        {mean_acc:.2f}  (+1 free token from the verify step)")
    print(f"  effective length     {mean_acc + 1:.2f} tokens per target forward")
    print("\n  distribution:")
    for k in sorted(per_pos):
        bar = "#" * int(40 * per_pos[k] / len(all_acc))
        print(f"    {k} accepted  {per_pos[k]:>4}  {bar}")

    BW, TGT = 86.5e9, 3.535e9
    t_t = TGT / BW * 1000
    t_d = 24.9e6 / BW * 1000
    cyc = args.gamma * t_d + t_t
    proj = 1000 / (cyc / (mean_acc + 1))
    print(f"\n  target step {t_t:.1f} ms, draft step {t_d:.2f} ms, cycle {cyc:.1f} ms")
    print(f"  PROJECTED  {proj:.1f} tok/s  vs 24.5 baseline  = {proj / 24.5:.2f}x")
    print("\n  This is arithmetic on a measured acceptance length, not a")
    print("  benchmark. The real number comes from the decode loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

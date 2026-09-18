#!/usr/bin/env python3
"""
divergence_text.py — what are the divergent tokens AS LANGUAGE, and do the two
sequences realign afterwards?

TWO QUESTIONS, AND WHY THE SECOND ONE IS THE IMPORTANT ONE

The obvious question is what token 121 actually says. Worth knowing: if the
model is torn between " the" and " a", or between "Because" and "Since", the
divergence is a coin-flip between two equally good continuations and the
practical cost is zero. If it is torn between "increases" and "decreases", it
is not.

The sharper question is the one about REORDERING, and it is a real diagnostic
rather than a curiosity. Two very different things produce "the outputs
differ":

  NUMERICAL TIE       At one position the top-2 logits are within a bfloat16
                      ULP. The two runs pick different tokens, and from there
                      they are decoding different sequences — they free-run
                      apart and never realign. Token counts are unrelated.
                      Nothing is broken; the model was indifferent.

  BOOKKEEPING BUG     A token is dropped, duplicated, emitted early, or two are
                      swapped. The tell is that the sequences REALIGN: after a
                      small edit the rest matches exactly. Every accept/emit
                      off-by-one has this signature — and we have already seen
                      it once in this project, as a dropped token from the
                      rotating-cache rollback.

Those are distinguishable without any theory. Run a diff over the two TOKEN ID
lists and look at the shape of the edit script:

    one 'replace' then 'equal' to the end        -> bug (substitution)
    'insert'/'delete' of 1 then 'equal'          -> bug (dropped/duplicated)
    spec[j],spec[j+1] == base[j+1],base[j]       -> bug (transposition)
    one 'replace' and NOTHING equal after it     -> tie, working as designed

A sentence that reads correctly with two words reordered is exactly the
transposition case, and that would be a BUG — a correct-looking output is not
the same as a lossless one. This checks for it explicitly instead of assuming.

    ../bin/python divergence_text.py --gamma 3 --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import difflib
import math
from pathlib import Path

import mlx.core as mx

import roofline as RF
from measure_acceptance import PROMPTS, enable_kv_capture, text_model
from spec_generate import greedy_baseline, spec_decode, use_plain_kv_caches


def show(tok, ids) -> str:
    """Decode with the whitespace visible — ' the' and 'the' are different
    tokens and the difference is invisible in ordinary printing."""
    return repr(tok.decode(list(ids)))


def logits_at(target, tm, ids, prefix, block):
    """
    Logits for the position that follows `prefix`, with the last `block` tokens
    of the prefix fed in ONE forward.

    block=1 reproduces the greedy baseline's path (GEMV in the output matmul);
    block=gamma+1 reproduces the speculative verify's path (GEMM). Everything
    else about the context is identical.
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    use_plain_kv_caches(cache)
    h = tm(mx.array([ids]), cache=cache)
    mx.eval(h)
    i = 0
    while i < len(prefix):
        step = 1 if (len(prefix) - i) > block else (len(prefix) - i)
        h = tm(mx.array([prefix[i: i + step]]), cache=cache)
        i += step
    # Project over the WHOLE block and index the row afterwards. Slicing to
    # h[:, -1:] first made this a 1-row GEMV in every case, so the "N rows
    # (verify / GEMM)" column was never a GEMM -- it differed from the 1-row
    # column only in the attention path. spec_decode does as_linear(hv) over
    # all g+1 rows; this now matches it.
    lg = tm.embed_tokens.as_linear(h)[0, -1]
    mx.eval(lg)
    return lg


def topk_table(tok, lg, k=5):
    order = mx.argsort(-lg)[:k]
    mx.eval(order)
    ids = [int(v) for v in order.tolist()]
    vals = [float(lg[i].item()) for i in ids]
    scale = max(abs(vals[0]), 1e-6)
    ulp = 2.0 ** (math.floor(math.log2(scale)) - 7)
    rows = []
    for i, v in zip(ids, vals):
        rows.append((i, show(tok, [i]), v, (vals[0] - v) / scale, (vals[0] - v) / ulp))
    return rows, ulp


def classify(base, spec):
    """Return (label, detail) describing the edit between two token lists."""
    sm = difflib.SequenceMatcher(None, base, spec, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    equal_after = 0
    first = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal" and first is None:
            first = (tag, i1, i2, j1, j2)
        elif first is not None and tag == "equal":
            equal_after += i2 - i1
    if first is None:
        return "IDENTICAL", "", 0, ops
    tag, i1, i2, j1, j2 = first
    n_base, n_spec = i2 - i1, j2 - j1
    if tag == "replace" and n_base == 1 and n_spec == 1:
        label = "SUBSTITUTION of 1 token"
    elif tag == "delete":
        label = f"DELETION of {n_base} token(s) from the speculative output"
    elif tag == "insert":
        label = f"INSERTION of {n_spec} extra token(s) in the speculative output"
    else:
        label = f"REPLACE {n_base} -> {n_spec} tokens"
    return label, f"at index {i1}", equal_after, ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--context", type=int, default=24, help="tokens of text to show")
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

        base, margins, _ulps = greedy_baseline(target, tm, ids, args.tokens, want_margins=True)
        st = {}
        spec = spec_decode(target, tm, drafter, ids, args.tokens, args.gamma, st, ref=base)

        if base == spec:
            print("  identical — nothing to inspect.")
            verdicts.append("identical")
            continue

        j = next(i for i in range(min(len(base), len(spec))) if base[i] != spec[i])
        print(f"  first difference at generated token {j}")
        print(f"  baseline top-2 margin there: {margins[j]:.3e}")

        # ---------------------------------------------------- as language
        lo = max(0, j - args.context)
        print("\n  --- shared text up to the split ---")
        print("  ..." + tok.decode(base[lo:j]))
        print("\n  baseline chose  ", show(tok, [base[j]]), f"  id {base[j]}")
        print("  speculative chose", show(tok, [spec[j]]), f"  id {spec[j]}")
        print("\n  --- baseline continues ---")
        print("  " + tok.decode(base[j: j + args.context * 2]).replace("\n", "\\n"))
        print("\n  --- speculative continues ---")
        print("  " + tok.decode(spec[j: j + args.context * 2]).replace("\n", "\\n"))

        # ------------------------------------- reordering / edit-shape test
        print("\n  --- EDIT SHAPE (is this a reorder, a drop, or a fork?) ---")
        swapped = (j + 1 < min(len(base), len(spec))
                   and base[j] == spec[j + 1] and base[j + 1] == spec[j])
        label, where, equal_after, ops = classify(base, spec)
        print(f"  first edit           {label} {where}")
        print(f"  tokens matching AFTER the first edit: {equal_after}"
              f" of {len(base) - j - 1} remaining")
        print(f"  distinct edit regions in the whole sequence: {len(ops)}")
        print(f"  adjacent transposition (base[j],base[j+1] swapped): "
              f"{'YES' if swapped else 'no'}")

        if swapped:
            v = "REORDERING BUG"
            print("\n  -> TRANSPOSITION. The two tokens are emitted in the wrong")
            print("     order. Readable output, but not lossless — this is an")
            print("     accept/emit ordering bug, not numerical noise.")
        elif equal_after > 0.5 * (len(base) - j - 1):
            v = "STRUCTURAL BUG"
            print(f"\n  -> THE SEQUENCES REALIGN ({equal_after} tokens match after the")
            print("     edit). A numerical tie cannot do that: once two greedy runs")
            print("     pick different tokens they condition on different text and")
            print("     have no reason to ever agree again. This is a dropped,")
            print("     duplicated or misplaced token in the bookkeeping.")
        else:
            v = "FORK (consistent with a numerical tie)"
            print("\n  -> THEY NEVER REALIGN. One position differs and everything")
            print("     after is a different continuation of a different prefix.")
            print("     That is the signature of a coin-flip at a near-tie, not of")
            print("     a bookkeeping error.")
        verdicts.append(v)

        # ------------------------------- what the two paths actually scored
        print("\n  --- top-5 at that position, scored BOTH ways ---")
        prefix = base[:j]
        for name, block in (("1 row  (baseline / GEMV)", 1),
                            (f"{args.gamma + 1} rows (verify / GEMM)", args.gamma + 1)):
            lg = logits_at(target, tm, ids, prefix, block)
            rows, ulp = topk_table(tok, lg, 5)
            print(f"\n  {name}   [bf16 ULP = {ulp:.4f}]")
            print(f"    {'id':>8} {'token':<14}{'logit':>10}{'gap rel':>11}{'gap ULP':>10}")
            for i, s, v, rel, u in rows:
                mark = ""
                if i == base[j]:
                    mark = "  <- baseline"
                elif i == spec[j]:
                    mark = "  <- speculative"
                print(f"    {i:>8} {s:<14}{v:>10.4f}{rel:>11.2e}{u:>10.2f}{mark}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for i, v in enumerate(verdicts):
        print(f"  prompt {i}: {v}")
    if all(v.startswith("FORK") or v == "identical" for v in verdicts):
        print("\n  No reordering, no realignment, no dropped tokens. Every")
        print("  divergence is a fork at a near-tie. Combined with a positive")
        print("  shape_stability result the loop is correct and the tie threshold")
        print("  should be set from measurement.")
    else:
        print("\n  At least one divergence has a STRUCTURAL signature. That part is")
        print("  a bug in the accept/emit path and numerical noise does not explain it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

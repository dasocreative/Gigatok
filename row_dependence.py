#!/usr/bin/env python3
"""
row_dependence.py — the (M, row) result, reduced to its minimal form.

WHAT fork_validity PART C ACTUALLY SHOWED

Read its two grids down the COLUMNS rather than across the rows:

    prompt 0, position 121          prompt 1, position 82
      M    r0    r1    r2    r3       M    r0    r1    r2    r3
      1  base     -     -     -       1  base     -     -     -
      2  SPEC  SPEC     -     -       2  base  SPEC     -     -
      3  SPEC  SPEC  SPEC     -       3  base  SPEC  base     -
      4  SPEC  SPEC  SPEC  base       4  base  SPEC  base  base
      ...                             ...

Every column is CONSTANT below M=1. r1 is SPEC at M=2,3,4,5,6,7,8 without
exception; r3 is base at M=4..8 without exception. The same holds for all eight
columns in both grids, and the SPEC-cell counts confirm it exactly: 24 = 7+7+6+
0+0+3+0+1 and 11 = 7+4.

So the conclusion printed under those grids — "the argmax is a function of
(M, row), not of M alone" — is true but far weaker than the data. The argmax is
a function of the ROW INDEX ALONE, independent of the block size, with M=1
special only because a single row dispatches a GEMV kernel instead of a GEMM.

That also RETRACTS the earlier reading carried in fork_validity's own docstring:
"powers of two flip and 3 does not, which is a kernel tiling signature." That
pattern was an artefact of shape_stability testing one alignment per block size —
with block b, the deciding token lands at row (j-1) mod b, so varying b varied r.
It looked like a function of b because r was moving underneath. It is not.

WHY THE STRONGER VERSION MATTERS

1. It is mechanistically plausible in a way the weaker one is not. If the GEMM
   pads M up to the 8-row simdgroup_matrix tile, then row r occupies the same
   position in the same tile whatever M is, accumulates in the same order, and
   rounds the same way. Constant columns are exactly what that predicts.
2. It says the flip is NOT a property of speculative decoding. Nothing about
   drafting, accepting or trimming enters it — only where a token happens to sit
   in whatever block scores it. In spec_decode that row index is a function of
   the entire acceptance history, which is why the same position flips in one run
   and not another and why margin alone never predicted it.
3. If the row index behaves the same way along the BATCH axis, the finding
   generalises past this recipe to any batched forward — batched serving, and
   mlx-vlm's own batch-4/8 dispatch. That is a testable claim and part B tests it.

THE EXPERIMENT

fork_validity varied the real block contents, so attention and the projection
moved together. Here the hidden state is computed ONCE per position and then
replicated: every row of every block is bit-identical input. Any difference
across rows is the output-projection kernel and nothing else — no attention, no
cache, no context.

  A. sequence axis   as_linear on [1, M, D] built from M copies of h, read at row r
  B. batch axis      as_linear on [B, 1, D] built from B copies of h, read at row b
  C. all at-risk positions, not only the two that flipped: every position whose
     baseline top-2 margin is under --threshold gets the same sweep, which turns
     "2 of 320 flipped" into a structural rate over the whole tie band.

Nothing is timed. The mx.eval barriers exist only to force the graph before a
host-side .item() read; there is no timed region to protect.

    ../bin/python row_dependence.py --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import json
import statistics as stats

import mlx.core as mx

import roofline as RF
from measure_acceptance import PROMPTS, text_model
from spec_generate import greedy_baseline
from fork_validity import top2


def sweep(tm, h, max_m):
    """Score one hidden state at every (axis, size, index) cell.

    h is [1, 1, D]. Replicating it means every row is the same input vector, so
    the only thing that varies is where that vector sits in the matmul.
    """
    cells = {}
    for M in range(1, max_m + 1):
        x = mx.concatenate([h] * M, axis=1)          # [1, M, D]
        lg = tm.embed_tokens.as_linear(x)
        mx.eval(lg)
        for r in range(M):
            cells[("seq", M, r)] = top2(lg[0, r])
    for B in range(1, max_m + 1):
        x = mx.concatenate([h] * B, axis=0)          # [B, 1, D]
        lg = tm.embed_tokens.as_linear(x)
        mx.eval(lg)
        for b in range(B):
            cells[("batch", B, b)] = top2(lg[b, 0])
    return cells


def column_constant(cells, axis, max_m):
    """True if argmax at each index is the same for every size > index (M>=2)."""
    for r in range(max_m):
        ids = {cells[(axis, M, r)][0]
               for M in range(max(r + 1, 2), max_m + 1) if (axis, M, r) in cells}
        if len(ids) > 1:
            return False, r
    return True, None


def grid_str(cells, axis, max_m, base_id, tok):
    lines = ["    " + "".join(f"{('i' + str(r)):>8}" for r in range(max_m))]
    for M in range(1, max_m + 1):
        row = []
        for r in range(max_m):
            if (axis, M, r) not in cells:
                row.append(f"{'-':>8}")
            else:
                bid = cells[(axis, M, r)][0]
                row.append(f"{('base' if bid == base_id else 'ALT'):>8}")
        lines.append(f" {M:>2} " + "".join(row))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--max-m", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=2.3e-2)
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    from mlx_lm.models.cache import make_prompt_cache

    print("loading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)

    records = []
    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        print("\n" + "=" * 78)
        print(f"PROMPT {pi}: {prompt}")
        print("=" * 78)

        base, margins, _ulps = greedy_baseline(target, tm, ids, args.tokens, want_margins=True)
        targets = [j for j, m in enumerate(margins) if m < args.threshold]
        print(f"  {len(targets)} of {len(base)} positions under {args.threshold:.2e}: {targets}")

        # One forward pass. After feeding base[k-1] the hidden state predicts
        # base[k], so sweep at exactly the step where each target is decided.
        cache = make_prompt_cache(target)
        h = tm(mx.array([ids]), cache=cache)
        mx.eval(h)
        results = {}
        for k in range(len(base)):
            if k in targets:
                results[k] = sweep(tm, h[:, -1:, :], args.max_m)
            h = tm(mx.array([[int(base[k])]]), cache=cache)
            mx.eval(h)

        for j in sorted(results):
            cells = results[j]
            bid1 = cells[("seq", 1, 0)][0]          # the GEMV / baseline answer
            alts = {c[0] for c in cells.values()} - {bid1}
            mgs = [c[1] for c in cells.values()]
            print(f"\n  position {j}  margin {margins[j]:.3e}  "
                  f"base {bid1} {tok.decode([bid1])!r}")
            if not alts:
                print("    argmax identical in all 72 cells — margin small but decisive")
            else:
                for a in sorted(alts):
                    print(f"    alternative {a} {tok.decode([a])!r}")
                ok_s, bad_s = column_constant(cells, "seq", args.max_m)
                ok_b, bad_b = column_constant(cells, "batch", args.max_m)
                print("    sequence axis  [1, M, D], read at row r:")
                print(grid_str(cells, "seq", args.max_m, bid1, tok))
                print("    batch axis     [B, 1, D], read at row b:")
                print(grid_str(cells, "batch", args.max_m, bid1, tok))
                print(f"    columns constant (seq)   {'YES' if ok_s else f'NO at r{bad_s}'}")
                print(f"    columns constant (batch) {'YES' if ok_b else f'NO at r{bad_b}'}")
                same_axes = all(
                    cells[("seq", M, r)][0] == cells[("batch", M, r)][0]
                    for M in range(1, args.max_m + 1) for r in range(M))
                print(f"    batch axis == sequence axis  {'YES' if same_axes else 'NO'}")
                print(f"    margin range across cells    {min(mgs):.3e} .. {max(mgs):.3e}")
                if min(mgs) == 0.0:
                    print("    -> some cells have an EXACT bf16 tie: the two candidates")
                    print("       are the same representable number and argmax falls to")
                    print("       the lower id. That is the strongest possible evidence")
                    print("       that this is rounding, not a decision.")

            records.append({
                "record_type": "row_dependence", "schema_version": 1,
                "prompt_index": pi, "position": j, "margin": margins[j],
                "base_id": bid1, "n_alternatives": len(alts),
                "row_dependent": bool(alts),
                "seq_columns_constant": column_constant(cells, "seq", args.max_m)[0],
                "batch_columns_constant": column_constant(cells, "batch", args.max_m)[0],
                "batch_matches_seq": all(
                    cells[("seq", M, r)][0] == cells[("batch", M, r)][0]
                    for M in range(1, args.max_m + 1) for r in range(M)),
                "margin_min_cell": min(c[1] for c in cells.values()),
                "margin_max_cell": max(c[1] for c in cells.values()),
                "threshold": args.threshold,
            })

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n = len(records)
    dep = [r for r in records if r["row_dependent"]]
    print(f"  at-risk positions swept          {n}")
    print(f"  row-DEPENDENT (argmax moves)     {len(dep)}"
          f"  ({100 * len(dep) / max(n, 1):.1f}% of the tie band)")
    print(f"  columns constant, sequence axis  "
          f"{sum(r['seq_columns_constant'] for r in records)}/{n}")
    print(f"  columns constant, batch axis     "
          f"{sum(r['batch_columns_constant'] for r in records)}/{n}")
    print(f"  batch axis behaves as sequence   "
          f"{sum(r['batch_matches_seq'] for r in records)}/{n}")
    if dep:
        print(f"  median margin, row-dependent     "
              f"{stats.median([r['margin'] for r in dep]):.3e}")
    print("\n  If columns are constant everywhere, state the result as: the argmax")
    print("  at a tied position is a deterministic function of the ROW INDEX, not")
    print("  of the block size. If the batch axis matches, it is not a property of")
    print("  speculative decoding at all — it is a property of every batched")
    print("  forward, including batched serving and mlx-vlm's batch-4/8 dispatch.")

    with open(args.out, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  appended {len(records)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# DRAFT — upstream issue for ml-explore/mlx

**Status: DRAFT ONLY. Not posted.** Posting requires explicit human approval.

**Revision 3, 2026-09-08.** Supersedes revision 2 (kept at `mlx-upstream-issue-draft.rev2.bak`).
Six corrections applied — see "Corrections" at the end. **No root cause is claimed anywhere.**

## BLOCKER before posting — resolve this first

Revision 2 asserted that issue #3553 was closed 2026-08-05 as completed, with a closing comment
citing `qmv_wide`. **That assertion is currently unverified.** Three independent HTML fetches of
the issue page report it **Open** with no comments visible, and the GitHub REST API was
unreachable from both checking environments (403 / no egress). One of the two readings is wrong.

Run this and paste the result before doing anything else:

```
curl -s https://api.github.com/repos/ml-explore/mlx/issues/3553 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ('state','state_reason','closed_at','comments')})"
```

- **If CLOSED** → post this as a **new** issue, as drafted below.
- **If OPEN** → post the body below as a **comment on #3553** instead, and delete the sentence
  in "Summary" that says the qmv_fast path was superseded (keep the rest: our step is still at a
  different M, and no mechanism link is claimed either way).

Also re-verify the binary evidence locally, since revision 2's framing rests on it:

```
strings ~/mlx-env/lib/python3.12/site-packages/mlx/lib/mlx.metallib | grep -c qmv_wide
```

**Nothing below depends on the answer.** The measurements stand either way; only the routing and
one sentence change.

---

## Proposed title

`[Performance] Quantized forward cost is a staircase in M, not a ramp: +16.2 ms at M=6→7 then +1.0 ms at M=7→8 (Gemma 4 E4B 4-bit, M3)`

## Proposed labels

`performance`

---

## Body

### Summary

On a fanless M3 MacBook Air, the cost of a batched quantized forward does not grow smoothly with
the number of rows. It is a **staircase**: some added rows are nearly free while others are very
expensive, and the two kinds sit next to each other. The largest single jump is **+16.21 ms at
M=6→7**, immediately followed by **+1.01 ms at M=7→8**. There is also an earlier step at
**M=4** (+4.91 ms, against +0.91 and +1.88 ms for the two rows before it).

Single-token decode runs at the memory roofline. Multi-row forwards — the shape a
speculative-decoding verify pass produces — do not.

Reported as a **characterisation, not a diagnosis.** I have not identified the responsible kernel
or dispatch path, and I make no claim that this is related to #3553: that issue's step was at
**M=3** on the `qmv_fast` path, and this build already contains the `qmv_wide` kernels that
superseded it. **The steps here are at M=4 and M=6→7, not M=3.**

### Environment

| | |
|---|---|
| chip | Apple M3, MacBook Air, 4P+4E CPU, 10 GPU cores, 16 GB unified, **fanless** |
| macOS | 26.6.1 |
| mlx | **0.32.2** |
| mlx-lm | 0.31.3 |
| Python | 3.12.14 |

### Model

`mlx-community/gemma-4-e4b-it-OptiQ-4bit` — affine 4-bit, `group_size` 64, per-layer embeddings
kept at 8-bit. `hidden_size` 2560, `intermediate_size` 10240, 42 layers, 8 heads, `head_dim` 256,
`vocab_size` 262144, `sliding_window` 512.

### The measurement

In-model forward, real weights, two context lengths. Every timed region ends in an `mx.eval` on
the value it produces; cache build and trim sit outside the region. **Forward only — the LM head
is excluded from both the times and the floor.**

| k (rows) | ctx 128 (ms) | marginal | ctx 2048 (ms) | vs forward floor (31.71 ms) |
|---:|---:|---:|---:|---:|
| 1 | 36.78 | — | 38.74 | 1.16× |
| 2 | 37.70 | +0.91 | 42.97 | 1.19× |
| 3 | 39.58 | +1.88 | 45.00 | 1.25× |
| 4 | 44.49 | **+4.91** | 50.69 | 1.40× |
| 5 | 50.14 | +5.65 | 57.56 | 1.58× |
| 6 | 59.09 | +8.95 | 68.48 | 1.86× |
| 7 | 75.31 | **+16.21** | 83.71 | 2.38× |
| 8 | 76.32 | **+1.01** | 85.17 | 2.41× |

Extending further (single run, earlier session): k=9 91.9, k=10 92.4 (**+0.55**), k=11–16
≈ 111–116 ms — flattening again.

**The staircase is the striking feature.** Cheap marginal steps land at k=8, k=10, and from k=12
onward; the expensive ones cluster at k=4 and k=6→7. I am deliberately not offering an
explanation for the pattern — I have no profiling data and would only be guessing.

**Reproducibility:** run in two independent sessions, days apart, from a cool machine. Agreement
is **within 2.6 ms at every k, and within 1 ms at six of the eight points.**

### Byte accounting and floors

Active bytes streamed per decode token: **3.535 GB**, of which the dense (matmul'd) part is
**2.822 GB**. The remaining **46 % of the checkpoint is gather-only per-layer-embedding tables**
— row-gathered, never matmul'd, never streamed. Counting full checkpoint size as the roofline
denominator overstates it badly for this architecture.

Against a **measured 89 GB/s** (not the ~100 GB/s the spec sheet implies):

- forward-only floor = 2.822 GB / 89 GB/s = **31.71 ms** ← the table above uses this
- LM head = 0.713 GB / 89 GB/s = 8.01 ms
- full verify floor = 39.72 ms

### Why I trust the ceiling and the byte split

**Two independent implementations, byte-identical weights, agreeing to 0.1 %.** In one session,
on the same machine, at batch-1 plain decode:

| implementation | tok/s | effective GB/s | % of 89 GB/s |
|---|---|---|---|
| mlx-lm 0.31.3 | 23.56 / 24.17 | 83.3 / 85.5 | **93.6 / 96.0** |
| mlx-vlm 0.6.17 | 23.54 | 83.2 | **93.5** |

Active bytes/token were derived **independently inside each virtual environment** from the actual
parameter tree, not shared between them, and came out identical (3,535,067,220). Two separately
written decode loops landing within 0.1 % of each other is the reason I believe both the
89 GB/s ceiling and the gather-only split, rather than either being an artefact of my harness.

### How much headroom is real — bounded, not a single number

The forward at k=8 is **2.41×** the weight-read floor. That ratio assumes **perfect overlap** of
weight streaming and compute, so it is an *upper* bound on the opportunity. With **no overlap**
the floor rises to ~75.8 ms and the same measurement is only **1.01×** it.

**The true headroom lies between those bounds and I cannot narrow it with this data.** I state
the range because quoting only the optimistic ratio would overstate the case.

### A microbenchmark warning for anyone investigating

> **A quantized-matmul bandwidth probe below roughly 32 MB of working set measures dispatch and
> cache, not DRAM.** The same kernel on this machine measures **35 GB/s at 8.8 MB** and
> **90 GB/s at 566 MB**.

Not hypothetical: my own harness's isolated `mx.quantized_matmul` sweep ran on a 14.7 MB weight,
went cache-resident, and produced an M-vs-time curve with apparent plateaus at row *pairs*. **I
am deliberately not reporting that curve or its tiling interpretation**, because it is below the
threshold and did not reproduce against the in-model marginals above. Only the in-model numbers
in this issue stream the full 3.5 GB.

### Why this shape matters

M=2…8 forwards are exactly what speculative-decoding verify passes, small-batch serving and beam
search produce. On this machine the staircase sets the optimal speculative block size: cost rises
faster than acceptance past k=4, capping the achievable speedup at ~1.74× over a stock-decode
baseline. GPU compute utilisation at batch 1 is 5–6 %, so this is not an ALU limit.

### Known gaps

1. **No minimal reproducer.** These are in-model measurements. I can produce an isolated one
   sized above the 32 MB threshold — say the word.
2. **Kernel dispatch unconfirmed.** I have not verified which kernel these shapes dispatch to.
   No claim is made.
3. **Sweep ordering is not randomised.** The k sweep runs ascending on a **fanless** part, so
   higher k are measured warmer. Two-session agreement argues against a large thermal component,
   but both sessions shared the ascending order, so ordering bias is **not excluded**. A
   descending and shuffled re-run is the obvious control and I have not done it.
4. **Single machine, single model.** Fanless M3 Air, one architecture, no cross-check on Pro/Max
   parts or another model family.
5. The 46 % gather-only split is specific to Gemma 4's per-layer embeddings.

### What would help

Whether an isolated `quantized_matmul` repro across M=1…16 at the MLP shapes (2560→10240) would
be useful, or whether the in-model curve is enough to act on.

---

## Corrections from revision 2 (internal note, not for posting)

1. **The #3553-closure premise is unverified** and is now a stated blocker with a command to
   settle it, plus both posting routes. Revision 2 asserted it as fact.
2. **Retitled around the staircase.** The +16.21 / +1.01 ms adjacent pair at M=6→7→8 is a far
   stronger signal than the M=4 step revision 2 led with, and is what a maintainer would open
   first. M=4 demoted to a second feature.
3. **The "~61.5 GB/s = 69 % of ceiling" claim is removed from the issue body.** It came from an
   **in-loop, ctx-186, fused-barrier** measurement, while the table beside it is an **isolated
   sweep at ctx 128 / 2048**. Putting them in one document without labelling was the same
   apples-to-oranges error revision 2 had just corrected one section earlier.
4. **The roofline figure is now 93.5–96.0 %, never 97 %.** Recipe 11 re-measured it in a
   controlled session; CLAUDE.md's stored 97 % is stale and is corrected there too.
5. **Sweep ordering added as a declared gap.** Ascending-only on a fanless part is exactly the
   thermal confound that already caused a 7× attribution error in Phase 0.
6. **The mlx-vlm corroboration is promoted into the body** as its own section. Two independently
   written decode loops agreeing to 0.1 % on independently derived bytes is the strongest reason
   a maintainer should trust the byte accounting, and revision 2 buried it in a parenthesis.

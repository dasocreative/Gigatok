# Recipe 04 — "One ULP Apart", the losslessness write-up

*Renamed from `recipe-04-writeup.md`. External-facing version of Recipe 03's headline,
published as the artifact **"One ULP Apart"**. Audience: MLX / Apple Silicon practitioners.
Working detail and the full killed-hypothesis record stay in `03-mtp-speculative-decode.md`.*

---

## 1. The disagreement

| source | claim | evidence published |
|---|---|---|
| **mlx-vlm 0.6.17** | "Quality matches the target at temperature 0 (byte-identical greedy output)." | none found in the shipped package (§7) |
| **MLX-OptiQ** | "bf16 precision drift in multi-token verify … **not lossless**." | max logit difference 0.679688 |

Same mechanism, same OptiQ 4-bit checkpoint family. Neither published a rate, a margin, an
edit shape, or a mechanism.

## 2. The result

> Speculative decoding with the Gemma 4 MTP drafter is **lossless in the sense that matters**:
> every emitted token is one the target itself produced, and every emitted tail is a valid
> greedy continuation of its own prefix — verified by construction. It is **not** byte-identical
> to sequential greedy decoding. Divergence occurs only where the target's top-2 margin falls
> below **3 bfloat16 ULP**, at **3.12 %** of positions (10 of 320), of which **20 %** (2 of 320,
> **0.63 %**) actually flip — both at exactly 1.00 and 2.00 ULP. The cause is the transformer
> forward, not the output projection. Unchanged by `final_logit_softcapping`. Measured at
> contexts below the 512-token sliding window.

Median top-2 margin across the run is **15–60× the threshold**.

## 3. The two divergences

| prompt | pos | baseline | speculative | margin | edit shape |
|---|---|---|---|---|---|
| 0 | 121 | `41152 ' Ensure'` | `108 '\n\n'` | 1.00 ULP | clean fork, never realigns |
| 1 | 82 | `5396 ' actual'` | `1938 ' information'` | 2.00 ULP | skip-vs-emit, reads as a deletion |

### The skip-vs-emit tie

At position 82 the tied candidates are `' actual'` and `' information'` — and the baseline's
*next* token after `' actual'` **is** `' information'`. "The actual information stored" vs
"The information stored". The candidates are **a token and its own successor**, so the fork's
edit shape is a **deletion** and the sequences realign immediately.

**This is how a lossless implementation produces output that looks like it dropped a token.**
Any diff-based check reports a deletion followed by resynchronisation — the signature of a
bookkeeping bug. Settling it needs a test that ignores the diff: teacher-force
`base[:j] + [spec[j]]`, free-run, check it reproduces `spec[j+1:]` token for token, with the
mirror control on `base[j]`. Both pass on both prompts.

## 4. Where the sensitivity lives

Sweeping every (M, row) cell and **reading the grids down the columns**: every column is
constant below M=1. The argmax depends on *where the token sits in the block*, not on how large
the block is. M=1 is separate only because one row dispatches GEMV instead of GEMM.

**Not the projection.** Replicate one hidden state across every row and project it — every row
a bit-identical input. **0 of 10 at-risk positions move**, 72 cells each, both axes, including
both positions that *do* flip and positions at margin exactly zero.

**The transformer.** Row 0 against a one-row forward of the same token in the same context:

| | |
|---|---:|
| cells where row 0's hidden state differs | **96 of 96** |
| first divergent layer | **0 or 1**, never deeper |
| logit `max\|Δ\|` | **3–9 ULP**, vs 1 ULP for the projection alone |
| cells that flip the argmax | 7 of 96 |

Row 0 is causally independent of the rows after it, so: **the same token, in the same context,
produces a bitwise-different hidden state depending on how many tokens shared its forward
pass.** Present in the first decoder layer, roughly constant after — not rounding compounding
over depth.

**The drift is universal; the flip is rare. The tie band is where the drift matters, not where
it occurs.**

## 5. Why the threshold is in ULP

`ulp/scale` ranges over a factor of two depending on where the top logit sits in its binade.
One divergent position has a top logit of ~39.25 (ULP 0.25); elsewhere the scale is ~28.5
(ULP 0.125). A single relative threshold of `2.3e-2` means **2.9 to 5.9 ULP** depending on
position — a 2×-ragged band reported as one number.

A ULP threshold tightened R from 4.38 % to **3.12 %** and made it uniform. It also survives the
softcap exactly, where the relative margin does not:

| position | rel margin | after softcap | ULP | ULP after softcap |
|---|---|---|---|---|
| p0 · 121 | 6.369e-03 | 4.831e-03 | 1.00 | 1.00 |
| p1 · 82 | 1.550e-02 | 1.053e-02 | 2.00 | 2.00 |

`final_logit_softcapping = 30.0` changed **zero** argmaxes, so the result holds on the
production decode path.

## 6. What was ruled out

Eight explanations for the divergence, measured and killed: centroid shortlist, per-step eval
barriers, batched-verify inequivalence, sliding-window rollback desync, the ~1e-7 noise-floor
assumption, the realignment heuristic, the accept/emit arithmetic, cache class. Full record in
Recipe 03 §4.

Two *explanations of the mechanism* were also killed — the power-of-two tiling signature and
the output-projection tile theory. **Both were written into the project docs before they were
tested.** The claim therefore states *where* the dependence is and deliberately not *why*.

## 7. Reconciling the two claims

**The "byte-identical" claim.**
`mlx_vlm/speculative/drafters/gemma4_assistant/parity_check.py` builds a fake target embedding
from `mx.random.normal`, random shared K/V and random input embeds; runs one forward and one
`draft_block`; prints tensor shapes plus the mean and std of the logits. **It never loads a
target model, never decodes, and never compares tokens against anything.** Across all 76
speculative-related source files in the installed wheel there is no `lossless`, no
`byte-identical`, no `allclose`, no `array_equal` — the only `mismatch` hits are config
compatibility guards and the accept loop's own first-mismatch logic. A shape smoke test
carrying the name of a correctness test.

*Caveat:* a repository CI suite would not ship in a wheel. What can be said is that **nothing
in the shipped package tests it.**

**The "not lossless" claim.** A max logit difference of 0.679688 is a real measurement of the
wrong quantity. The maximum over a 262,144-token vocabulary is attained at whichever token has
the largest logit, not at the two that decide the argmax. Flipping requires
`Δ(top1) − Δ(top2)` to exceed the margin, and the top two move far less than the worst case —
exactly why 96 of 96 cells drift while only 7 flip. If their logits sit on a comparable scale,
0.679688 is ≈5.4 ULP, inside the measured 3–9 ULP transformer-path range.

**This project made the identical error early on** — a diagnostic compared a one-sided
`max|Δ|` against a two-sided event and reported "noise floor below the margins → real bug".

## 8. Speed

Median of 5 passes, warmup discarded, 25 s cooldown, arm order alternated, apps closed:

| | speculative | baseline | speedup |
|---|---|---|---|
| **median** | **40.40 tok/s** | 20.17 tok/s (per-token sync) | 2.04× |
| vs Recipe 02 pipelined | | 23.60 | 1.71× |
| **vs stock `stream_generate` (declared baseline)** | | **24.37** | **1.66×** |

Per prompt: 39.28 (cv 4.2 %) and 43.27 (cv 2.1 %); baselines cv 1.0–1.4 %. Arm-order control
agrees within 2.5 %. One pass ran ~10 % slow on both prompts — a transient the median absorbs.

**Publish 1.66×.** Recipe 02's rule: *never quote a per-token-sync number as throughput; it is
a latency instrument and costs 18 %.* The 2.04× figure compares against exactly such a
baseline.

### Contention inflates the speedup, it does not understate it

| run | condition | spec | baseline | speedup | n |
|---|---|---|---|---|---|
| A | unrecorded | 49.27 | 19.95 | 2.47× | 1 |
| B | loaded | 41.03 | 18.81 | 2.18× | 1 |
| C | loaded | 40.19 | 18.47 | 2.18× | 1 |
| **D** | **quiet** | **40.40** | **20.17** | **2.04×** | **5** |

Closing the apps moved the **baseline** +8 % and left the speculative arm unchanged (+0.5 %).
The baseline reads the full weight set once per token and is bandwidth-bound; the speculative
loop amortises one read over ~2.6 emitted tokens. Contention degrades the baseline and barely
touches the speculative arm, **inflating** the ratio. Direction solid (8 % gap against a
1.0–1.4 % cv); magnitude not, since the loaded condition is n=1.

**Retire 2.47×.** Six subsequent measurements never exceeded 43.67 tok/s.

## 9. What this does not establish

- **Long contexts.** Everything runs to 186 positions against a 512-token sliding window, so
  the rotating cache never wraps. The cache-class control and the rollback hypothesis both
  inherit that limit.
- **Other decode paths.** The hidden-state shape-dependence plausibly reaches
  prefill-vs-decode, chunked prefill and batched serving. None measured.
- **Sampling.** Everything is temperature 0.
- **Other models.** One target, one drafter, one quantisation, two prompts.
- **The loaded speed case.** n=1.

## 10. Environment

```
Target    mlx-community/gemma-4-e4b-it-OptiQ-4bit
Drafter   mlx-community/gemma-4-E4B-it-assistant-bf16
Runtime   mlx 0.32.2 · mlx-lm 0.31.3 · transformers 5.16.1 · Python 3.12.14
Machine   MacBook Air M3, 16 GB, fanless — batch 1, temperature 0

Model     42 layers — 35 sliding_attention, 7 full_attention
          24 caches — 20 rotating, 4 plain
          18 layers share KV and hold no cache of their own
          sliding_window 512 · softcap 30.0 · vocab 262,144
```

**Corrections recorded:** mlx-vlm 0.6.17 does **not** depend on mlx-lm (a clean install leaves
`mlx_lm` absent) — the separate control venv is still right practice, but the stated reason was
wrong. And `mlxutil.host_info()` read `getattr(mx, "__version__", None)`, which mlx does not
define, so **every row in `runs.jsonl` before 2026-08-29 records a null mlx version.**

Checkpoint caveat: stock 4-bit Gemma 4 conversions quantise the per-layer embeddings to 4 bits
and are unusable; OptiQ keeps them at 8 bits. 46 % of the checkpoint is gather-only per-layer
embedding that never streams — a roofline from parameter count alone is wrong by 85 %.

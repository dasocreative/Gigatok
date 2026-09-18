# Metal Inference — recipe index

**Machine of record:** MacBook Air M3, 4P+4E CPU, 10-core GPU, 16 GB, fanless.
macOS 26.6.1 · MLX 0.32.2 · mlx-lm 0.31.3 · Python 3.12.14 — **never upgrade in `~/mlx-env`**.
**Model of record:** `mlx-community/gemma-4-e4b-it-OptiQ-4bit`

---

## The numbers that matter

| | value | where |
|---|---|---|
| Calibrated bandwidth ceiling | **89 GB/s** usable (spec 100) | 01 |
| Active bytes per decode token | **3.535 GB** — 46 % gather-only PLE | 01 |
| **Declared baseline** | **24.37 tok/s** stock `mlx_lm.stream_generate` | 02 |
| Decode vs roofline | **93.5–96.0 %** (Recipe 11, controlled session; the older 97 % is stale) | 02, 11 |
| **Declared speculative result** | **42.46 tok/s = 1.74×** vs the declared baseline | 08 |
| **Cross-runtime, plain decode** | ours 23.56/24.17 vs **mlx-vlm 0.6.17 23.54 tok/s** — byte-identical weights, **0.1 % apart**. No kernel advantage. | 11 |
| **Cross-runtime, speculative** | ours 42.94/41.62 vs mlx-vlm **34.39** = **+22.9 % ours**; speedup over each runtime's own plain: ours **1.72–1.82×**, mlx-vlm **1.41–1.46×** | 11 |
| Losslessness | lossless in the sense that matters; **not** bitwise. 3 ULP band, 3.12 % exposure, 0.63 % flips | 03, 04 |
| Acceptance | 1.587 at γ=3 · 1.858 under confidence scheduling · **neither converts to throughput** | 09 |
| Cycle split at k=4 | forward 48.76 ms (**65 %** of floor) · LM head 7.27 (at roofline) · host 4.5 (**7.4 %**) | 08, 09 |
| `t(k)` shape | **a staircase, not a ramp**: +0.91, +1.88, **+4.91** (M=4), +5.65, +8.95, **+16.21** (M=6→7), **+1.01** (M=7→8); cheap again at k=10 and k≥12 | 05, 12 |
| Verify headroom | **bounded, not a point**: k=8 is 2.41× the floor under perfect stream/compute overlap, 1.01× under none | 12 |
| Compute utilisation at batch 1 | **5–6 %** | 02 |

**Rules this project keeps having to relearn:**

1. **Never quote a per-token-sync number as throughput.** Every speedup names its baseline.
2. **State a ceiling's regime before quoting it.** Nine invalid ceilings/comparisons so far.
   Each caught by arithmetic, none by the code.
3. **Never size a phase against an isolated microbenchmark.** Phase 0's premise was wrong by 4×.
4. **Check what the winning configuration actually is before writing the verdict string.**
   Phase 2's "GO" was a chain, not a tree.
5. **Check a plan item's premise before executing it.** "Comment on #3553" was refuted on two
   grounds before a word was drafted.
6. **The statistic must match the question** — including inside gates. Recipe 11's pooled-CV gate
   was measuring the prompt difference, not steadiness.

---

## Recipes

| # | doc | state | outcome |
|---|---|---|---|
| **01** | `01-harness-and-roofline.md` | CLOSED | The instrument. Cancelled recipes 1–3 by measurement. |
| **02** | `02-baseline-measurement.md` | CLOSED | Decode at roofline across three contexts and three engines. Baseline 24.37 tok/s. `mx.async_eval` worth +18 % with no kernel work. |
| **03** | `03-mtp-speculative-decode.md` | CLOSED | Gemma 4 MTP drafter on mlx-lm. Losslessness settled; eight loop hypotheses and four mechanism explanations killed. |
| **04** | `04-losslessness-writeup.md` | **draft — publish** | Artifact "One ULP Apart". |
| **05** | `05-quantized-matmul-headroom.md` | CLOSED | Kernel headroom investigation. **Its #3553 framing is superseded — see 12.** |
| **06** | `06-tblink-plan-review.md` | done | Plan assessed; six corrections, one fatal gate. |
| **07** | `07-tblink-v1-plan.md` | superseded | Phases 0–2 as written; outcomes in 08 and 09. |
| **08** | `08-phase0-outcome.md` | CLOSED | 42.46 tok/s = 1.74×. Host overhead 7.4 %, not 32 %. Target 48 unreachable in Tier 1. |
| **09** | `09-phase1-phase2-outcome.md` | CLOSED | Phase 1: acceptance +17.1 %, throughput +0 %. Phase 2: NO-GO on width. **Tier 1 finished.** |
| **11** | `11-cross-runtime-comparison.md` | **CLOSED** | The founding question, answered. Plain decode identical to mlx-vlm (0.1 %); our speculative loop **+22.9 %**. Prediction held on all three clauses. |
| **12** | `12-upstream-contribution.md` | **draft, not posted** | `t(k)` is a staircase; steps at M=4 and M=6→7. `#3553` framing refuted. Blocker: issue state unverified. |

Cancelled by Recipe 01's measurements: decode fusion (~3 % envelope), quantized GEMV tuning
(q4 and fp16 both within 4 % of ceiling), KV precision (148 MiB at 8K against 3.5 GB of weights).

---

## Live targets, in expected-value order

1. **Unblock and route the upstream draft** (`../../reports/mlx-upstream-issue-draft.md`):
   verify #3553's open/closed state via the GitHub API, re-verify `qmv_wide` in the pinned build,
   then post as a new issue or as a comment accordingly.
2. **Descending + shuffled k sweep** to exclude thermal ordering bias before the staircase is
   published. Ascending-only on a fanless part is the Phase 0 confound again.
3. **Isolated reproducer** above the 32 MB threshold, k=1…16 — the first thing a maintainer will
   ask for.
4. **Publish 04.** Strongest original result, still a draft.
5. **Long contexts.** Everything sits below the 512-token sliding window; the rotating cache has
   never wrapped.

Not pursued: writing Metal kernels; porting DSpark / DFlash-2 / EAGLE-3 (mlx-vlm ships all
three); switching models; wide verification (measured, 09); further γ scheduling (measured, 09).

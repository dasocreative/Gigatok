# Recipe 03 — Gemma 4 MTP drafter + speculative decode (CLOSED)

*Renamed from `recipe-04-mtp-spec-decode.md`.*

**Machine** MacBook Air M3, 16 GB unified, fanless. 4P+4E CPU, 10 GPU cores.
**Env** Python 3.12.14, mlx 0.32.2, mlx-lm 0.31.3, transformers 5.16.1.
**Models** target `mlx-community/gemma-4-e4b-it-OptiQ-4bit`, drafter `mlx-community/gemma-4-E4B-it-assistant-bf16`.

> The venv interpreter is a symlink out of the connected folder in the original dev setup, so
> a sandboxed agent shell cannot execute it and it has no Metal access. **Every benchmark here
> was run by hand in Terminal, on-device.**

---

## 0. Process notes

**Diagnostics persist to `runs.jsonl`.** Earlier ones printed to stdout and exited, which is
how this file once listed as "queued" experiments whose results were already cited in code.

**Mechanism claims are not measurements.** The §5 *finding* survived every test. Four
*explanations* were killed (§4b), each written into these docs before being tested.

**Preflight gates fail closed on benign variation.** Three raised false alarms: a pin check
reading `mx.__version__` (which does not exist) that condemned a healthy environment; an
app-contention refusal; and a drift check that flagged ordinary battery discharge (57 → 56 %)
and voided a good run. **Every new gate needs a stated benign case it must not fire on.**

---

## 1. Where this stands

| | |
|---|---|
| Losslessness | **SETTLED.** `fork_validity` A and B pass on both prompts |
| Mechanism | **LOCALISED.** Transformer forward, not the projection. Layer 0–1, 3–9 ULP vs 1 ULP for the projection |
| N (threshold) | **3.0 bf16 ULP.** Both flips at exactly 1.00 and 2.00 |
| R (exposure) | **10 of 320 = 3.12 %** |
| Flip rate | 2 of 320 = **0.63 %**; conditional 2/10 = 20 % |
| Softcap | **CLOSED.** 0 argmax changes; ULP count survives exactly |
| Cache class | **CLOSED**, but only below the sliding window (§5g) |
| **Speed** | **40.40 tok/s = 1.66×** vs the declared 24.37 baseline (§7) |

### Model configuration, as installed (measured, not assumed)

42 layers — 35 `sliding_attention`, 7 `full_attention`. 24 caches from `make_prompt_cache` —
20 `RotatingKVCache`, 4 `KVCache`. **18 kv-shared layers holding no cache.** sliding_window
512, pattern 5, `final_logit_softcapping` 30.0, vocab 262,144.

---

## 2. Measured facts

- **Verify cost:** `t(k) = 37.2 + 6.75·k ms` — **superseded past k=10**, see Recipe 05.
- **Acceptance decays with depth**: 2.46 at 128 tokens, 2.19 at 256. 1.59 at γ=3, 2.29 at γ=8.
- **The centroid shortlist wins**: 0.39 ms vs 1.67 ms for the dense 134 MB head.
- **30–33 % of cycles accept zero tokens** and pay the full verify.
- **Drafter confidence is calibrated.** P(accept) 0.333 → 0.974 across buckets (spread 0.641)
  over 311 leading-run positions; n-weighted mean 0.714 reproduces mean-accepted 2.29 at γ=8.
- **`mx.compile` on the draft step is bounded at ≤9 %.** Draft 4.4–5.3 ms vs verify 55.6–57.2.
- Draft-phase CPU exceeds wall time (118–120 %) — graph construction is the bottleneck within
  the draft phase. Verify 34–41 % CPU.

### Retracted: the verify cost model

`confidence_calibration.py` fitted `verify_ms = 17.15 + 8.641·k`. **The intercept is
physically impossible**: 3,535,067,220 active bytes at 89 GB/s is a 39.7 ms floor; 17.15 ms
implies 206 GB/s. Three faults: three in-loop points with no cooldown against a visibly convex
curve (residuals +2.99, −4.49, +1.49); measured against γ=8, the *worst* of three configs
(γ=2 → 46.9 tok/s, γ=4 → 50.4, γ=8 → 32.3, so **a schedule must beat ~50, not 33**); and
break-even is on the cumulative product `q = Π p_j`, not the current step's confidence.

### Retracted: the contention asymmetry

Earlier revisions claimed contention costs the speculative arm ~18 % against the baseline's
~7 %, and concluded a busy laptop *understates* the speedup. **The 5-pass run falsified it.**
Closing the apps moved the baseline +8 % and left the speculative arm flat. Contention
**inflates** the speedup. §7.

---

## 3. Files

`gemma4_assistant.py` (the drafter — install into `mlx_lm/models/`) · `measure_acceptance.py` ·
`spec_generate.py` (loop, baseline, losslessness, ULP-thresholded) · `shape_stability.py` (run) ·
`divergence_text.py` (run, superseded) · `fork_validity.py` (run, PASS) ·
`control_variables.py` (run, both controls pass) · `confidence_calibration.py` (run) ·
`schedule_decision.py` (**not run**) · `row_dependence.py` (run, NULL) ·
`hidden_shape_drift.py` (run, POSITIVE) · `speed_5pass.py` (run) · `setup_control.sh`.

---

## 4. Hypotheses KILLED — about the loop

| # | hypothesis | how it died |
|---|---|---|
| 1 | The centroid shortlist is slow | 0.39 ms vs 1.67 ms dense. |
| 2 | Per-step `mx.eval` barriers are the cycle cost | Removed. Cycle time did not move. |
| 3 | Batched verify ≠ sequential decode | Identical predictions at all 4 positions from an identical cache. |
| 4 | `RotatingKVCache` rollback desyncs | Swapped for plain `KVCache`. Same failure cycles (27, 46). **Valid only below the sliding window — §5g.** |
| 5 | Noise floor ~1e-7, so the divergences are logic bugs | max abs Δ = exactly 1.00 bf16 ULP. |
| 6 | Realignment ⇒ bookkeeping bug | `fork_validity` B: both tails are valid greedy continuations. |
| 7 | Cache class is a hidden second variable | Identical over all 160 positions, both prompts. |
| 8 | The accept/emit arithmetic is wrong | All three invariants OK on both prompts. |

## 4b. Hypotheses KILLED — about the *mechanism*

| # | explanation | how it died |
|---|---|---|
| 9 | Power-of-two block sizes flip, 3 does not | The exhaustive grid is column-constant; block size does not matter. Artefact of testing one alignment per block size. |
| 10 | The **output projection** pads M to the 8-row tile | `row_dependence.py`: 0 of 10 at-risk positions move, 72 cells each, both axes. |
| 11 | The softcap widens the production tie band | ctrl 2: **0 argmax changes**; ULP survives exactly. The arithmetic assumed a ~28.5 logit scale; the actual top logit at position 121 is **~39.25**, a binade higher. |
| 12 | Contention hurts the CPU-bound speculative arm more | `speed_5pass`: baseline +8 %, speculative +0.5 %. Direction backwards. |

---

## 5. The finding

| prompt | pos | base | spec | ULP | edit shape |
|---|---|---|---|---|---|
| 0 | 121 | 41152 `' Ensure'` | 108 `'\n\n'` | **1.00** | clean fork |
| 1 | 82 | 5396 `' actual'` | 1938 `' information'` | **2.00** | skip-vs-emit, +1 shift for 2 tokens |

**A — the arithmetic.** `emitted == drafts[:n] + [bonus]` OK, `n == true match length` OK,
`bonus == target_preds[n]` OK, both prompts.

**B — the decisive test.** Teacher-force `base[:j] + [spec[j]]`, free-run: reproduces
`spec[j+1:]` token for token; reverse control on `base[j]` reproduces `base[j:]`.

**The skip-vs-emit tie.** Prompt 1's tied candidates are `' actual'` and `' information'` —
and the baseline's *next* token after `' actual'` **is** `' information'`. The candidates are a
token and its own successor, so the fork's **edit shape is a deletion** and the sequences
realign immediately. This is how a lossless implementation produces output that looks like it
lost a token. `divergence_text.py`'s heuristic is structurally wrong for this class of tie —
**do not re-run it.**

**Where the shape sensitivity is.** Not the projection (0 of 10 positions move when one hidden
state is replicated across rows, 72 cells each, both axes). The transformer:
**96 of 96 cells** differ, first divergent layer **0 or 1**, drift flat across M=2…8, logit
`max|Δ|` **3–9 ULP** vs 1 ULP for the projection. **Drift is universal; flips are rare.**

**Why 96/96 drift but 7/96 flip.** `max|Δ|` over a 262 k vocabulary is attained at the largest
logit, not the top-2. Flipping needs `Δ(top1) − Δ(top2)`. Same one-sided-vs-two-sided error
that made `debug_verify`'s original verdict untrustworthy — second time it bit.

## 5f. The softcap — CLOSED, and the binade correction

**0 positions change argmax** under `logit_softcap(30.0, ·)`. ULP survives exactly
(1.00 → 1.00, 2.00 → 2.00) while the relative margin does not (6.369e-03 → 4.831e-03).
Position 121's top logit is ~39.25, in the [32,64) binade (ULP 0.25); the cap takes it to
~25.9, into [16,32) (ULP 0.125). Gap and ULP halved together — coincidence, not law.

**Consequence:** the threshold moved from relative margin to ULP. `rel = ulps · ulp/scale` with
`ulp/scale ∈ (2^-8, 2^-7]`, so a fixed `2.3e-2` means **2.9 to 5.9 ULP** depending on binade.
`spec_generate.py` now uses `--tie-ulps 3.0`. **R fell from 4.38 % to 3.12 %.**

## 5g. Scope limit — everything is below the sliding window

Context reaches 186 positions against a 512 window, so **the ring has never wrapped**. Ctrl 1's
pass holds below the window; **killed hypothesis 4 inherits the limit** — it shows rollback did
not cause *these* divergences, not that it is safe once the ring evicts.

## 5e. The claim

> Lossless in the sense that matters — every emitted token is one the target produced, every
> emitted tail a valid greedy continuation of its own prefix, verified by construction. **Not**
> byte-identical. Divergence only below **3 bf16 ULP**, at **3.12 %** of positions, of which
> **20 %** (0.63 % overall) flip. Cause: the transformer forward, not the projection — 96 of 96
> cells, layer 0–1, 3–9 ULP vs 1 ULP. Unchanged by `final_logit_softcapping`. Measured below
> the 512-token sliding window.

## 5h. Ecosystem context

mlx-vlm 0.6.17 has a complete implementation **and also ships DSpark, DFlash-2, EAGLE-3 and
draft trees** — the mechanism ports that remained on this roadmap are catching up, not
contribution. Its `parity_check.py` runs the drafter on random tensors with a fake target and
never compares tokens; across all 76 speculative source files in the wheel there is no
`lossless`, `allclose` or `array_equal`. OptiQ's 0.679688 is ≈5.4 ULP — inside our measured
3–9 ULP range, consistent with measuring the same effect with a one-sided statistic.

`gemma4_assistant` is still **not** in any mlx-lm release — only open PR #1276, model class
only. Control setup: `setup_control.sh` pins mlx-vlm 0.6.17 in a separate venv built from the
main venv's own Python so the interpreter matches. **Correction:** mlx-vlm does *not* depend on
mlx-lm; the separate venv is right practice, the stated reason was wrong.

---

## 6. What remains

Optional: 5-pass `--allow-contention` to pin the contention magnitude; the long-context regime;
`schedule_decision.py` (never run — and Recipe 05 now supplies a correct `t(k)`).

---

## 7. Speed

| | speculative | baseline | speedup |
|---|---|---|---|
| median of 5, quiet | **40.40** | 20.17 (per-token sync) | 2.04× |
| vs Recipe 02 pipelined | | 23.60 | 1.71× |
| **vs stock `stream_generate`** | | **24.37** | **1.66×** |

**1.66× is the publishable number.** Recipe 02: *never quote a per-token-sync number as
throughput — it is a latency instrument and costs 18 %.*

Per prompt 39.28 (cv 4.2 %) / 43.27 (cv 2.1 %); baselines cv 1.0–1.4 %. Arm order agrees within
2.5 %. **Contention inflates the ratio** — the loaded runs gave 2.18× because the baseline
suffers and the speculative arm does not.

Discipline: warmup discarded, 25 s cooldown, medians ≥5, temp 0. Gate on Low Power Mode and
<30 % charge, not on being plugged in; ordinary discharge is not drift. Report the statistic
that matches the question — argmax flips need `Δ(top1) − Δ(top2)`, not `max|Δ|`.

**Provenance bug, fixed 2026-08-29:** `mlxutil.host_info()` read `getattr(mx,"__version__")`,
which mlx does not define — **every row in `runs.jsonl` before that date records a null mlx
version.** Now read from installed distribution metadata.

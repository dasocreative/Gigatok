# TBlink v1 — the plan to actually start (2026-08-30)

Adapted from the submitted TBlink plan, rebuilt on measured ground. Rationale for every
change is in `06-tblink-plan-review.md`. **Models unchanged from Recipes 02–05** — every
calibration we own is on them.

```
Target    mlx-community/gemma-4-e4b-it-OptiQ-4bit
Drafter   mlx-community/gemma-4-E4B-it-assistant-bf16   (Google's MTP head, already working)
Machine   MacBook Air M3, 16 GB, fanless · mlx 0.32.2 / mlx-lm 0.31.3 — pinned
Baseline  24.37 tok/s, stock mlx_lm.stream_generate
Current   40.40 tok/s = 1.66x
```

---

## 1. Explicitly NOT a DFlash-2 port

The submitted plan's Track 3 is DFlash-2's shape with different kernels: block-diffusion
drafter, multi-layer feature tap, linear block, verify. **TBlink v1 does none of that**, for
reasons that are engineering, not legal:

| DFlash-2 does | TBlink v1 does instead | why |
|---|---|---|
| Trains a block-diffusion drafter | **Uses Gemma 4's shipped MTP head** | We already have it running with measured acceptance (1.59 @ γ=3, 2.29 @ γ=8) and a calibrated confidence signal (P(accept) 0.333 → 0.974 across buckets). Training a drafter to reach parity is months to arrive at the same place. |
| Taps 5 intermediate layers to DRAM | **Uses the target's final hidden state**, already returned by `Gemma4TextModel.__call__` | Measured: no tap needed, no serialisation, no fused kernel to write. The plan's "DRAM write penalty" is a cost we never pay. |
| Drafts a linear block of K=5 | **Branches on drafter uncertainty at shallow depth** | Our data: acceptance *decays with depth* (2.46@128 → 2.19@256) while 30–33 % of cycles accept **zero**. Depth is the wrong axis. |
| Verifies linearly | **Verifies a small tree in one forward** | Same forward, more candidates, targeted at the measured failure mode. |

The one idea we take from the submitted plan is **multi-branch verification**. Everything
around it is derived from our own measurements, and the drafter is Google's, not a
reimplementation of anyone's.

---

## 2. The measurement that sets the design

`t(k)` and acceptance, both measured on this machine:

| k | t(k) ms | emitted/cycle | ms/token |
|---|---|---|---|
| 1 | 35.97 | 1.00 | 36.0 |
| 3 | 39.73 | ~2.0 † | 19.9 |
| **4 (γ=3)** | **44.09** | **2.59** | **17.0** ← optimum |
| 6 | 61.48 | ~3.0 † | 20.5 |
| 9 (γ=8) | 92.16 | 3.29 | 28.0 |
| 16 | 111.6 | ~4.0 † | 27.9 |

† interpolated; only γ=3 and γ=8 acceptance are measured.

**Three conclusions, and one of them retracts an earlier recommendation.**

1. **k=4 is the operating point.** The curve is shallow around it and rises steeply after.
2. **The k ≥ 11 saturation is real but not exploitable.** A previous revision recommended
   retargeting tree width to M ≈ 11–16 because five verify positions are free there. The
   economics do not support it: a 16-wide tree needs ~6.6 emitted tokens/cycle to beat γ=3.
   **Retracted.** Stay at k ≈ 4–6.
3. **The gap is overhead, not width.** 17.0 ms/token predicted at k=4 is **58.8 tok/s**;
   measured is 40.4. **~32 % is host overhead** — graph construction, cache trim, shared-KV
   slice, Python. That is the biggest lever in the project and it needs no new mechanism.

And the failure mode worth attacking: **30–33 % of cycles accept zero tokens** and pay the
full 44 ms verify to emit one token. That is where a second branch earns its slot, not depth.

---

## 3. What TBlink v1 is

> A speculative loop on Gemma 4's own MTP drafter that (a) removes host-side serialisation,
> and (b) spends 1–2 extra verify slots on **breadth at the first uncertain draft position**,
> chosen by the drafter's calibrated confidence, to rescue the third of cycles that currently
> accept nothing.

Tier 1 throughout. No Metal. `mx.fast.metal_kernel` only if Phase 3 measures a win that MLX
ops cannot deliver — and Recipe 05 says that is unlikely, since MLX's own maintainers own the
relevant dispatch heuristic and have [issue #3553](https://github.com/ml-explore/mlx/issues/3553)
open on it.

---

## 4. Phases

### Phase 0 — reclaim the overhead (no new mechanism)

**Entry:** none. **Exit:** measured tok/s at γ=3, medians ≥5, quiet machine.

1. `mx.async_eval` the speculative loop. Recipe 02 precedent: the same change took the
   baseline 20.03 → 23.60, **+18 %, zero kernel work**. Our loop still uses per-token
   `mx.eval`.
2. `mx.compile` the draft step. Bounded at ≤9 % by measurement, but drafting runs at
   118–120 % CPU so it is the phase most exposed to host cost.
3. Profile what remains of the 32 % gap: cache `trim`, `slice_shared_kv`, the Python accept
   loop.

**Target: 48–52 tok/s (≈2.0× the declared baseline) with no change to the algorithm.**
If Phase 0 alone reaches the submitted plan's 1.8 × gate, the rest is upside, not necessity.

### Phase 1 — confidence-scheduled γ

**Entry:** Phase 0 measured. **Exit:** `schedule_decision.py` output, γ policy fixed.

`schedule_decision.py` has never run and now has a correct `t(k)` to run against. Break-even
is the cumulative product `q = Π p_j`, not per-step confidence. The calibration already
exists: P(accept) climbs 0.333 → 0.974 across drafter-confidence buckets over 311 positions,
n-weighted mean 0.714 reproducing measured mean-accepted independently.

Expected: γ varies 2–5 by position instead of fixed 3. Modest, and it is the prerequisite for
Phase 2 — the same confidence signal decides *where to branch*.

### Phase 2 — the tree, small and targeted

**Entry:** Phases 0–1. **Exit:** measured acceptance and tok/s vs fixed-γ.

Not a general tree. A **2-branch tree at the first draft position whose confidence falls below
the Phase-1 threshold**, taking the drafter's top-2 rather than top-1 there. Cost: k=4 → k=6,
+40 % verify. Break-even: emitted must rise 2.59 → 3.6.

Whether that clears is an open question and Phase 2 must be allowed to fail. The argument for
trying: a third of cycles currently emit exactly one token for 44 ms, and the drafter's
confidence is calibrated well enough to identify them in advance.

Implementation is MLX ops — a block-diagonal-plus-causal mask over the branch, built with
`mx.where` on the existing attention mask. **No `tree_attention.metal`.**

### Phase 3 — measure, then decide about kernels

Only after 0–2 are measured. Recipe 05's decomposition (`verify_decomposition.py`) reruns at
the new operating point. Write MSL only if a specific shape sits ≥2× above *both* floors and
MLX ops cannot reach it.

**Also in Phase 3, and cheap:** comment on MLX #3553 with our M3 + Gemma 4 + roofline-framed
data. It is an open issue with an unidentified cause and we have a second chip, a second model
family, and a bandwidth framing they lack.

---

## 5. Acceptance gates

| gate | value |
|---|---|
| **Correctness** | Every emitted token is one the target produced, and every emitted tail is a valid greedy continuation of its own prefix — `fork_validity.py` A **and** B, both directions. Divergence permitted only below **3 bf16 ULP**; report the measured rate. |
| **NOT a gate** | ~~Bitwise identity with the baseline~~. Recipe 03 proved it unachievable: the transformer forward is shape-dependent in 96 of 96 tested cells, 0.63 % of positions flip. Any method verifying M ≥ 2 rows inherits this. |
| **Throughput** | Named against **24.37 tok/s** stock `stream_generate`. Never against a per-token-sync baseline — Recipe 02: *"it is a latency instrument and costs 18 %."* Phase 0 target 2.0×; Phase 2 target 2.2× and permitted to fail. |
| **Discipline** | Medians ≥5, temp 0, warmup discarded, 25–40 s cooldown, apps closed, host state recorded. **Contention inflates speculative ratios** — quiet is the conservative measurement, not the flattering one. |
| **Memory** | Active footprint under 10 GB. Currently ~5.5 GB (target 4-bit + bf16 drafter + KV). |
| **Roofline** | Every phase states measured/roofline before proposing the next. This is what cancelled three recipes and caught a 206 GB/s intercept. |

---

## 6. Metrics per run → `runs.jsonl`

`record_type`, `schema_version`, chip, macOS, **mlx version from distribution metadata**
(the old `getattr(mx,"__version__")` probe wrote null on every row before 2026-08-29), model +
quantisation, prompt/gen lengths, TTFT, prefill and decode tok/s, effective GB/s and % of
roofline, **mean accepted and acceptance rate per step**, tree width and branch positions
(Phase 2), peak memory, `power_source`, Low Power Mode, `top_processes`.

---

## 7. What this plan deliberately does not do

- **No custom Metal in Phases 0–2.** Recipe 05: headroom 16–130 %, and the maintainers own the
  dispatch heuristic.
- **No new drafter.** Gemma 4's MTP head works, is measured, and is Google's.
- **No feature tap.** The target already returns the hidden state the drafter consumes.
- **No model switch.** Every calibration we own — 89 GB/s ceiling, 3.535 GB active bytes, KV
  `f = 0.55`, acceptance curves, losslessness characterisation — is on this pair.
- **No wide verification.** §2, conclusion 2.
- **No throughput claim before Phase 0 is measured five times on a quiet machine.**

The submitted plan's ambition survives intact. What changed is the order: measure the overhead
we already know about before building mechanisms to work around it.

# Review — Project TBlink plan, against measured ground (2026-08-30)

Reviewing the TBlink engine plan against what this project has actually measured.
**Verdict: the premise is right and Recipe 0 predicted it. The sequencing, the target
window, and two acceptance gates are wrong.** Six corrections, one of which is fatal to a
stated gate.

---

## 0. Where TBlink is RIGHT, and why our own data supports it

TBlink's core bet is that batch-1 decode leaves the GPU idle and that wider parallel
verification converts idle compute into tokens. **Recipe 0 measured exactly this and said so:**

> dense transformer params 3.97 B · FLOPs/token 7.9 GFLOP · compute @ 3–4 TFLOPS = 2.0–2.7 ms
> against 41.5 ms of memory time. **GPU compute utilisation at batch 1: 5–6 %.**
> *"Headroom for γ ≈ 15–20 before compute binds."*

Recipe 5 then confirmed it empirically: `t(k)` **saturates flat from k=11 to k=16** —
five extra verify positions for free while FLOPs grow 45 %.

So the plan is not shifted from the mission. It is the mission. The problems are all in *how*.

---

## 1. FATAL — the bitwise-equivalence gate cannot be met

> *"Assert 100 % bitwise token equivalence against the baseline target at temperature T = 0.0"*
> *"Losslessness: Output sequence at T=0 must match the unaccelerated base model token-for-token."*

**Recipe 3 (MTP) proved this is unachievable on this hardware, for reasons unrelated to any
implementation.** The transformer forward is shape-dependent: the same token, in the same
context, produces a bitwise-different hidden state depending on how many tokens shared its
forward pass — **96 of 96 tested cells**, entering at decoder layer 0–1, worth 3–9 bf16 ULP at
the logits. Where the target's top-2 margin is under ~3 ULP, the argmax flips. Measured rate:
**3.12 % of positions at risk, 0.63 % actually flip.**

Any speculative method that verifies M ≥ 2 rows inherits this. TBlink verifies M = K+1 and
then 8–10 tree candidates, so it will hit it harder, not less.

**TBlink would fail its own acceptance gate on day one, for a reason that has nothing to do
with TBlink.** Note that Recipe 0's original entry criteria contained the same error
("greedy output must match `output_sha256` token-for-token"); Recipe 3 corrected it.

**Restate the gate as:**

> Every emitted token is one the target itself produced, and every emitted tail is a valid
> greedy continuation of its own prefix (teacher-force + free-run test, both directions).
> Divergence permitted only where the target's top-2 margin is below N bf16 ULP; report N and
> the measured rate R. Bitwise identity is *not* the criterion.

---

## 2. The target window M ∈ [4, 8] is the worst possible choice

> *"GPU Configuration | 10 Cores | Narrow-batch GEMM optimization (M ∈ [4, 8])"*
> *"verifying 8–10 candidate tokens simultaneously in a single narrow-batch GEMM pass"*

Recipe 5 measured `mx.quantized_matmul` bandwidth against the calibrated 89 GB/s ceiling:

| M | 1 | 2 | 3 | **4** | 6 | **8** | 11 | 16 |
|---|---|---|---|---|---|---|---|---|
| % of roofline | 100 | 101 | 95 | **78** | 50 | **39** | 26 | 26 |

**M ∈ [4,8] is precisely where the kernel is collapsing.** This is not our idiosyncrasy —
[MLX issue #3553](https://github.com/ml-explore/mlx/issues/3553) is open on it: a non-linear
cost step at M=3 on asymmetric shapes, with the author noting existing PRs *"leave M=3–9 on
asymmetric shapes unaddressed."* A full Qwen 3.6-27B forward there runs **1.38× slower at M=3
than M=1**. TBlink's Phase-1 model is Qwen.

**The correction is favourable.** `t(k)` saturates at k ≥ 11, so the right verification width
is **M ≈ 11–16, not 4–8**. Positions 11 → 16 cost nothing. That is a *stronger* argument for
2D tree attention than the plan makes: a tree wide enough to fill 16 slots is priced the same
as a linear block of 11. Design the tree to the measured free window.

---

## 3. Both throughput targets are unfounded, and the baseline is unnamed

> *"Minimum 1.8× speedup on general text; minimum 2.5× on code/reasoning over baseline
> autoregressive decode"*

Recipe 0 declares the baseline explicitly — **24.37 tok/s** stock `mlx_lm.stream_generate` —
and states a rule this project has been violating:

> **"Rule: never quote a per-token-sync number as throughput. It is a latency instrument and
> it costs 18 %."**

Against 24.37: **1.8× = 43.9 tok/s, 2.5× = 60.9 tok/s.** Recipe 0's own projection table says
~62 tok/s requires an **acceptance length of 3.0**. Measured acceptance with a shipped,
Google-trained MTP drafter: **1.59 at γ=3, 2.29 at γ=8.** So the 2.5× target needs roughly
double the acceptance a production drafter achieves, from a from-scratch block-diffusion
drafter.

The targets need an acceptance model before they are credible. Speedup ≈
(mean accepted + 1) × t_baseline / t_verify — and the plan measures neither term.

**Our own number, corrected by the same rule:** 40.40 tok/s speculative is **1.66×** against
the 24.37 baseline, not the 2.04× we have been quoting against a per-token-sync baseline.
Recipe 0 forbade that comparison and we made it anyway.

---

## 4. The bandwidth figure is the spec sheet, not the machine

> *"Memory Bandwidth | ≈ 100 GB/s theoretical peak"*

Recipe 0 calibrated it: **89 GB/s usable** (q4 GEMV 92.8 streaming, read plateau 96.4,
denominator clamped to 89 where the fit over-extrapolated). Using 100 in a plan whose entire
premise is bandwidth saturation overstates the headroom by 12 % and will make every
"% of peak" claim optimistic.

Also inherited: *"any GEMV probe under ~32 MB is launch-bound"* — the same kernel reads
35 GB/s at 8.8 MB and 90 GB/s at 566 MB. TBlink's benchmark suite must exclude small probes
or it will report impossible efficiencies.

---

## 5. The dispatch-bubble diagnosis is correct — the proposed fix is out of order

> *"CPU–GPU dispatch bubbles caused by host-side verification and sequence-length indexing"*
> → `atomic_rollback.metal`, GPU-native rollback

The diagnosis is right and it is the **largest lever this project has found**: our speculative
loop's draft phase runs at 118–120 % CPU (multithreaded graph construction), and measured
verify + draft costs predict ~53 tok/s against 40.4 actual — a **~25 % gap** that is pure host
overhead.

But Recipe 0 already fixed this class of problem for **zero kernel work**:

> `--sync per-token` vs `mx.async_eval`: **+18 % throughput, no kernel changes.** Overhead per
> token fell 9.11 ms → 1.62 ms, **82 % recovered by overlap alone.**

**Our speculative loop still uses per-token `mx.eval` internally.** That +18 % is sitting
unclaimed. Writing `atomic_rollback.metal` before trying `mx.async_eval` inverts the project's
own Tier-1-before-Tier-2 rule, which exists precisely because Recipes 1 and 2 were cancelled
after the roofline showed kernels could not pay.

---

## 6. Switching to Qwen2.5-7B abandons measured ground

Everything calibrated here is on `gemma-4-e4b-it-OptiQ-4bit`: the 89 GB/s ceiling, the 3.535 GB
active-bytes split (46 % gather-only PLE, a number a parameter count gets wrong by 85 %), the
KV model (`f = 0.55`, 45 % of shared re-reads absorbed by SLC), acceptance curves, a working
drafter, and the losslessness characterisation.

Qwen2.5-7B-4bit at ~4.3 GB active is ~20.7 tok/s at 89 GB/s — a **slower** baseline than
Gemma's 24.37, which flatters speedup ratios while lowering absolute throughput. Fine as a
second model for generality; expensive as the first.

---

## 7. Two things the plan is missing entirely

**No roofline instrument.** There is no "when are we done" calculation anywhere in the plan.
That instrument is what cancelled Recipes 1–3 here and what caught an impossible 206 GB/s
verify intercept before it shipped as a 1.635× claim. A 2.5× target with no ceiling
calculation cannot know whether it is reachable.

**No acceptance measurement plan.** All three "core innovations" (transition lattice, tree
attention, feature tap) aim at raising acceptance, yet acceptance appears only as a reported
metric, never as a measured input to the design. Recipe 0's entry criterion for speculative
work was *"measure real acceptance length before building anything — the cheapest possible
probe."* That still applies.

**Minor:** the 5.0 s thermal reset is too short — Recipe 0 used 40 s cooldown and Recipe 3's
speed harness uses 25 s on this chassis. Recipe 0 also found the bimodality it was chasing was
CPU/GPU overlap, **not** thermal, so gating on thermal state alone may watch the wrong
variable. Host contention proved to matter more, and it is asymmetric: it degrades the
bandwidth-bound baseline more than the speculative arm, **inflating** measured speedups.

---

## 8. Recommended revision — same goal, reordered

| phase | change from the plan |
|---|---|
| **0 (new)** | Port the roofline instrument to the chosen model. Compute the byte split and calibrated ceiling before anything else. Non-negotiable; it is what makes every later number checkable. |
| **0b (new)** | `mx.async_eval` the speculative loop. Recipe 0 precedent: +18 %, zero kernel work. Do this before any Metal is written. |
| **1** | Keep the feature tap — but **measure the DRAM write penalty it claims to fix** before fusing it in registers. That premise is currently unverified. |
| **2** | Keep the drafter and lattice. Add the missing step: **measure acceptance** before tuning anything downstream of it. |
| **3** | Retarget tree width to **M ≈ 11–16**, the measured free window, not 4–8. Then `tree_attention.metal` is justified by a measured saturation rather than an assumption. |
| **3b** | Drop `atomic_rollback.metal` until 0b is measured. If async_eval recovers the 25 % gap, the kernel is unnecessary. |
| **4** | Name the baseline (stock `stream_generate`, 24.37). Restate targets against it. 40 s cooldown. Record host contention. |
| **gate** | Replace bitwise equivalence with the ULP-band criterion (§1). |

## 9. What TBlink adds that we do not have

Genuine, and worth keeping:

- **2D tree attention.** We never tried multi-branch verification. Combined with the k ≥ 11
  saturation it is the best-founded idea in the plan.
- **Fused feature tap in register space.** Plausible; unmeasured either way.
- **Top-16 transition lattice.** Addresses the conditional-independence weakness of block
  diffusion, which is a real limitation.
- **Thermal/power telemetry via IOKit.** We record `power_source` and Low Power Mode but not
  thermal state; on a fanless chassis that is a gap worth closing.

**The plan's ambition is right and its diagnosis is largely right. It just skips the
measurement layer that this project exists to provide, and it aims its central optimisation at
the one batch-size window MLX currently handles worst.**

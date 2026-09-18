# Recipe 09 — Phase 1 & Phase 2 outcome

**Status:** both CLOSED. Both are measured negatives on throughput. Neither changed the
declared result. Both were killed by their own instrumentation before any Metal was written,
which is the outcome the phase design was built to produce.

**Declared result, unchanged:** **42.46 tok/s = 1.74×** the declared 24.37 baseline, lossless
within the 3.0 bf16 ULP band, `fork_validity` A+B PASS.

Records: `phase1_finding` (20260830-141834), `tree_feasibility` and
`tree_feasibility_corrected` (20260902-1331/1335), `declared_result` (20260902-133154).

---

## 1. Requirements check against `OBJECTIVES.md` §4

| # | target as written | result | met |
|---|---|---|---|
| 0 | Phase 0 — reclaim host overhead. **48–52 tok/s (≈2.0×)**, no algorithm change | 42.46 tok/s = 1.74×; premise refuted (host overhead 7.4 %, not 32 %) | **NO — accepted as unreachable in Tier 1, with proof** |
| 1 | Phase 1 — confidence-scheduled γ, break-even on the cumulative product | acceptance **+17.1 %** (1.587 → 1.858); throughput **42.46 → 42.46** | **NO on throughput. YES on the mechanism.** |
| 2 | Phase 2 — small tree, 2 branches at first low-confidence position, k=4→6. Break-even needs emitted 2.59 → 3.6. *Permitted to fail.* | width costs **−20.4 %**; best shape under a tree-favouring cost model is a **chain** | **NO — NO-GO, exercised the permission to fail** |
| — | Losslessness gate, every phase | PASS at γ=3 fixed and at γ_max=5 / thr=0.70 | **YES** |
| — | Value neutrality of every kept change | byte-identical via `spec_sha256`, both prompts | **YES** |

Three of three throughput targets missed. Every one of them missed *for a reason the harness
could state*, and none of them cost an implementation that had to be thrown away.

---

## 2. Phase 1 — confidence-scheduled γ

**The signal is real.** Drafter confidence predicts acceptance cleanly: P(accept) rises
0.333 in the [0.00, 0.50) bucket to 0.974 in [0.99, 1.00) — a spread of 0.641. Break-even
confidence q* = 0.422. This is a well-calibrated drafter and the scheduling premise was sound.

**Acceptance moved. Throughput did not.**

| arm | acceptance | spec tok/s | baseline arm tok/s | normalised ratio |
|---|---|---|---|---|
| fixed γ=3 | 1.587 | 42.46 | 20.48 | 2.073 |
| sched γ_max=4, thr 0.70 | — | 42.37 | 20.32 | 2.085 |
| sched γ_max=5, thr 0.70 | 1.858 (**+17.1 %**) | 42.46 | 20.19 | 2.103 |

Net gain, machine-normalised: **+1.4 %** — inside the noise of a fanless M3.

**Why the acceptance gain does not convert.** Two costs absorb it exactly.

1. **Single-barrier truncation must still run all γ_max drafter steps.** 5 × 1.338 = 6.98 ms
   against ~4.3 ms at fixed γ=3. You pay for the drafting you then throw away.
2. **The truncation decision needs the confidences on the host**, which reinstates the very
   barrier P0.3 removed. Draft no longer fuses into verify.

Mean drafted rises to ~3.0, so k ≈ 4.0 — *the same verify cost as fixed γ=3* — while drafter
cost and one barrier are added. The schedule buys tokens it has already paid for.

**The adaptive variant is worse.** Stopping the drafter early genuinely saves drafter steps,
but costs ~3.7 barriers per cycle at ~0.8 ms each. Measured **slower** than fixed γ=3 even on
cycles that both accepted more (1.71 vs 1.46) and used fewer verify slots (k=3.66 vs 4).

**The simulation that said otherwise was benchmarking against a straw man.**
`confidence_calibration.py`'s sweep predicted 47.06 tok/s at 1.549×. Its baseline was
**fixed γ=8** (30.39 tok/s) — an operating point nobody runs. Against the k=4 optimum the
schedule has almost no room, exactly as the cost curve implies: *at the optimum, shortening
the block moves you below it.*

**Kept:** `--conf-threshold`, **default 0.0 = fixed γ**, verified byte-identical to
pre-Phase-1. The mechanism stays in the codebase, off, correct, and cheap to re-enable if the
cost structure ever changes (a cheaper drafter, or a barrier-free truncation).

---

## 3. Phase 2 — the small tree

Run as a **feasibility study before implementation**: measured drafter coverage at 308 real
draft positions, depths 1–6, widths 1/2/4/8/16, priced with the measured cost model
(`lm_head_ms` 7.3, `draft_ms_per_node` 1.338, `host_ms` 4.6) **deliberately biased in the
tree's favour**.

**Coverage does rise with width**, as the mechanism predicts — at depth 1: 0.750 (W1) →
0.851 (W2) → 0.909 (W4) → 0.945 (W8) → 0.958 (W16).

**And it still loses.**

| shape | ms/token | tok/s | k | E[accepted] |
|---|---|---|---|---|
| best chain (1,1,1,1) | 20.57 | 48.62 | 4 | 1.982 |
| best widened (2,1) | 24.76 | 40.39 | 4 | 1.478 |

**Width penalty: −20.4 %. `width_pays: false`.**

k rises faster than `t(k)` forgives: k=4 costs 51.4 ms, k=8 costs 86.0 ms (**+67 %**) to buy
**+0.27** expected accepted tokens. The optimal shape under a cost model built to flatter trees
is a **chain at depth 3–4** — which independently reproduces the k=4 optimum by a completely
different method.

> **Correction of record.** The first `tree_feasibility` row printed verdict **GO**, because
> the best shape beat the current operating point. But that best shape *was a chain*. The
> question Phase 2 asked was whether **width** pays. `tree_feasibility_corrected` supersedes
> the verdict string. This is the eighth invalid-comparison catch in the project and the third
> where a script compared the right numbers under the wrong question.

---

## 4. Where the remaining time actually is

Diagnosed in `verify_gap_diagnosis` (20260830-135649) and unchanged by Phases 1–2.

| phase | ms/cycle | at | note |
|---|---|---|---|
| transformer forward | 48.76 | **65 %** of its 31.71 ms floor | **all 17.05 ms of headroom is here** |
| LM head | 7.27 | ~110 % of floor | at roofline — nothing to take |
| host python | 4.5 | — | 7.4 % of cycle |
| **cycle** | **58.4** | 71 % overall | emits 2.587 → 42.5 tok/s (measured 42.46) |

Also settled: the loop's own ingredients — KV-capture wrapper, plain KV caches, shared_kv
slicing — cost **0.02 ms total**. The gap is not loop-specific. And context 128 → 186 costs
**0.04 ms**: context length is not the variable.

**To reach 48.0 tok/s the cycle must emit 2.80 tokens — mean accepted 1.80 out of a hard
maximum of 3.0.** Phase 1 reached 1.858 accepted and it *still* did not convert, because the
schedule that produced it also raised the cycle cost. That closes the algorithmic route at
this operating point.

**The one thing left is the 4-token verify's bandwidth efficiency: 61.5 GB/s = 69 % of the
89 GB/s ceiling, against 97 % for stock 1-token decode.** Batching 4 tokens through the verify
is *less* bandwidth-efficient per byte than decoding one. That is a kernel property, it is the
same dispatch-heuristic family as MLX issue #3553, and it is Tier 2.

---

## 5. What this costs the roadmap

Phases 0, 1 and 2 were the whole Tier 1 programme in `07-tblink-v1-plan.md`. All three are
now closed and none of them moved throughput. **1.74× is the number, and Tier 1 is finished.**

The honest reading: the speculative loop is running at ~97 % of what its own cost model says
it can do. There is no bookkeeping left to reclaim and no algorithmic shape at k ≈ 4 that
beats the chain. Any further gain has to come from making the verify kernel stream at closer
to 89 GB/s — which is exactly the thing the project already decided not to own, because the
MLX maintainers own that dispatch heuristic and have an open issue on it.

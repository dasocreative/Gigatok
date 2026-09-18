# Metal Inference — objectives, revised 2026-09-08 (eleventh pass)

**Mission.** Find an optimised runtime for local LLMs on Apple Silicon.

**Status: the founding question is answered.** Tier 1 is closed at 1.74×, and Recipe 11 settled
whether that is worth anything against the runtimes people actually use. Index:
`docs/recipes/00-INDEX.md`.

```
Target    mlx-community/gemma-4-e4b-it-OptiQ-4bit
Drafter   mlx-community/gemma-4-E4B-it-assistant-bf16
Machine   MacBook Air M3, 16 GB, fanless · mlx 0.32.2 / mlx-lm 0.31.3 — pinned, never upgrade
Baseline  24.37 tok/s (stock mlx_lm.stream_generate) — the declared baseline
DECLARED  42.46 tok/s = 1.74x, lossless within the 3.0 bf16 ULP band
vs mlx-vlm  plain decode 0.1 % apart · speculative +22.9 % ours
```

---

## 1. The answer to the question the project was built on

**No kernel advantage — and never any.** Plain decode, byte-identical weights, independently
derived active-bytes inside each venv: ours 23.56 / 24.17 tok/s, mlx-vlm 0.6.17 **23.54**. Two
separately written decode loops, **0.1 % apart**, both at 93.5–96.0 % of the 89 GB/s ceiling.

**But the loop is real.** Same weights, same drafter architecture: ours **42.94 / 41.62** tok/s
against mlx-vlm's **34.39** at its own default block — **+22.9 %**. Speedup over each runtime's
own plain baseline: ours **1.72–1.82×**, mlx-vlm **1.41–1.46×**.

**Both halves go in every external write-up.** Quoting the second without the first is the kind
of claim this project exists to avoid.

Scope limit, stated loudly: ollama was absent and LM Studio could not load the checkpoint
(version skew in its vendored mlx-vlm 0.6.5, not a bad checkpoint). **This is MLX-vs-MLX and says
nothing about llama.cpp/GGUF**, which is the stack most people run.

---

## 2. Where the project stands

| recipe | state | outcome |
|---|---|---|
| 01 harness + roofline | CLOSED | The instrument. Cancelled fusion / GEMV / KV recipes by measurement. |
| 02 baseline | CLOSED | Decode at roofline. Baseline 24.37 tok/s. `mx.async_eval` worth **+18 %** with no kernel work. Compute utilisation at batch 1 is **5–6 %**. |
| 03 MTP speculative decode | CLOSED | Losslessness settled; eight loop hypotheses and four mechanism explanations killed. |
| 04 losslessness write-up | **draft — publish** | Artifact "One ULP Apart". |
| 05 kernel headroom | CLOSED | Kernel headroom investigation. **Its #3553 framing is superseded by 12.** |
| 06 TBlink review | done | Six corrections, one fatal gate. |
| 07 TBlink v1 plan | superseded | Phases 0–2 as written. |
| 08 Phase 0 outcome | CLOSED | 1.74×. Host overhead **7.4 %, not 32 %** — premise refuted by its own instrumentation. |
| 09 Phase 1 + 2 outcome | CLOSED | Phase 1: acceptance +17.1 %, throughput +0 %. Phase 2: NO-GO on width. **Tier 1 finished.** |
| **11 cross-runtime** | **CLOSED** | The founding question. Prediction held on all three clauses. |
| **12 upstream contribution** | **draft, not posted** | `t(k)` is a staircase. #3553 framing refuted. Blocker: issue state unverified. |

**Honest framing, updated.** Recipe 03 measured an existing mechanism rather than optimising it.
What is ours: the characterisation, the harness, the roofline discipline, four well-instrumented
negatives that say where the time is *not* — **and now the speculative loop itself**, which is
the one thing that measurably beats the comparable implementation.

---

## 3. Where the remaining time is

At k=4 (γ=3) per cycle: forward **48.76 ms at 65 %** of its 31.71 ms floor (all 17.05 ms of
headroom), LM head 7.27 ms at roofline, host python 4.5 ms (7.4 %). Cycle 58.4 ms, emits 2.587
→ 42.5 predicted against 42.46 measured. **No bookkeeping left to reclaim.**

**The shape of the remaining lever, corrected.** `t(k)` is a **staircase, not a ramp**: marginals
+0.91, +1.88, **+4.91** (M=4), +5.65, +8.95, **+16.21** (M=6→7), then **+1.01** (M=7→8), cheap
again at k=10 and k≥12. **The headroom is a range, not a number**: k=8 is 2.41× the floor under
perfect stream/compute overlap and 1.01× under none, and this data cannot narrow it.

**Retracted and still retracted:** the ~32 % host-overhead claim (measured 7.4 %); wide
verification at M ≈ 11–16; the `tree_feasibility` GO verdict (its winner was a chain);
**and the "same dispatch-heuristic family as MLX #3553" framing** — #3553's step is at M=3 on the
superseded `qmv_fast` path, ours is at M=4 and M=6→7. That was a mechanism claim made in chat
without measurement, and the harness killed it.

---

## 4. Live targets, in expected-value order

1. **Unblock and route the upstream draft** (`reports/mlx-upstream-issue-draft.md` rev 3).
   Verify #3553's open/closed state via the GitHub API and re-verify `qmv_wide` in the pinned
   build, then post as a new issue or as a comment accordingly. **Push, do not fork** — see 12 §6.
2. **Descending + shuffled k sweep.** The staircase is currently measured ascending-only on a
   fanless part, and both reproducing sessions shared that ordering. Thermal bias is not
   excluded, and this is the Phase 0 confound again.
3. **Isolated reproducer** above the 32 MB threshold, k=1…16 — the first thing a maintainer asks
   for.
4. **Publish 04.**
5. **Fix the pooled-CV gate** (Recipe 11 §4): it measures the prompt difference, not steadiness.
6. **Long-context regime.** Everything sits below the 512-token sliding window, so the rotating
   cache has never wrapped. Both the losslessness band and the verify economics inherit that.

**Explicitly not pursued.** Writing Metal kernels. Porting DSpark / DFlash-2 / EAGLE-3 (mlx-vlm
ships all three). Switching models. Wide verification (measured, 09). Further γ scheduling
(measured, 09). Working around LM Studio's vendored engine (it would stop being LM Studio).

---

## 5. Standing rules

- **Never quote a per-token-sync number as throughput.** Every speedup names its baseline.
- **State a ceiling's regime before quoting it.** Nine invalid ceilings/comparisons so far — a
  memory-limited "compute ceiling", a sub-32 MB bandwidth probe, `getattr(mx,"__version__")`, an
  app-contention refusal, a battery-discharge drift check, a verdict averaged across a knee, a
  width recommendation that ignored the cost curve, a tree verdict whose winner was a chain, and
  a stale 97 % roofline. **Each caught by arithmetic, none by the code.**
- **Never size a phase against an isolated microbenchmark.** Phase 0's premise was wrong by 4×.
- **Run timed stages from a cool start, and randomise sweep order on a fanless part.**
- **A fused barrier hides what it fused.**
- **Check what the winning configuration actually is before writing the verdict string.**
- **Check a plan item's premise before executing it.** "Comment on #3553" was refuted on two
  grounds before a word was drafted.
- **Mechanism claims are not measurements.** Five killed so far — the most recent one was mine,
  made in chat.
- **Report the statistic that matches the question — including inside gates.** Recipe 11's
  pooled-CV gate was measuring the prompt difference, not steadiness.
- **Prove two runtimes are doing identical work before timing them.** mlx-vlm's processor does
  not apply the Gemma chat template where mlx-lm does; an id fingerprint caught it pre-flight.
- **Report a bound as a range when the data only supports a range.**
- **Contention inflates speculative ratios.** Quiet is conservative, not flattering.
- **Every diagnostic persists its verdict to `runs.jsonl`; the record is append-only.**
- **Losslessness is the ULP-band criterion, never bitwise identity.**

# Phase 0 — outcome (2026-08-30). Measured negative on throughput; premise refuted.

Run autonomously by Claude Code on the M3. Full report:
`reports/phase0-conclusion.md`. Machine record: `phase0_finding` in `runs.jsonl`.

## Result

| criterion | verdict | value |
|---|---|---|
| losslessness | **PASS** | within band, 3.12 % of positions |
| fork_validity A+B | **PASS** | tails are valid greedy continuations |
| acceptance | **PASS** | 1.587 (per-prompt 1.46 / 1.71) vs 1.55 threshold |
| **throughput** | **FAIL** | **42.46 tok/s = 1.74×** vs a 48.0 target |

Net gain from the two kept changes, thermally controlled: **+1.35 %** (41.57 → 42.46).

**Kept, both verified byte-identical via `spec_sha256` on both prompts:**
- **P0.1** — drafted ids stay on device; `draft()` returns a stacked `(γ,)` array, the verify
  input is built with `mx.concatenate`, preds and drafts fetched in one read. **γ+2 host
  round-trips per cycle → 1.**
- **P0.3** — the intermediate draft barrier removed; draft and verify fuse into one graph with
  one barrier.

## The premise was wrong — and so was my arithmetic

`tasks/PHASE0.md` budgeted the work against *"~32 % is host overhead, not kernels."* Measured
in-loop:

| phase | ms/cycle | |
|---|---|---|
| draft graph build | 0.8 | python |
| verify graph build | 3.7 | python, 42 layers |
| **GPU barrier** | **57.5** | **92.6 %** |
| tail — 24 `trim` + 42-layer KV walk | 0.1 | python |
| cycle | 62.1 | |

**Host overhead is 7.4 %.** Ceiling with *all* host python removed: 45.9 tok/s — still 2.1
short of 48. P0.4 was budgeted as "the per-cycle Python tax"; measured at 0.1 ms, there is no
tax.

**Where my 32 % came from.** I derived it from `verify_decomposition.py`'s isolated
`t(k=4) = 44.09 ms` — a pre-built cache at fixed ctx 128, the same filler token fed k times,
trim time subtracted. In-loop the same verify costs far more. **An isolated microbenchmark was
used as the loop's cost model.** That is the same class of error as quoting a ceiling without
its regime, and it set a target that was never reachable.

## Caveat on the diagnosis — the barrier now conflates two things

The report attributes the whole 57.5 ms barrier to streaming 3.535 GB of target weights and
concludes the verify runs at **61.5 GB/s = 69 %** of the 89 GB/s ceiling.

But **P0.3 fused draft and verify into one graph with one barrier** (`mx.eval(combo)`,
`spec_generate.py:298`). The drafter's GPU work — 4 layers × γ=3 steps, each attending over
~186 positions of the target's shared KV — is now *inside* that barrier and cannot be
separated. "Draft wall fell 3.8 → 0.8 ms" is graph-build time; the GPU work did not vanish, it
moved.

So 69 % understates verify efficiency by whatever the drafter costs on the GPU. Reconciling:
44.09 (isolated verify) + drafter GPU ≈ 57.5 implies the drafter is ~10 ms of GPU, well above
the ~4 ms its old wall time suggested.

**Why this matters:** Recipe 05 measured `mx.quantized_matmul` at M=4 running at **69.4 GB/s =
78 %** of the ceiling in isolation. If the in-loop verify alone is also ~78 %, two independent
measurements agree and the diagnosis is solid. If it stays at 69 %, there is a ~10 ms/cycle
residual the isolated benchmark does not capture — roughly 15 % of throughput sitting in an
unattributed bucket.

**One cheap measurement settles it:** time the drafter's GPU work alone with its own barrier,
outside the loop, and subtract. Do this before Phase 1 sizes anything against the cycle model.

## Two defects in my instrumentation, found by the agent

1. **The gate could not pass.** Its acceptance criterion regex-matched `mean accepted` from
   stdout and took the **first** occurrence — prompt 0's per-prompt 1.46 — while the 1.55
   threshold was calibrated against the cross-prompt summary, 1.59. It reported FAIL on
   unmodified code every run. Now reads `runs.jsonl`, as the throughput criterion already did.
   Per-prompt values recorded, since the mean hides a 1.46 / 1.71 spread.
2. **My staged-gate instructions caused thermal contamination.** Chaining `loss → fork → speed`
   back to back meant the speed stage inherited ~7 minutes of accumulated load and drooped in
   passes 4–5 (39.32 median, baseline arm 19.83). An identical-code control from a cool start
   gave 41.57 with baseline arm 20.32. **Uncontrolled, P0.1 would have been credited +6.9 %
   instead of its true +0.9 %.** The baseline arm — unchanged code in every run — is what
   exposed it.

The second is the better catch. It is the same failure the project already knew about
(fanless M3, cooldowns) reintroduced by my own harness design, and it was caught by an in-run
control rather than by suspicion.

## Standing rules, updated

- **Never size a phase against an isolated microbenchmark.** Instrument the loop.
- **Run timed stages from a cool start**, not chained after other stages. Add cooldown between
  gate stages, not only between passes.
- **A fused barrier hides what it fused.** Any optimisation that merges graphs must either keep
  a separable timing path or state what became unobservable.

## Next

1. **Split the barrier** (above) before Phase 1 uses the cycle model for anything.
2. **Phase 1 — γ re-optimisation** must use **in-loop** `t(k)`, not the isolated curve. The
   isolated curve is now known to understate in-loop verify by ~30 %, so the k=4 optimum should
   be re-derived rather than assumed.
3. The remaining headroom is the verify barrier's bandwidth efficiency — which is Recipe 05's
   finding and [MLX issue #3553](https://github.com/ml-explore/mlx/issues/3553), an open
   upstream issue. **That argues for commenting on #3553 with two independent confirmations,
   not for writing a local kernel.**

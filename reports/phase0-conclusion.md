# Phase 0 — conclusion: the 48 tok/s target is unreachable in Tier 1

**Result: 42.46 tok/s = 1.74x the declared 24.37 baseline.** Losslessness, fork_validity and
acceptance PASS. Throughput FAILS against the 48.0 target, and the measurement below is the
case that it cannot be reached without leaving Tier 1.

## The phase's premise was wrong

PHASE0.md budgeted the work against "~32 % is host overhead, not kernels." Timing the cycle on
both sides of the barrier instead of around the whole verify gives:

| phase | ms/cycle | what it is |
|---|---|---|
| draft graph build | 0.8 | python |
| verify graph build | 3.7 | python, 42 layers |
| **GPU barrier** | **57.5** | **92.6 % of the cycle** |
| tail — 24 `trim` + 42-layer KV walk | 0.1 | python |
| **cycle** | **62.1** | |

Host overhead is **7.4 %**, not 32 %. So:

    ceiling if ALL host python were removed = 42.46 / (1 - 0.074) = 45.9 tok/s
    shortfall at that impossible limit      = 2.1 tok/s

No arrangement of Tier-1 host optimisation reaches 48.0. P0.2 (`mx.compile`) can only attack
the 4.5 ms of graph build; its ceiling is 7.4 % and it cannot reach even that, because the
target's forward runs against a growing KV cache and would recompile per shape.

P0.4 was budgeted as "the per-cycle Python tax." Measured: **0.1 ms**. There is no tax.

## Where the time actually is

The 4-token verify barrier is 57.5 ms streaming 3.535 GB of active weights:

    61.5 GB/s = 69 % of the calibrated 89 GB/s ceiling
    vs 86.1 GB/s = 97 % for stock 1-token decode

Batching 4 tokens through the verify is *less* bandwidth-efficient than decoding one. That is a
kernel property, not a host one, and Phase 0 forbids addressing it. It is also the single
largest identified headroom in the loop: at roofline the verify would be 39.7 ms, not 57.5.

## What was kept

Both changes are value-neutral — verified by `spec_sha256` byte-identical on both prompts, with
identical accept histograms, so losslessness and fork_validity cannot have moved.

- **P0.1** — drafted ids stay on device. `draft()` returns a stacked `(gamma,)` array; the
  verify input is built with `mx.concatenate`; `preds` and drafts are fetched in one combined
  read. **gamma+2 host round-trips per cycle -> 1.**
- **P0.3** — the intermediate draft barrier is removed. The drafted ids are only ever gather
  indices, so nothing host-side needs them before the verify barrier; draft and verify now fuse
  into one graph with one barrier. Draft wall fell 3.8 -> 0.8 ms.

Net, normalised against the unchanged baseline arm: **+1.35 %** (41.57 -> 42.46 raw).

## Two measurement defects found and fixed

1. **The gate could not pass.** Its acceptance criterion `re.search`ed `mean accepted` from
   stdout and matched the FIRST occurrence — prompt 0's per-prompt 1.46 — while its 1.55
   threshold was calibrated against the cross-prompt summary, 1.59. It reported FAIL on
   unmodified code on every run. It now reads `runs.jsonl`, as the throughput criterion already
   did. Threshold unchanged; per-prompt values now recorded, since the mean hides a 1.46/1.71
   spread.

2. **The first before-state was thermally contaminated.** It chained loss -> fork -> speed back
   to back, so the speed stage inherited ~7 minutes of accumulated load and drooped in passes
   4-5 (39.32 median, baseline arm 19.83). An identical-code control from a cool start gave
   **41.57** with baseline arm 20.32. Uncontrolled, P0.1 would have been credited **+6.9 %**
   instead of its true **+0.9 %**. The baseline arm is unchanged code in every run and is the
   control that exposes this; the speedup ratio (2.04 -> 2.04) is thermally invariant and showed
   the same thing.

## Recommendation

Close Phase 0 as a measured negative on throughput. The remaining headroom is the verify
barrier at 69 % of roofline, which is Phase 2 / kernel territory, not Tier 1. Raising gamma is
also off the table here: CLAUDE.md's cost curve has t(k) rising faster than acceptance past
k=4, and PHASE0.md fixes gamma at 3.

---

# Phase 1 — confidence scheduling: acceptance rises, throughput does not

**Result: 42.46 tok/s — identical to fixed gamma=3.** Acceptance rose 1.587 -> 1.858 (+17 %).
Losslessness and fork_validity PASS at gamma_max=5, thr=0.70.

The calibration was a genuine GO: P(accept) runs 0.333 in the [0.00,0.50) confidence bucket to
0.974 in [0.99,1.00), a spread of 0.641 against a break-even q* of 0.422. The drafter does know
when it is about to be rejected. The signal is real; the throughput is not.

## Why a 17 % acceptance gain converts to 0 %

| config | tok/s | baseline arm | spec/base |
|---|---|---|---|
| fixed gamma=3 | 42.46 | 20.48 | 2.073 |
| schedule gamma_max=4, thr 0.70 | 42.37 | 20.32 | 2.085 |
| schedule gamma_max=5, thr 0.70 | 42.46 | 20.19 | 2.103 |

Two costs absorb it:

1. **Single-barrier truncation pays every drafter step.** At gamma_max=5 that is 5 x 1.338 =
   6.98 ms against ~4.3 ms at fixed gamma=3.
2. **The truncation decision needs the confidences on the host**, which reinstates the barrier
   P0.3 removed — draft no longer fuses into verify.

And mean drafted settles at ~3.0, so k ~ 4.0: the *same verify cost* as fixed gamma=3, with
extra draft steps and a barrier added on top.

The **adaptive** variant (stop the drafter early) was measured and rejected: it saves drafter
steps but costs ~3.7 barriers/cycle at ~0.8 ms each, and came in slower than fixed gamma=3 even
on cycles that both accepted more (1.71 vs 1.46) and used fewer verify slots (k=3.66 vs 4).

## The simulation was misleading, and why

`confidence_calibration`'s sweep predicted 47.06 tok/s at 1.549x — but its baseline was **fixed
gamma=8** (30.39 tok/s), an operating point nobody runs. Against the k=4 optimum the schedule
has almost no room, which is exactly what the cost curve already implied: at the optimum,
shortening the block moves you below it. A ratio is only as good as its denominator.

## Kept

`--conf-threshold`, **default 0.0** = fixed gamma, verified byte-identical to pre-Phase-1 output.
Off by default because it does not pay; retained because the acceptance result (1.86 at
gamma_max=5) is the input Phase 2 needs — a tree exploits exactly the high-confidence branches
this calibration located.

## What is left

The transformer forward runs at **65 % of its 31.71 ms bandwidth floor — 17 ms/cycle**. That is
now the only large headroom in the loop, and it is kernel work.

---

# Phase 2 — the tree: width never pays

**NO-GO.** The best widened shape is **+20 % worse** than the best chain, with the cost model
deliberately biased in the tree's favour.

Gated before any tree code existed, the same way confidence_calibration gated Phase 1 -- and for
the reason CLAUDE.md already records: a tree-width recommendation that ignored the cost curve has
been killed here once before.

## Coverage — measured, teacher-forced

P_W[d] = P(target's actual token is in the drafter's top-W | the prefix up to d is correct).
Teacher forcing is what makes the depth-d numbers conditional on a correct prefix, which is the
only regime a tree operates in.

| depth | W=1 | W=2 | W=4 | W=8 | W=16 |
|---|---|---|---|---|---|
| 1 | 0.750 | 0.851 | 0.909 | 0.945 | 0.958 |
| 3 | 0.724 | 0.821 | 0.886 | 0.932 | 0.948 |
| 6 | 0.666 | 0.776 | 0.847 | 0.896 | 0.929 |

Width really does capture more of the target's behaviour. That was never the question.

## Economics — why it still loses

| shape | k | E[acc] | ms/token |
|---|---|---|---|
| (1,1,1,1) chain | 4 | 1.982 | **20.57** |
| (2,1) | 4 | 1.478 | 24.76 |
| (2,1,1,1) | 8 | 2.248 | 31.19 |
| (4,1) | 8 | 1.579 | 39.28 |
| (2,2) | 6 | 1.552 | 32.37 |

Every branch must be scored, so k grows multiplicatively while coverage grows by fractions.
Doubling k from 4 to 8 costs **+67 %** on t(k) (51.4 -> 86.0 ms) and buys **+0.27** expected
accepted tokens. The superlinear region of the verify curve is exactly where a tree wants to
live, and it cannot afford the rent.

The best chain lands at depth 3-4, independently reproducing the k=4 optimum the recipe found by
a different route.

## Caveats, stated

The shape model charges t(k) for k DRAFTED nodes, while the loop scores k+1 positions (the
current token rides along). That off-by-one understates cost for every shape, chains and trees
alike, so the width verdict is unaffected -- but it is why the model predicts 48.6 tok/s for the
depth-3 chain where the loop measures 42.46. Correcting it gives ~45.0, and the sweep's remaining
~4 ms optimism covers the rest.

# mlx-bench — project memory

Speculative-decoding research harness for MLX on Apple Silicon. **You can execute here:**
run everything as `../bin/python <script>` from this directory (the venv is one level up).

## Division of labour — read this before deciding anything

This project runs on two surfaces and they are **not interchangeable**:

- **Cowork chat (the human + Claude there)** owns *interpretation*: objectives, what a result
  means, what counts as met, what gets measured next, and every change to this file's
  "Current work order". Reasoning about direction happens there.
- **You, Claude Code, here** own *execution*: writing scripts, running them, recording rows in
  `runs.jsonl`, and reporting what the numbers say. **You do not redefine the objective, do not
  substitute a different target when one proves hard, and do not open a new phase.**

If a measurement contradicts the work order — which has happened in all three phases so far —
that is a **result to report**, not a licence to re-scope. Write the finding, persist it,
say plainly that the order's premise is refuted, and stop. The re-scope decision is made in
Cowork chat and comes back to you as an updated work order.

**Never modify the harness to make a target reachable.** Three phases have closed as measured
negatives and the project is stronger for each one.

## HARD CONSTRAINTS — violating these invalidates the whole project

1. **NEVER upgrade `mlx` or `mlx-lm` in `~/mlx-env`.** Pinned at mlx 0.32.2 / mlx-lm 0.31.3 /
   transformers 5.16.1 / Python 3.12.14. Every row in `runs.jsonl` was measured against them.
   Anything needing a different version goes in a separate venv (see `setup_control.sh`).
2. **Never edit `runs.jsonl` except by appending.** It is the measurement record.
3. **Do not change `gemma4_assistant.py`'s numerics** without re-running the losslessness gate.

## Machine and models

```
MacBook Air M3 · 4P+4E CPU · 10 GPU cores · 16 GB unified · FANLESS
Target   mlx-community/gemma-4-e4b-it-OptiQ-4bit
Drafter  mlx-community/gemma-4-E4B-it-assistant-bf16
```

Gemma needs `<bos>`; without it output degenerates into repetition that looks like a broken
checkpoint. Stock mlx-lm 4-bit Gemma 4 conversions quantise the per-layer embeddings to 4 bits
and are unusable — OptiQ keeps them at 8. Screen with `check_model.py`.

## Measured ground — do not re-derive, do not contradict without new measurement

| fact | value |
|---|---|
| Calibrated bandwidth ceiling | **89 GB/s** usable (spec sheet says 100 — do not use 100) |
| Active bytes per decode token | **3.535 GB**; 46 % of the checkpoint is gather-only PLE that never streams |
| Decode vs roofline | **93.5-96.0 %** (Recipe 11, controlled session; the older 97 % is stale) — batch-1 decode is finished; kernels cannot help there |
| **Cross-runtime, plain decode** | mlx-lm 23.56/24.17 vs **mlx-vlm 0.6.17 23.54 tok/s** — byte-identical weights, independently derived bytes, **0.1 % apart**. NO kernel advantage. |
| **Cross-runtime, speculative** | ours 42.94/41.62 vs mlx-vlm **34.39** (its default block) = **+22.9 % ours**. Per-runtime speedup over own plain: ours **1.72-1.82x**, mlx-vlm **1.41-1.46x**. |
| **DECLARED BASELINE** | **24.37 tok/s**, stock `mlx_lm.stream_generate` |
| per-token-sync baseline | 20.03 tok/s — **a latency instrument, never quote as throughput** |
| pipelined baseline | 23.60 tok/s (`mx.async_eval`, +18 % over per-token, zero kernel work) |
| Current speculative | **42.46 tok/s = 1.74×** the declared baseline (was 40.40) |
| Verify cost **(FORWARD ONLY — excludes the LM head)** | `t(k)`: k=1 36.0 ms · k=4 44.1 · k=9 92.2 · saturates flat k=11→16 at ~111 ms |
| **Full verify at k=4** (what the loop pays) | **56.0 ms** = forward 48.8 + LM head 7.3 (ctx 186) |
| Verify vs its floor | floor 39.72 = **forward 31.71 + head 8.01**. Forward runs at **65 %**, head at **~110 % (at roofline)**. All 17 ms of headroom is the forward. |
| Acceptance | 1.59 at γ=3, 2.29 at γ=8. **30–33 % of cycles accept ZERO.** |
| Optimum | **k=4 (γ=3)** — 17.0 ms/token predicted. Cost rises faster than acceptance past it. |
| GPU compute utilisation at batch 1 | 5–6 % |

**`verify_decomposition.py`'s `ratio` column is apples-to-oranges** — it divides a FORWARD-ONLY
time by a floor that includes the LM head, so k=4 reads 1.110 when forward-vs-forward-floor is
1.39. `verify_inloop_gap.py` measures the split. The loop's own ingredients (KV-capture wrapper,
plain KV caches, shared_kv slicing) cost **0.02–0.80 ms total** — they are not the gap.

**The Phase 0 gap — REFUTED BY MEASUREMENT (2026-08-30).** The claim below was that k=4
predicts 44.09 ms / 2.59 emitted = 17.0 ms/token = 58.8 tok/s against a measured 40.4, and that
"~32 % is host overhead, not kernels." **It is not.** Timing the cycle either side of the
barrier (`spec_generate` now reports build/wait/tail) gives, per cycle:

| phase | ms | |
|---|---|---|
| draft graph build | 0.8 | python |
| verify graph build | 3.7 | python |
| **GPU barrier** | **57.5** | **92.6 % of the cycle** |
| tail (24 `trim` + 42-layer KV walk) | 0.1 | python — P0.4 is free, not a tax |
| cycle | 62.1 | |

Host overhead is **7.4 %, not 32 %**. Driving ALL host python to zero yields **45.9 tok/s** —
still **2.1 short of the 48.0 target**. 48 is unreachable in Tier 1; the gap is the barrier.
The 4-token verify streams at **61.5 GB/s = 69 %** of the 89 GB/s ceiling, against **97 %** for
stock 1-token decode. Batching 4 tokens through the verify is *less* bandwidth-efficient than
decoding one — that is a kernel property, and it is where the remaining time is.

## Losslessness — the criterion is NOT bitwise identity

Bitwise identity with sequential greedy decoding is **unachievable on this hardware** and must
never be an acceptance gate. Measured: the transformer forward is shape-dependent — the same
token in the same context produces a bitwise-different hidden state depending on how many
tokens shared its forward pass (96 of 96 tested cells, entering at decoder layer 0–1, worth
3–9 bf16 ULP at the logits). Where the target's top-2 margin is under ~3 ULP the argmax flips:
**3.12 % of positions at risk, 0.63 % actually flip.**

**The gate is:** every emitted token is one the target itself produced, AND every emitted tail
is a valid greedy continuation of its own prefix (`fork_validity.py` parts A and B, both
directions). Divergence permitted only below **3 bf16 ULP** — `spec_generate.py --tie-ulps 3.0`.

## Standing rules — each of these was learned by getting it wrong

- **Every speedup names its baseline.** Against 24.37, not against 20.03.
- **State a ceiling's regime before quoting it.** Seven invalid ceilings so far: a
  memory-limited "compute ceiling", a sub-32 MB bandwidth probe, `getattr(mx,"__version__")`
  (mlx defines no such attribute — it wrote null into every runs.jsonl row for weeks), an
  app-contention refusal, a battery-discharge drift check, a verdict averaged across a knee,
  and a tree-width recommendation that ignored the cost curve.
- **Any GEMV/matmul bandwidth probe under ~32 MB measures dispatch, not DRAM.** Same kernel:
  35 GB/s at 8.8 MB, 90 GB/s at 566 MB.
- **Report the statistic that matches the question.** `max|Δ|` over a vocabulary is one-sided
  and attained at the wrong token; argmax flips need `Δ(top1) − Δ(top2)`.
- **Contention inflates speculative ratios** — the baseline is bandwidth-bound and suffers more
  than the speculative arm. A quiet machine is the conservative measurement.
- **Fanless M3:** warmup discarded, 25–40 s cooldown, medians of ≥5, temperature 0.
  Gate on Low Power Mode and <30 % charge — not on being plugged in. Ordinary battery
  discharge is not host-state drift.
- **A gate that cries wolf is worse than no gate.** Every new preflight check needs a stated
  benign case it must NOT fire on.
- **mlx's `nn.Module` subclasses `dict`** — `vars(m)` is empty. Walk `tree_flatten(model.parameters())`.
- **Mechanism claims are not measurements.** Do not write an explanation into a doc before
  measuring it; four have been killed that way already.

## Files

`mlxutil.py` shims/probes · `roofline.py` byte split from the real parameter tree ·
`bwprobe.py` calibrated GB/s · `bench.py` Recipe-0 harness · `check_model.py` checkpoint screen
`gemma4_assistant.py` **the drafter** (install into `mlx_lm/models/`) · `measure_acceptance.py`
`spec_generate.py` the speculative loop + losslessness · `speed_5pass.py` the publishable speed
`fork_validity.py` the losslessness proof · `schedule_decision.py` **never run** ·
`verify_decomposition.py` kernel headroom · `runs.jsonl` every measurement

## Running under the Claude desktop app

**Claude Desktop being open is EXPECTED, not contamination.** Recipe 02's declared 24.37
baseline was itself measured with it resident (it appears in 30/30 baseline rows; Chrome
appears in 0 of 262). Measuring without it would compare a quieter speculative run against a
Claude-resident baseline and inflate the ratio — exactly the asymmetry we measured, since
contention hurts the bandwidth-bound baseline more than the speculative arm.

**Quit Chrome** (quit, not just close the window) before any timed run. That one is a hard
refusal in `speed_5pass.py`'s preflight.

**Commands time out; the gate is staged.** A full gate loads the model three times and runs
one ~6-minute benchmark. Run one stage per command:

```
../bin/python phase0_gate.py --stage loss      # ~1.5 min
../bin/python phase0_gate.py --stage fork      # ~4 min
../bin/python phase0_gate.py --stage speed     # ~6 min
../bin/python phase0_gate.py --stage verdict   # instant — reads the persisted stages
```

Each stage persists to `runs.jsonl`; `verdict` reads them back. Nothing is recomputed. If a
stage times out anyway, raise the command timeout rather than reducing `--passes` — fewer than
5 passes is not a median under this project's rules.

## Declared result (accepted 2026-08-30)

**42.46 tok/s = 1.74× the declared 24.37 baseline**, lossless within the 3.0 ULP band,
`fork_validity` A+B pass. Phase 0's 2.0× target is **accepted as not met**: it was set from a
cost model now known to be understated (t(k) forward-only, compared against a floor that
includes the LM head). The corrected model predicts 42.5 and the loop measures 42.46.

## Phase 1 & 2 — CLOSED 2026-09-02, both measured negatives

**Phase 1 (confidence-scheduled gamma).** Signal is real: P(accept) 0.333 in [0.00,0.50) ->
0.974 in [0.99,1.00), break-even q*=0.422. Acceptance **1.587 -> 1.858 (+17.1 %)**.
Throughput **42.46 -> 42.46**. Net machine-normalised +1.4 %, inside M3 noise.
Two costs absorb the gain exactly: (1) single-barrier truncation must still run ALL gamma_max
drafter steps (5 x 1.338 = 6.98 ms vs ~4.3 at fixed gamma=3); (2) the truncation decision needs
the confidences on the HOST, reinstating the barrier P0.3 removed. Mean drafted ~3.0 so k~4.0 —
the SAME verify cost — plus drafter cost plus a barrier.
The adaptive variant (stop the drafter early) is **slower**: ~3.7 barriers/cycle at ~0.8 ms.
`confidence_calibration.py`'s 47.06 tok/s prediction was against **fixed gamma=8** (30.39 tok/s),
an operating point nobody runs. At the k=4 optimum, shortening the block moves you below it.
**Kept:** `--conf-threshold`, DEFAULT 0.0 = fixed gamma, byte-identical to pre-Phase-1.

**Phase 2 (the tree) — NO-GO on width.** Feasibility measured BEFORE implementation: coverage
at 308 real draft positions, depths 1-6, widths 1/2/4/8/16, priced with the measured cost model
(lm_head 7.3 ms, draft 1.338 ms/node, host 4.6 ms) **deliberately biased toward the tree**.
Coverage does rise with width (depth 1: 0.750 W1 -> 0.958 W16) and it still loses:

| shape | ms/token | tok/s | k | E[acc] |
|---|---|---|---|---|
| best chain (1,1,1,1) | 20.57 | 48.62 | 4 | 1.982 |
| best widened (2,1) | 24.76 | 40.39 | 4 | 1.478 |

**Width penalty -20.4 %.** k=4 costs 51.4 ms, k=8 costs 86.0 ms (+67 %) to buy +0.27 expected
accepted tokens. The optimum is a **chain at depth 3-4** — an independent reproduction of the
k=4 optimum by a different method.
**Correction of record:** the first `tree_feasibility` row printed GO because the best shape beat
the operating point — but that shape WAS A CHAIN. `tree_feasibility_corrected` supersedes it.

**Tier 1 is finished. 1.74x is the number.** Phases 0, 1 and 2 were the whole Tier 1 programme;
none moved throughput. The loop runs at ~97 % of what its own cost model allows. The only lever
left is the verify kernel's bandwidth efficiency (69 % of ceiling at k=4 vs 97 % for 1-token
decode) — same dispatch-heuristic family as MLX #3553, and Tier 2.

## Current work order — none open

Recipe 10 (cross-runtime) is **CLOSED**; see `recipes/11-cross-runtime-comparison.md` in the
Claude project and the `crossruntime_finding` row (20260902-202512). The pre-registered
prediction **held on all three clauses**.

**Plan item "comment on MLX #3553" is REFUTED** (`issue3553_premise_check` 20260902-210128) and
replaced by `reports/mlx-upstream-issue-draft.md` **revision 3**, which is NOT posted and carries
a stated blocker at the top: **#3553's open/closed state is unverified** — three HTML fetches say
Open, the row that says Closed could not be independently confirmed, and the GitHub API was
unreachable from both checking environments. **Settle that before posting anything.**

Outstanding, in order — none of these is authorised to start without a Cowork work order:

1. **Verify #3553's state** via the GitHub API (command is in the draft), then route the draft:
   new issue if closed, comment on #3553 if open.
2. **Re-verify `qmv_wide`** in the pinned build's `mlx.metallib`. Revision 2's framing rests on
   it and it could not be checked from the review side (`~/mlx-env` is outside the mounted folder).
3. **Descending + shuffled k sweep.** The current sweep is ascending-only on a fanless part and
   both reproducing sessions shared that ordering, so thermal bias is not excluded. Same class as
   the Phase 0 contamination that caused a 7x attribution error.
4. **Isolated reproducer** for the upstream draft, sized above the 32 MB threshold, k=1..16.
5. **Fix the pooled-CV gate.** It flagged 5.5 % as "not publishable" by pooling two prompts whose
   speculative rates genuinely differ (p0 ~40.6, p1 ~44.6 tok/s), so it was measuring the PROMPT
   DIFFERENCE, not steadiness. Per-prompt CV is <=2.1 % everywhere. Statistic did not match the
   question. No median changes.
6. **Publish `recipes/04-losslessness-writeup.md`.**
7. **Long contexts** — everything measured sits below the 512-token sliding window.

**A well-measured failure closes a phase honestly; drifting into Metal to force a pass does not.**
Three phases closed that way, and Recipe 10 answered the project's founding question. Keep doing that.

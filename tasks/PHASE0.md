# Phase 0 — reclaim the host overhead

**Goal:** raise the speculative loop from 40.4 to **≥48 tok/s** without changing the
algorithm, without touching Metal, and without breaking losslessness.

**Why this first:** at the measured operating point k=4 (γ=3), verify costs 44.09 ms and the
loop emits 2.59 tokens per cycle — 17.0 ms/token, i.e. **58.8 tok/s predicted**. The loop
measures 40.4. The missing ~32 % is host-side: graph construction, host round-trips, Python
per-cycle work. Recipe 02 already measured the precedent — `mx.async_eval` alone took the
plain decode baseline from 20.03 to 23.60 tok/s, **+18 %, zero kernel changes**.

If Phase 0 alone reaches ~48 tok/s that is **2.0× the declared 24.37 baseline**, which clears
the original project target before any new mechanism is built.

---

## Candidate optimisations, in expected-value order

These are hypotheses, not instructions. Measure each independently; keep what pays.

### P0.1 — stop round-tripping draft tokens through Python ints

`measure_acceptance.draft()` ends with:

```python
mx.eval(toks)                       # one barrier, good
return [int(t.item()) for t in toks] # ...then gamma host reads
```

and `spec_generate.spec_decode()` then does:

```python
xv = mx.array([[int(cur.item())] + drafts])   # another host read, then a rebuild
```

That is **γ+1 device→host reads per cycle** to construct a tensor that could stay on device.
Build `xv` with `mx.concatenate` from the arrays already in `toks`. The accepted-token compare
still needs one host read of `preds`, which is unavoidable — but it should be the *only* one.

Numerically identical (same values, same order), so losslessness cannot change. Verify anyway.

### P0.2 — `mx.compile` the draft step

Drafting runs at **118–120 % CPU** (`process_time` exceeds wall — graph construction is
multithreaded and still the bottleneck). Draft is 4.4–5.3 ms of a ~60 ms cycle, so the ceiling
on this alone is ≤9 %; it is worth doing because it is the phase most exposed to host cost.

`mx.compile` can change numerics. Run the losslessness gate after.

### P0.3 — overlap the next cycle's graph build with the current verify

`mx.async_eval` on the verify forward so the drafter's graph for cycle N+1 is constructed while
cycle N's verify is still on the GPU. This is the direct analogue of what gave Recipe 02 +18 %.

### P0.4 — the per-cycle Python tax

Per cycle the loop does: 24 `c.trim(n)` calls, a `collect_shared_kv` walk over 42 layers, a
`slice_shared_kv` that allocates new arrays, and a Python accept loop. Profile before
optimising — this may be noise next to P0.1–P0.3, or it may be most of the remainder.

---

## Method

1. **Measure the current state first.** `../bin/python phase0_gate.py --baseline-only` writes a
   reference row. Do not skip this; you need a same-session before/after.
2. Change **one** thing. Run the gate. Record the result.
3. If it does not pay, revert it. A change that does not move the median is not neutral — it is
   surface area.
4. Keep going until the gate prints ALL PASS or you run out of candidates.

## Gate — staged, because a full run exceeds a command timeout

```
../bin/python phase0_gate.py --stage loss      # ~1.5 min  losslessness + accept rate
../bin/python phase0_gate.py --stage fork      # ~4 min    fork_validity A+B
../bin/python phase0_gate.py --stage speed     # ~6 min    median of 5
../bin/python phase0_gate.py --stage verdict   # instant   reads the persisted stages
```

After a change that cannot affect numerics (P0.1 is one — same values, same order), `loss` and
`fork` do not need re-running every iteration; `speed` then `verdict` is enough. After
`mx.compile` (P0.2), which **can** change numerics, run all four.

It checks three things and prints PASS/FAIL for each:

| criterion | threshold |
|---|---|
| **Losslessness** | every divergence below 3.0 bf16 ULP, and `fork_validity` A+B pass |
| **Throughput** | median of 5 ≥ **48.0 tok/s** (2.0× the declared 24.37 baseline) |
| **No regression** | mean accepted ≥ 1.55 at γ=3 (currently 1.59) — catches a "speedup" that quietly drafts worse |

Every run appends to `runs.jsonl` and writes `reports/phase0-<timestamp>.md`.

## Rules for this phase

- **No Metal.** No `mx.fast.metal_kernel`. Tier 1 only.
- **No algorithm change.** γ stays 3, the drafter is untouched, the accept rule is untouched.
  Confidence scheduling is Phase 1; the tree is Phase 2.
- **Do not optimise the baseline.** `greedy_baseline()` is a reference, not a target. Making it
  faster invalidates the comparison.
- **Quit Chrome before any timed run** (quit, not just close the window). Contention inflates
  speculative ratios by hurting the bandwidth-bound baseline more than the speculative arm.
- **Claude Desktop staying open is fine and expected** — the declared baseline was measured
  with it resident. Do not quit it "to be safe"; that would make the comparison inconsistent.
- If the gate's throughput criterion proves unreachable, **say so with the measurement** and
  stop. A well-measured "the remaining overhead is irreducible in Tier 1" is a real result and
  closes the phase honestly. Do not drift into Metal to force a pass.

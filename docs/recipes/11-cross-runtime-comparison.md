# Recipe 11 — cross-runtime comparison (Recipe 10 work order)

**Status: CLOSED.** Session `r10-20260902-200543`. Pre-registered prediction **HELD on all three
clauses**. Records: `crossruntime_prediction` (20260902-195147), 62 × `crossruntime_run`,
3 × `crossruntime_summary`, `crossruntime_finding` (20260902-202512).

This is the question the project was built on, asked for the first time with an instrument good
enough to answer it.

---

## 1. The verdict, in one line

**We have no kernel-quality advantage and never did — and the harness is still not "the same
number with more ceremony," because the speculative loop is measurably better than the best
comparable MLX implementation.**

---

## 2. What was measured

Byte-identical weights (`mlx-community/gemma-4-e4b-it-OptiQ-4bit`), the same drafter
architecture (`gemma4_assistant` → mlx-vlm `draft_kind='mtp'`), one session, 45 s between
runtimes, ours measured **on both sides of** mlx-vlm as a drift bracket.

| runtime | plain decode | % of 89 GB/s | speculative | speedup over its **own** plain |
|---|---|---|---|---|
| ours (mlx-lm 0.31.3 + this harness) | **23.56** / 24.17 tok/s | 93.58 / 96.02 | **42.94** / 41.62 | **1.72–1.82×** |
| mlx-vlm 0.6.17 | **23.54** tok/s | 93.49 | 34.39 (its default block) · 33.30 (matched γ) | **1.41–1.46×** |

- **Plain decode agrees to 0.1 %.** Two independent implementations, independently derived
  active-bytes (3,535,067,220 per token, computed inside each venv by the same
  `roofline.param_breakdown`), both at 93.5–96.0 % of the calibrated ceiling.
- **Speculative differs by +22.9 %** in our favour at mlx-vlm's own default block size.
- The session **independently brackets both declared numbers** — plain 23.56/24.17 around the
  declared 24.37, speculative 42.94/41.62 around the declared 42.46.

**Byte reporting.** Our speculative arm is reported on an *amortised* basis —
`(target active + drafted × drafter active) / (accepted + 1)` = 1.549 GB per emitted token,
66.5 GB/s, 74.7 % of ceiling. mlx-vlm's speculative bytes are recorded **"NOT DERIVABLE"**: its
generator yields `(token, logprobs)` only and exposes no accepted/drafted counters, so traffic
per emitted token cannot be derived. Reported as not derivable rather than estimated — the plain
3.535 GB figure would have been wrong there.

---

## 3. Prediction status

| clause | status |
|---|---|
| 1 — plain decode converges once normalised by measured bytes/token | **HELD** (0.1 % apart) |
| 2 — our speedup is the loop, not the kernels | **HELD** (kernels identical; loops 22.9 % apart) |
| 3 — falsifier: a runtime beats us on normalised bytes/token | **NOT TRIGGERED** |

Registered before the first timed run, as the work order required.

---

## 4. Two hazards caught before they became results

**The tokeniser hazard.** mlx-vlm's processor does **not** apply the Gemma chat template
(11 ids) where mlx-lm does (26 ids). Feeding each runtime "the same prompt string" would have
compared **different work** and the difference would have been invisible in the output. Caught
by a shared `crossruntime_ids.json` id fingerprint before any timed run; both runtimes were then
fed the identical chat-template ids (prompt sha `0419483c336c009b` / `3d4f237fa066ea2a`, 160
tokens each).

**The pooled-CV gate is wrong.** The harness flagged 5.5 % CV (our speculative arm, block 1) as
"not publishable." That is a gate artefact, not instability: it pools two prompts whose
speculative rates genuinely differ (p0 ≈ 40.6, p1 ≈ 44.6 tok/s), so the pooled CV measures the
**prompt difference**, not steadiness. Per-prompt CV is ≤ 2.1 % everywhere. **The statistic did
not match the question** — the project's own recurring failure, this time inside a gate. Fix the
gate; no median changes.

---

## 5. What could not be measured, and why it matters

- **ollama: ABSENT.** Not on PATH, no `/Applications/Ollama.app`, no `~/.ollama`, nothing in
  `/opt/homebrew/bin`. Not installed — approval not sought.
- **LM Studio 0.4.21+2: PRESENT, cannot load the checkpoint.** Zero LLMs on disk (only a bundled
  embedder). The OptiQ 4-bit was hardlinked into `~/.lmstudio/models/` at zero extra disk and
  **LM Studio indexed it** (arch `gemma4`, 6.56 GB) — then its MLX engine failed to load it.
  Cause: LM Studio vendors **mlx-vlm 0.6.5**, whose Gemma-4 vision/audio tower definitions demand
  **1411 parameters the OptiQ checkpoint does not carry** (quantised-activation
  `input_max`/`input_min`/`linear.weight`/`output_max`/`output_min` families), loaded with
  `strict=True`. Our control venv's **mlx-vlm 0.6.17 loads the identical files without error** —
  so this is **version skew in LM Studio's vendored engine, not a bad checkpoint**. Not worked
  around: editing LM Studio's vendored engine would stop it being LM Studio as shipped.

**So "cross-runtime" here means MLX-vs-MLX.** This recipe says nothing about llama.cpp/GGUF
kernel quality — which is the stack most people actually run. State that wherever the +22.9 % is
quoted.

---

## 6. Honest limits on the +22.9 %

1. **mlx-vlm's acceptance is not exposed**, so its deficit cannot be split between *lower
   acceptance* and *higher per-cycle cost*. That is the obvious next measurement, and until it
   exists we know the gap but not its cause.
2. **mlx-vlm's mtp path is general-purpose** across many drafter families; ours is specialised to
   this target/drafter pair. The +22.9 % is a measured fact about the two implementations **as
   shipped**, not a claim that mlx-vlm is poorly written.
3. **Our speculative arm includes prefill** in its wall clock (`spec_decode` returns its tokens
   in one batch) while every other arm excludes it. On a 26-token prompt this **understates** our
   speculative arm by ~1 % — conservative direction, stated rather than corrected.
4. Machine drift across the session was ≤ 3.1 %, **in opposite directions per arm** (plain
   +2.61 %, speculative −3.08 %). mlx-vlm sits between the two "ours" blocks, so it is bracketed
   rather than compared against a single stale figure.

---

## 7. What this does to the project's story

Recipe 03's honest framing said what was ours was "the characterisation, the harness, the
roofline discipline." **Recipe 11 adds one more: the loop itself.** Against the only comparable
MLX speculative implementation on this machine, on the same weights and the same drafter, ours
emits 22.9 % more tokens per second — 1.72–1.82× over its own baseline against 1.41–1.46×.

It simultaneously closes the door on the other half. Plain decode is **the same number** across
implementations to within a tenth of a percent. There is no kernel advantage to claim, to build,
or to market. Any remaining gain is in MLX, and MLX is shared by everyone.

Both halves belong in any external write-up. Quoting the first without the second is the kind of
claim this project exists to avoid.

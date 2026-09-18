# Gigatok

Measured speculative-decoding research for LLM inference on Apple Silicon (MLX). Every number
in this repo comes from a benchmark you can re-run, appended to an append-only log
(`runs.jsonl`), not from a slide.

```
Target     mlx-community/gemma-4-e4b-it-OptiQ-4bit
Drafter    mlx-community/gemma-4-E4B-it-assistant-bf16  (Gemma 4's own MTP head)
Machine    MacBook Air M3, 16 GB, fanless · mlx 0.32.2 / mlx-lm 0.31.3
Baseline   24.37 tok/s (stock mlx_lm.stream_generate)
Result     42.46 tok/s = 1.74x, lossless within a 3-ULP band
```

## What this is

Batch-1 decode on Apple Silicon is memory-bandwidth bound: throughput is roughly
`bandwidth ÷ bytes of active weight read per token`. Once decode is already near that ceiling —
and on this machine it measures at 93.5–96.0 % of it — the only way to go faster is to move
**fewer bytes per emitted token**. That's speculative decoding: a small drafter proposes several
tokens, the target model verifies them all in one forward pass, and every accepted token was
"free" in the sense that it didn't cost its own full memory read.

This repo builds and measures that loop using the target model's **own shipped multi-token-
prediction (MTP) head** as the drafter — no custom model, no training — and treats every claim as
something to verify, not assume. Three optimization phases were tried and closed as measured
negatives; the reasons they failed are as much the finding as the 1.74× that stuck.

## Headline results

| claim | value | where |
|---|---|---|
| Calibrated memory-bandwidth ceiling | **89 GB/s** usable (spec sheet says 100 — don't use 100) | `docs/recipes/01-harness-and-roofline.md` |
| Plain decode vs roofline | **93.5–96.0 %** | `docs/recipes/02-baseline-measurement.md`, `11-cross-runtime-comparison.md` |
| Declared speculative result | **42.46 tok/s = 1.74×** the declared baseline | `docs/recipes/08-phase0-outcome.md` |
| Losslessness | Every emitted token is one the target itself produced; **not** bitwise-identical to sequential greedy decode. Divergence only below a 3-bfloat16-ULP margin, at 3.12 % of positions, 0.63 % actually flip | `docs/recipes/04-losslessness-writeup.md` |
| Cross-runtime, plain decode | Byte-identical weights, independently written decode loop: ours vs **mlx-vlm 0.6.17**, **0.1 % apart**. No kernel advantage — and never was | `docs/recipes/11-cross-runtime-comparison.md` |
| Cross-runtime, speculative | Our loop is **+22.9 %** faster than mlx-vlm's comparable speculative path on the same weights and drafter | `docs/recipes/11-cross-runtime-comparison.md` |
| Kernel headroom | Independently reproduced an open MLX dispatch-heuristic issue (`ml-explore/mlx#3553`) on different hardware and a different model; found the real shape is a staircase, not a ramp | `docs/recipes/05-quantized-matmul-headroom.md`, `12-upstream-contribution.md` |

Full index, with every number and its source measurement: **`docs/recipes/00-INDEX.md`**.

## Why this might be useful to you

- **A measurement harness for MLX**, not just a demo: bandwidth calibration that separates
  dispatch overhead from real DRAM traffic, a byte-accurate roofline built from the model's real
  parameter tree (not a naive parameter count — for this model that's wrong by 85 %), and a
  losslessness criterion for speculative decoding stated in bfloat16 ULPs instead of an
  unachievable bitwise-identity bar.
- **A working speculative-decoding loop on MLX** using a shipped MTP drafter head, with the
  verification and acceptance bookkeeping needed to keep it lossless.
- **A record of what didn't work and why** — decode fusion, GEMV kernel tuning, KV-cache
  precision changes, confidence-scheduled drafting, and tree-shaped verification were all tried
  and closed as measured negatives. Each closure states the number that killed it.

## Quick start

```bash
git clone https://github.com/dasocreative/Gigatok.git
cd Gigatok

# create/point a venv at MLX (pin to the versions above — the whole harness is calibrated on them)
VENV=/path/to/your/mlx-venv ./setup.sh

./smoke.sh                 # ~3 min end-to-end sanity check on a small model
./run_matrix.sh prep       # bandwidth probe + roofline computation
./run_matrix.sh pipelined  # the throughput pass — this is the number to quote
./run_matrix.sh dash       # streamlit dashboard over runs.jsonl
```

**Always quote the `pipelined` number.** A per-token-sync measurement is a latency instrument,
not a throughput one — it costs ~18 % versus overlapped (`mx.async_eval`) execution, and every
number in this repo that looks like a "speedup" states explicitly which baseline it's against.

For the speculative loop specifically, see `docs/recipes/03-mtp-speculative-decode.md` for setup
(the drafter head needs to be installed into your `mlx-lm` package) and
`docs/recipes/04-losslessness-writeup.md` for what "lossless" means here and how it's verified.

## Repository layout

```
*.py, *.sh          the harness itself — bandwidth probes, roofline calculation, the
                     speculative loop, the losslessness/acceptance gates, benchmarking scripts
runs.jsonl           every measurement this project has made, schema-versioned, append-only
prompts/             exact prompt text used for the recorded benchmarks
tasks/               work-order documents that scoped specific benchmarking phases
reports/             standalone findings, including a draft upstream issue for MLX
docs/OBJECTIVES.md   where the project stands, right now, in plain language
docs/recipes/        the numbered write-ups — one per investigation, each with its
                     hypothesis, method, result, and (where it applies) retraction
```

`docs/recipes/00-INDEX.md` is the map: every recipe's status, the numbers that matter, and the
list of measurement mistakes this project made and corrected along the way — kept visible
deliberately, because the corrections are as informative as the results.

## Ground rules this project holds itself to

- Every speedup states its baseline. A per-token-sync number is never quoted as throughput.
- Every bandwidth ceiling states the regime it was measured in — a probe under ~32 MB measures
  dispatch overhead, not DRAM bandwidth, on this hardware.
- A mechanism claim is not a measurement. Several early hypotheses about *why* a number looked
  the way it did were written up, tested, and retracted; the retraction is left in the docs.
- Losslessness for speculative decoding is stated as a ULP-band criterion, never as bitwise
  identity — the latter is provably unreachable on this hardware for reasons unrelated to any
  implementation bug (see `docs/recipes/04-losslessness-writeup.md`).

## Scope and honest limits

Everything here is measured on one machine (a fanless MacBook Air M3, 16 GB), one target/drafter
pair, and contexts below the model's 512-token sliding window. The cross-runtime comparison is
MLX-vs-MLX (mlx-lm vs mlx-vlm); llama.cpp/GGUF, which is what most people actually run locally,
is out of scope here. None of the headline numbers are claimed to generalize past what's stated
above without new measurement — `docs/OBJECTIVES.md` lists exactly what's still open.

## License

MIT — see `LICENSE`.

# Recipe 01 — the measurement harness and the roofline (CLOSED)

*Renamed from `recipe-00-harness-roofline.md`. Content unchanged except this header and the
status table, which now reflects the closed state of recipes 03–05.*

Measurement harness and optimization recipes for MLX inference on M-series Macs.

**Machine of record:** MacBook Air M3 (4P+4E CPU, 10-core GPU), 16 GB, fanless.
macOS 26.6.1 · MLX 0.32.2 · mlx-lm 0.31.3 · Python 3.12.14
**Model of record:** `mlx-community/gemma-4-e4b-it-OptiQ-4bit`

---

## Status

| recipe | state | outcome |
|---|---|---|
| 01 — harness, roofline, dashboard | **CLOSED** | decode measured at **97 % of roofline**, 24.37 tok/s |
| — decode fusion | **CANCELLED** | only ~3 % headroom exists; cannot pay for itself |
| — quantized GEMV kernels | **CANCELLED** | q4 and fp16 both reach ~92 GB/s, within 4 % of ceiling |
| — KV precision (fp16 → 8-bit) | **DEPRIORITISED** | KV is 148 MiB at 8K vs 3.5 GB of weights |
| 03 — Gemma 4 MTP drafter on MLX | **CLOSED** | 1.66× vs declared baseline; losslessness characterised |
| 05 — quantized matmul headroom | **CLOSED** | reproduced open MLX issue #3553 independently |
| — confidence-scheduled verification | queued | `schedule_decision.py` never run |
| — block drafter (DFlash-style) | dropped | mlx-vlm ships it |
| — serving loop | queued | chunked prefill, prefix cache, paged KV |

Recipes for fusion/GEMV/KV were cancelled **because of what this harness measured**, not by
guesswork. That is the return on building the instrument first.

## The baseline

| prompt | decode tok/s | effective GB/s | % of roofline |
|---:|---:|---:|---:|
| 128 | **24.37** | 86.6 | 97.3 % |
| 2,048 | **24.55** | 86.5 | 97.2 % |
| 8,192 | **22.86** | 86.5 | 97.2 % |

Machine ceiling: **89 GB/s** usable (spec 100). Active weight bytes per token: **3.535 GB** —
46 % of the checkpoint is a gather-only per-layer embedding table that never streams. Fixed
dispatch cost: 220 µs per isolated op + barrier.

Anything faster now has to move *fewer bytes per accepted token*. That is speculative
decoding, and nothing else.

---

## Files

```
mlxutil.py            MLX version shims, sysctl, powermetrics, contention, compat patches
roofline.py           bytes/token from the real parameter tree and real cache objects
bwprobe.py            achievable GB/s: size sweep + GEMV fit separating dispatch from DRAM
bench.py              the harness -> runs.jsonl
check_model.py        screen a checkpoint (PLE quantization, byte split, coherence)
dashboard.py          streamlit view over runs.jsonl
setup.sh smoke.sh run_matrix.sh
runs.jsonl            all measurements, schema_version 1
prompts/              exact prompt text used for the recorded baseline
```

Recipe 03/05 additions: `gemma4_assistant.py` (the drafter), `measure_acceptance.py`,
`spec_generate.py`, `shape_stability.py`, `fork_validity.py`, `control_variables.py`,
`row_dependence.py`, `hidden_shape_drift.py`, `speed_5pass.py`, `verify_decomposition.py`,
`schedule_decision.py`, `confidence_calibration.py`, `setup_control.sh`.

## Running

```bash
cd ~/mlx-env/mlx-bench       # or: cd Gigatok && VENV=/path/to/your/mlx-venv ./setup.sh
./setup.sh                 # deps + API check
./smoke.sh                 # 3-min end-to-end validation on a 1B model
./run_matrix.sh prep       # bandwidth + roofline + memory plan
./run_matrix.sh pipelined  # the throughput pass — quote THIS number
./run_matrix.sh dash       # dashboard
```

**Always quote the `pipelined` number.** `per-token` sync is a latency instrument and costs
18 % of throughput to obtain.

Environment: quit Chrome (not just close it). Claude Desktop at ~1 GB is fine. Low Power Mode
must be off — hard abort. Battery vs AC does not matter on Apple Silicon, but do not mix the
two within a comparison; `power_source` is recorded per run. Wait for XProtect/Spotlight to
finish after a model download.

---

## Hard-won lessons

* **Measure the byte split; never trust a parameter count.** 46 % of this model is
  gather-only. A naive roofline was wrong by 85 %.
* **Probe KV from the model's own cache objects.** The textbook formula is wrong four separate
  ways on hybrid attention.
* **Any GEMV bandwidth probe under ~32 MB measures dispatch, not DRAM.** The same kernel reads
  35 GB/s at 8.8 MB and 90 GB/s at 566 MB.
* **CPU/GPU overlap was worth 18 %** — more than any kernel change available.
* **Gemma needs `<bos>`.** Without it the model degenerates into repetition that looks exactly
  like a broken checkpoint. Cost an afternoon and a wrong verdict on the 12B.
* **Stock mlx-lm Gemma 4 conversions quantize the PLE table to 4 bits and are unusable.**
  Screen every checkpoint with `check_model.py` first.
* **Bimodal clustering and intra-run decay** catch what monotonic-drift tests miss.
* **macOS compresses before it swaps, silently.** Record swapout and compressor deltas per run
  or a bad run averages in unnoticed.

## Known gaps

* `gemma4_unified` (the 12B) needs the compat remap and has not been retested since the BOS
  fix — its earlier "broken" verdict is probably wrong.
* mlx-lm 0.31.3 `mlx_lm.server` crashes on sliding-window models from worker threads. Affects
  the serving-loop recipe, not this harness.

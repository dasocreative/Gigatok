# Recipe 02 — baseline established, roadmap revised (CLOSED)

*Renamed from `recipe-00-CLOSED.md`. Content unchanged.*

M3 Air 16 GB fanless (4P+4E, 10-core GPU) · `mlx-community/gemma-4-e4b-it-OptiQ-4bit`
MLX 0.32.2 / mlx-lm 0.31.3 · battery 98 %, Low Power Mode off
128 / 2,048 / 8,192 prompt × 256 gen · median of 5 + warmup · temp 0 · 40 s cooldown

## THE NUMBER

**MLX decode runs at 97 % of the calibrated memory-bandwidth roofline.**

| prompt | per-token sync | our pipelined | stock mlx-lm | effective GB/s | % roofline |
|---:|---:|---:|---:|---:|---:|
| 128 | 20.43 | 24.12 | **24.37** | 86.6 | **97.3 %** |
| 2,048 | 20.05 | 23.60 | **24.55** | 86.5 | **97.2 %** |
| 8,192 | 19.29 | 22.54 | **22.86** | 86.5 | **97.2 %** |

Three engines, three contexts, agreeing on ~86.5 GB/s against an 89 GB/s ceiling. Spread
within a cell ≤ 1 %.

**Baseline to beat: 24.37 tok/s** @ 128/256 greedy.

## Independent validation

Stock `mlx_lm.stream_generate` vs our manual engine: decode agrees to within 1.0 / 3.9 / 1.4 %.
The harness measures what mlx-lm actually does.

| | ours | stock | note |
|---|---:|---:|---|
| prefill tok/s @ 2048 | 565 | 530 | **+6.6 %** — tail-logits prefill |
| prefill tok/s @ 128 | 391 | 348 | **+12.4 %** |
| peak memory @ 8192 | 6.79 GiB | 7.58 GiB | **−0.79 GiB** |
| decode | 23.60 | 24.55 | −3.9 %, see below |

The prefill and memory wins come from evaluating **cache state, not logits**, on every prefill
chunk, then forwarding the final token alone. Stock computes full `[1, chunk, 262144]` logits
on the last chunk and slices. Worth carrying into the serving loop.

The decode deficit was most likely the **generation stream**: mlx-lm runs decode inside
`with mx.stream(generation_stream)`; our manual engine used the default. Now matched.

## The overhead was CPU/GPU serialisation, not kernels

`--sync per-token` (one hard barrier per token) vs `--sync pipelined` (`mx.async_eval`):
**+18 % throughput, no kernel changes.** Overhead per token fell from 9.11 ms to 1.62 ms —
**82 % recovered by overlap alone.** CPU time during decode is 54–56 % of wall (23.2 ms per
41.5 ms token); pipelining hides it.

This also explains the earlier bimodality (fast ~22.9 / slow ~20.1 clusters, on battery and AC
alike): those were runs where CPU and GPU happened to overlap better. Pipelined beats every
fast excursion and its spread collapses to ≤0.9 %. Not a power state, not P-core placement.

**Rule: never quote a per-token-sync number as throughput.** It is a latency instrument and it
costs 18 %.

## Bytes per token, and why the naive roofline is useless here

| | bytes |
|---|---:|
| total parameters | 6.081 GiB |
| dense — streams every token | 2.628 GiB |
| gather-only PLE — resident, one row/token | 2.789 GiB |
| tied embedding, read in full as lm_head | 680 MiB |
| **active per decode token** | **3.535 GB** |

7.46 B logical parameters at 7.00 effective bits/weight. **46 % never streams.** Naive
"6.081 GiB ÷ bandwidth" predicts 14.2 tok/s against a true ~26 — wrong by 85 %, in the
direction that makes real measurements look impossible.

## KV: solved, and not a bottleneck

42 layers, 24 caches (`num_kv_shared_layers: 18`), period 6 (5 sliding : 1 global), final
layer global. 20 × `RotatingKVCache` @ 2048 B/token capped at 1.0 MiB (= 512 window);
4 × `KVCache` @ 4096 B/token uncapped. Sharing map: `cache[22]` serves 16 layers, `cache[23]`
serves 4.

Measured 128 → 8,192 slowdown 5.9 %, between the 4.2 % optimistic and 7.3 % pessimistic
bounds. Solving for the fraction reaching DRAM: **f = 0.55**, at which effective bandwidth is
72.43 / 72.42 / 72.43 GB/s — three independent measurements to 0.01 GB/s. **~45 % of shared-KV
re-reads are absorbed by the SLC.**

148 MiB at 8K against 3.5 GB of weights. 4 GiB headroom; 256K context fits.

## Calibrated bandwidth

q4 GEMV **92.8 GB/s** streaming, fp16 92.1, read plateau 96.4 (R² = 1.0000; denominator
clamped to 89 where the fit over-extrapolated past the read plateau). Fixed dispatch cost
**220 µs** per isolated op + barrier.

Same q4 kernel: 35.4 GB/s at 8.8 MB, 89.6 GB/s at 566 MB. Small matrices measure dispatch, not
DRAM — any GEMV probe under ~32 MB is launch-bound and must be excluded from the fit.

---

## ROADMAP: two recipes cancelled, one promoted

**CANCELLED — decode fusion.** 3 % envelope. RMSNorm+QKV+RoPE fusion cannot pay for itself.

**CANCELLED — quantized GEMV kernel tuning.** q4 and fp16 both reach ~92 GB/s, within 4 % of
the read ceiling. Hand-written MSL cannot recover what is not lost.

**DEPRIORITISED — KV precision (FP16 → 8-bit).** 148 MiB at 8K against 3.5 GB of weights, with
45 % of shared re-reads never reaching DRAM.

**PROMOTED — speculative decoding is the only remaining lever.** It is the sole mechanism that
reduces *bytes per accepted token* rather than trying to move bytes faster.

### Compute headroom: verification is nearly free here

| | |
|---|---:|
| dense transformer params | 3.97 B (PLE 2.82 B, embed 0.67 B are gathers) |
| FLOPs/token | 7.9 GFLOP |
| compute @ 3–4 TFLOPS | 2.0–2.7 ms |
| memory time | 41.5 ms |
| **GPU compute utilisation at batch 1** | **5–6 %** |

A verification forward over γ+1 tokens reads the same weights once — memory time unchanged,
compute scales with γ. Headroom for **γ ≈ 15–20** before compute binds.

**This overturns the caution the project opened with.** I expected a 10-core M3 to understate
speculative gains because verification would compete for scarce compute. The opposite holds:
so much of this model is gather-only PLE and small bandwidth-bound matmuls that compute sits
~94 % idle at batch 1.

| accept length | projected tok/s (15 % haircut) |
|---:|---:|
| 2.0 | ~41 |
| 2.5 | ~51 |
| 3.0 | ~62 |
| 4.0 | ~82 |

*(Recipe 05 later confirmed the compute-headroom prediction empirically: `t(k)` saturates flat
from k=11 to k=16.)*

## Entry criteria for the speculative recipe

1. Measure real acceptance length before building anything. Cheapest possible probe.
2. ~~Losslessness: greedy output must match `output_sha256` token-for-token.~~
   **SUPERSEDED by Recipe 03** — bitwise identity is unachievable on this hardware; the
   criterion is the ULP-band statement instead.
3. Report acceptance length AND acceptance rate per step, not just tok/s.

## Harness lessons worth keeping

* Measure the byte split; never trust a parameter count. 46 % of this model is gather-only.
* Probe KV from the model's own cache objects; the formula is wrong four different ways on
  hybrid architectures.
* Any GEMV bandwidth probe under ~32 MB measures dispatch, not DRAM.
* Bimodal clustering and intra-run ITL decay catch what monotonic-drift tests miss.
* Record swapout/compressor deltas per run — one 8,192 run hit 16,588 swapouts and was
  auto-excluded.
* Gemma needs `<bos>`; a missing one produces repetition that looks exactly like a broken
  checkpoint.
* mlx-lm's stock Gemma 4 conversions quantize PLE to 4 bits and are unusable; screen every
  checkpoint structurally before benchmarking.

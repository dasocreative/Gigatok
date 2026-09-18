# Recipe 05 — quantized matmul headroom on M-series (CLOSED)

*Renamed from `recipe-05-kernel-headroom.md`.*

**Question.** Recipe 02 measured batch-1 decode at 97 % of the calibrated roofline, which
closed the fusion/GEMV/KV recipes. MLX is MIT-licensed and takes PRs. So: is there anywhere
the roofline does *not* bind, where a better Metal kernel would pay?

**Answer.** Yes — and it is a **known, open, unresolved MLX issue** that we reproduced
independently on different hardware and a different model before finding it. See §6.

Script: `verify_decomposition.py`. Records: `record_type: "verify_decomposition"`.

---

## 1. Findings that stand

### The quantized kernel's bandwidth collapses in the multi-row path

`embed_tokens` [262144 × 5120], 4-bit g128, **713 MB** — matching `roofline.json`'s
`lm_head_bytes = 713,031,680` exactly, which confirms the packing solver:

| M | 1 | 2 | 3 | 4 | 6 | 8 | 11 | 16 |
|---|---|---|---|---|---|---|---|---|
| GB/s | 89.4 | 89.9 | 84.9 | 69.4 | 44.6 | 35.0 | 23.4 | 23.4 |
| % of 89 GB/s roof | 100 | 101 | 95 | 78 | 50 | 39 | 26 | 26 |

M=1 landing exactly on the calibrated ceiling validates the method. Efficiency then falls
**3.8×**. Flat M=1→2, knee at M=3–4. Reproducible across three runs.

### t(k) saturates at k ≥ 11

| k | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|
| ms (ctx 128) | 92.2 | 94.3 | 111.2 | 114.3 | 115.9 | 114.8 | 111.7 | 111.6 |

Five extra verify positions free while FLOPs grow 45 %. Cost per position minimised at the
top: **6.98 ms at k=16 vs 9.84 at k=8**. Reproducible to ~3 %. Empirically confirms Recipe
02's prediction of headroom for γ ≈ 15–20.

**This invalidates the cost model past the knee.** `t(k) = 37.2 + 6.75·k` predicts 145 ms at
k=16; measured 111.6 — 30 % over-prediction. Fitted at k ≤ 10; γ≈3 and the DSpark sizing
inherit the error.

### Compute ceiling, measured for the first time

**2.92 TFLOP/s** (2048² bf16 GEMM, cache-resident). Headroom at the worst k (k=9) is bounded
**1.16× to 2.30×** — measured 92.2 ms against a 79.7 ms no-overlap floor and a 40.0 ms
perfect-overlap floor. Quoting either endpoint alone overstates the case.

---

## 2. Findings WITHDRAWN

**The M=12 path switch.** `up_proj` [10240 × 2560], 67× per forward, appeared to jump 2.3×
faster at M=12 in one run; the next run showed a drop at M=2 instead. **The weight is 14.7 MB
— below the 32 MB floor this project set after the 2.4 MB GEMV probe measured dispatch instead
of DRAM.** Cache-resident across reps, so the curve depends on what else is in cache. All
conclusions from it withdrawn, including a prediction that `t(12) < t(10)`.

Measuring the dominant decoder shapes validly needs a cache-defeating design — rotating over
many distinct weights, nothing under 32 MB.

---

## 3. Measurement errors made here (all caught by arithmetic, not by the script)

| error | how it showed |
|---|---|
| Module walk via `vars(m)` | mlx's `nn.Module` subclasses `dict`; found **0** quantized linears in a fully quantized model. |
| One global `(bits, group_size)` | OptiQ is mixed-precision; `w.shape[1]·32/scales.shape[1] = 512` fits 4-bit/g128 *and* 8-bit/g64. |
| Gather-only tensors in a FLOP model | `embed_tokens_per_layer` is row-gathered, never matmul'd. |
| "Compute ceiling" that was memory-bound | FLOPs ÷ a bandwidth-limited time gave 1.409 TFLOP/s and a ratio of ~1.0 — two wrong numbers agreeing. |
| Bandwidth claim on a 14.7 MB weight | §2. The project had already written this rule. |
| Verdict averaged across the knee | Printed "marginal cost is FLAT, no headroom" *and* "2.30× its roofline" together. |

**Pattern: reaching for a comparison before checking the regime it is valid in.**

---

## 4. What the findings actually depend on

| parameter | verdict |
|---|---|
| **MLX's qmv/qmm dispatch threshold** | **The mechanism.** MLX routes below `vector_limit` to `qmv_fast_impl` and above it to `qmm`. PR #2031 (merged 2025-04-03) tuned this and states the optimum "is both machine and matrix size dependent … this is only an approximation." |
| **Matrix asymmetry** | Decisive. Issue #3553 reports square shapes show **no** step; asymmetric ones do. Our [262144 × 5120] is 51:1 — the most asymmetric weight in the model, and shows the effect most strongly. |
| **Machine** | Real. PR #2031 cites an M4 Max crossing point of 14 at size 4352; #3553 is on M4 Pro; ours is a 10-core M3. Per-machine by design. |
| **Quantisation** | Relevant. #3553 is group_size=64 / bits=4; ours 4-bit g128 (LM head) and g64 (decoder). |
| **Model** | Only through shape. #3553 reproduces on Qwen 3.6-27B, we see it on Gemma 4. |
| **Our harness** | **Not a factor.** Below our code: #3553 reproduces in isolated `mx.quantized_matmul` calls and in a stock forward. |
| **MLX version** | Ours 0.32.2; the issue is open, so presumed present. |

---

## 5. Recipe 02 comparison — the ambiguity, resolved from data already recorded

`runs.jsonl` stores `top_processes` per run, so this needed no new benchmark.

| tag | n | median tok/s | Chrome seen | Claude seen |
|---|---|---|---|---|
| `baseline-pertoken` (Recipe 02) | 30 | **20.03** | 0/30 | 30/30 |
| `baseline-pipelined` (Recipe 02) | 15 | 23.60 | 0/15 | 15/15 |
| `stock-stream` (Recipe 02) | 15 | 24.37 | 0/15 | 15/15 |
| `spec_decode` baseline (ours) | 24 | 20.43 | 0/24 | — |
| `spec_speed` baseline (ours) | 10 | **20.17** | 0/10 | — |

**Our harness baseline is not slower than Recipe 02 — it is the same code path and the same
number.** 20.17 against 20.03. What differs is the **strategy**: pipelining takes 20.03 →
23.60, and stock `stream_generate` reaches 24.37.

| against | tok/s | speedup |
|---|---|---|
| in-process per-token baseline (same strategy) | 20.17 | 2.04× |
| Recipe 02 pipelined | 23.60 | 1.71× |
| **stock mlx-lm `stream_generate` (declared baseline)** | **24.37** | **1.66×** |

**1.66× is the number Recipe 02's own rule requires** — "never quote a per-token-sync number
as throughput."

*Caveat:* `top_processes` records only the top 8 processes above 200 MB, so "Chrome 0/262" is
suggestive, not conclusive. It supports *similar* conditions, not quiet ones.

**The consequence.** Our speculative loop uses per-token `mx.eval` internally, exactly like the
baseline that gained ~20 % from pipelining. Measured verify + draft predicts ~53 tok/s against
40.4 actual — a **~25 % host-overhead gap** with a measured precedent for closing it.

---

## 6. Prior art — we reproduced an open MLX issue

**[Issue #3553](https://github.com/ml-explore/mlx/issues/3553), "`qmv` kernel: non-linear cost
step at M=3 for large MLP shapes" — OPEN, cause unidentified.**

| | issue #3553 | ours |
|---|---|---|
| op | `mx.quantized_matmul(transpose=True)` | same |
| quant | group_size 64, bits 4 | 4-bit g128 / g64 |
| M=1→2 | "nearly flat (~2 %)" | 89.4 → 89.9 GB/s (flat) |
| knee | sharp step at M=3, 28–37 % | M=3 −5 %, M=4 −22 % |
| shape sensitivity | square shows no step; asymmetric does | 51:1 asymmetric shows it strongly |
| full forward | Qwen 3.6-27B **1.38× slower at M=3 than M=1** | `t(k)` superlinear to k≈11 |
| hardware | M4 Pro | M3, 10-core |
| dispatch path | `qmv_fast_impl`, below `vector_limit` | consistent |

The author notes existing PRs "address adjacent regimes but leave M=3–9 on asymmetric shapes
unaddressed." [PR #2031](https://github.com/ml-explore/mlx/pull/2031) tuned qmv/qmm dispatch
(M2 Ultra, Mistral 7B: batch 6 31.97 → 19.40 ms) and concedes the optimum is machine- and
shape-dependent. [PR #1503](https://github.com/ml-explore/mlx/pull/1503) added batched
quantized matmul and fast small QMV.

**What our data adds:** a second chip (M3 10-core vs M4 Pro), a second model family (Gemma 4
vs Qwen), the **bandwidth framing** — 89.4 → 23.4 GB/s against a *calibrated* 89 GB/s roofline,
turning "it gets slower" into "it drops to 26 % of the memory ceiling" — and the **saturation
at k ≥ 11**, which the issue does not report and which has a direct consequence for
speculative-decoding cost models.

---

## 7. Next, in order

1. **Pipeline the speculative loop** (`mx.async_eval`). Precedent: Recipe 02's baseline
   20.03 → 23.60 for zero kernel work. Largest lever.
2. **`schedule_decision.py`** — never run; would now run against a correct `t(k)`.
3. **Comment on MLX issue #3553** with the M3 + Gemma 4 + roofline-framed data.
4. **Cache-defeating sweep** of the dominant decoder shapes, if reopening the kernel question.
5. Not recommended: writing a Metal kernel. The maintainers own this dispatch heuristic and
   have an open issue on it; a one-machine tiling patch from outside is unlikely to land.

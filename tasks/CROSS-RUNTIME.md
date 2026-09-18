# Recipe 10 — cross-runtime comparison

**Owner of the question:** Cowork chat. **Owner of the measurement:** you.
**Do not re-scope.** If the design below turns out to be unmeasurable, report why and stop.

## The question

Does this harness produce a better result than the runtimes people already use, or the same
number with more ceremony? Never answered. It is the reason the project exists.

## Pre-registered prediction — commit to this BEFORE measuring

At batch 1 all of these are memory-bandwidth-bound; stock MLX decode measures 97 % of the
89 GB/s ceiling. Therefore:

1. Plain decode across all runtimes lands **within a few percent once normalised by measured
   active bytes per token**.
2. Our 1.74x comes **entirely from speculative decoding**, not from kernel quality.
3. Any runtime beating us on **normalised bytes-per-token** falsifies (1) and is a real finding.

Persist this prediction as a `crossruntime_prediction` row before the first timed run.

## Step 0 — inventory (do this first, it may be the whole answer for today)

Report, without installing anything:
- Is `ollama` on PATH? `ollama list` — which models, which quantisation.
- Is LM Studio installed (`/Applications/LM Studio.app`)? Its CLI is `lms`. What is in its
  model directory, and does the build expose draft-model speculative decoding?
- Is `~/mlx-vlm-control` present and working (`setup_control.sh` built it, mlx-vlm 0.6.17)?

**Do not install a runtime without asking.** Report the inventory and wait if anything is
missing. Installing ollama or LM Studio is fine when approved — they are outside `~/mlx-env`
and cannot touch the pins. **Nothing may be installed into `~/mlx-env`.**

## Step 1 — the honest-comparison problem, and how to handle it

These runtimes will not be running identical weights. ollama and LM Studio are llama.cpp/GGUF;
we are MLX 4-bit OptiQ with 8-bit per-layer embeddings. Different quantisation means different
bytes per token, which means **raw tok/s is not a comparison**.

For every runtime, measure and report all four:

| column | how |
|---|---|
| raw tok/s | median of >=5 passes, temperature 0 |
| active bytes/token | from the actual on-disk quant + the same PLE/gather exclusion `roofline.py` applies. Do not assume; derive it per runtime. |
| effective GB/s | bytes/token x tok/s |
| % of 89 GB/s | the comparable number |

State every weight mismatch explicitly in the write-up. A runtime whose quant we cannot
account for byte-for-byte is reported as **"not comparable"**, not estimated.

## Step 2 — arms

- **A. plain decode** — every runtime, no speculation. This is the kernel-quality comparison.
- **B. speculative** — only where the runtime supports it (LM Studio draft models; mlx-vlm MTP;
  ours). Report acceptance where the runtime exposes it, "not exposed" where it does not.

Ours is already measured: 24.37 plain / 42.46 speculative. Re-measure both in the same session
as the others so machine state is shared — do not compare against a stored row from another day.

## Step 3 — conditions (identical for every runtime, no exceptions)

Chrome quit. Claude Desktop resident is expected and correct. AC power, not Low Power Mode.
**Cool start**, warmup discarded, 25-40 s cooldown between passes, medians of >=5, temperature 0,
same two prompts, same 160 gen tokens. Alternate arm order across passes.
Record `power`, `memory_pressure_pct` and `top_processes` on every row, as `speed_5pass.py` does.

## Step 4 — persist and report

Every run appends a `crossruntime_run` row. Finish with one `crossruntime_finding` row carrying
the verdict, the prediction's status (held / falsified), and the per-runtime table.
Write `reports/recipe10-crossruntime.md`.

## Acceptance criteria

- [ ] Inventory reported before any install.
- [ ] Prediction persisted before the first timed run.
- [ ] Every runtime reported with all four columns, or explicitly marked not comparable.
- [ ] Every weight/quantisation mismatch stated.
- [ ] All runtimes measured in one session under identical conditions.
- [ ] Prediction explicitly marked held or falsified. **Falsified is a fine outcome — say so.**

## Failure modes this project has already paid for

- Quoting a per-token-sync figure as throughput (cost us an 18 % error and a 2.04x claim).
- Comparing against an operating point nobody runs (Phase 1's gamma=8 straw man).
- Averaging across a knee.
- Writing a verdict string without checking what the winning configuration actually is
  (Phase 2 printed GO for a chain).
- Sizing anything from an isolated microbenchmark.

# Recipe 12 — upstream contribution to ml-explore/mlx

**Status: DRAFT, NOT POSTED.** Artifact: `reports/mlx-upstream-issue-draft.md` **revision 3**
(revision 2 preserved at `.rev2.bak`). Records: `issue3553_premise_check` (20260902-210128),
`upstream_issue_draft` rev 2 (20260902-210401) and rev 3 (20260908-230700),
`tk_step_finding` (20260903-000454).

---

## 1. The plan item that was refuted before it was executed

The standing plan said: *"comment on MLX issue #3553 with the M3 + Gemma 4 + roofline framing and
the 69 %-vs-97 % verify-batching observation."* The premise was checked first, and failed on two
grounds:

1. **The `#3553` mechanism link was a mechanism claim, not a measurement.** #3553 reported a step
   at **M=3** on the `qmv_fast` path (M4 Pro, Qwen3.6-27B). Our pinned mlx 0.32.2 reportedly
   already contains the `qmv_wide` kernel family that superseded that path — and **our step is at
   M=4, not M=3.** Different location, different kernel generation, no link asserted.
2. **The issue may already be closed.** See the blocker below.

**This was my error, made in this project's own chat**, and it is the fifth mechanism claim the
harness has killed. The rule it violated is the project's own: *mechanism claims are not
measurements.*

## 2. BLOCKER — the issue's state is unverified

Revision 2 asserted #3553 was closed 2026-08-05 as completed with a `qmv_wide` closing comment.
**An independent check could not confirm it.** Three HTML fetches of the issue page report
**Open** with no comments visible; the GitHub REST API returned 403 / no egress from both
checking environments. One reading is wrong.

Settle it before posting:

```
curl -s https://api.github.com/repos/ml-explore/mlx/issues/3553 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ('state','state_reason','closed_at','comments')})"
```

**Closed** → post as a new issue. **Open** → post the body as a comment on #3553 and drop the one
sentence about the `qmv_fast` path being superseded. No measurement depends on the answer.

Also outstanding: re-verify `qmv_wide` in the pinned build
(`strings ~/mlx-env/.../mlx.metallib | grep -c qmv_wide`). Revision 2's framing rests on it and
it could not be checked from the review side — `~/mlx-env` sits outside the mounted folder.

## 3. The finding the draft actually reports

**`t(k)` is a staircase, not a ramp.** In-model forward, real weights, forward-only times against
a forward-only floor of 31.71 ms:

| k | ctx 128 (ms) | marginal | vs floor |
|---:|---:|---:|---:|
| 1 | 36.78 | — | 1.16× |
| 2 | 37.70 | +0.91 | 1.19× |
| 3 | 39.58 | +1.88 | 1.25× |
| 4 | 44.49 | **+4.91** | 1.40× |
| 5 | 50.14 | +5.65 | 1.58× |
| 6 | 59.09 | +8.95 | 1.86× |
| 7 | 75.31 | **+16.21** | 2.38× |
| 8 | 76.32 | **+1.01** | 2.41× |

Cheap marginals also at k=10 (+0.55) and from k=12 onward. **The adjacent +16.21 / +1.01 pair at
M=6→7→8 is the strongest signal in the dataset** and is what revision 3 leads with; the M=4 step
is demoted to a second feature. **No mechanism is offered** — there is no profiling data, and
anything else would be a guess.

**Reproducibility:** two independent sessions days apart, cool machine, within 2.6 ms at every k
and within 1 ms at six of eight points.

**Headroom is a range, not a number.** k=8 is **2.41×** the floor assuming perfect overlap of
weight streaming and compute, and **1.01×** assuming none. The truth is between and this data
cannot narrow it. Earlier framings that quoted only the optimistic ratio overstated the case.

## 4. Corrections applied in revision 3

1. `#3553`-closure asserted as fact → **stated blocker** with the settling command and both
   posting routes.
2. Title led on the M=4 step → **retitled around the staircase**.
3. The "~61.5 GB/s = 69 % of ceiling" claim **removed from the body**: it was an in-loop, ctx-186,
   fused-barrier measurement sitting unlabelled beside an isolated ctx-128/2048 sweep — the
   apples-to-oranges error the same document had corrected one section earlier.
4. Roofline figure **93.5–96.0 %, never 97 %** (Recipe 11 re-measured it; CLAUDE.md corrected).
5. **Sweep ordering declared as a gap.** Ascending-only on a fanless part, and both reproducing
   sessions shared that ordering, so thermal bias is not excluded. Same class as the Phase 0
   contamination that produced a 7× attribution error.
6. **mlx-vlm corroboration promoted into the body.** Two independently written decode loops on
   byte-identical weights, with active-bytes derived independently inside each venv, agreeing to
   **0.1 %** at 93.5–96.0 % of ceiling — the strongest reason a maintainer should trust the byte
   accounting. Revision 2 buried it in a parenthesis.

## 5. What the draft deliberately withholds

- **Part A's isolated `mx.quantized_matmul` sweep** and its "tiling quantum" interpretation. The
  weight was 14.7 MB, below this project's 32 MB floor, so it went cache-resident and measured
  dispatch rather than DRAM. Withdrawn — and the *threshold warning itself* is included in the
  issue as a service to whoever investigates next.
- **Any dispatch claim.** Which kernel our shapes actually select is unverified. Presence of a
  kernel family in the binary is not proof of dispatch.

## 6. Position on forking

**Push, do not fork.** MLX is MIT with no CLA; fork-and-PR is the documented process; performance
changes are expected to carry before/after benchmarks, and [PR #2031](https://github.com/ml-explore/mlx/pull/2031)
is precedent for a merged dispatch-heuristic tune backed by exactly that evidence.

A maintained fork would be actively harmful here: every number this project owns is anchored to
pinned mlx 0.32.2, and a divergent kernel makes our results incomparable to everyone else's
forever. Recipe 11 also showed the ecosystem is converging on the same MLX — so an upstream fix
reaches every Mac running local LLMs, while a fork reaches one laptop.

**Sequence:** observational report first (zero risk, establishes credibility), then a benchmark
script into `benchmarks/python/` that makes M=2…8 measurable, and only then a heuristic or kernel
PR. Any MLX built for testing goes in a **separate venv** — `~/mlx-env` stays pinned.

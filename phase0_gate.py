#!/usr/bin/env python3
"""
phase0_gate.py — the objective stop condition for Phase 0.

WHY A GATE SCRIPT

Autonomous iteration needs a criterion the machine can evaluate, not a judgement call. Without
one you either stop early on a number that looked good once, or churn forever. This runs the
same three checks every time, on the same prompts, and prints PASS/FAIL per criterion.

It measures; it does not optimise. It has no opinion about how the speedup was obtained.

THE THREE CRITERIA

  losslessness   every divergence below --tie-ulps (3.0), via spec_generate --verify-lossless.
                 NOT bitwise identity -- that is unachievable on this hardware (the transformer
                 forward is shape-dependent, 0.63 % of positions flip at <=2 ULP) and must
                 never be a gate.
  throughput     median of 5 passes >= --target (48.0 tok/s = 2.0x the DECLARED baseline of
                 24.37 tok/s, stock mlx_lm.stream_generate). Never compare against the
                 per-token-sync baseline: it is a latency instrument and costs 18 %.
  no regression  mean accepted >= --min-accepted (1.55; currently 1.59 at gamma=3). Catches an
                 "optimisation" that bought time by drafting worse.

    ../bin/python phase0_gate.py                 # full gate
    ../bin/python phase0_gate.py --baseline-only # record a before-state, skip the verdict
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DECLARED_BASELINE = 24.37          # stock mlx_lm.stream_generate, Recipe 02


def run(cmd: list[str], label: str) -> str:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}\n  $ {' '.join(cmd)}\n", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    print(out, flush=True)
    print(f"  [{label}: {time.perf_counter() - t0:.0f}s, exit {p.returncode}]", flush=True)
    if p.returncode != 0:
        print(f"  !! {label} exited {p.returncode} — gate cannot evaluate this criterion.")
    return out


def grab(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def tail_rows(path: str, record_type: str, n: int) -> list[dict]:
    """Last n rows of a record_type. The gate reads runs.jsonl rather than only parsing
    stdout, so a formatting change in a script cannot silently blind it."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("record_type") == record_type:
            rows.append(r)
    return rows[-n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="../bin/python")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--target", type=float, default=48.0,
                    help="tok/s to clear = 2.0x the declared 24.37 baseline")
    ap.add_argument("--tie-ulps", type=float, default=3.0)
    ap.add_argument("--min-accepted", type=float, default=1.55)
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="PHASE 1 confidence schedule; 0 = fixed gamma")
    ap.add_argument("--baseline-only", action="store_true",
                    help="record a before-state and skip the verdict")
    ap.add_argument("--skip-fork", action="store_true",
                    help="skip fork_validity (slow); only if losslessness already passed today")
    # STAGES. A full gate runs three scripts, each loading the model (~40 s) and one of them
    # taking ~5 minutes — well past the command timeout an agent runs under. Each stage is a
    # separate invocation that persists its result to runs.jsonl; `--stage verdict` then reads
    # them back and decides. Nothing is recomputed and nothing is held in memory between calls.
    ap.add_argument("--stage", default="all",
                    choices=("all", "loss", "fork", "speed", "verdict"),
                    help="run one stage per command when under a timeout; "
                         "loss ~1.5min, fork ~4min, speed ~6min, verdict instant")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    py = args.python
    results, log = {}, []
    st = args.stage
    want = {"all": {"loss", "fork", "speed"}, "loss": {"loss"}, "fork": {"fork"},
            "speed": {"speed"}, "verdict": set()}[st]

    def remember(key, payload):
        """Persist a stage result so a later --stage verdict can read it back."""
        with open("runs.jsonl", "a") as f:
            f.write(json.dumps({"record_type": "phase0_stage", "schema_version": 1,
                                "stage": key, "stamp": stamp, "payload": payload},
                               default=str) + "\n")

    def recall(key):
        rows = tail_rows("runs.jsonl", "phase0_stage", 200)
        for r in reversed(rows):
            if r.get("stage") == key:
                return r.get("payload")
        return None

    # ---------------------------------------------------------- 1. losslessness
    if "loss" not in want:
        for k in ("losslessness", "acceptance"):
            v = recall(k)
            if v:
                results[k] = v
    else:
        out = run([py, "spec_generate.py", "--verify-lossless",
               "--gamma", str(args.gamma), "--tokens", str(args.tokens),
               "--prompts", str(args.prompts), "--tie-ulps", str(args.tie_ulps),
               "--conf-threshold", str(args.conf_threshold)],
              "losslessness + accept rate")
        log.append(out)

        exact = "LOSSLESS         EXACT" in out
        passed_band = "LOSSLESS         PASS" in out
        failed = "LOSSLESS         FAILED" in out
        band_frac = grab(r"tie band\s+\d+ of \d+ baseline positions \(([\d.]+)%\)", out)

        # ACCEPTANCE COMES FROM runs.jsonl, NOT FROM stdout.
        #
        # spec_generate prints "mean accepted" once PER PROMPT (the in-loop figure for
        # that prompt) and once more in the SUMMARY (the fmean across prompts). A bare
        # re.search takes the FIRST match, which is prompt 0's number. At gamma=3 /
        # 160 tokens that is 1.46, while the summary -- and CLAUDE.md's documented
        # ground truth -- is 1.59. The threshold 1.55 was calibrated against the
        # summary, so the gate reported FAIL on completely unmodified code, on every
        # run, for a criterion whose whole job is to notice CHANGE. A gate that cries
        # wolf is worse than no gate.
        #
        # Verified deterministic at temperature 0: every historical gamma=3/160-token
        # row in runs.jsonl is p0=1.462, p1=1.712 with identical accept histograms.
        # The threshold is untouched at --min-accepted; only the number it is compared
        # against is now the one the docstring always claimed it was.
        arows = [r for r in tail_rows("runs.jsonl", "spec_decode", 400)
                 if r.get("gamma") == args.gamma and r.get("gen_tokens") == args.tokens]
        accs = [r["mean_accepted"] for r in arows[-args.prompts:]
                if r.get("mean_accepted") is not None]
        mean_acc = stats.fmean(accs) if accs else None
        results["losslessness"] = {
            "pass": bool((exact or passed_band) and not failed),
            "detail": ("EXACT" if exact else "within band" if passed_band
                       else "FAILED" if failed else "could not parse"),
            "tie_band_pct": band_frac,
        }
        results["acceptance"] = {
            "pass": mean_acc is not None and mean_acc >= args.min_accepted,
            "mean_accepted": mean_acc, "threshold": args.min_accepted,
            # Per-prompt, because the mean hides a real spread (1.46 vs 1.71 at
            # gamma=3): a change that helps one prompt and hurts the other is
            # invisible in the mean and obvious here.
            "per_prompt": [round(a, 4) for a in accs],
        }
        remember("losslessness", results["losslessness"])
        remember("acceptance", results["acceptance"])

    # -------------------------------------------------- 2. the structural proof
    if "fork" not in want:
        v = recall("fork_validity")
        if v:
            results["fork_validity"] = v
    elif not args.skip_fork:
        out = run([py, "fork_validity.py", "--gamma", str(args.gamma),
                   "--tokens", str(args.tokens), "--prompts", str(args.prompts)],
                  "fork_validity A+B (is every emitted tail a valid greedy continuation?)")
        log.append(out)
        bad = "bookkeeping bug" in out or "not a greedy continuation" in out
        good = "valid greedy continuation" in out
        results["fork_validity"] = {
            "pass": bool(good and not bad),
            "detail": "tails are valid greedy continuations" if good and not bad
                      else "FAILED — see cycle dump" if bad else "could not parse",
        }
        remember("fork_validity", results["fork_validity"])
    else:
        results["fork_validity"] = {"pass": None, "detail": "skipped"}

    # ------------------------------------------------------------- 3. throughput
    if "speed" not in want:
        v = recall("throughput")
        if v:
            results["throughput"] = v
        med = (v or {}).get("median_tok_s")
        drift = (v or {}).get("host_state_drift")
    else:
        out = run([py, "speed_5pass.py", "--passes", str(args.passes),
               "--gamma", str(args.gamma), "--tokens", str(args.tokens),
               "--prompts", str(args.prompts),
               "--conf-threshold", str(args.conf_threshold)],
                  "throughput, median of %d" % args.passes)
        log.append(out)

        rows = tail_rows("runs.jsonl", "spec_speed", args.passes * args.prompts)
        spec = [r["spec_tok_s"] for r in rows if r.get("spec_tok_s")]
        med = stats.median(spec) if spec else None
        drift = any(r.get("host_state_drift") for r in rows)
        results["throughput"] = {
            "pass": med is not None and med >= args.target and not drift,
            "median_tok_s": med, "target": args.target,
            "vs_declared_baseline": (med / DECLARED_BASELINE) if med else None,
            "host_state_drift": drift,
        }
        remember("throughput", results["throughput"])

    # ----------------------------------------------------------------- verdict
    print("\n" + "=" * 74)
    print("PHASE 0 GATE" + (f"  [stage: {st}]" if st != "all" else "")
          + ("  (baseline record only)" if args.baseline_only else ""))
    print("=" * 74)
    for k, v in results.items():
        mark = "----" if v["pass"] is None else ("PASS" if v["pass"] else "FAIL")
        extra = {kk: vv for kk, vv in v.items() if kk != "pass"}
        print(f"  [{mark}] {k:<14} {extra}")

    if med:
        print(f"\n  {med:.2f} tok/s = {med / DECLARED_BASELINE:.2f}x the declared baseline "
              f"({DECLARED_BASELINE} tok/s, stock mlx_lm.stream_generate)")
        print(f"  Phase 0 target {args.target:.1f} tok/s = "
              f"{args.target / DECLARED_BASELINE:.2f}x")
        print("  (Never quote this against the 20.03 per-token-sync baseline.)")
    if drift:
        print("\n  !! host state changed mid-run — the throughput number is not usable.")

    need = {"losslessness", "acceptance", "fork_validity", "throughput"}
    missing = need - set(results)
    checked = [v["pass"] for v in results.values() if v.get("pass") is not None]
    allpass = bool(checked) and all(checked) and not missing
    if missing and st in ("all", "verdict"):
        print(f"\n  incomplete — no result on file for: {sorted(missing)}")
        print("  run the missing stages, then: ../bin/python phase0_gate.py --stage verdict")
    if args.baseline_only:
        print("\n  Before-state recorded. Re-run without --baseline-only after each change.")
    elif allpass:
        print("\n  *** ALL PASS — Phase 0 complete. Stop optimising; write up what paid. ***")
    else:
        print("\n  Not yet. Change ONE thing, re-run. If a change does not move the median,")
        print("  revert it — a neutral change is added surface area, not a neutral outcome.")

    # ------------------------------------------------------------------ report
    Path("reports").mkdir(exist_ok=True)
    rp = Path(f"reports/phase0-{stamp}.md")
    rp.write_text(
        f"# Phase 0 gate — {stamp}\n\n"
        f"declared baseline {DECLARED_BASELINE} tok/s (stock `mlx_lm.stream_generate`)\n"
        f"target {args.target} tok/s · gamma {args.gamma} · {args.tokens} tokens × "
        f"{args.prompts} prompts · {args.passes} passes\n\n"
        "## verdict\n\n```\n"
        + "\n".join(
            f"[{'----' if v['pass'] is None else 'PASS' if v['pass'] else 'FAIL'}] {k:<14} "
            f"{ {kk: vv for kk, vv in v.items() if kk != 'pass'} }"
            for k, v in results.items())
        + "\n```\n\n## raw output\n\n```\n" + "\n\n".join(log) + "\n```\n")
    print(f"\n  report: {rp}")

    with open("runs.jsonl", "a") as f:
        f.write(json.dumps({
            "record_type": "phase0_gate", "schema_version": 1, "stamp": stamp,
            "declared_baseline_tok_s": DECLARED_BASELINE,
            "target_tok_s": args.target, "gamma": args.gamma,
            "baseline_only": bool(args.baseline_only),
            "all_pass": allpass, "results": results,
        }, default=str) + "\n")
    return 0 if (allpass or args.baseline_only) else 1


if __name__ == "__main__":
    sys.exit(main())

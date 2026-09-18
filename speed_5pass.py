#!/usr/bin/env python3
"""
speed_5pass.py — the publishable speed number, under the discipline the recipe
requires and that no run so far has met.

WHY A SEPARATE SCRIPT

Three speed numbers exist and none is publishable: 2.47x once on a quiet machine,
2.18x twice with a browser and a desktop app running, all n=1 against a rule of
medians over at least five passes. spec_generate.py is the CORRECTNESS tool -- it
loads two models, generates a baseline, verifies losslessness and appends a
divergence record. Bolting a pass loop onto it would confuse a settled question
with an open one. Losslessness is settled; this measures time and nothing else.

WHAT IT CONTROLS THAT A BARE RE-RUN DOES NOT

  warmup discarded      the first pass pays for lazy graph construction, Metal
                        pipeline compilation and page-ins that no later pass pays
                        again. Averaging it in is how a benchmark understates
                        itself and hides its own variance.
  cooldown between      fanless M3 Air. Back-to-back passes measure the heat of
                        the previous pass as much as the work of the current one.
  ALTERNATING ARM ORDER baseline-first on odd passes, speculative-first on even.
                        Within a pass the second arm runs on a hotter chip, so a
                        fixed order biases one arm systematically. Alternating
                        cancels it to first order and the per-order medians below
                        show whether it mattered.
  preflight gate        Low Power Mode caps CPU/GPU frequency outright and low
                        charge limits peak draw. Both are refused rather than
                        recorded, because a throttled pass is not a slow
                        measurement, it is a different machine.
  state drift check     power source, battery and Low Power Mode are sampled
                        before AND after every pass. Anything that changes
                        mid-run invalidates the comparison and says so.

Being on battery is NOT a reason to refuse: Apple Silicon delivers the same
performance on battery as on AC (see mlxutil.power_info). Mixing states across
passes you later compare IS. Plugging in mid-run is the failure mode, not battery.

Reports medians, spread and coefficient of variation per arm, and refuses to
present a median it does not trust.

    ../bin/python speed_5pass.py --passes 5 --gamma 3 --tokens 160 --prompts 2
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import mlx.core as mx

import mlxutil as U
import roofline as RF
from measure_acceptance import PROMPTS, enable_kv_capture, text_model
from spec_generate import greedy_baseline, spec_decode


def preflight(args) -> dict:
    """Refuse to measure on a machine that is not in a measurable state."""
    pw = U.power_info()
    host = U.host_info()
    con = U.contention_report()

    print("=" * 74)
    print("PREFLIGHT")
    print("=" * 74)
    print(f"  chip            {host.get('chip')}")
    print(f"  macOS / python  {host.get('macos')} / {host.get('python')}")
    print(f"  mlx / mlx-lm    {host.get('mlx_version')} / {host.get('mlx_lm_version')}")
    print(f"  power           {pw.get('power_source')}  "
          f"{pw.get('battery_percent')}%  low_power_mode={pw.get('low_power_mode')}")
    print(f"  memory pressure {con.get('memory_pressure_pct')}%")

    tops = con.get("top_processes") or []
    if tops:
        print("  top processes by RSS:")
        for t in tops[:6]:
            name = t.get("command") or t.get("name") or "?"
            mb = t.get("rss_mb") or t.get("mb") or "?"
            print(f"    {str(mb):>7} MB  {name}")

    # WHICH APPS ACTUALLY MATTER — settled from runs.jsonl, not from intuition.
    #
    # Recipe 02's entire baseline was measured with Claude Desktop resident: it appears in
    # 30/30 baseline-pertoken rows, 15/15 pipelined, 15/15 stock-stream. The DECLARED
    # 24.37 tok/s baseline therefore INCLUDES it. Chrome appears in 0 of 262 rows.
    #
    # So refusing to measure with Claude Desktop open was wrong twice over: it blocks the
    # normal working setup, and measuring without it would compare a quieter speculative run
    # against a Claude-resident baseline — inflating the ratio by exactly the asymmetry we
    # measured (contention hurts the bandwidth-bound baseline more than the speculative arm).
    #
    # Chrome stays a hard refusal. The harness README said this from the start: "quit Chrome
    # (not just close it). Claude Desktop at ~1 GB is fine."
    noisy = [t for t in tops
             if any(k in str(t.get("command") or t.get("name") or "").lower()
                    for k in ("chrome", "safari", "firefox", "slack"))]
    resident = [t for t in tops
                if "claude" in str(t.get("command") or t.get("name") or "").lower()]
    if resident:
        print("  Claude Desktop resident — expected; the declared baseline was measured")
        print("  with it running. Recorded, not refused.")
    if noisy and not args.allow_contention:
        print("\n  !! CONTENDING APPS STILL RUNNING:")
        for t in noisy:
            print(f"     {t.get('command') or t.get('name')}")
        print("     Contention is NOT symmetric: the baseline is bandwidth-bound and")
        print("     loses ~7%, while drafting is CPU-bound (CPU time exceeds wall time)")
        print("     and the speculative arm loses ~18%. A speedup measured here")
        print("     understates itself and is not comparable to a published number.")
        print("     Quit them (not just close the window), or pass --allow-contention to")
        print("     measure the loaded case deliberately.")
        raise SystemExit(2)

    fatal = []
    if pw.get("low_power_mode"):
        fatal.append("Low Power Mode is ON - it caps CPU/GPU frequency outright")
    pct = pw.get("battery_percent")
    if pct is not None and pct < 30:
        fatal.append(f"battery at {pct}% - below 30% the SoC limits peak power draw")
    if fatal:
        print()
        for f in fatal:
            print(f"  !! {f}")
        raise SystemExit(2)

    print("\n  state OK - measuring")
    return {"power": pw, "host": host, "contention": con,
            "claude_resident": bool(resident)}


def timed(fn) -> tuple:
    """Wall time around one fully-evaluated generation.

    The barrier belongs INSIDE the timed region and at its end: both
    greedy_baseline and spec_decode already force evaluation per step via
    mx.eval before their host-side .item() reads, so the returned token list is
    a host object and no lazy GPU work outlives the call. Timing therefore needs
    no trailing eval, and adding one would measure an empty queue.
    """
    U.clear_cache()
    mx.eval(mx.zeros(1))          # drain anything the previous arm left queued
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-4-e4b-it-OptiQ-4bit")
    ap.add_argument("--assistant", default="mlx-community/gemma-4-E4B-it-assistant-bf16")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=25.0,
                    help="Seconds idle between passes. Fanless M3 Air.")
    ap.add_argument("--allow-contention", action="store_true",
                    help="Measure the loaded case deliberately.")
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="PHASE 1 confidence schedule; 0 = fixed gamma (unchanged behaviour)")
    ap.add_argument("--tag", default="spec-speed-5pass")
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    env = preflight(args)

    from mlx_lm.models import gemma4_assistant

    print("\nloading target ...")
    target, tok = RF.load_model(args.target)
    tm = text_model(target)
    enable_kv_capture()
    print("loading drafter ...")
    cfg, path = RF.find_config(args.assistant)
    drafter = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(cfg))
    w = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        w.update(mx.load(str(f)))
    drafter.load_weights(list(w.items()), strict=True)
    mx.eval(drafter.parameters())

    prompts = []
    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        ids, _ = RF.build_eval_prompt(tok, prompt)
        prompts.append((pi, prompt, ids))

    def one_pass(n: int, warm: bool):
        """One pass over every prompt. Arm order alternates by pass index."""
        spec_first = (n % 2 == 1)
        pw0 = U.power_info()
        rows = []
        for pi, prompt, ids in prompts:
            if spec_first:
                t_spec, _ = timed(lambda: spec_decode(
                    target, tm, drafter, ids, args.tokens, args.gamma, {},
                    conf_threshold=args.conf_threshold))
                t_base, _ = timed(lambda: greedy_baseline(
                    target, tm, ids, args.tokens))
            else:
                t_base, _ = timed(lambda: greedy_baseline(
                    target, tm, ids, args.tokens))
                t_spec, _ = timed(lambda: spec_decode(
                    target, tm, drafter, ids, args.tokens, args.gamma, {},
                    conf_threshold=args.conf_threshold))
            rows.append({
                "prompt_index": pi, "prompt_tokens": len(ids),
                "spec_tok_s": args.tokens / t_spec,
                "baseline_tok_s": args.tokens / t_base,
                "speedup": t_base / t_spec,
                "arm_order": "spec_first" if spec_first else "baseline_first",
            })
        pw1 = U.power_info()
        # Only MEANINGFUL state changes. battery_percent ticks down on any run
        # that lasts minutes and is on battery -- that is discharge, not drift,
        # and flagging it voids every honest measurement ever taken unplugged.
        # What actually changes the machine: the power source flipping, Low
        # Power Mode toggling, or charge crossing the 30% floor where the SoC
        # starts limiting peak draw.
        drift = {}
        for k in ("power_source", "low_power_mode"):
            if pw0.get(k) != pw1.get(k):
                drift[k] = (pw0.get(k), pw1.get(k))
        b0, b1 = pw0.get("battery_percent"), pw1.get("battery_percent")
        if b0 is not None and b1 is not None and b0 >= 30 > b1:
            drift["battery_crossed_30pct"] = (b0, b1)
        label = "warmup" if warm else f"pass {n}"
        for r in rows:
            print(f"  {label:<8} prompt {r['prompt_index']}  "
                  f"spec {r['spec_tok_s']:6.2f}  base {r['baseline_tok_s']:6.2f}  "
                  f"{r['speedup']:5.2f}x  [{r['arm_order']}]")
        if drift:
            print(f"  !! host state changed during {label}: {drift}")
        return rows, drift

    print("\n" + "=" * 74)
    print(f"MEASURING — 1 warmup + {args.passes} passes, {args.cooldown:.0f}s cooldown")
    print("=" * 74)

    one_pass(0, warm=True)          # discarded: graph build, pipeline compile, page-ins

    records, drifts = [], []
    for n in range(1, args.passes + 1):
        time.sleep(args.cooldown)
        rows, drift = one_pass(n, warm=False)
        if drift:
            drifts.append((n, drift))
        for r in rows:
            r["pass_index"] = n
        records.extend(rows)

    # ------------------------------------------------------------- report
    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)

    def summarize(name, key, rows):
        vals = [r[key] for r in rows]
        med = stats.median(vals)
        cv = (stats.pstdev(vals) / med * 100) if med and len(vals) > 1 else 0.0
        print(f"  {name:<22}{med:7.2f}   "
              f"min {min(vals):6.2f}  max {max(vals):6.2f}  "
              f"spread {100 * (max(vals) - min(vals)) / med:4.1f}%  cv {cv:4.1f}%")
        return med, cv, min(vals), max(vals)

    out = {}
    for pi, prompt, _ in prompts:
        rows = [r for r in records if r["prompt_index"] == pi]
        print(f"\n  prompt {pi}: {prompt}")
        out[f"p{pi}_spec"] = summarize("speculative tok/s", "spec_tok_s", rows)
        out[f"p{pi}_base"] = summarize("baseline tok/s", "baseline_tok_s", rows)
        out[f"p{pi}_speedup"] = summarize("speedup", "speedup", rows)

    print("\n  across all prompts:")
    spec_med = stats.median([r["spec_tok_s"] for r in records])
    base_med = stats.median([r["baseline_tok_s"] for r in records])
    sp_med = stats.median([r["speedup"] for r in records])
    print(f"    speculative   {spec_med:6.2f} tok/s")
    print(f"    baseline      {base_med:6.2f} tok/s   (in-process, per-token eval)")
    print(f"    SPEEDUP       {sp_med:6.2f}x")

    # Arm-order control: if these disagree, intra-pass thermal drift is real.
    bf = [r["speedup"] for r in records if r["arm_order"] == "baseline_first"]
    sf = [r["speedup"] for r in records if r["arm_order"] == "spec_first"]
    if bf and sf:
        print(f"\n  arm-order control: baseline-first {stats.median(bf):.2f}x  "
              f"vs spec-first {stats.median(sf):.2f}x")
        if abs(stats.median(bf) - stats.median(sf)) / sp_med > 0.05:
            print("    -> the two orders disagree by more than 5%. Intra-pass thermal")
            print("       drift is biasing whichever arm runs second. Report the")
            print("       order-balanced median above, not a single-order number.")

    # bench.py's discipline: show the shape of the spread, do not trust a
    # summary statistic. One transient pass inflates cv while leaving the median
    # untouched -- which is the whole reason the median is the estimator here.
    print("\n  per-pass speculative tok/s (median is robust to a single transient):")
    for pi, _p, _i in prompts:
        vals = [(r["pass_index"], r["spec_tok_s"])
                for r in records if r["prompt_index"] == pi]
        med = stats.median([v for _n, v in vals])
        cells = "  ".join(
            f"{n}:{v:5.2f}{'*' if abs(v - med) / med > 0.05 else ' '}" for n, v in vals)
        print(f"    prompt {pi}   {cells}")
    outliers = [(r["pass_index"], r["prompt_index"], r["spec_tok_s"])
                for r in records
                if abs(r["spec_tok_s"] - stats.median(
                    [x["spec_tok_s"] for x in records
                     if x["prompt_index"] == r["prompt_index"]])) /
                stats.median([x["spec_tok_s"] for x in records
                              if x["prompt_index"] == r["prompt_index"]]) > 0.05]
    if outliers:
        bad = sorted({n for n, _p, _v in outliers})
        print(f"    * pass(es) {bad} deviate >5% from the median.")
        print("      A transient on one pass is absorbed by the median and is not a")
        print("      reason to discard the run; a TREND across passes would be.")

    worst_cv = max(v[1] for v in out.values())
    print()
    if drifts:
        print("  !! host state changed mid-run — this median is NOT publishable:")
        for n, d in drifts:
            print(f"     pass {n}: {d}")
    elif worst_cv > 5.0:
        print(f"  !! worst coefficient of variation {worst_cv:.1f}% > 5%. The machine")
        print("     was not in a steady state. Re-run with a longer cooldown before")
        print("     publishing this number.")
    else:
        print(f"  Stable: worst cv {worst_cv:.1f}%. Publishable, with conditions named:")
        print(f"    {args.passes} passes, median, temperature 0, batch 1, gamma={args.gamma},")
        print(f"    {env['power'].get('power_source')} power at "
              f"{env['power'].get('battery_percent')}%, "
              f"{'contended' if args.allow_contention else 'quiet machine'}.")
        print("    Name the baseline: this is against the in-process greedy baseline")
        print("    with a per-token mx.eval and both models resident, NOT against")
        print("    Recipe 0's standalone 24.5 tok/s.")

    with open(args.out, "a") as f:
        for r in records:
            r.update({
                "record_type": "spec_speed", "schema_version": 1, "tag": args.tag,
                "gamma": args.gamma, "gen_tokens": args.tokens,
                "conf_threshold": args.conf_threshold,
                "passes": args.passes, "cooldown_s": args.cooldown,
                "allow_contention": bool(args.allow_contention),
                "power": env["power"], "host": env["host"],
                "claude_desktop_resident": bool(env.get("claude_resident")),
                "memory_pressure_pct": env["contention"].get("memory_pressure_pct"),
                "top_processes": env["contention"].get("top_processes"),
                "host_state_drift": bool(drifts),
            })
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  appended {len(records)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

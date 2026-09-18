#!/usr/bin/env python3
"""
crossruntime.py -- Recipe 10, the cross-runtime comparison.

THE QUESTION (from tasks/CROSS-RUNTIME.md)

  Does this harness produce a better result than the runtimes people already
  use, or the same number with more ceremony?

WHY THIS IS A SEPARATE SCRIPT FROM speed_5pass.py

speed_5pass.py measures OUR two arms against each other and reports the
speculative arm against `greedy_baseline` -- the per-token-sync instrument
(20.03 tok/s), which CLAUDE.md forbids quoting as throughput. A cross-runtime
comparison needs a different plain arm: stock `mlx_lm.stream_generate`, the
DECLARED 24.37 baseline, because that is what another runtime's plain decode
is comparable to. It also has to run inside two different virtualenvs, which
speed_5pass.py cannot do.

THE REPORTING RULE THIS SCRIPT EXISTS TO ENFORCE

Raw tok/s alone is not a comparison. A runtime using a smaller quant reads
fewer bytes, wins on tok/s, and is LESS efficient. So every arm carries:

    raw tok/s | active bytes/token | effective GB/s | % of the 89 GB/s ceiling

`active bytes/token` is DERIVED, never assumed: roofline.param_breakdown walks
the actual loaded parameter tree in whichever runtime is running, applying the
same PLE/gather-only exclusion in both. That is why this script imports
roofline from the mlx-bench directory even when running under the mlx-vlm venv.

WHAT % OF CEILING MEANS, AND WHERE IT DOES NOT APPLY

For PLAIN decode, bytes/token is the weight traffic of one forward, so
bytes x tok/s is real DRAM traffic and %-of-89 is meaningful. This is the
kernel-quality comparison and the actual test of the pre-registered prediction.

For SPECULATIVE decode it is NOT. A speculative loop reads the target weights
once per verify and emits several tokens from it, so traffic per EMITTED token
is lower than 3.535 GB. Multiplying an emitted-token rate by the plain-decode
byte count would produce an effective GB/s above the 89 GB/s ceiling -- an
impossible number, and exactly the class of invalid ceiling this project has
already been burned by seven times. So the speculative arms report an AMORTISED
byte count where the runtime exposes the acceptance data needed to derive it,
and report it as not derivable where it does not. Never the plain figure.

USAGE -- one process per runtime, because a 7 GB model on a 16 GB machine
cannot be resident twice.

    ../bin/python crossruntime.py --runtime ours   --session <id>
    ~/mlx-vlm-control/bin/python crossruntime.py --runtime mlxvlm --session <id>

Both append `crossruntime_run` rows to runs.jsonl carrying the shared session
id, so the analysis step can prove they came from one session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as stats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mlx.core as mx

import mlxutil as U
import roofline as RF

CEILING_GB_S = 89.0          # calibrated, NOT the 100 on the spec sheet
TARGET = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
ASSISTANT = "mlx-community/gemma-4-E4B-it-assistant-bf16"

# Copied verbatim from measure_acceptance.PROMPTS. Duplicated rather than
# imported because measure_acceptance imports mlx_lm, which does not exist in
# the mlx-vlm control venv. The `ours` path asserts they still match.
PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that reverses a linked list.",
    "Summarise the causes of the French Revolution.",
    "What is the difference between a mutex and a semaphore?",
]


def ids_sha(ids) -> str:
    """Fingerprint of the exact token ids fed to the model. Both runtimes record
    it; if they differ, the runtimes were not given the same work and the
    comparison is void."""
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def eff_gbs(bytes_per_token: float, tok_s: float) -> float:
    return bytes_per_token * tok_s / 1e9


# ------------------------------------------------------------------ preflight
def preflight(args) -> dict:
    pw = U.power_info()
    host = U.host_info()
    con = U.contention_report()

    print("=" * 74)
    print(f"PREFLIGHT -- runtime={args.runtime}  session={args.session}")
    print("=" * 74)
    print(f"  chip            {host.get('chip')}")
    print(f"  macOS / python  {host.get('macos')} / {host.get('python')}")
    print(f"  mlx             {host.get('mlx_version')}")
    print(f"  power           {pw.get('power_source')}  {pw.get('battery_percent')}%  "
          f"low_power_mode={pw.get('low_power_mode')}")
    print(f"  memory pressure {con.get('memory_pressure_pct')}%")

    tops = con.get("top_processes") or []
    if tops:
        print("  top processes by RSS:")
        for t in tops[:6]:
            print(f"    {str(t.get('rss_mb') or t.get('mb') or '?'):>7} MB  "
                  f"{t.get('command') or t.get('name') or '?'}")

    # Chrome is the hard refusal (README + speed_5pass). Claude Desktop resident
    # is EXPECTED, not contamination -- the declared baseline included it.
    noisy = [t for t in tops
             if any(k in str(t.get("command") or t.get("name") or "").lower()
                    for k in ("chrome", "safari", "firefox", "slack"))]
    if noisy and not args.allow_contention:
        print("\n  !! CONTENDING APPS RUNNING:")
        for t in noisy:
            print(f"     {t.get('command') or t.get('name')}")
        print("     Contention is not symmetric across arms. Quit them, or pass")
        print("     --allow-contention to measure the loaded case deliberately.")
        raise SystemExit(2)

    fatal = []
    if pw.get("low_power_mode"):
        fatal.append("Low Power Mode is ON -- caps CPU/GPU frequency outright.")
    bp = pw.get("battery_percent")
    if bp is not None and bp < 30:
        fatal.append(f"battery at {bp}% -- under 30% the SoC limits peak draw.")
    if fatal:
        print()
        for f in fatal:
            print(f"  !! {f}")
        raise SystemExit(2)

    print("  preflight OK\n")
    return {"power": pw, "host": host, "contention": con}


def state_drift(pw0: dict, pw1: dict) -> dict:
    """Only MEANINGFUL changes. Battery ticking down on a multi-minute unplugged
    run is discharge, not host-state drift -- CLAUDE.md is explicit that
    treating it as drift voids every honest measurement taken on battery."""
    drift = {}
    for k in ("power_source", "low_power_mode"):
        if pw0.get(k) != pw1.get(k):
            drift[k] = (pw0.get(k), pw1.get(k))
    b0, b1 = pw0.get("battery_percent"), pw1.get("battery_percent")
    if b0 is not None and b1 is not None and b0 >= 30 > b1:
        drift["battery_crossed_30pct"] = (b0, b1)
    return drift


# ------------------------------------------------------------ runtime: OURS
class OursRuntime:
    """mlx-lm 0.31.3 in ~/mlx-env. Plain arm is stock `mlx_lm.stream_generate`
    -- the DECLARED 24.37 baseline, not the per-token-sync instrument."""

    name = "ours (mlx-lm 0.31.3 + this harness)"
    key = "ours"

    def __init__(self, args):
        from measure_acceptance import PROMPTS as MA_PROMPTS, enable_kv_capture, text_model
        assert list(MA_PROMPTS) == PROMPTS, (
            "PROMPTS drifted from measure_acceptance.PROMPTS -- the two runtimes "
            "would no longer be given identical work.")
        from mlx_lm.models import gemma4_assistant

        print("loading target ...")
        self.target, self.tok = RF.load_model(args.target)
        self.tm = text_model(self.target)
        enable_kv_capture()

        print("loading drafter ...")
        cfg, path = RF.find_config(args.assistant)
        self.drafter = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(cfg))
        w = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            w.update(mx.load(str(f)))
        self.drafter.load_weights(list(w.items()), strict=True)
        mx.eval(self.drafter.parameters())

        self.cfg, _ = RF.find_config(args.target)
        self.pb = RF.param_breakdown(self.target, self.cfg)
        self.dpb = RF.param_breakdown(self.drafter, cfg)
        self.gamma = args.gamma
        self.versions = {
            "mlx": U.host_info().get("mlx_version"),
            "mlx_lm": U.host_info().get("mlx_lm_version"),
        }

    def build_ids(self, prompt):
        ids, how = RF.build_eval_prompt(self.tok, prompt)
        return list(ids), how

    def arms(self):
        return ["plain", "spec"]

    def run(self, arm, ids, max_tokens):
        if arm == "plain":
            return self._plain(ids, max_tokens)
        return self._spec(ids, max_tokens)

    def _plain(self, ids, max_tokens):
        """Stock mlx_lm.stream_generate. Timed the same way as every other
        runtime here: from the FIRST emitted token, so prefill is excluded and
        the number is a decode rate."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        U.clear_cache()
        t0 = time.perf_counter()
        t_first = None
        toks = []
        last = None
        for resp in stream_generate(self.target, self.tok, mx.array(ids),
                                    max_tokens=max_tokens,
                                    sampler=make_sampler(temp=0.0)):
            if t_first is None:
                t_first = time.perf_counter()
            toks.append(resp.token)
            last = resp
        t_end = time.perf_counter()
        return {
            "tokens": len(toks),
            "decode_s": (t_end - t_first) if t_first else None,
            "wall_s": t_end - t0,
            "ttft_s": (t_first - t0) if t_first else None,
            "lib_generation_tps": getattr(last, "generation_tps", None),
            "extra": {},
        }

    def _spec(self, ids, max_tokens):
        from spec_generate import spec_decode
        st = {}
        U.clear_cache()
        t0 = time.perf_counter()
        out = spec_decode(self.target, self.tm, self.drafter, ids, max_tokens,
                          self.gamma, st)
        t_end = time.perf_counter()
        # spec_decode does its own prefill then emits; it returns the full token
        # list at the end rather than streaming, so there is no per-token first
        # mark. wall == decode + prefill. Recorded honestly as such.
        return {
            "tokens": len(out),
            "decode_s": None,
            "wall_s": t_end - t0,
            "ttft_s": None,
            "lib_generation_tps": None,
            "extra": {
                "gamma": self.gamma,
                "cycles": st.get("cycles"),
                "mean_accepted": st.get("mean_accepted"),
                "mean_drafted": st.get("mean_drafted"),
                "acceptance_exposed": True,
            },
        }


# --------------------------------------------------------- runtime: MLX-VLM
class MlxVlmRuntime:
    """mlx-vlm 0.6.17 in ~/mlx-vlm-control. Same mlx 0.32.2, same weights, and
    -- for the speculative arm -- the same drafter architecture
    (`gemma4_assistant`, which mlx-vlm maps to draft_kind='mtp').

    That is what makes this the controlled comparison: identical bytes, so any
    difference is loop and kernel quality, not quantisation."""

    name = "mlx-vlm 0.6.17"
    key = "mlxvlm"

    def __init__(self, args):
        from mlx_vlm import load
        from mlx_vlm.speculative.drafters import (
            DRAFTER_KIND_BY_MODEL_TYPE, validate_drafter_compatibility)

        print("loading target ...")
        self.model, self.processor = load(args.target)
        self.tok = getattr(self.processor, "tokenizer", self.processor)

        print("loading drafter ...")
        self.drafter, _ = load(args.assistant)
        dcfg = getattr(self.drafter, "config", None)
        mt = (dcfg.get("model_type") if isinstance(dcfg, dict)
              else getattr(dcfg, "model_type", None))
        self.draft_kind = DRAFTER_KIND_BY_MODEL_TYPE.get(mt)
        validate_drafter_compatibility(self.model, self.drafter, self.draft_kind)
        self.drafter_model_type = mt
        self.configured_block = int(getattr(dcfg, "block_size",
                                            (dcfg or {}).get("block_size", 0)
                                            if isinstance(dcfg, dict) else 0) or 0)

        self.cfg, _ = RF.find_config(args.target)
        self.pb = RF.param_breakdown(self.model, self.cfg)
        dc, _ = RF.find_config(args.assistant)
        self.dpb = RF.param_breakdown(self.drafter, dc)
        self.gamma = args.gamma
        import mlx_vlm
        self.versions = {"mlx": U.host_info().get("mlx_version"),
                         "mlx_vlm": getattr(mlx_vlm, "__version__", None)}

    def build_ids(self, prompt):
        ids, how = RF.build_eval_prompt(self.tok, prompt)
        return list(ids), how

    def arms(self):
        # `spec_default` is mlx-vlm at ITS OWN default block size -- the number a
        # user actually gets. `spec_matched` forces our gamma so the loops are
        # compared at the same block. Reporting only one of the two would repeat
        # Phase 1's straw-man-operating-point mistake, in one direction or other.
        return ["plain", "spec_default", "spec_matched"]

    def _step_kwargs(self, arm):
        if arm == "plain":
            return {}
        kw = {"draft_model": self.drafter, "draft_kind": self.draft_kind}
        if arm == "spec_matched":
            kw["draft_block_size"] = self.gamma
        return kw

    def run(self, arm, ids, max_tokens):
        from mlx_vlm.generate.ar import generate_step

        U.clear_cache()
        t0 = time.perf_counter()
        t_first = None
        n = 0
        kw = self._step_kwargs(arm)
        for tok, _lp in generate_step(mx.array([ids]), self.model, None, None,
                                      max_tokens=max_tokens, temperature=0.0, **kw):
            if t_first is None:
                t_first = time.perf_counter()
            n += 1
            if n >= max_tokens:
                break
        t_end = time.perf_counter()
        extra = {}
        if arm != "plain":
            extra = {
                "draft_kind": self.draft_kind,
                "drafter_model_type": self.drafter_model_type,
                "draft_block_size": kw.get("draft_block_size", "runtime default"),
                "configured_block_size": self.configured_block,
                # mlx-vlm's speculative generator yields (token, logprobs) only.
                # It exposes no accepted/drafted counters to the caller, and
                # instrumenting it would add host syncs to the loop under test.
                "acceptance_exposed": False,
            }
        return {
            "tokens": n,
            "decode_s": (t_end - t_first) if t_first else None,
            "wall_s": t_end - t0,
            "ttft_s": (t_first - t0) if t_first else None,
            "lib_generation_tps": None,
            "extra": extra,
        }


RUNTIMES = {"ours": OursRuntime, "mlxvlm": MlxVlmRuntime}


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=sorted(RUNTIMES))
    ap.add_argument("--session", required=True,
                    help="Shared id proving every runtime was measured in ONE session.")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--assistant", default=ASSISTANT)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--allow-contention", action="store_true")
    ap.add_argument("--ids-file", default="crossruntime_ids.json",
                    help="Shared token ids. The FIRST runtime to run writes it; every "
                         "later runtime loads it verbatim. Two runtimes' tokenizers can "
                         "disagree about the chat template (mlx-lm applies it, mlx-vlm's "
                         "processor does not), and different ids mean different work -- "
                         "which is not a comparison. This makes the ids a shared input.")
    ap.add_argument("--smoke", action="store_true",
                    help="One un-timed pass to prove the code path runs. Writes nothing.")
    ap.add_argument("--tag", default="recipe10-crossruntime")
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    env = preflight(args)
    rt = RUNTIMES[args.runtime](args)

    prompts = []
    idsf = Path(args.ids_file)
    shared = json.loads(idsf.read_text()) if idsf.exists() else None
    if shared is not None:
        print(f"  loading shared token ids from {idsf}")
    built = []
    for pi, p in enumerate(PROMPTS[: args.prompts]):
        own_ids, how = rt.build_ids(p)
        if shared is not None:
            rec = shared["prompts"][pi]
            assert rec["prompt"] == p, f"ids file prompt {pi} does not match PROMPTS"
            ids = list(rec["ids"])
            if list(own_ids) != ids:
                # Not an error -- it is the whole reason the file exists. Say so
                # loudly and record it, then use the shared ids so both runtimes
                # do identical work.
                print(f"  note: this runtime's own tokenisation of prompt {pi} differs "
                      f"({len(own_ids)} ids, {how}); using the shared ids "
                      f"({len(ids)} ids, {rec['how']}) so the work is identical.")
            how = rec["how"] + " (shared)"
        else:
            ids = list(own_ids)
            built.append({"prompt": p, "ids": ids, "how": how})
        prompts.append((pi, p, ids, how))
        print(f"  prompt {pi}: {len(ids)} ids  sha={ids_sha(ids)}  [{how}]")
    if shared is None:
        idsf.write_text(json.dumps(
            {"note": "Shared token ids for Recipe 10. Written by the first runtime to "
                     "run; every later runtime loads these verbatim so all runtimes are "
                     "given byte-identical work.",
             "source_runtime": rt.key, "session": args.session,
             "prompts": built}, indent=1))
        print(f"  wrote shared token ids -> {idsf}")

    print(f"\n  active bytes/token (derived from the loaded parameter tree):")
    print(f"    total        {U.fmt_bytes(rt.pb.total_bytes)}")
    print(f"    gather-only  {U.fmt_bytes(rt.pb.gather_only_bytes)}  (resident, not streamed)")
    print(f"    ACTIVE       {U.fmt_bytes(rt.pb.active_bytes_per_token)}"
          f"  = {rt.pb.active_bytes_per_token/1e9:.4f} GB")
    print(f"    drafter active {rt.dpb.active_bytes_per_token/1e6:.1f} MB")

    if args.smoke:
        print("\nSMOKE -- one short run per arm, nothing recorded\n")
        for arm in rt.arms():
            pi, p, ids, _ = prompts[0]
            r = rt.run(arm, ids, 24)
            dur = r["decode_s"] or r["wall_s"]
            print(f"  {arm:<14} {r['tokens']:3d} tok in {dur:5.2f}s "
                  f"-> {r['tokens']/dur:6.2f} tok/s  {r['extra']}")
        print("\nsmoke OK")
        return 0

    arms = rt.arms()
    records = []
    drifts = []

    def one_pass(n: int, warm: bool):
        # Rotate arm order every pass: within a pass the later arm runs on a
        # hotter chip, so a fixed order biases one arm systematically.
        order = arms[n % len(arms):] + arms[: n % len(arms)]
        pw0 = U.power_info()
        con = U.contention_report()
        rows = []
        for pi, p, ids, how in prompts:
            for arm in order:
                r = rt.run(arm, ids, args.tokens)
                dur = r["decode_s"] if r["decode_s"] else r["wall_s"]
                tps = r["tokens"] / dur if dur else None
                rows.append({
                    "arm": arm, "prompt_index": pi,
                    "prompt_tokens": len(ids), "prompt_ids_sha": ids_sha(ids),
                    "prompt_build": how,
                    "gen_tokens_requested": args.tokens,
                    "gen_tokens_actual": r["tokens"],
                    "tok_s": tps,
                    "decode_s": r["decode_s"], "wall_s": r["wall_s"],
                    "ttft_s": r["ttft_s"],
                    "lib_generation_tps": r["lib_generation_tps"],
                    "timing_basis": ("first-token-to-last (prefill excluded)"
                                     if r["decode_s"] else
                                     "wall clock incl. prefill (arm does not stream)"),
                    "arm_order": order,
                    **r["extra"],
                })
        pw1 = U.power_info()
        d = state_drift(pw0, pw1)
        label = "warmup" if warm else f"pass {n}"
        for r in rows:
            print(f"  {label:<8} p{r['prompt_index']} {r['arm']:<14} "
                  f"{r['gen_tokens_actual']:3d} tok  {r['tok_s']:6.2f} tok/s")
        if d:
            print(f"  !! host state changed during {label}: {d}")
        return rows, d, {"power": pw0, "contention": con}

    print("\n" + "=" * 74)
    print(f"WARMUP (discarded) then {args.passes} timed passes, "
          f"{args.cooldown:.0f}s cooldown between")
    print("=" * 74)

    one_pass(0, warm=True)

    for n in range(1, args.passes + 1):
        print(f"  cooling {args.cooldown:.0f}s ...")
        time.sleep(args.cooldown)
        rows, d, st = one_pass(n, warm=False)
        if d:
            drifts.append((n, d))
        for r in rows:
            r.update({"pass": n,
                      "memory_pressure_pct": st["contention"].get("memory_pressure_pct"),
                      "top_processes": st["contention"].get("top_processes"),
                      "power": st["power"]})
        records.extend(rows)

    # ------------------------------------------------------------ summarise
    print("\n" + "=" * 74)
    print(f"MEDIANS -- {rt.name}")
    print("=" * 74)
    summary = {}
    for arm in arms:
        vals = [r["tok_s"] for r in records if r["arm"] == arm and r["tok_s"]]
        if not vals:
            continue
        med = stats.median(vals)
        cv = 100.0 * (stats.pstdev(vals) / med) if med else float("nan")
        plain_bytes = rt.pb.active_bytes_per_token
        row = {"median_tok_s": med, "n": len(vals), "cv_pct": cv,
               "min": min(vals), "max": max(vals)}
        if arm == "plain":
            g = eff_gbs(plain_bytes, med)
            row.update({"active_bytes_per_token": plain_bytes,
                        "effective_gb_s": g,
                        "pct_of_ceiling": 100.0 * g / CEILING_GB_S})
            print(f"  {arm:<14} {med:6.2f} tok/s  cv {cv:4.1f}%   "
                  f"{plain_bytes/1e9:.3f} GB/tok  {g:5.2f} GB/s  "
                  f"{100.0*g/CEILING_GB_S:5.1f}% of {CEILING_GB_S:.0f}")
        else:
            # Amortised traffic per EMITTED token, derivable only where the
            # runtime exposes acceptance. Never the plain figure -- see header.
            acc = [r.get("mean_accepted") for r in records
                   if r["arm"] == arm and r.get("mean_accepted")]
            if acc:
                ma = stats.median(acc)
                emitted_per_cycle = ma + 1.0        # accepted + the bonus token
                drafted = stats.median([r.get("mean_drafted") for r in records
                                        if r["arm"] == arm and r.get("mean_drafted")])
                per_cycle = plain_bytes + drafted * rt.dpb.active_bytes_per_token
                amort = per_cycle / emitted_per_cycle
                g = eff_gbs(amort, med)
                row.update({"mean_accepted": ma, "mean_drafted": drafted,
                            "amortised_bytes_per_emitted_token": amort,
                            "effective_gb_s": g,
                            "pct_of_ceiling": 100.0 * g / CEILING_GB_S,
                            "bytes_basis": "amortised: (target active + drafted x drafter active) "
                                           "/ (mean_accepted + 1)"})
                print(f"  {arm:<14} {med:6.2f} tok/s  cv {cv:4.1f}%   "
                      f"{amort/1e9:.3f} GB/tok(amort)  {g:5.2f} GB/s  "
                      f"{100.0*g/CEILING_GB_S:5.1f}% of {CEILING_GB_S:.0f}   "
                      f"acc {ma:.2f}")
            else:
                row.update({"amortised_bytes_per_emitted_token": None,
                            "effective_gb_s": None, "pct_of_ceiling": None,
                            "bytes_basis": "NOT DERIVABLE -- runtime does not expose "
                                           "acceptance, so traffic per emitted token "
                                           "cannot be derived. Plain-decode bytes/token "
                                           "would be wrong here and is deliberately omitted."})
                print(f"  {arm:<14} {med:6.2f} tok/s  cv {cv:4.1f}%   "
                      f"bytes/token NOT DERIVABLE (acceptance not exposed)")
        summary[arm] = row

    worst_cv = max((v["cv_pct"] for v in summary.values()), default=0.0)
    print()
    if drifts:
        print("  !! host state changed mid-run -- these medians are NOT publishable:")
        for n, d in drifts:
            print(f"     pass {n}: {d}")
    elif worst_cv > 5.0:
        print(f"  !! worst cv {worst_cv:.1f}% > 5%. Not a steady state; re-run "
              f"with a longer cooldown before publishing.")
    else:
        print(f"  Stable: worst cv {worst_cv:.1f}%.")

    meta = {
        "record_type": "crossruntime_run", "schema_version": 1,
        "recipe": "10-cross-runtime", "session": args.session,
        "runtime": rt.key, "runtime_name": rt.name, "versions": rt.versions,
        "tag": args.tag, "target": args.target, "assistant": args.assistant,
        "gen_tokens": args.tokens, "passes": args.passes,
        "cooldown_s": args.cooldown, "temperature": 0.0,
        "active_bytes_per_token": rt.pb.active_bytes_per_token,
        "drafter_active_bytes_per_token": rt.dpb.active_bytes_per_token,
        "param_total_bytes": rt.pb.total_bytes,
        "param_gather_only_bytes": rt.pb.gather_only_bytes,
        "ceiling_gb_s": CEILING_GB_S,
        "host": env["host"], "host_state_drift": bool(drifts),
        "claude_desktop_resident": any(
            "claude" in str(t.get("command") or t.get("name") or "").lower()
            for t in (env["contention"].get("top_processes") or [])),
        "allow_contention": bool(args.allow_contention),
    }
    with open(args.out, "a") as f:
        for r in records:
            f.write(json.dumps({**meta, **r}, default=str) + "\n")
        f.write(json.dumps({**meta, "record_type": "crossruntime_summary",
                            "summary": summary, "worst_cv_pct": worst_cv}, default=str) + "\n")
    print(f"\n  appended {len(records)} crossruntime_run rows + 1 summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

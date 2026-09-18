#!/usr/bin/env python3
"""
check_model.py — is this checkpoint usable, before you spend an hour on it?

Screens a model in about a minute:

  1. Loads it (applying our model_type remap + non-text weight drops).
  2. Reports whether the PER-LAYER EMBEDDING table is quantized, and at what
     bits. This is the specific known defect behind "MLX quantized Gemma 4
     produces garbage": mlx-lm's gemma4_text.quant_predicate special-cases only
     `router.proj` and returns True for everything else, so PLE gets quantized
     to 4 bits along with the weight matrices. PLE is a lookup table whose rows
     ARE the signal — 4-bit affine quantization destroys it, and the model
     still loads perfectly and still produces fluent-looking tokens.
  3. Reports the dense / gather-only / non-text byte split, which is what the
     decode roofline actually depends on.
  4. Runs a greedy coherence check and prints a verdict.

    ../bin/python check_model.py --model mlx-community/gemma-4-e4b-it-OptiQ-4bit
    ../bin/python check_model.py --model <repo> --prompt "Explain gravity briefly."
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import mlxutil as U
import roofline as RF

PLE_MARKERS = ("embed_tokens_per_layer", "per_layer_embed", "embed_per_layer")


def leaf_modules(model):
    try:
        return tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module))
    except Exception:
        return []


def quant_report(model) -> dict:
    """Which modules are quantized, at what bits — with PLE called out."""
    rows, ple, embed = [], [], []
    for path, mod in leaf_modules(model):
        bits = getattr(mod, "bits", None)
        gs = getattr(mod, "group_size", None)
        kind = type(mod).__name__
        rec = {"path": path, "kind": kind, "bits": bits, "group_size": gs}
        rows.append(rec)
        low = path.lower()
        if any(m in low for m in PLE_MARKERS):
            ple.append(rec)
        elif low.endswith("embed_tokens"):
            embed.append(rec)
    return {"all": rows, "ple": ple, "embed": embed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Feed a bare completion prompt (still with BOS) instead of "
                         "the model's chat template.")
    ap.add_argument("--remap-model-type", action="append", metavar="FROM=TO")
    args = ap.parse_args()

    remap = {}
    for pair in (args.remap_model_type or []):
        if "=" in pair:
            k, v = pair.split("=", 1)
            remap[k.strip()] = v.strip()

    cfg, local = RF.find_config(args.model)
    print("=" * 78)
    print(f"CHECKING  {args.model}")
    print("=" * 78)
    if cfg:
        tc = cfg.get("text_config", cfg)
        print(f"  model_type          {cfg.get('model_type')}  /  text: {tc.get('model_type')}")
        print(f"  layers x hidden     {tc.get('num_hidden_layers')} x {tc.get('hidden_size')}")
        print(f"  quantization        {cfg.get('quantization')}")
        print(f"  PLE dims            hidden_per_layer={tc.get('hidden_size_per_layer_input')} "
              f"vocab_per_layer={tc.get('vocab_size_per_layer_input')}")

    print("\nLoading ...")
    try:
        model, tokenizer = RF.load_model(args.model, remap)
    except Exception as e:
        print(f"\nFAIL — will not load: {type(e).__name__}: {e}")
        return 2
    U.reset_peak_memory()

    # ---- PLE quantization check -------------------------------------
    qr = quant_report(model)
    print("\n" + "=" * 78)
    print("PER-LAYER EMBEDDING (PLE) QUANTIZATION")
    print("=" * 78)
    verdict_ple = "n/a"
    if not qr["ple"]:
        print("  no per-layer embedding table in this model — the PLE defect does not apply")
        verdict_ple = "n/a"
    else:
        for r in qr["ple"]:
            q = "NOT quantized" if r["bits"] is None else f"QUANTIZED at {r['bits']} bits (group {r['group_size']})"
            print(f"  {r['path']:<44}{r['kind']:<20}{q}")
        bits = [r["bits"] for r in qr["ple"] if r["bits"] is not None]
        if not bits:
            print("\n  GOOD: PLE is kept at full precision.")
            verdict_ple = "good"
        elif min(bits) <= 4:
            print(f"\n  BAD: PLE quantized to {min(bits)} bits. This is the known defect —")
            print("  the table's rows are the signal, and 4-bit affine quantization")
            print("  destroys them. The model loads and generates fluent nonsense.")
            verdict_ple = "bad"
        else:
            print(f"\n  MARGINAL: PLE at {min(bits)} bits. Better than 4, still lossy.")
            print("  Judge by the coherence check below.")
            verdict_ple = "marginal"
    for r in qr["embed"]:
        q = "not quantized" if r["bits"] is None else f"{r['bits']} bits"
        print(f"  (main embed_tokens: {q})")

    # ---- byte split --------------------------------------------------
    pb = RF.param_breakdown(model, cfg)
    print("\n" + "=" * 78)
    print("BYTE SPLIT")
    print("=" * 78)
    print(f"  total parameters          {U.fmt_bytes(pb.total_bytes)}")
    print(f"  dense (streams per token) {U.fmt_bytes(pb.dense_bytes)}")
    print(f"  gather-only (resident)    {U.fmt_bytes(pb.gather_only_bytes)}")
    print(f"  main embedding            {U.fmt_bytes(pb.embed_bytes)}  tied={pb.tied_embeddings}")
    print(f"  non-text (excluded)       {U.fmt_bytes(pb.non_text_bytes)}")
    print(f"  ACTIVE BYTES / TOKEN      {U.fmt_bytes(pb.active_bytes_per_token)}")
    if pb.total_bytes:
        pct = 100.0 * pb.gather_only_bytes / pb.total_bytes
        if pct > 5:
            print(f"  -> {pct:.0f}% of this checkpoint never streams during decode.")

    # ---- coherence ---------------------------------------------------
    print("\n" + "=" * 78)
    print("COHERENCE CHECK")
    print("=" * 78)
    try:
        print(f"  prompt : {args.prompt!r}")
        txt = RF.sanity_generate(model, tokenizer, n=args.tokens, prompt=args.prompt,
                                 use_chat_template=not args.no_chat_template)
        print(f"  output : {txt!r}")
        # Second pass with the other prompt path. If one is coherent and the
        # other is not, the model is fine and the prompt format was the problem.
        txt2 = RF.sanity_generate(model, tokenizer, n=args.tokens, prompt=args.prompt,
                                  use_chat_template=args.no_chat_template, verbose=False)
        other = "bare+bos" if not args.no_chat_template else "chat template"
        print(f"  output ({other}): {txt2!r}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return 3

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if verdict_ple == "bad":
        print("  REJECT — PLE is 4-bit quantized. Even if the text above looks")
        print("  plausible, quality is compromised. Find a PLE-safe conversion.")
        rc = 1
    elif verdict_ple == "marginal":
        print("  JUDGE BY OUTPUT — PLE is quantized above 4 bits. If the text reads")
        print("  correctly this checkpoint is usable; note the caveat in runs.jsonl.")
        rc = 0
    else:
        print("  PLE is fine. Accept if the text above is a coherent continuation.")
        rc = 0
    print(f"  peak memory during load+gen: {U.fmt_bytes(U.peak_memory())}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

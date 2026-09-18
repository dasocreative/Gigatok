#!/usr/bin/env python3
"""
roofline.py — calibrated bytes-per-token derivation for a specific MLX model.

Design decision, and it is the important one in this file:

  We do NOT derive KV bytes from a formula over config.json fields.
  We instantiate the model's OWN cache objects via make_prompt_cache(), run
  short real prefills, and measure the resident bytes of each cache's `.state`.

Reason: Gemma 4 Unified is a hybrid — interleaved sliding-window local layers
and periodic full-attention global layers, and mlx-lm builds a heterogeneous
cache list (KVCache for full-attention layers, RotatingKVCache(max_size=
sliding_window) for sliding layers). The global layers additionally use a
different head dim (`global_head_dim`) and their own KV head count
(`num_global_key_value_heads`), and the model card describes "unified Keys and
Values" on those layers. A naive 2 * L * H_kv * D * bytes formula is wrong on
every one of those counts, and would silently overstate KV by several x.

Measuring the actual cache objects is immune to all of that. It also
automatically handles: capped rotating windows, K==V sharing (same array
object appears twice), cross-layer KV sharing (same cache object appears in the
list twice), and quantized KV caches.

Two different numbers are reported, and they are not the same thing:

  kv_resident_bytes(L)        deduplicated by array identity — what actually
                              occupies unified memory. This is what the memory
                              guard must use.
  kv_read_bytes_per_token(L)  NOT deduplicated across layers — every layer that
                              attends reads its keys/values from DRAM, even if
                              two layers share one buffer. This is what the
                              bandwidth roofline must use.

Usage:
    python roofline.py --model mlx-community/gemma-4-12B-it-4bit
    python roofline.py --model <path> --contexts 128,2048,8192,16384 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten

import mlxutil as U

# Substrings that mark a parameter as belonging to a non-text tower. Gemma 4
# 12B is "encoder-free" multimodal (raw image patches / audio projected by
# lightweight linear layers), so these are small — but they are still bytes
# that never move during text decode, and counting them inflates the roofline.
NON_TEXT_MARKERS = (
    "vision", "visual", "image", "audio", "speech",
    "multi_modal", "mm_", "_projector", "projector",
)

EMBED_MARKERS = ("embed_tokens", "tok_embeddings", "wte", "embed.weight")

# Tables that are ONLY ever row-gathered, never matmul'd. They occupy unified
# memory but contribute ~one row of traffic per token, so counting them as
# active weight bytes overstates the roofline denominator badly.
#
# Gemma 4 Unified has a big one: `embed_tokens_per_layer`, an
# nn.Embedding(vocab_size_per_layer_input, num_hidden_layers *
# hidden_size_per_layer_input). At 262144 x 48 x 256 that is ~3.2B parameters —
# a large fraction of the "12B" parameter count that never streams during
# decode. (Verified in mlx_lm/models/gemma4_text.py, mlx-lm 0.31.3.)
GATHER_ONLY_MARKERS = (
    "embed_tokens_per_layer", "per_layer_embed", "embed_per_layer",
    "altup_embed", "altup_proj_embed",
)


# --------------------------------------------------------------------------
# model / config loading
# --------------------------------------------------------------------------


def load_model(path_or_repo: str, remap: Optional[dict] = None):
    from mlx_lm import load

    applied = U.patch_model_remapping(remap)
    for k, v in applied:
        print(f"  [compat] model_type '{k}' is not implemented in this mlx-lm; "
              f"routing it to mlx_lm.models.{v}")
        print(f"           This is a local patch, not upstream behaviour. "
              f"Verify the generation sanity check below before trusting numbers.")
    U.patch_multimodal_weight_drop()
    model, tokenizer = load(path_or_repo)
    return model, tokenizer


def bos_id(tokenizer) -> Optional[int]:
    for attr in ("bos_token_id",):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            return v
    inner = getattr(tokenizer, "_tokenizer", None)
    v = getattr(inner, "bos_token_id", None) if inner is not None else None
    return v if isinstance(v, int) else None


def build_eval_prompt(tokenizer, prompt: str, use_chat_template: bool = True):
    """
    Turn a prompt string into token ids the model was actually trained to see.

    THIS IS NOT A DETAIL. Gemma is trained with a leading <bos> and, for the
    -it checkpoints, with the <start_of_turn> chat template. Feed it a bare
    completion string with add_special_tokens=False and it degenerates into
    repetition loops that look exactly like a broken checkpoint. Getting this
    wrong makes a perfectly good model look destroyed.

    Returns (ids, how) where `how` names the path taken, so the caller can
    print it and you can see what the model was fed.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )
            if isinstance(out, str):
                ids = tokenizer.encode(out)
                return list(ids), "chat_template->encode"
            return list(out), "chat_template"
        except Exception:
            pass
    # Plain completion, but WITH special tokens so <bos> is present.
    try:
        ids = list(tokenizer.encode(prompt))
        how = "encode(default specials)"
    except Exception:
        ids, how = [], "failed"
    b = bos_id(tokenizer)
    if b is not None and (not ids or ids[0] != b):
        ids = [b] + ids
        how += " +manual bos"
    return ids, how


def sanity_generate(model, tokenizer, n: int = 24, prompt: str = "The capital of France is",
                    use_chat_template: bool = True, verbose: bool = True) -> str:
    """
    Generate a few greedy tokens and return them.

    For a checkpoint loaded through a hand-added model_type remap this is not a
    nicety: a shape-compatible but semantically wrong load produces fluent
    garbage while every bandwidth number still looks reasonable.

    But the reverse trap is just as real — a correct model fed a malformed
    prompt also produces garbage. So the prompt construction is printed.
    """
    from mlx_lm.models.cache import make_prompt_cache

    ids, how = build_eval_prompt(tokenizer, prompt, use_chat_template)
    if not ids:
        return "<tokenizer produced no ids>"
    if verbose:
        b = bos_id(tokenizer)
        has_bos = b is not None and ids[0] == b
        print(f"  prompt path : {how}")
        print(f"  first ids   : {ids[:10]}  (bos_token_id={b}, present={has_bos})")
        if b is not None and not has_bos:
            print("  !! no BOS at position 0 — Gemma degenerates without it")
    x = mx.array([list(ids)])
    cache = make_prompt_cache(model)
    logits = prefill(model, x, cache, step=256, tail_logits=True)
    y = mx.argmax(logits, axis=-1)
    U.barrier(y)
    out = [int(y.item())]
    for _ in range(n - 1):
        logits = model(y.reshape(1, 1), cache=cache)[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        U.barrier(y)
        out.append(int(y.item()))
    U.clear_cache()
    try:
        return tokenizer.decode(out)
    except Exception:
        return repr(out)


def find_config(path_or_repo: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Return (config dict, resolved local snapshot path). Never guesses values.

    mlx-lm moved this around: `get_model_path` existed in older releases and is
    GONE in 0.31.3, replaced by `hf_repo_to_path` (local snapshot lookup) and
    `load_config` (which also folds in generation_config.json). `_download`
    fetches if the repo is not cached. All four are tried, newest first.
    """
    p = Path(path_or_repo)
    if (p / "config.json").exists():
        try:
            from mlx_lm.utils import load_config

            return load_config(p), str(p)
        except Exception:
            return json.loads((p / "config.json").read_text()), str(p)

    # mlx-lm >= 0.31: local snapshot, then download if needed
    for fn_name in ("hf_repo_to_path", "_download", "get_model_path"):
        try:
            import mlx_lm.utils as MU

            fn = getattr(MU, fn_name, None)
            if fn is None:
                continue
            r = fn(path_or_repo)
            r = r[0] if isinstance(r, (tuple, list)) else r
            r = Path(r)
            if (r / "config.json").exists():
                try:
                    return MU.load_config(r), str(r)
                except Exception:
                    return json.loads((r / "config.json").read_text()), str(r)
        except Exception:
            continue

    try:
        from huggingface_hub import hf_hub_download

        f = hf_hub_download(path_or_repo, "config.json")
        return json.loads(Path(f).read_text()), str(Path(f).parent)
    except Exception:
        return None, None


def model_revision(local_dir: Optional[str]) -> Optional[str]:
    """Best-effort HF snapshot hash, so runs.jsonl pins an exact revision."""
    if not local_dir:
        return None
    p = Path(local_dir)
    for cand in (p,) + tuple(p.parents)[:4]:
        if cand.parent.name == "snapshots":
            return cand.name
    return None


# --------------------------------------------------------------------------
# weight accounting
# --------------------------------------------------------------------------


@dataclass
class ParamBreakdown:
    total_bytes: int
    text_bytes: int
    non_text_bytes: int
    embed_bytes: int
    gather_only_bytes: int
    dense_bytes: int
    lm_head_bytes: int
    tied_embeddings: bool
    active_bytes_per_token: int
    by_top_level: Dict[str, int] = field(default_factory=dict)
    non_text_keys: List[str] = field(default_factory=list)
    gather_only_keys: List[str] = field(default_factory=list)


def param_breakdown(model, cfg: Optional[dict] = None) -> ParamBreakdown:
    """
    Split the parameter tree into what actually streams from DRAM per decode
    token, versus what is merely resident.

    Three buckets that are NOT the same thing:
      dense        matmul'd every token -> full bytes read
      gather_only  row-gathered every token -> ~one row read, but fully resident
      non_text     vision/audio towers -> never touched on the text path

    The main embedding table straddles this: when tie_word_embeddings is true it
    is gather-only on the input side AND read in full as the output projection,
    so it counts once toward active bytes. When untied it is gather-only, and a
    separate lm_head sits in `dense`.
    """
    flat = tree_flatten(model.parameters())
    total = text = nontext = embed = gather = dense = lm_head = 0
    by_top: Dict[str, int] = {}
    nt_keys: List[str] = []
    go_keys: List[str] = []
    lm_head_present = False

    for k, v in flat:
        if not isinstance(v, mx.array):
            continue
        n = v.nbytes
        total += n
        top = k.split(".")[0]
        by_top[top] = by_top.get(top, 0) + n
        kl = k.lower()

        if any(m in kl for m in NON_TEXT_MARKERS):
            nontext += n
            nt_keys.append(k)
            continue
        text += n

        if any(m in kl for m in GATHER_ONLY_MARKERS):
            gather += n
            go_keys.append(k)
            continue
        if any(m in kl for m in EMBED_MARKERS):
            embed += n
            continue
        dense += n
        if ".lm_head." in kl or kl.startswith("lm_head."):
            lm_head += n
            lm_head_present = True

    tied = not lm_head_present
    if cfg is not None:
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        tw = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings"))
        if isinstance(tw, bool):
            tied = tw

    active = dense + (embed if tied else 0)

    return ParamBreakdown(
        total_bytes=total,
        text_bytes=text,
        non_text_bytes=nontext,
        embed_bytes=embed,
        gather_only_bytes=gather,
        dense_bytes=dense,
        lm_head_bytes=lm_head if lm_head else (embed if tied else 0),
        tied_embeddings=tied,
        active_bytes_per_token=active,
        by_top_level=dict(sorted(by_top.items(), key=lambda kv: -kv[1])),
        non_text_keys=nt_keys[:16],
        gather_only_keys=go_keys[:8],
    )


# --------------------------------------------------------------------------
# prefill (the eval-barrier placement matters — read this)
# --------------------------------------------------------------------------


def cache_arrays(cache) -> List[mx.array]:
    out: List[mx.array] = []
    for c in cache:
        out.extend(U.flat_arrays(getattr(c, "state", None)))
    return out


def prefill(model, ids: mx.array, cache, step: int = 512, tail_logits: bool = True):
    """
    Chunked prefill with the correct barrier.

    BARRIER PLACEMENT — the single easiest way to misreport MLX numbers:

      * Per chunk we barrier on the CACHE STATE, not on logits. The model
        returns logits for every position in the chunk; a [1, 512, 262144]
        fp16 logits tensor is 268 MiB and an lm_head matmul we would throw
        away. Because MLX is lazy, evaluating only the cache state means that
        matmul is never scheduled at all. mlx-lm does the same thing.

      * The last token is forwarded on its own so the lm_head runs on exactly
        one row. That is what a real decode step costs, so TTFT measured this
        way is TTFT, not TTFT + wasted vocab projection.

      * Nothing is timed until the barrier returns. mx.eval() blocks until the
        command buffers complete.

    Returns logits for the final position, shape [1, vocab], unevaluated.
    """
    n = int(ids.shape[-1])
    if tail_logits:
        body = n - 1
        i = 0
        while i < body:
            j = min(i + step, body)
            model(ids[:, i:j], cache=cache)
            U.barrier(cache_arrays(cache))   # cache only — never logits
            i = j
        return model(ids[:, body:n], cache=cache)[:, -1, :]

    # mlx-lm-equivalent path: all but the final chunk are cache-only; the final
    # chunk computes logits for every one of its positions and slices the last.
    # Kept for apples-to-apples comparison with stock mlx_lm.generate.
    i = 0
    logits = None
    while i < n:
        j = min(i + step, n)
        if j < n:
            model(ids[:, i:j], cache=cache)
            U.barrier(cache_arrays(cache))
        else:
            logits = model(ids[:, i:j], cache=cache)[:, -1, :]
        i = j
    return logits


# --------------------------------------------------------------------------
# measured KV model
# --------------------------------------------------------------------------


@dataclass
class CacheProfile:
    index: int
    cls: str
    bytes_per_token: float      # slope below the cap
    cap_bytes: Optional[float]  # None = grows without bound
    shared_group: int           # caches sharing one object share a group id
    k_is_v: bool                # K and V are the same array object


@dataclass
class KVModel:
    profiles: List[CacheProfile]
    probe_lengths: List[int]
    measured: Dict[int, Dict[str, int]]
    # How many transformer layers attend against each cache entry. 1 each for a
    # conventional model. Gemma 4 Unified has num_kv_shared_layers > 0: the tail
    # layers own no cache and re-attend against an earlier layer's K/V, so a
    # handful of buffers are read many times per token.
    multiplicity: Optional[List[int]] = None
    n_layers: Optional[int] = None

    def _bytes(self, p: "CacheProfile", ctx: int) -> float:
        b = p.bytes_per_token * ctx
        if p.cap_bytes is not None:
            b = min(b, p.cap_bytes)
        return b

    def resident_bytes(self, ctx: int) -> float:
        """Deduplicated: what actually sits in unified memory at context `ctx`.
        This is the number the memory guard must use."""
        seen = set()
        total = 0.0
        for p in self.profiles:
            if p.shared_group in seen:
                continue
            seen.add(p.shared_group)
            total += self._bytes(p, ctx)
        return total

    def read_bytes_per_token(self, ctx: int, shared: str = "pessimistic") -> float:
        """
        Bandwidth-side number: bytes of K/V pulled per decode step.

        shared="pessimistic"  every layer's read goes to DRAM, including the
                              KV-shared tail layers re-reading the same buffer.
                              Correct if the buffer is too big for the SLC.
        shared="optimistic"   each distinct buffer is read once; re-reads by
                              shared layers are assumed to hit cache. Correct
                              for a small windowed buffer at short context.

        Report the range. Where your measured effective GB/s falls inside it
        tells you how much of the shared-KV traffic the cache is absorbing —
        which is itself a finding, and it is what a fused multi-layer
        verification kernel would go after later.
        """
        if shared == "optimistic" or not self.multiplicity:
            seen, total = set(), 0.0
            for p in self.profiles:
                if p.shared_group in seen:
                    continue
                seen.add(p.shared_group)
                total += self._bytes(p, ctx)
            return total
        total = 0.0
        for i, p in enumerate(self.profiles):
            m = self.multiplicity[i] if i < len(self.multiplicity) else 1
            total += self._bytes(p, ctx) * m
        return total

    def mean_read_bytes(self, prompt_tokens: int, gen_tokens: int,
                        shared: str = "pessimistic") -> float:
        """Mean KV bytes read per decode step across the generated range —
        context grows from P to P+G during a run, so a single-point sample is
        wrong by several percent at long context."""
        if gen_tokens <= 0:
            return self.read_bytes_per_token(prompt_tokens, shared)
        return sum(
            self.read_bytes_per_token(prompt_tokens + i, shared)
            for i in range(gen_tokens)
        ) / gen_tokens

    def max_context_within(self, budget_bytes: float) -> int:
        lo, hi = 1, 1 << 22
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.resident_bytes(mid) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        return lo


def _measure_cache_bytes(cache) -> Tuple[List[int], List[int], List[bool], List[int]]:
    """Per-cache: (raw bytes, dedup bytes, k_is_v, object id)."""
    raw, dedup, kisv, oids = [], [], [], []
    for c in cache:
        state = getattr(c, "state", None)
        arrs = list(U.flat_arrays(state))
        raw.append(sum(a.nbytes for a in arrs))
        seen, d = set(), 0
        for a in arrs:
            if id(a) in seen:
                continue
            seen.add(id(a))
            d += a.nbytes
        dedup.append(d)
        kisv.append(len(arrs) >= 2 and arrs[0] is arrs[1])
        oids.append(id(c))
    return raw, dedup, kisv, oids


def profile_kv(
    model,
    tokenizer,
    probe_lengths: Tuple[int, int, int] = (64, 1024, 2176),
    prefill_step: int = 512,
    verbose: bool = True,
) -> KVModel:
    """
    Run three short prefills and fit each cache's growth.

    Probe lengths must straddle the sliding window (default 1024) so capped
    caches are detectable: if bytes stop growing between the 2nd and 3rd probe,
    the cache is windowed and we record the cap.
    """
    from mlx_lm.models.cache import make_prompt_cache

    vocab = getattr(tokenizer, "vocab_size", None) or 1000
    measured: Dict[int, Dict[str, int]] = {}
    per_len: Dict[int, Tuple[List[int], List[int], List[bool], List[int]]] = {}

    for L in probe_lengths:
        U.clear_cache()
        ids = mx.array(
            [[(i * 7919 + 13) % max(64, min(vocab, 30000)) for i in range(L)]]
        )
        cache = make_prompt_cache(model)
        logits = prefill(model, ids, cache, step=prefill_step, tail_logits=True)
        U.barrier(logits, cache_arrays(cache))
        raw, dedup, kisv, oids = _measure_cache_bytes(cache)
        per_len[L] = (raw, dedup, kisv, oids)
        measured[L] = {
            "kv_raw_bytes": sum(raw),
            "kv_dedup_bytes": sum(dedup),
            "n_caches": len(raw),
        }
        if verbose:
            print(
                f"  probe L={L:>5}: raw={U.fmt_bytes(sum(raw))}  "
                f"dedup={U.fmt_bytes(sum(dedup))}  caches={len(raw)}"
            )
        del cache, logits
        U.clear_cache()

    L0, L1, L2 = probe_lengths
    r0, r1, r2 = per_len[L0][0], per_len[L1][0], per_len[L2][0]
    kisv = per_len[L2][2]
    oids = per_len[L2][3]
    classes = [type(c).__name__ for c in _dummy_cache_classes(model)]

    group_of: Dict[int, int] = {}
    groups: List[int] = []
    for oid in oids:
        if oid not in group_of:
            group_of[oid] = len(group_of)
        groups.append(group_of[oid])

    profiles: List[CacheProfile] = []
    for i in range(len(r2)):
        capped = r2[i] <= r1[i] * 1.001
        if capped:
            cap = float(max(r1[i], r2[i]))
            # Slope must come from a probe BELOW the cap. Fitting across the
            # saturation point averages the growing and flat regimes and
            # understates bytes/token badly: for a 512-token window sampled at
            # 64 and 1024, the fit returns ~956 B/token when the true rate is
            # 2048. The smallest probe is below the window by construction, so
            # r0/L0 is exact.
            if r0[i] > 0 and r0[i] < cap * 0.999:
                slope = r0[i] / float(L0)
            elif r1[i] > 0 and r1[i] < cap * 0.999:
                slope = r1[i] / float(L1)
            else:
                slope = cap / float(L0)   # window smaller than the first probe
        else:
            cap = None
            slope = r2[i] / float(L2)
        profiles.append(
            CacheProfile(
                index=i,
                cls=classes[i] if i < len(classes) else "?",
                bytes_per_token=slope,
                cap_bytes=cap,
                shared_group=groups[i],
                k_is_v=bool(kisv[i]),
            )
        )
    mult, n_layers = layer_cache_multiplicity(model, len(profiles))
    return KVModel(profiles=profiles, probe_lengths=list(probe_lengths),
                   measured=measured, multiplicity=mult, n_layers=n_layers)


def layer_cache_multiplicity(model, n_caches: int) -> Tuple[List[int], Optional[int]]:
    """
    How many layers attend against each cache entry.

    mlx-lm's Gemma 4 builds `previous_kvs`: a per-layer index into the cache
    list. With num_kv_shared_layers > 0 the tail layers are remapped to the last
    layer of the same type below the sharing boundary, so `make_cache()` returns
    fewer entries than there are layers and two buffers can serve many layers.
    Any model exposing the same attribute is handled; everything else gets 1s.
    """
    # previous_kvs can sit several wrappers down. For Gemma 4 it is
    # Model -> language_model -> model -> previous_kvs, i.e. two levels below
    # the object mlx-lm hands back, which a one-level lookup silently misses
    # and then reports "no sharing" for a model that shares 18 of 42 layers.
    holder, prev = _find_attr(model, "previous_kvs")
    if prev is not None:
        mult = [0] * n_caches
        for idx in prev:
            if isinstance(idx, int) and 0 <= idx < n_caches:
                mult[idx] += 1
        if sum(mult):
            layers = getattr(holder, "layers", None)
            return mult, (len(layers) if layers is not None else len(prev))
    _, layers = _find_attr(model, "layers")
    return [1] * n_caches, (len(layers) if layers is not None else None)


def _find_attr(root, name: str, max_depth: int = 5):
    """Breadth-first hunt for an attribute through common wrapper names."""
    seen, queue = set(), [(root, 0)]
    while queue:
        obj, d = queue.pop(0)
        if obj is None or id(obj) in seen or d > max_depth:
            continue
        seen.add(id(obj))
        v = getattr(obj, name, None)
        if v is not None:
            return obj, v
        for child in ("language_model", "model", "text_model", "transformer",
                      "decoder", "base_model"):
            queue.append((getattr(obj, child, None), d + 1))
    return None, None


def kvmodel_from_json(path: str) -> KVModel:
    """Reuse a profile produced by `roofline.py --json`, so bench.py does not
    repeat the three probe prefills on every invocation."""
    d = json.loads(Path(path).read_text())
    kvd = d["kv"] if "kv" in d else d
    profs = [CacheProfile(**p) for p in kvd["profiles"]]
    measured = {int(k): v for k, v in kvd.get("measured", {}).items()}
    return KVModel(profiles=profs, probe_lengths=kvd.get("probe_lengths", []),
                   measured=measured, multiplicity=kvd.get("multiplicity"),
                   n_layers=kvd.get("n_layers"))


def _dummy_cache_classes(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


# --------------------------------------------------------------------------
# roofline arithmetic
# --------------------------------------------------------------------------


def ceiling_tok_s(active_bytes: float, kv_read_bytes: float, achievable_gbs: float) -> float:
    """Batch-1 decode is bandwidth-bound: tok/s = B_eff / bytes read per token."""
    per_token = active_bytes + kv_read_bytes
    return (achievable_gbs * 1e9) / per_token


def summarize(
    pb: ParamBreakdown,
    kv: KVModel,
    contexts: List[int],
    achievable_gbs: float,
    headroom_bytes: Optional[float],
) -> dict:
    rows = []
    for c in contexts:
        res = kv.resident_bytes(c)
        rd = kv.read_bytes_per_token(c)
        rows.append(
            {
                "context": c,
                "kv_resident_bytes": res,
                "kv_read_bytes_per_token": rd,
                "bytes_per_token": pb.active_bytes_per_token + rd,
                "ceiling_tok_s": ceiling_tok_s(pb.active_bytes_per_token, rd, achievable_gbs),
            }
        )
    out = {
        "active_weight_bytes_per_token": pb.active_bytes_per_token,
        "achievable_gbs_used": achievable_gbs,
        "contexts": rows,
        "kv_profiles": [asdict(p) for p in kv.profiles],
    }
    if headroom_bytes:
        out["max_context_within_headroom"] = kv.max_context_within(headroom_bytes)
        out["headroom_bytes"] = headroom_bytes
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrated bytes/token roofline for an MLX model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="128,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--achievable-gbs", type=float, default=None,
                    help="Measured GB/s from bwprobe.py. Required for tok/s ceilings.")
    ap.add_argument("--prefill-step", type=int, default=512)
    ap.add_argument("--probe-lengths", default="64,1024,2176")
    ap.add_argument("--activation-slack-gb", type=float, default=0.8)
    ap.add_argument("--remap-model-type", action="append", metavar="FROM=TO",
                    help="Route an unimplemented config model_type to an existing "
                         "mlx_lm.models module. Repeatable. gemma4_unified=gemma4 "
                         "is applied automatically.")
    ap.add_argument("--no-sanity-generate", action="store_true",
                    help="Skip the post-load coherence check. Don't.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    probes = tuple(int(x) for x in args.probe_lengths.split(","))
    if len(probes) != 3:
        print("--probe-lengths needs exactly 3 values", file=sys.stderr)
        return 2

    host = U.host_info()
    print("=" * 78)
    print("HOST")
    print("=" * 78)
    for k, v in host.items():
        if k.endswith("_bytes") and isinstance(v, int):
            v = f"{v} ({U.fmt_bytes(v)})"
        print(f"  {k:38s} {v}")
    if host["iogpu_wired_limit_mb"] == 0:
        print("  note: iogpu.wired_limit_mb == 0 means SYSTEM DEFAULT, not zero.")

    cfg, local = find_config(args.model)
    print()
    print("=" * 78)
    print("CONFIG (read from config.json — nothing here is assumed)")
    print("=" * 78)
    if cfg is None:
        print("  !! could not locate config.json; KV measurement still valid")
    else:
        tc = cfg.get("text_config", cfg)
        interesting = [
            "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "global_head_dim", "num_global_key_value_heads", "num_kv_shared_layers",
            "attention_k_eq_v", "hidden_size_per_layer_input",
            "vocab_size_per_layer_input", "final_logit_softcapping",
            "sliding_window", "sliding_window_pattern", "layer_types",
            "rope_theta", "rope_local_base_freq", "partial_rotary_factor",
            "rope_parameters", "vocab_size", "tie_word_embeddings", "torch_dtype",
        ]
        for k in interesting:
            for src, tag in ((tc, "text"), (cfg, "root")):
                if k in src:
                    v = src[k]
                    if isinstance(v, list) and len(v) > 12:
                        from collections import Counter
                        v = f"len={len(v)} {dict(Counter(v))} head={v[:8]}"
                    print(f"  [{tag}] {k:32s} {v}")
                    break
        q = cfg.get("quantization") or cfg.get("quantization_config")
        if q:
            print(f"  [root] quantization                   {q}")

    print()
    print("Loading model ...")
    remap = {}
    for pair in (args.remap_model_type or []):
        if "=" in pair:
            k, v = pair.split("=", 1)
            remap[k.strip()] = v.strip()
    model, tokenizer = load_model(args.model, remap)
    U.reset_peak_memory()
    resident_after_load = U.active_memory()

    if not args.no_sanity_generate:
        print("\n" + "=" * 78)
        print("GENERATION SANITY CHECK  (read this before trusting anything below)")
        print("=" * 78)
        try:
            txt = sanity_generate(model, tokenizer)
            print(f"  prompt : 'The capital of France is'")
            print(f"  output : {txt!r}")
            print("  If that is not coherent continuation, the checkpoint did not load")
            print("  correctly and every number below is meaningless. Stop and say so.")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print("  The model loaded but cannot generate. Do not trust the numbers below.")

    pb = param_breakdown(model, cfg)
    print()
    print("=" * 78)
    print("WEIGHT BYTES (measured from the loaded parameter tree)")
    print("=" * 78)
    for k, v in pb.by_top_level.items():
        print(f"  {k:38s} {U.fmt_bytes(v)}")
    print(f"  {'-' * 60}")
    print(f"  {'total parameters':38s} {U.fmt_bytes(pb.total_bytes)}")
    print(f"  {'non-text towers (excluded)':38s} {U.fmt_bytes(pb.non_text_bytes)}")
    print(f"  {'text path':38s} {U.fmt_bytes(pb.text_bytes)}")
    print(f"  {'  dense (matmul every token)':38s} {U.fmt_bytes(pb.dense_bytes)}")
    print(f"  {'  gather-only tables (resident)':38s} {U.fmt_bytes(pb.gather_only_bytes)}")
    print(f"  {'  main embedding matrix':38s} {U.fmt_bytes(pb.embed_bytes)}")
    print(f"  {'tied embeddings':38s} {pb.tied_embeddings}")
    print(f"  {'lm_head read per token':38s} {U.fmt_bytes(pb.lm_head_bytes)}")
    print(f"  {'ACTIVE BYTES / DECODE TOKEN':38s} {U.fmt_bytes(pb.active_bytes_per_token)}")
    if pb.gather_only_bytes:
        pctg = 100.0 * pb.gather_only_bytes / max(1, pb.total_bytes)
        print(f"  -> {U.fmt_bytes(pb.gather_only_bytes)} ({pctg:.0f}% of the model) is "
              f"row-gathered, not streamed.")
        print(f"     Counting it as active weight bytes would understate the roofline "
              f"by that much. Keys: {pb.gather_only_keys[:3]}")
    if pb.non_text_keys:
        print(f"  excluded (first few): {pb.non_text_keys[:6]}")
    # Cross-check against mlx-lm's own accounting: get_total_parameters()
    # unpacks quantized weights back to logical parameter counts, so this is
    # the honest "how many B params is this really" number.
    try:
        from mlx_lm.utils import get_total_parameters, compute_bits_per_weight

        tp = get_total_parameters(model)
        bpw = compute_bits_per_weight(model)
        print(f"  {'logical parameters (mlx-lm)':38s} {tp / 1e9:.2f} B")
        print(f"  {'effective bits per weight':38s} {bpw:.2f}")
        if pb.total_bytes:
            print(f"  {'implied bytes if all were dense':38s} {U.fmt_bytes(pb.total_bytes)}")
    except Exception as e:
        print(f"  (mlx-lm parameter cross-check unavailable: {type(e).__name__})")
    print(f"  resident after load (mx active): {U.fmt_bytes(resident_after_load)}")

    print()
    print("=" * 78)
    print("KV MEASUREMENT (real cache objects, not a formula)")
    print("=" * 78)
    kv = profile_kv(model, tokenizer, probes, args.prefill_step)

    from collections import Counter
    kinds = Counter(
        (p.cls, round(p.bytes_per_token, 1), None if p.cap_bytes is None else round(p.cap_bytes))
        for p in kv.profiles
    )
    print("\n  distinct cache kinds:")
    for (cls, slope, cap), n in kinds.items():
        capstr = "uncapped" if cap is None else f"cap {U.fmt_bytes(cap)}"
        print(f"    {n:>3} x {cls:<20s} {slope:>10.1f} B/token  {capstr}")
    if any(p.k_is_v for p in kv.profiles):
        n = sum(1 for p in kv.profiles if p.k_is_v)
        print(f"    -> {n} cache(s) report K and V as the SAME array (unified K/V)")
    ngroups = len({p.shared_group for p in kv.profiles})
    if ngroups != len(kv.profiles):
        print(f"    -> same cache object appears at {len(kv.profiles)} slots, {ngroups} buffers")
    if kv.n_layers and kv.n_layers != len(kv.profiles):
        hot = [(i, m) for i, m in enumerate(kv.multiplicity or []) if m > 1]
        print(f"    -> KV SHARING: {kv.n_layers} layers, only {len(kv.profiles)} caches.")
        if hot:
            print(f"       {len(hot)} buffer(s) serve multiple layers: "
                  f"{[f'cache[{i}] x{m}' for i, m in hot][:6]}")
            print("       resident KV counts each buffer once; read-per-token counts "
                  "every layer that attends against it.")
        else:
            print("       !! could not locate the layer->cache map (previous_kvs), so "
                  "read-per-token is counting each buffer ONCE.")
            print("       With fewer caches than layers that is a LOWER BOUND on KV "
                  "traffic. Treat the ceiling as optimistic.")

    mws = host["max_recommended_working_set_bytes"]
    headroom = None
    if mws and resident_after_load:
        headroom = mws * 0.92 - resident_after_load - args.activation_slack_gb * U.GB

    print()
    print(f"  {'ctx':>7} {'KV resident':>13} {'read/tok opt':>13} {'read/tok pess':>14} "
          f"{'ceiling hi':>11} {'ceiling lo':>11}")
    bw = args.achievable_gbs
    for c in contexts:
        res = kv.resident_bytes(c)
        rd_o = kv.read_bytes_per_token(c, "optimistic")
        rd_p = kv.read_bytes_per_token(c, "pessimistic")
        hi = f"{ceiling_tok_s(pb.active_bytes_per_token, rd_o, bw):.2f}" if bw else "  --  "
        lo = f"{ceiling_tok_s(pb.active_bytes_per_token, rd_p, bw):.2f}" if bw else "  --  "
        flag = ""
        if headroom is not None and res > headroom:
            flag = "  <-- EXCEEDS HEADROOM"
        print(f"  {c:>7} {U.fmt_bytes(res):>13} {U.fmt_bytes(rd_o):>13} "
              f"{U.fmt_bytes(rd_p):>14} {hi:>11} {lo:>11}{flag}")
    print("\n  opt = each KV buffer read once per token (shared re-reads hit cache)")
    print("  pess = every attending layer's read goes to DRAM. Truth is between;")
    print("  where your measured effective GB/s lands inside this band is a result.")
    if not bw:
        print("\n  ceilings blank: pass --achievable-gbs <measured> from bwprobe.py")
    if headroom is not None:
        print(f"\n  KV headroom      : {U.fmt_bytes(headroom)}"
              f"  (0.92 x working set - weights - {args.activation_slack_gb} GiB activations)")
        print(f"  max context fit  : {kv.max_context_within(headroom)} tokens")

    if args.json:
        payload = {
            "host": host,
            "model": args.model,
            "revision": model_revision(local),
            "config_present": cfg is not None,
            "params": asdict(pb),
            "kv": {"profiles": [asdict(p) for p in kv.profiles],
                   "probe_lengths": kv.probe_lengths,
                   "measured": kv.measured,
                   "multiplicity": kv.multiplicity,
                   "n_layers": kv.n_layers},
            "table": summarize(pb, kv, contexts, bw or 0.0, headroom),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

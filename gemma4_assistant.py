"""
gemma4_assistant.py — MLX implementation of the Gemma 4 MTP drafter.

Drop into mlx_lm/models/ (or keep here and register via mlxutil), which makes
`model_type: gemma4_assistant` loadable.

ECOSYSTEM STATUS (verified 2026-08-28, correcting an earlier claim here that
"no MLX runtime has this" — mlx-vlm does):

  mlx-lm        NOT SUPPORTED in any release, and not on main. Exists only in
                open PR #1276 (filed 2026-05-14, still unmerged), which adds
                the model class alone — its author states the speculative
                decoding integration would be a separate PR, which does not
                exist. This file is what makes the drafter usable on mlx-lm.
  mlx-vlm       FULLY IMPLEMENTED, mlx_vlm/speculative/drafters/gemma4_assistant/,
                with batched dispatch. Claims byte-identical greedy output.
                We use it as an external control, not as a dependency.
  mlx-swift-lm  NOT SUPPORTED (issues #279, #282 open).
  mlx-engine    NOT SUPPORTED (lmstudio-ai/mlx-engine #323 open).
  ollama        Supported via MLX since PR #15980, merged 2026-05-05.

Note that mlx-vlm claims byte-identical greedy output while MLX-OptiQ, running
the same mechanism on the same checkpoint family, reports "bf16 precision drift
in multi-token verify ... not lossless". Both cannot be right. Settling that is
Recipe 4's headline objective; see recipes/recipe-04-mtp-spec-decode.md.

Ported from transformers 5.16.1 `models/gemma4_assistant/modeling_gemma4_assistant.py`
and its caller in `generation/candidate_generator.py`, both read directly rather
than inferred.

============================ ARCHITECTURE ============================

This drafter is NOT a standalone model, which is why mlx-lm's generic
`--draft-model` cannot drive it. Two things make it dependent on the target:

1. It has NO key/value projections. Confirmed in the checkpoint: 12 attention
   tensors over 4 layers = q_proj, o_proj, q_norm only. Its layers attend over
   the TARGET's K/V — `shared_kv_states`, one entry per layer_type (the last
   sliding layer and the last full-attention layer of the target). That is the
   same `previous_kvs` mechanism gemma4_text already implements, extended
   across models. Hence `num_kv_shared_layers == num_hidden_layers == 4`, so
   the drafter allocates zero caches of its own.

2. Its input is the target's state:

       inputs_embeds = concat([last_token_embedding, last_hidden_state], -1)

   both `backbone_hidden_size` (2560) wide, hence pre_projection's 5120 input.

Forward:

    [B, L, 5120] --pre_projection--> [B, L, 256]
                 --4x DecoderLayer (attending target K/V)-->
                 --norm--> h [B, L, 256]
      post_projection(h) -> [B, L, 2560]   fed back as the next step's
                                           "last_hidden_state" so the drafter
                                           can run several steps without the
                                           target
      masked_embedding(h, lm_head.weight) -> logits

`lm_head.weight` is tied to `model.embed_tokens.weight` [262144, 256].

======================= THE VOCABULARY SHORTLIST =======================

`masked_embedding` is why drafting is nearly free. Instead of a 262144-wide
output projection it keeps:

    centroids       Linear(256 -> 2048)
    token_ordering  int buffer [262144] — a permutation grouping the vocab into
                    2048 clusters of 128 tokens

Per token it scores 2048 centroids, takes the top 32, and only scores the
32 x 128 = 4096 tokens in those clusters. That reads 4096 x 256 x 2 = 2.1 MB
instead of the full 134 MB embedding table — 64x less traffic.

DELIBERATE DEVIATION FROM THE REFERENCE: torch scatters those 4096 logits into
a full [B, L, 262144] tensor padded with (min - 1). We do not, by default.
Materialising 262144 floats per draft token to then argmax over 4096 real
values is pure waste on a bandwidth-bound machine — it would cost more than the
shortlist saved. `__call__` returns (candidate_ids, candidate_logits) and the
caller argmaxes over the shortlist. Pass `dense=True` for the padded tensor when
you need to compare against a reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from . import gemma4_text


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_assistant"
    text_config: dict = None
    backbone_hidden_size: int = 2560
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    use_ordered_embeddings: bool = True
    tie_word_embeddings: bool = True
    vocab_size: int = 262144

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        self.text_config.setdefault("vocab_size", self.vocab_size)


class MaskedEmbedder(nn.Module):
    """Centroid-shortlisted output head. See module docstring."""

    def __init__(self, args: ModelArgs, hidden_size: int, vocab_size: int):
        super().__init__()
        self.num_centroids = args.num_centroids
        self.top_k = args.centroid_intermediate_top_k
        self.vocab_size = vocab_size
        self.per_centroid = vocab_size // args.num_centroids
        self.centroids = nn.Linear(hidden_size, args.num_centroids, bias=False)
        # Loaded from the checkpoint: a permutation of the vocabulary grouped
        # into clusters. Not trained, not quantized — keep as ints.
        self.token_ordering = mx.zeros((vocab_size,), dtype=mx.int32)

    def __call__(self, h: mx.array, lm_head_weight: mx.array, dense: bool = False):
        """
        h: [B, L, D].  Returns (candidate_ids, candidate_logits), each
        [B, L, top_k * per_centroid] — or a dense [B, L, vocab] when asked.
        """
        B, L, D = h.shape
        centroid_logits = self.centroids(h)                       # [B, L, C]

        # Top-k centroids. argpartition gives unordered indices, which is all we
        # need — the shortlist is a set, not a ranking.
        idx = mx.argpartition(-centroid_logits, kth=self.top_k - 1, axis=-1)
        top = idx[..., : self.top_k]                              # [B, L, k]

        ordering = self.token_ordering.reshape(self.num_centroids, self.per_centroid)
        cand = ordering[top]                                      # [B, L, k, per_centroid]
        cand = cand.reshape(B, L, self.top_k * self.per_centroid)  # [B, L, M]

        emb = lm_head_weight[cand.reshape(-1)]                    # [B*L*M, D]
        emb = emb.reshape(B, L, -1, D)                            # [B, L, M, D]
        logits = (emb @ h[..., None]).squeeze(-1)                 # [B, L, M]

        if not dense:
            return cand, logits

        fill = mx.min(logits) - 1.0
        out = mx.full((B, L, self.vocab_size), fill, dtype=logits.dtype)
        return mx.put_along_axis(out, cand, logits, axis=-1)


class AssistantBackbone(nn.Module):
    """
    The drafter's own 4-layer stack, named `model` so checkpoint keys line up
    (`model.embed_tokens`, `model.layers.N.*`, `model.norm`).

    Built from gemma4_text.DecoderLayer directly rather than Gemma4TextModel.
    That is not a style choice: Gemma4TextModel.__init__ raises KeyError on this
    config. With num_kv_shared_layers == num_hidden_layers its `previous_kvs`
    loop indexes a `kvs_by_type` map built from range(0) — always empty. The
    drafter takes all its K/V from outside, so that machinery has nothing to do.
    """

    def __init__(self, tc: "gemma4_text.ModelArgs"):
        super().__init__()
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = [
            gemma4_text.DecoderLayer(tc, i) for i in range(tc.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.layer_types = list(tc.layer_types)

    def __call__(self, h: mx.array, shared_kv: dict, mask=None, offset: int = 0):
        """
        `offset` is the ABSOLUTE position of the drafter's query token, and it
        must be an int — mx.fast.rope rejects None.

        Getting it right matters. The drafter consumes the embedding of the
        token the target just produced (position p+1, not yet in the target's
        cache) together with the hidden state at position p. So its query sits
        at p+1 = the number of tokens the target has consumed. Each further
        draft step advances one more position, while the keys it attends to
        stay fixed at 0..p — which is exactly a causal next-token query.

        The shared keys were already rotated by the target at their own
        positions, so only the query is roped here.
        """
        for layer, ltype in zip(self.layers, self.layer_types):
            kv = shared_kv.get(ltype)
            if kv is None:
                raise ValueError(
                    f"shared_kv_states is missing '{ltype}'. The drafter has no "
                    f"K/V projections of its own; it must be given the target's "
                    f"K/V for the last layer of each layer_type."
                )
            m = mask.get(ltype) if isinstance(mask, dict) else mask
            h, _, _ = layer(h, m, None, per_layer_input=None,
                            shared_kv=kv, offset=offset)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        tc = gemma4_text.ModelArgs.from_dict(args.text_config)
        self.text_args = tc

        self.model = AssistantBackbone(tc)
        self.pre_projection = nn.Linear(
            2 * args.backbone_hidden_size, tc.hidden_size, bias=False
        )
        self.post_projection = nn.Linear(
            tc.hidden_size, args.backbone_hidden_size, bias=False
        )
        self.masked_embedding = (
            MaskedEmbedder(args, tc.hidden_size, tc.vocab_size)
            if args.use_ordered_embeddings else None
        )
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

    @property
    def lm_head_weight(self) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.weight
        return self.lm_head.weight

    def __call__(
        self,
        last_token_embedding: mx.array,   # [B, L, backbone_hidden] from the TARGET
        last_hidden_state: mx.array,      # [B, L, backbone_hidden] from the TARGET
        shared_kv: dict,                  # {layer_type: (keys, values)} from the TARGET
        mask: Optional[Any] = None,
        offset: int = 0,                  # absolute position of the query token
        dense: bool = False,
        use_shortlist: bool = True,
    ):
        """
        Returns (next_backbone_state, candidate_ids, candidate_logits).

        `next_backbone_state` is post_projection's output — feed it back as
        `last_hidden_state` on the following draft step so the drafter can run
        several steps without touching the target.
        """
        x = mx.concatenate([last_token_embedding, last_hidden_state], axis=-1)
        h = self.pre_projection(x)
        h = self.model(h, shared_kv, mask=mask, offset=offset)
        nxt = self.post_projection(h)

        # The shortlist reads 4096 scattered 512-byte rows out of a 134 MB
        # table. That is 2.1 MB of DATA but thousands of random accesses, and
        # random access is the one thing this memory system is worst at. The
        # dense path reads the whole table sequentially — 64x the bytes, but
        # streaming at ~86 GB/s. Which wins is an empirical question on this
        # hardware, not a foregone conclusion, so both are selectable.
        if self.masked_embedding is not None and use_shortlist:
            out = self.masked_embedding(h, self.lm_head_weight, dense=dense)
        else:
            logits = h @ self.lm_head_weight.T
            if dense:
                return nxt, logits
            ids = mx.broadcast_to(
                mx.arange(logits.shape[-1]).reshape(1, 1, -1), logits.shape
            )
            out = (ids, logits)
        if dense:
            return nxt, out
        cand, logits = out
        return nxt, cand, logits

    def sanitize(self, weights):
        # token_ordering is an integer permutation buffer, not a weight; make
        # sure nothing downstream tries to quantize or cast it.
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Never quantize the centroid table or the ordering buffer: the
            # shortlist is an index structure, and the same 4-bit-PLE mistake
            # that breaks stock Gemma 4 conversions applies here.
            if "masked_embedding" in path:
                return False
            return True
        return predicate

    @property
    def layers(self):
        return self.model.layers

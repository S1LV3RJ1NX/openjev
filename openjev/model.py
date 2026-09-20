"""The model: a shared state prefix, block-diagonal attention, one scoring head.

    logits = scorer(h[marker_positions])     # scorer is Linear(d, 1), shared
    probs  = softmax(logits within a question's own markers)

The scoring head has **no per-class parameters**. A new label set changes the
input, not the weights, which is the property that makes a runtime-variable
menu possible and is the prerequisite for zero-shot transfer to unseen schemas.

On the attention mask: this first version builds an explicit 4D additive mask
and runs SDPA. That is correct everywhere and fast enough to train on, but it
gets no block-sparsity speedup — the cost is still O(L^2). FlexAttention's
`BlockMask` is the version that actually exploits the structure, and it is
CUDA-only. Correctness first, then the microbenchmark, then the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .schema import Answer


def causal_block_and_mask_fn(block_id: torch.Tensor):
    """Block isolation for a decoder backbone.

    A causal model already prevents the state prefix from seeing the question
    blocks, because the state comes first. What causality does *not* prevent
    is question block 2 attending to question block 1, which would make an
    answer depend on which other questions happened to be asked. So the same
    rule as the encoder, composed with the model's own causal mask:

        attend if the key is in the state prefix, or in my own block.

    Note this is strictly cheaper than the encoder case: the state is encoded
    once as an ordinary prefix, which is exactly what a KV cache is for.
    """

    def fn(batch_idx, head_idx, q_idx, kv_idx):
        bi = block_id[batch_idx, q_idx]
        bj = block_id[batch_idx, kv_idx]
        return (bj == 0) | (bi == bj)

    return fn


def block_and_mask_fn(block_id: torch.Tensor):
    """An `and_mask_function` for transformers' masking utilities.

    Returns True where query token i may attend to key token j: j is in the
    shared state prefix (block 0), or i and j are in the same question block.

    This is the supported extension point. ModernBERT rebuilds its masks from
    a 2D padding mask unless it is handed a ready-made dict, and it alternates
    global and sliding-window layers — so ANDing our condition into each of
    its own masks keeps that structure intact, where substituting a single 4D
    mask would silently make every layer global.
    """

    def fn(batch_idx, head_idx, q_idx, kv_idx):
        bi = block_id[batch_idx, q_idx]
        bj = block_id[batch_idx, kv_idx]
        return (bj == 0) | (bi == bj)

    return fn


def block_diagonal_mask(
    block_id: torch.Tensor, attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """(B, 1, L, L) additive mask: see the state, see your own block, nothing else.

    `block_id` is 0 for the shared state prefix, 1..n for question blocks, and
    -1 for padding. A token may attend to position j when

        j is real, and (j is in the state prefix or j is in the same block as i)

    Question blocks therefore cannot see each other. We verified on a
    production system that this isolation holds in practice — adding sibling
    questions that explicitly assert an answer moved the target's distribution
    less than re-running the identical request did — so it is the behaviour to
    reproduce, not merely a plausible design.
    """
    b_i = block_id.unsqueeze(-1)  # (B, L, 1) the querying token
    b_j = block_id.unsqueeze(-2)  # (B, 1, L) the attended token

    valid_j = attention_mask.unsqueeze(-2)              # (B, 1, L)
    allowed = (b_j == 0) | (b_i == b_j)                 # state prefix, or same block
    allowed = allowed & valid_j

    # Padding rows attend nowhere, which would produce NaN from a fully-masked
    # softmax. Let them attend to themselves; their outputs are discarded.
    dead = ~allowed.any(dim=-1, keepdim=True)
    eye = torch.eye(block_id.shape[-1], dtype=torch.bool, device=block_id.device)
    allowed = allowed | (dead & eye)

    neg = torch.finfo(dtype).min
    return torch.zeros_like(allowed, dtype=dtype).masked_fill_(~allowed, neg).unsqueeze(1)


def grouped_log_softmax(logits: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Log-softmax over each question's own markers, all questions at once.

    `logits` and `group` are flat and ragged: question 1 may have 77 options
    and question 2 only 2. A scatter-based logsumexp handles that in one pass
    without padding to the widest menu.
    """
    m = torch.full((n_groups,), float("-inf"), device=logits.device, dtype=logits.dtype)
    m = m.scatter_reduce(0, group, logits, reduce="amax", include_self=True)
    shifted = logits - m[group]
    denom = torch.zeros(n_groups, device=logits.device, dtype=logits.dtype)
    denom = denom.scatter_add(0, group, shifted.exp())
    return shifted - denom[group].log()


@dataclass
class OpenJevOutput:
    log_probs: torch.Tensor   # (n_markers,) log P(option | its question)
    marker_group: torch.Tensor
    hidden: torch.Tensor | None = None


class OpenJev(nn.Module):
    def __init__(
        self,
        backbone: str = "answerdotai/ModernBERT-base",
        dropout: float = 0.1,
        vocab_size: int | None = None,
        scorer: str = "linear",
        yes_token: str = " yes",
        no_token: str = " no",
        tokenizer=None,
    ):
        """
        scorer="linear"  a fresh Linear(d,1) over marker hidden states. Has to
                         be trained; no zero-shot ability by construction.
        scorer="mlm"     reuse the pretrained masked-LM head at the marker and
                         read off logit(yes) - logit(no). Adds no parameters,
                         so the model can be evaluated zero-shot. This is
                         UniMC's mechanism (arXiv:2210.08590) and it requires
                         the marker to be the tokenizer's own mask token.
        """
        super().__init__()
        self.scorer_kind = scorer
        cfg = AutoConfig.from_pretrained(backbone)
        # flash-attn cannot take an arbitrary mask; SDPA can. ModernBERT's
        # torch.compile path is also unreliable on non-CUDA backends, and it
        # is a config field rather than a from_pretrained kwarg.
        if hasattr(cfg, "reference_compile"):
            cfg.reference_compile = False

        d = cfg.hidden_size
        if scorer == "mlm":
            from transformers import AutoModelForMaskedLM

            lm = AutoModelForMaskedLM.from_pretrained(
                backbone, config=cfg, attn_implementation="sdpa"
            )
            self.backbone = lm.model
            self.lm_head = lm.head
            self.lm_decoder = lm.decoder
            if tokenizer is None:
                raise ValueError("scorer='mlm' needs a tokenizer to resolve yes/no ids")
            self.yes_id = _single_token_id(tokenizer, yes_token)
            self.no_id = _single_token_id(tokenizer, no_token)
            self.scorer = nn.Identity()
        else:
            self.backbone = AutoModel.from_pretrained(
                backbone, config=cfg, attn_implementation="sdpa"
            )
            self.scorer = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d, 1),
            )
            nn.init.zeros_(self.scorer[-1].bias)

        if vocab_size is not None and vocab_size != cfg.vocab_size:
            self.backbone.resize_token_embeddings(vocab_size)

    def _mask_mapping(self, ids, attn, blocks):
        """ModernBERT's own masks, with block isolation ANDed into each."""
        from transformers.masking_utils import (
            create_bidirectional_mask,
            create_bidirectional_sliding_window_mask,
        )

        cfg = self.backbone.config
        probe = torch.empty(
            (*ids.shape, 1), dtype=self.backbone.dtype, device=ids.device
        )
        kwargs = {
            "config": cfg,
            "inputs_embeds": probe,
            "attention_mask": attn,
            "and_mask_function": block_and_mask_fn(blocks),
        }
        return {
            "full_attention": create_bidirectional_mask(**kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**kwargs),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> OpenJevOutput:
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        blocks = batch["block_id"]

        mapping = self._mask_mapping(ids, attn, blocks)
        h = self.backbone(input_ids=ids, attention_mask=mapping).last_hidden_state

        flat = h.reshape(-1, h.shape[-1])
        markers = flat.index_select(0, batch["marker_flat"])

        if self.scorer_kind == "mlm":
            # What would the pretrained LM predict at this mask: yes or no?
            vocab = self.lm_decoder(self.lm_head(markers))
            logits = vocab[:, self.yes_id] - vocab[:, self.no_id]
        else:
            logits = self.scorer(markers).squeeze(-1)

        lp = grouped_log_softmax(logits, batch["marker_group"], int(batch["n_groups"]))
        return OpenJevOutput(log_probs=lp, marker_group=batch["marker_group"])

    # -- losses ------------------------------------------------------------

    @staticmethod
    def cross_entropy(out: OpenJevOutput, target_index: torch.Tensor) -> torch.Tensor:
        """Plain cross-entropy against the correct marker of each question.

        Deliberately not policy gradient. Any objective built on a strictly
        proper scoring rule over predicted probabilities shares its optimum
        with cross-entropy — the logarithmic score *is* cross-entropy up to
        sign — and is differentiable, so the gradient is available in closed
        form. Sampling estimators exist for rewards you cannot differentiate
        through; this is not one.
        """
        return -out.log_probs[target_index].mean()

    @staticmethod
    def soft_cross_entropy(
        out: OpenJevOutput, target_probs: torch.Tensor
    ) -> torch.Tensor:
        """For distillation, where the target is a distribution not a label."""
        return -(target_probs * out.log_probs).sum() / out.marker_group.max().add(1)

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor], packed: list) -> list[dict[str, Answer]]:
        """Full-precision probabilities, deliberately unquantized.

        A production endpoint we measured rounds to 0.01, which puts 71.9% of
        returned values at a hard zero and makes log-odds, log loss and
        tail-risk reasoning unavailable. It also creates ties that make some
        coverage levels unreachable by thresholding. We do not do that.
        """
        out = self(batch)
        probs = out.log_probs.exp().cpu()
        results, cursor, g = [], 0, 0
        for p in packed:
            answers: dict[str, Answer] = {}
            for qid, k in zip(p.question_ids, p.n_options):
                vals = probs[cursor : cursor + k].tolist()
                labels = _labels_for(p, qid, k)
                answers[qid] = Answer(dict(zip(labels, vals)))
                cursor += k
                g += 1
            results.append(answers)
        return results


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        alt = tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
        if len(alt) == 1:
            return alt[0]
        raise ValueError(f"{text!r} is not a single token ({ids}); pick another word")
    return ids[0]


def _labels_for(packed, qid: str, k: int) -> list[str]:
    """Option label strings, in the order they were packed."""
    meta = getattr(packed, "labels", None)
    if meta and qid in meta:
        return meta[qid]
    return [str(i) for i in range(k)]

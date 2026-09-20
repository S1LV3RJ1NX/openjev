"""Turning a (state, questions) pair into one packed sequence.

The layout, which is the whole design:

    [CLS] <state tokens> [SEP] │ Q1 tokens ⟨M⟩opt ⟨M⟩opt … [SEP] │ Q2 … │ …
          └─ shared prefix ────┘ └──── question block 1 ─────┘   └─ block 2 ─┘

Every question block attends to the state prefix and to itself, never to a
sibling block. One encoder pass answers every question, and the state is
tokenized once rather than once per question.

`⟨M⟩` is a marker token placed immediately before each option's text. The
option's hidden state at that position is what the scoring head reads, and a
softmax runs over the markers belonging to one question. Because the option
text is *input*, the label set can change at runtime without touching a single
weight — which is what makes zero-shot menus possible at all.

This module only builds tensors. The masking and scoring live in `model.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from .schema import Choice, Noul, Question, Score

MARKER = "[unused0]"  # resolved against the tokenizer in Packer.__init__


def render_state(state: str | dict) -> str:
    if isinstance(state, str):
        return state
    # A JSON state is flattened to text. Measured on a production endpoint, a
    # dict and its flattened form give identical answers at a constant small
    # token delta, so there is no structured path worth building.
    return json.dumps(state, ensure_ascii=False, indent=None)


def option_text(label: str, description: object | None) -> str:
    """Both halves carry meaning, so both go in.

    Measured: descriptive keys with no descriptions, and opaque keys carrying
    the real descriptions, scored the same (75.0% each, n=40). The model reads
    semantics from wherever they sit.
    """
    if description is None:
        return label
    if not isinstance(description, str):
        description = json.dumps(description, ensure_ascii=False)
    return f"{label}: {description}"


def question_options(q: Question) -> list[str]:
    if isinstance(q, Choice):
        return [option_text(k, v) for k, v in q.criteria.items()]
    if isinstance(q, Score):
        return [option_text(str(i), c) for i, c in enumerate(q.criteria)]
    if isinstance(q, Noul):
        crit = q.criteria or {}
        return [option_text("no", crit.get("false")), option_text("yes", crit.get("true"))]
    raise TypeError(f"unknown question type {type(q)}")


@dataclass
class Packed:
    """One example, ready for the model.

    `block_id` is the key to everything downstream: 0 marks the shared state
    prefix, and 1..n mark question blocks. The attention mask is built from it,
    and so is the per-question softmax grouping.
    """

    input_ids: torch.Tensor      # (L,)
    block_id: torch.Tensor       # (L,) 0 = state prefix, i = question i
    marker_pos: torch.Tensor     # (n_options_total,) indices into the sequence
    marker_block: torch.Tensor   # (n_options_total,) which question each marker belongs to
    question_ids: list[str]      # block i (1-indexed) is question_ids[i-1]
    n_options: list[int]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


class Packer:
    """Builds `Packed` sequences. Stateless apart from the tokenizer."""

    def __init__(
        self,
        tokenizer,
        max_len: int = 8192,
        max_state_len: int | None = None,
        marker: str | None = None,
    ):
        self.tok = tokenizer
        self.max_len = max_len
        # The state is truncated, never the options: a truncated option is
        # silently unscoreable, whereas a truncated state merely loses context.
        self.max_state_len = max_state_len or max_len // 2

        # With MLM-head scoring the marker must be the tokenizer's own mask
        # token, or the pretrained head is being asked about a token it has
        # never predicted at and the zero-shot ability evaporates. With a
        # learned scorer any spare token will do.
        if marker is None:
            marker = MARKER
            if marker not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": [marker]})
        self.marker = marker
        self.marker_id = tokenizer.convert_tokens_to_ids(marker)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def _ids(self, text: str, limit: int | None = None) -> list[int]:
        out = self.tok(text, add_special_tokens=False)["input_ids"]
        return out[:limit] if limit else out

    def pack(self, state: str | dict, questions: dict[str, Question]) -> Packed:
        ids: list[int] = [self.cls_id]
        block: list[int] = [0]

        state_ids = self._ids(render_state(state), self.max_state_len)
        ids += state_ids + [self.sep_id]
        block += [0] * (len(state_ids) + 1)

        marker_pos: list[int] = []
        marker_block: list[int] = []
        qids: list[str] = []
        n_options: list[int] = []

        for b, (qid, q) in enumerate(questions.items(), start=1):
            qids.append(qid)
            instr = self._ids(q.instructions or qid)
            ids += instr
            block += [b] * len(instr)

            opts = question_options(q)
            n_options.append(len(opts))
            for opt in opts:
                marker_pos.append(len(ids))
                marker_block.append(b)
                opt_ids = self._ids(opt)
                ids += [self.marker_id] + opt_ids
                block += [b] * (len(opt_ids) + 1)

            ids.append(self.sep_id)
            block.append(b)

            if len(ids) > self.max_len:
                raise ValueError(
                    f"packed sequence is {len(ids)} tokens, over max_len={self.max_len}. "
                    f"Total length is the only budget: option count and option text "
                    f"length are not separate limits. Shorten the state, the "
                    f"descriptions, or split the questions across calls."
                )

        return Packed(
            input_ids=torch.tensor(ids, dtype=torch.long),
            block_id=torch.tensor(block, dtype=torch.long),
            marker_pos=torch.tensor(marker_pos, dtype=torch.long),
            marker_block=torch.tensor(marker_block, dtype=torch.long),
            question_ids=qids,
            n_options=n_options,
        )

    def collate(self, batch: list[Packed]) -> dict[str, torch.Tensor]:
        """Right-pad to the longest sequence in the batch.

        Marker positions are shifted into the flat batch index space so the
        scoring head can gather them with a single `index_select`.
        """
        L = max(len(p) for p in batch)
        B = len(batch)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        block_id = torch.full((B, L), -1, dtype=torch.long)   # -1 = padding
        attn = torch.zeros((B, L), dtype=torch.bool)

        flat_pos, flat_block, flat_row, group = [], [], [], []
        g = 0
        for i, p in enumerate(batch):
            n = len(p)
            input_ids[i, :n] = p.input_ids
            block_id[i, :n] = p.block_id
            attn[i, :n] = True
            flat_pos.append(p.marker_pos + i * L)
            flat_block.append(p.marker_block)
            flat_row.append(torch.full_like(p.marker_block, i))
            # A global group id per (example, question), so every question's
            # options can be softmaxed independently with one scatter.
            for qi, k in enumerate(p.n_options):
                group.append(torch.full((k,), g, dtype=torch.long))
                g += 1

        return {
            "input_ids": input_ids,
            "block_id": block_id,
            "attention_mask": attn,
            "marker_flat": torch.cat(flat_pos),
            "marker_block": torch.cat(flat_block),
            "marker_row": torch.cat(flat_row),
            "marker_group": torch.cat(group),
            "n_groups": torch.tensor(g),
            "seq_len": torch.tensor(L),
        }

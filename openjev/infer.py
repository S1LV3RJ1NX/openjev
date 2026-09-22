"""Load a checkpoint and ask it questions, in two lines.

    from openjev import DecisionModel, Choice, Noul

    model = DecisionModel.from_pretrained("s1lv3rj1nx/openjev-general-lora")
    out = model.answer("I need a refill", {"intent": Choice(...)})
    out["intent"].label

Everything the model needs to be reproduced at inference time travels in the
checkpoint: the backbone, whether there is a residual head, the preamble and
option template it was trained with, and the fitted temperatures. Scoring the
same weights under a different prompt is a different system, so none of that
is left to the caller to remember.

Both architectures load through here and the checkpoint says which it is. The
LoRA decoder is the recommended one, so where this module has to pick a
default with no checkpoint to read it from, it picks the decoder's; the
encoder's defaults are left exactly as they were, because changing them would
move published numbers for checkpoints already in the wild.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ckpt import load_into, lora_rank, wants_head
from .encode import Packer
from .schema import Answer, Question

# The framing every decoder checkpoint here was trained and evaluated under.
# Only reached when a checkpoint does not carry its own `option_template`.
#
# It had `{q}` in it, which `Packer.pack` never fills: it formats the template
# with `opt` alone, so `str.format` raised KeyError('q') and every decoder
# checkpoint without a stored template failed to answer anything. The string
# was also not the one the models were trained with, which would have been the
# quieter half of the bug, since a checkpoint scored under a different prompt
# is a different system.
DEFAULT_DECODER_TEMPLATE = "\nOption: {opt}\nIs this the correct answer to the question? answer"

# Backbone to fall back on when a checkpoint does not name its own. Every
# checkpoint `ckpt.save` writes does name one, so this only reaches
# hand-assembled files, and it is keyed on the architecture because the two
# are not interchangeable: building a causal LM from encoder weights fails
# rather than degrading.
DECODER_BACKBONE = "Qwen/Qwen3-1.7B"
ENCODER_BACKBONE = "answerdotai/ModernBERT-base"

# Sequence budget, per architecture. This is a ceiling and not a cost: the
# packer raises if (state + instructions + every option) exceeds it, and
# `collate` pads to the longest sequence in the batch, so a larger number
# buys headroom and no memory.
#
# The decoder gets the larger one because large menus are the capability it
# is recommended for. The general adapter was trained at 4096 and the
# held-out suite is scored at 4096, so serving it at 2048 both truncated
# more of the state than training did and refused menus the published
# numbers were measured on. The encoder keeps 2048, which is what it has
# always had here.
DECODER_MAX_LEN = 4096
ENCODER_MAX_LEN = 2048


class DecisionModel:
    """A trained typed-decision model, ready to answer questions."""

    def __init__(self, model, packer: Packer, temperatures: dict[str, float] | None = None,
                 meta: dict[str, Any] | None = None):
        self.model = model
        self.packer = packer
        self.temperatures = temperatures or {}
        self.meta = meta or {}
        self.model.eval()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | None = None,
        max_len: int | None = None,
    ) -> DecisionModel:
        """Load from a local checkpoint, a directory, or a Hub repo id.

        `max_len` defaults to the architecture's budget above. Raise it for a
        menu that does not pack: the 151-way held-out task needs 6144. It is
        worth raising rather than trimming, because the packer reports the
        packed length and refuses instead of dropping options silently.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = _resolve(path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        from transformers import AutoTokenizer

        is_decoder = bool(ck.get("decoder"))
        backbone = ck.get("backbone") or (
            DECODER_BACKBONE if is_decoder else ENCODER_BACKBONE
        )
        tok = AutoTokenizer.from_pretrained(backbone)
        if max_len is None:
            max_len = DECODER_MAX_LEN if is_decoder else ENCODER_MAX_LEN

        packer = Packer(
            tok, max_len=max_len, max_state_len=max_len // 2,
            marker=":" if is_decoder else None,
            marker_after=is_decoder,
            option_template=ck.get("option_template")
            or (DEFAULT_DECODER_TEMPLATE if is_decoder else None),
            preamble=ck.get("preamble"),
        )

        if is_decoder:
            from .decoder import OpenJevDecoder

            model = OpenJevDecoder(
                backbone=backbone, tokenizer=tok, learned_head=wants_head(ck),
                lora_r=lora_rank(ck),
            ).to(device)
        else:
            from .model import OpenJev

            model = OpenJev(backbone=backbone, vocab_size=len(tok)).to(device)
        load_into(model, ck)
        # An unmerged adapter costs about 2x at inference for nothing:
        # measured p50 74ms unmerged against 40ms merged, on the same
        # checkpoint under the same load.
        if hasattr(model, "merge_adapter"):
            model.merge_adapter()
        return cls(
            model, packer, ck.get("temperatures"),
            {k: v for k, v in ck.items() if k != "state_dict"},
        )

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def answer(self, state: str | dict, questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer every question about one state, in a single forward pass."""
        return self.answer_batch([state], questions)[0]

    @torch.no_grad()
    def answer_batch(
        self, states: list[str | dict], questions: dict[str, Question], batch_size: int = 8,
    ) -> list[dict[str, Answer]]:
        device = next(self.model.parameters()).device
        results: list[dict[str, Answer]] = []
        for i in range(0, len(states), batch_size):
            chunk = states[i : i + batch_size]
            packed = [self.packer.pack(s, questions) for s in chunk]
            for p in packed:
                # predict() reads option labels off the packed object.
                p.labels = {qid: _labels(questions[qid]) for qid in p.question_ids}
            batch = self.packer.collate(packed)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            results.extend(self.model.predict(batch, packed))
        return results


def _labels(q: Question) -> list[str]:
    # Must agree with encode.question_options position for position. A local
    # reimplementation here returned criteria in *dict* order, so a noul whose
    # criteria happened to be written {"true": ..., "false": ...} had its two
    # probabilities swapped: `probabilities["true"]` read the "no" slot. On
    # held-out civil_comments that turned AUROC 0.712 into 0.288, which is
    # exactly 1 - 0.712, and looked like the model ranking toxic comments as
    # clean. One definition, imported, so the two cannot drift again.
    from .data import option_labels

    return option_labels(q)


def _resolve(path: str | Path) -> Path:
    """A local file, a directory holding model.pt, or a Hub repo id."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir() and (p / "model.pt").exists():
        return p / "model.pt"
    if p.exists():
        raise FileNotFoundError(f"{p} exists but holds no model.pt")
    # Not on disk, so treat it as a Hub repo id.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            f"{path} is not a local path and huggingface_hub is not installed, "
            f"so it cannot be fetched. pip install huggingface_hub"
        ) from e
    return Path(hf_hub_download(repo_id=str(path), filename="model.pt"))

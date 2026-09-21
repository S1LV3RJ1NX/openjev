"""Load a checkpoint and ask it questions, in two lines.

    from openjev import DecisionModel, Choice, Noul

    model = DecisionModel.from_pretrained("s1lv3rj1nx/openjev-router-healthcare")
    out = model.answer("I need a refill", {"intent": Choice(...)})
    out["intent"].label

Everything the model needs to be reproduced at inference time travels in the
checkpoint: the backbone, whether there is a residual head, the preamble and
option template it was trained with, and the fitted temperatures. Scoring the
same weights under a different prompt is a different system, so none of that
is left to the caller to remember.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ckpt import load_into, lora_rank, wants_head
from .encode import Packer
from .schema import Answer, Question

DEFAULT_DECODER_TEMPLATE = "\nQuestion: {q}\nIs the answer: {opt}\nAnswer"


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
        max_len: int = 2048,
    ) -> DecisionModel:
        """Load from a local checkpoint, a directory, or a Hub repo id."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = _resolve(path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        from transformers import AutoTokenizer

        backbone = ck.get("backbone", "answerdotai/ModernBERT-base")
        tok = AutoTokenizer.from_pretrained(backbone)
        is_decoder = bool(ck.get("decoder"))

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

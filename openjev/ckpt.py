"""Saving and loading checkpoints, with the guards that cost us a result.

Two failures motivated this module, both of which produced numbers rather
than errors:

  - A decoder checkpoint was built without its residual head, so
    `load_state_dict(strict=False)` dropped six tensors and evaluated the
    untrained readout. It reported 5.8x chance where the truth was 16.2x.
  - A decoder run overwrote the encoder checkpoint a published result
    depended on, because both defaulted to the mixture's name.

So loading raises rather than warns, and saving refuses to replace a
checkpoint of a different architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# A frozen-backbone decoder trains ~4.2M parameters and carries ~1.7B frozen
# ones that are already on the Hub. Storing the backbone makes a 17MB model
# into a 3.3GB file for no gain.
HEAD_PREFIXES = ("scorer.",)


def is_trainable_part(key: str) -> bool:
    """Whether a parameter belongs to the adapter rather than the backbone."""
    return key.startswith(HEAD_PREFIXES) or "lora_" in key


def head_only_state(state: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if is_trainable_part(k)}


def save(
    dest: Path | str,
    model,
    *,
    backbone: str,
    decoder: bool = False,
    head_only: bool = False,
    overwrite: bool = False,
    **meta: Any,
) -> Path:
    """Write a checkpoint, refusing to clobber a different architecture."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        try:
            prev = torch.load(dest, map_location="cpu", weights_only=False)
        except Exception:
            prev = {}
        if prev.get("backbone") and (
            prev["backbone"] != backbone or bool(prev.get("decoder")) != bool(decoder)
        ):
            raise SystemExit(
                f"{dest} holds a {prev['backbone']} "
                f"({'decoder' if prev.get('decoder') else 'encoder'}) checkpoint and "
                f"this is {backbone} ({'decoder' if decoder else 'encoder'}).\n"
                f"Use a different --out, or --overwrite if you mean it."
            )
    state = model.state_dict()
    if head_only:
        state = head_only_state(state)
        if not state:
            raise SystemExit(
                "head_only was requested but the model has no scorer weights. "
                "Saving this would produce an empty checkpoint."
            )
    torch.save(
        {"state_dict": state, "backbone": backbone, "decoder": bool(decoder),
         "head_only": bool(head_only), **meta},
        dest,
    )
    return dest


def load_into(model, ck: dict[str, Any]) -> None:
    """Load a checkpoint into a model, or raise explaining what went wrong.

    A head-only checkpoint legitimately leaves the backbone untouched, so the
    backbone keys are expected to be "missing". Everything it does carry must
    land, and nothing may be left over.
    """
    state = ck.get("state_dict")
    if not state:
        return
    # Checkpoints written while the decoder stored `_base` as an attribute
    # carry every tensor twice, once under `lm.` and once under `_base.`,
    # because nn.Module.__setattr__ registered the backbone a second time.
    # `_base` is a property now, so drop the duplicates rather than fail on
    # 392 keys that are already loaded under their real names.
    if any(k.startswith("_base.") for k in state) and not hasattr(type(model), "_base_is_module"):
        state = {k: v for k, v in state.items() if not k.startswith("_base.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if ck.get("head_only"):
        missing = [k for k in missing if is_trainable_part(k)]
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} "
            f"unexpected.\n  missing: {list(missing)[:4]}\n"
            f"  unexpected: {list(unexpected)[:4]}\n"
            f"A partially loaded model still produces a full results table, and "
            f"that table is wrong. Fix the model construction and re-run."
        )


def wants_head(ck: dict[str, Any]) -> bool:
    """Whether a checkpoint was trained with the residual scoring head."""
    if "learned_head" in ck:
        return bool(ck["learned_head"])
    return any(k.startswith(HEAD_PREFIXES) for k in (ck.get("state_dict") or {}))


def lora_rank(ck: dict[str, Any]) -> int:
    """The adapter rank a checkpoint needs, or 0. Rebuilding the model
    without it would drop every adapter tensor on load."""
    if ck.get("lora_r"):
        return int(ck["lora_r"])
    for k, v in (ck.get("state_dict") or {}).items():
        if "lora_A" in k and hasattr(v, "shape"):
            return int(v.shape[0])
    return 0

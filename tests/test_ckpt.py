"""Checkpoint round-trips, including the head-only decoder format.

The guards here each correspond to a failure that produced a plausible
number instead of an error.
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev.ckpt import head_only_state, load_into, save, wants_head  # noqa: E402


class Fake(nn.Module):
    """Stands in for a frozen backbone plus a small trained head."""

    def __init__(self, head: bool = True):
        super().__init__()
        self.lm = nn.Linear(64, 64)
        self.scorer = nn.Sequential(nn.Linear(64, 8), nn.GELU(), nn.Linear(8, 1)) if head else None


def test_head_only_checkpoint_is_tiny_and_reloads():
    with tempfile.TemporaryDirectory() as d:
        m = Fake()
        full = save(Path(d) / "full.pt", m, backbone="b", decoder=True)
        head = save(Path(d) / "head.pt", m, backbone="b", decoder=True, head_only=True)
        assert head.stat().st_size < full.stat().st_size / 4, (
            head.stat().st_size, full.stat().st_size)

        # A head-only checkpoint must load into a fresh model without
        # complaining about the backbone it deliberately does not carry.
        fresh = Fake()
        ck = torch.load(head, map_location="cpu", weights_only=False)
        assert wants_head(ck)
        load_into(fresh, ck)
        for a, b in zip(fresh.scorer.parameters(), m.scorer.parameters()):
            assert torch.equal(a, b)


def test_load_raises_when_the_head_is_missing_from_the_model():
    # The exact bug: a checkpoint with scorer weights loaded into a model
    # built without one. It must not pass silently.
    with tempfile.TemporaryDirectory() as d:
        p = save(Path(d) / "m.pt", Fake(head=True), backbone="b", decoder=True)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        try:
            load_into(Fake(head=False), ck)
        except SystemExit as e:
            assert "unexpected" in str(e)
        else:
            raise AssertionError("loading dropped the head without raising")


def test_save_refuses_to_clobber_another_architecture():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.pt"
        save(p, Fake(), backbone="encoder-x", decoder=False)
        try:
            save(p, Fake(), backbone="decoder-y", decoder=True)
        except SystemExit as e:
            assert "different" in str(e) or "holds a" in str(e)
        else:
            raise AssertionError("a decoder run overwrote an encoder checkpoint")
        # ...unless asked explicitly.
        save(p, Fake(), backbone="decoder-y", decoder=True, overwrite=True)


def test_head_only_refuses_to_write_an_empty_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        try:
            save(Path(d) / "m.pt", Fake(head=False), backbone="b",
                 decoder=True, head_only=True)
        except SystemExit as e:
            assert "empty" in str(e)
        else:
            raise AssertionError("wrote a checkpoint with no weights in it")


def test_wants_head_infers_from_weights_when_unflagged():
    assert wants_head({"state_dict": {"scorer.0.weight": 1}})
    assert not wants_head({"state_dict": {"lm.weight": 1}})
    # An explicit flag wins over inference.
    assert not wants_head({"learned_head": False, "state_dict": {"scorer.0.weight": 1}})


def test_head_only_state_keeps_only_the_head():
    s = head_only_state({"lm.a": 1, "scorer.0.weight": 2, "scorer.2.bias": 3})
    assert set(s) == {"scorer.0.weight", "scorer.2.bias"}


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:
            bad += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)

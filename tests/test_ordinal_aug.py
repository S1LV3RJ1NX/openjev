"""Ordinal augmentation must never move the gold off the right level.

Silent label corruption is the expensive failure here: it looks exactly like
"the ordinal head does not work", which is a conclusion we already drew once
from a bug rather than from evidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev.data import ORDINAL_FAMILIES, coarsen, scale_family  # noqa: E402


def test_coarsen_keeps_gold_inside_the_span_it_points_at():
    # The merged label is a span, "L2 to L4", so the check is that the gold
    # lies between the named endpoints, not that it appears in the string.
    for k in range(2, 10):
        levels = [f"L{i}" for i in range(k)]
        for m in range(2, k + 1):
            groups: dict[int, list[int]] = {}
            for gold in range(k):
                merged, new_gold = coarsen(levels, m, gold)
                assert len(merged) == m, (k, m, merged)
                assert 0 <= new_gold < m
                groups.setdefault(new_gold, []).append(gold)
            merged, _ = coarsen(levels, m, 0)
            for j, members in groups.items():
                lo, hi = min(members), max(members)
                expect = f"L{lo}" if lo == hi else f"L{lo} to L{hi}"
                assert merged[j] == expect, (k, m, j, merged[j], expect)
                # Contiguous: an ordinal group cannot skip a level.
                assert members == list(range(lo, hi + 1)), (k, m, members)


def test_coarsen_is_monotone():
    # Order is the whole content of an ordinal scale. A higher source level
    # must never map to a lower merged level.
    for k in range(2, 10):
        levels = [f"L{i}" for i in range(k)]
        for m in range(2, k + 1):
            mapped = [coarsen(levels, m, g)[1] for g in range(k)]
            assert mapped == sorted(mapped), (k, m, mapped)


def test_coarsen_uses_every_group():
    for k in range(2, 10):
        levels = [f"L{i}" for i in range(k)]
        for m in range(2, k + 1):
            assert {coarsen(levels, m, g)[1] for g in range(k)} == set(range(m))


def test_family_vocabularies_are_the_length_they_claim():
    for fam, by_len in ORDINAL_FAMILIES.items():
        for n, vocabs in by_len.items():
            for v in vocabs:
                assert len(v) == n, (fam, n, v)
                assert len(set(v)) == n, f"duplicate level in {fam}/{n}: {v}"


def test_family_detection_on_the_scales_we_actually_have():
    # These are the real scales in the mixture, plus the held-out target.
    assert scale_family(["negative", "neutral", "positive"]) == "valence"
    assert scale_family(["1 star", "2 star", "3 stars", "4 stars", "5 stars"]) == "quality"
    assert scale_family(["depth 5", "depth 6", "depth 7"]) == "magnitude"
    assert scale_family(["strongly disagree", "disagree", "agree"]) == "agreement"


def test_no_reword_can_silently_change_length():
    # A reworded scale is only substituted for one of the same length, so a
    # gold index stays valid. Guard the invariant the code relies on.
    for fam, by_len in ORDINAL_FAMILIES.items():
        for n, vocabs in by_len.items():
            for v in vocabs:
                assert scale_family(v) is not None, (fam, v)


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

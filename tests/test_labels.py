"""Label-set classification, checked against the real sets in the mixture.

A false positive here silently inverts a task's labels, so the negatives
matter as much as the positives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev.labels import negation_pair, noul_positive  # noqa: E402

# Real label sets from tasks/mixture_ord2 that are yes/no questions wearing a
# two-option menu, with the label that means "yes".
YES_NO = [
    (["Not-Related", "Related"], "Related"),
    (["not_equivalent", "equivalent"], "equivalent"),
    (["not_duplicate", "duplicate"], "duplicate"),
    (["not_paraphrase", "paraphrase"], "paraphrase"),
    (["no_hate_speech", "hate_speech"], "hate_speech"),
    (["notsarc", "sarc"], "sarc"),
    (["valid", "invalid"], "valid"),
    (["No.", "Yes."], "Yes."),
    (["unacceptable", "acceptable"], "acceptable"),
    (["formal", "informal"], "formal"),
]

# Real label sets from the same mixture that are genuinely two-way choices or
# two-level scales. Treating any of these as yes/no would invent a polarity.
NOT_YES_NO = [
    ["constituency", "national"],
    ["strengthener", "weakener"],
    ["NN", "NNS"],
    ["PAST", "PRES"],
    ["I", "O"],
    ["C", "O"],
    ["low", "high"],
    ["negative", "positive"],
    ["partisan", "neutral"],
    ["commissive", "directive"],
]


def test_finds_the_positive_label_in_real_negation_pairs():
    for names, positive in YES_NO:
        idx = noul_positive(names)
        assert idx is not None, f"missed {names}"
        assert names[idx] == positive, (names, names[idx], positive)


def test_leaves_genuine_two_way_choices_alone():
    for names in NOT_YES_NO:
        assert noul_positive(names) is None, f"wrongly treated {names} as yes/no"


def test_order_does_not_change_the_answer():
    for names, positive in YES_NO:
        flipped = list(reversed(names))
        idx = noul_positive(flipped)
        assert idx is not None and flipped[idx] == positive, flipped


def test_ignores_sets_that_are_not_pairs():
    assert negation_pair(["a"]) is None
    assert negation_pair(["a", "b", "c"]) is None
    assert negation_pair([]) is None


def test_ignores_labels_that_normalise_to_the_same_thing():
    # "not_x" and "NOT-X" are one label written twice, not a pair.
    assert negation_pair(["not_x", "NOT-X"]) is None
    assert negation_pair(["", ""]) is None


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

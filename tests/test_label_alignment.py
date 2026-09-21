"""Option labels must line up with the options actually scored.

The packer decides what text sits at each marker; `option_labels` decides
what each position is called. If they disagree the probabilities are
attached to the wrong options and every downstream number is quietly wrong
in a way that still looks like a plausible result.

This bit once: a noul whose criteria were written {"true": ..., "false": ...}
had its probabilities swapped, turning AUROC 0.712 into 0.288.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev.data import option_labels  # noqa: E402
from openjev.encode import question_options  # noqa: E402
from openjev.schema import Choice, Noul, Score  # noqa: E402


def test_counts_match_for_every_question_type():
    questions = [
        Choice(instructions="q", criteria={"a": "first", "b": "second", "c": "third"}),
        Score(instructions="q", criteria=["low", "mid", "high"]),
        Noul(instructions="q"),
        Noul(instructions="q", criteria={"true": "it applies", "false": "it does not"}),
        # The order that caused the bug: "true" written first.
        Noul(instructions="q", criteria={"false": "no", "true": "yes"}),
    ]
    for q in questions:
        assert len(option_labels(q)) == len(question_options(q)), q


def test_noul_polarity_is_positional_not_dict_order():
    # Both orderings must produce the same positions: index 0 false, 1 true.
    a = Noul(instructions="q", criteria={"true": "T", "false": "F"})
    b = Noul(instructions="q", criteria={"false": "F", "true": "T"})
    assert option_labels(a) == ["false", "true"]
    assert option_labels(b) == ["false", "true"]
    assert question_options(a) == question_options(b)
    # And the rendered option at the "true" index must be the affirmative one.
    assert question_options(a)[option_labels(a).index("true")] == "yes"


def test_choice_labels_follow_menu_order():
    q = Choice(instructions="q", criteria={"z": "last letter", "a": "first letter"})
    assert option_labels(q) == ["z", "a"]
    assert len(question_options(q)) == 2


def test_score_labels_are_level_indices_in_order():
    q = Score(instructions="q", criteria=["worst", "middle", "best"])
    assert option_labels(q) == ["0", "1", "2"]
    assert len(question_options(q)) == 3


def test_inference_helper_agrees_with_the_canonical_function():
    # infer._labels must not reimplement this.
    from openjev.infer import _labels

    for q in (
        Choice(instructions="q", criteria={"a": "1", "b": "2"}),
        Score(instructions="q", criteria=["l", "h"]),
        Noul(instructions="q", criteria={"true": "T", "false": "F"}),
        Noul(instructions="q"),
    ):
        assert _labels(q) == option_labels(q), q


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

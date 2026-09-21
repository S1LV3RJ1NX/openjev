"""The dataset must never raise, whatever the source data looks like.

Two runs died thousands of steps in on "packed sequence is N tokens, over
max_len". Finding a data fault after the expensive part is the worst case,
so the pathological shapes are pinned here where they cost a second.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from openjev.data import TaskDataset, _clip_options  # noqa: E402
from openjev.encode import Packer, question_options  # noqa: E402
from openjev.schema import Choice, Example, Task  # noqa: E402

TOK = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def _dataset(examples, questions, max_len=512):
    task = Task(name="t", description="d", questions=questions, examples=examples)
    return TaskDataset(task, Packer(TOK, max_len=max_len, max_state_len=max_len // 2),
                       shuffle_options=True, seed=0)


def test_clipping_shortens_a_self_describing_menu():
    # {option: option} is the MultipleChoice shape. Clipping only the value
    # made the two differ, defeating the duplicate collapse in option_text
    # and making the rendered option longer than the original.
    long = "x" * 3000
    q = Choice(instructions="q", criteria={long: long, "b" * 3000: "b" * 3000})
    before = sum(len(o) for o in question_options(q))
    clipped, gold = _clip_options(q, 100, long)
    after = sum(len(o) for o in question_options(clipped))
    assert after < before / 5, (before, after)
    assert gold in clipped.criteria, "gold must follow its clipped key"


def test_clipping_keeps_options_distinct():
    # Two options sharing a long prefix must not collapse onto one key.
    a, b = "same prefix " * 40 + "AAA", "same prefix " * 40 + "BBB"
    q = Choice(instructions="q", criteria={a: a, b: b})
    clipped, gold = _clip_options(q, 30, a)
    assert len(clipped.criteria) == 2, clipped.criteria
    assert gold in clipped.criteria


def test_passage_length_options_still_pack():
    # The exact shape that killed two runs: a long state and two options that
    # are themselves passages.
    opts = {("p" * 4000): ("p" * 4000), ("q" * 4000): ("q" * 4000)}
    ds = _dataset(
        [Example(state="s " * 2000, answers={"a": "p" * 4000})],
        {"a": Choice(instructions="which?", criteria=opts)},
    )
    s = ds[0]
    assert s.targets.get("a") is not None, "gold survived clipping"
    assert s.packed.labels["a"][s.targets["a"]] in s.packed.labels["a"]


def test_many_long_options_still_pack():
    opts = {f"{i}-" + "z" * 800: f"{i}-" + "z" * 800 for i in range(40)}
    ds = _dataset(
        [Example(state="state " * 500, answers={"a": "0-" + "z" * 800})],
        {"a": Choice(instructions="which?", criteria=opts)},
    )
    s = ds[0]
    assert s.targets.get("a") is not None


def test_ordinary_examples_are_untouched():
    q = Choice(instructions="which?", criteria={"refill": "another fill",
                                                "hours": "opening times"})
    ds = _dataset([Example(state="need a refill", answers={"a": "refill"})], {"a": q})
    s = ds[0]
    assert s.packed.labels["a"] == ["refill", "hours"] or \
           set(s.packed.labels["a"]) == {"refill", "hours"}
    assert s.targets["a"] == s.packed.labels["a"].index("refill")


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

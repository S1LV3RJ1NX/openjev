"""Banking77 as an in-task specialist task, with train and dev splits.

`tasks/heldout/banking77` is test-only on purpose: it is a generalization
probe. This builds a *separate* task with a real train split so we can run the
specialist experiment — does the scoring head learn anything — without
pretending afterwards that Banking77 is still held out for that checkpoint.

Test rows are the same 600 the held-out task uses, so the two are directly
comparable, and the train/dev rows come from the upstream train split only.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev import Choice, Example, Task  # noqa: E402

BASE = (
    "https://huggingface.co/datasets/mteb/banking77/resolve/"
    "refs%2Fconvert%2Fparquet/default"
)
CACHE = Path("/tmp/ojcache")
DEV_PER_CLASS = 2


def fetch(split: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"b77_{split}.parquet"
    if not p.exists():
        urllib.request.urlretrieve(f"{BASE}/{split}/0000.parquet", p)
    return pd.read_parquet(p)


def main() -> None:
    heldout = Task.load(ROOT / "tasks" / "heldout" / "banking77", "test")
    question = heldout.questions["intent"]
    assert isinstance(question, Choice)
    print(f"reusing the held-out schema: {len(question.criteria)} labels, "
          f"{len(heldout)} test rows")

    tr = fetch("train")
    dev_idx = (
        tr.groupby("label_text", group_keys=False)
        .apply(lambda g: g.sample(n=min(DEV_PER_CLASS, len(g)), random_state=0))
        .index
    )
    dev = tr.loc[dev_idx]
    train = tr.drop(index=dev_idx)

    def rows(df, tier):
        return [
            Example(
                state=r.text,
                answers={"intent": r.label_text},
                meta={"tier": tier, "source": "mteb/banking77"},
            )
            for r in df.itertuples()
        ]

    out = ROOT / "tasks" / "banking77_specialist"
    for split, df in (("train", train), ("dev", dev)):
        t = Task(
            name="banking77_specialist",
            description=(
                "Banking77 with a train split, for the in-task specialist "
                "experiment. NOT a generalization task: a model trained here "
                "may not quote tasks/heldout/banking77 as held out."
            ),
            questions={"intent": question},
            examples=rows(df, "train" if split == "train" else "dev"),
        )
        problems = t.validate()
        assert not problems, problems[:5]
        t.save(ROOT / "tasks", split=split)
        print(f"{split}: {len(t)} rows")

    t = Task(
        name="banking77_specialist",
        description="see task.json",
        questions={"intent": question},
        examples=heldout.examples,
    )
    t.save(ROOT / "tasks", split="test")
    print(f"test: {len(t)} rows (identical to tasks/heldout/banking77)")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

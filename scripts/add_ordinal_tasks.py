"""Add real ordinal tasks to a mixture, from datasets curated by hand.

`tasksource`'s classification family is almost all `choice`. Sweeping it
yielded 9 `score` tasks, six of which are the same three-level
negative/neutral/positive scale, against a held-out task wanting a five-level
judgement. Synthesizing the missing diversity was tried and measured as a
null result, so this fetches real ordinal data instead.

    uv run python scripts/add_ordinal_tasks.py --into tasks/mixture_noul

## What is deliberately not here

`openai/summarize_from_feedback` ships 1–7 Likert ratings of response quality
and would be the single best fit on paper. That is the problem: the held-out
task, HelpSteer, rates response helpfulness on a five-point scale. Training
on a near-twin of the evaluation task would move the number without moving
the ability it is supposed to measure. The same reasoning rules out every
other LLM-response-rating corpus.

What is here instead is ordinal judgement of a different thing entirely —
star ratings of apps and products — so a gain on HelpSteer would be transfer
rather than recognition.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.heldout import assert_training_mixture_clean  # noqa: E402
from openjev.schema import Example, Score, Task  # noqa: E402

STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

# name, config, split, text column, label column, level names, zero-based?
SOURCES = [
    ("app_reviews", None, "train", "review", "star", STARS, False),
    ("SetFit/amazon_reviews_multi_en", None, "train", "text", "label", STARS, True),
]

PROMPT = "How many stars does this review give?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True, help="mixture directory to add to")
    ap.add_argument("--max-rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    # These are added by hand, so they bypass the builder's filter. Check them
    # against the held-out suite before anything is written.
    assert_training_mixture_clean([s[0] for s in SOURCES])
    print(f"contamination guard: {len(SOURCES)} hand-added sources, no overlap")

    out = Path(args.into)
    rng = random.Random(args.seed)
    added = 0
    for name, cfg, split, text_col, label_col, levels, zero_based in SOURCES:
        try:
            d = load_dataset(name, cfg, split=split)
        except Exception as e:
            print(f"FAIL {name}: {str(e).splitlines()[0][:70]}")
            continue
        # Several of these ship sorted by label, so taking a head slice gives
        # one class and a task that teaches nothing.
        d = d.shuffle(seed=args.seed).select(range(min(args.max_rows, len(d))))

        examples, seen = [], set()
        for row in d:
            text = str(row.get(text_col) or "").strip()
            raw = row.get(label_col)
            if not text or raw is None or text in seen:
                continue
            seen.add(text)
            lvl = int(raw) if zero_based else int(raw) - 1
            if not 0 <= lvl < len(levels):
                continue
            examples.append(Example(
                state=text, answers={"rating": lvl},
                meta={"tier": "mixture_ordinal", "source": name},
            ))
        if len(examples) < 32:
            print(f"skip {name}: only {len(examples)} usable rows")
            continue

        rng.shuffle(examples)
        task = Task(
            name="ordinal_" + name.split("/")[-1].replace("-", "_"),
            description=f"curated ordinal {name}",
            questions={"rating": Score(instructions=PROMPT, criteria=list(levels))},
            examples=examples,
        )
        problems = task.validate()
        if problems:
            raise SystemExit(f"{name} invalid: {problems[:3]}")
        task.save(out, split="train")
        dist = {i: sum(1 for e in examples if e.answers["rating"] == i)
                for i in range(len(levels))}
        print(f"added {task.name:<38} {len(examples):>5} rows  K={len(levels)}  dist={dist}")
        added += 1

    print(f"\n{added} ordinal task(s) added to {out}")


if __name__ == "__main__":
    main()

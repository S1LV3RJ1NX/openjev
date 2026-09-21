"""Build a task directory from a CSV, so the first step is not writing JSON.

    uv run python scripts/make_task.py --csv mydata.csv \\
        --text-column message --label-column intent --name my_task

Produces tasks/<name>/ with train/dev/test and a `choice` question built from
the distinct labels. Descriptions are left as placeholders on purpose: writing
what distinguishes each label from its neighbours was worth ~+5 accuracy
points in our measurements, and no script can do it for you.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.schema import Choice, Example, Task  # noqa: E402


def humanize(label: str) -> str:
    return re.sub(r"[_\-]+", " ", str(label)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-column", required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--question-id", default="answer")
    ap.add_argument("--instructions", default="Which of these applies?")
    ap.add_argument("--out", default=str(ROOT / "tasks"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.csv)
    for col in (args.text_column, args.label_column):
        if col not in df.columns:
            raise SystemExit(f"column {col!r} not in CSV. Found: {list(df.columns)}")

    df = df[[args.text_column, args.label_column]].dropna()
    df = df[df[args.text_column].astype(str).str.strip().astype(bool)]
    labels = sorted(df[args.label_column].astype(str).unique())
    if not 2 <= len(labels) <= 255:
        raise SystemExit(f"need between 2 and 255 distinct labels, found {len(labels)}")

    counts = Counter(df[args.label_column].astype(str))
    thin = [l for l, c in counts.items() if c < 10]

    question = Choice(
        instructions=args.instructions,
        # Placeholders. The humanised label is a starting point, not a
        # description: our own ablation found restating the label name is
        # worth nothing, while stating the distinction is worth ~+5 points.
        criteria={l: f"TODO: what distinguishes {humanize(l)} from its neighbours" for l in labels},
    )

    # Stratified split, so rare labels appear in every split rather than
    # landing entirely in one of them.
    shuffled = df.sample(frac=1.0, random_state=args.seed)
    buckets: dict[str, list] = {"train": [], "dev": [], "test": []}
    per_label: dict[str, int] = {}
    for row in shuffled.itertuples():
        lab = str(getattr(row, args.label_column))
        i = per_label.get(lab, 0)
        per_label[lab] = i + 1
        n = counts[lab]
        cut_dev = max(1, int(n * args.dev_frac)) if n >= 10 else 0
        cut_test = max(1, int(n * args.test_frac)) if n >= 10 else 0
        split = "dev" if i < cut_dev else "test" if i < cut_dev + cut_test else "train"
        buckets[split].append(
            Example(
                state=str(getattr(row, args.text_column)),
                answers={args.question_id: lab},
                meta={"tier": "default", "source": Path(args.csv).name},
            )
        )

    out = Path(args.out)
    for split, examples in buckets.items():
        if not examples:
            continue
        t = Task(
            name=args.name,
            description=f"Built from {Path(args.csv).name} by scripts/make_task.py",
            questions={args.question_id: question},
            examples=examples,
        )
        problems = t.validate()
        if problems:
            raise SystemExit(f"validate() failed: {problems[:5]}")
        t.save(out, split=split)
        print(f"{split:>6}: {len(examples)} rows")

    print(f"\nwritten to {out / args.name}")
    print(f"{len(labels)} labels, {len(df)} rows")
    if thin:
        print(
            f"\n!! {len(thin)} label(s) have fewer than 10 examples: {thin[:6]}\n"
            f"   A class with a handful of examples tends to be learned as "
            f"'never predict this'. Ours scored 0.000 recall while looking\n"
            f"   accurate overall. Check per-class recall, not just accuracy."
        )
    print(
        f"\nNext: edit tasks/{args.name}/task.json and replace the TODO "
        f"descriptions.\nThen:\n"
        f"  uv run python scripts/train.py --task tasks/{args.name} --epochs 6"
    )


if __name__ == "__main__":
    main()

"""Re-type an already-built mixture in place, without re-downloading it.

A yes/no question emitted as a two-option `choice` menu does not teach the
`noul` primitive, and the builder's original detector recognised only eight
hardcoded label names. This rewrites the affected tasks from a built mixture
rather than spending forty minutes fetching it again.

    uv run python scripts/retype_mixture.py \\
        --in tasks/mixture_ord2 --out tasks/mixture_noul

Labels are preserved exactly: the gold becomes True on the positive label of
the pair and False on its negation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.labels import noul_positive  # noqa: E402
from openjev.schema import Noul, Task  # noqa: E402

SPLITS = ("train", "dev", "test")


def humanize(label) -> str:
    return re.sub(r"[_\-]+", " ", str(label)).strip().rstrip(".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dirs = sorted(p for p in src.iterdir() if (p / "task.json").exists())
    if not dirs:
        raise SystemExit(f"no tasks under {src}")

    converted, untouched, rows = [], 0, 0
    for d in dirs:
        # Load each split that exists; the mixture is usually train-only.
        splits = {s: Task.load(d, s) for s in SPLITS if (d / f"{s}.jsonl").exists()}
        if not splits:
            continue
        probe = next(iter(splits.values()))

        changes: dict[str, tuple[Noul, str]] = {}
        for qid, q in probe.questions.items():
            names = list(getattr(q, "criteria", {}) or {})
            if q.__class__.__name__ != "Choice" or len(names) != 2:
                continue
            idx = noul_positive(names)
            if idx is None:
                continue
            positive = names[idx]
            changes[qid] = (
                Noul(instructions=f'Is the answer "{humanize(positive)}"?'),
                positive,
            )

        if not changes:
            untouched += 1
            if not args.dry_run:
                for s, t in splits.items():
                    t.save(dst, split=s)
                    rows += len(t)
            continue

        for s, t in splits.items():
            for qid, (noul, positive) in changes.items():
                t.questions[qid] = noul
                for ex in t.examples:
                    if qid in ex.answers:
                        # The gold was the label string; it becomes the truth
                        # of "is it the positive one".
                        ex.answers[qid] = str(ex.answers[qid]) == str(positive)
            problems = t.validate()
            if problems:
                raise SystemExit(f"{d.name}/{s} invalid after retyping: {problems[:3]}")
            if not args.dry_run:
                t.save(dst, split=s)
                rows += len(t)

        converted.append((d.name, [c[1] for c in changes.values()]))

    print(f"converted {len(converted)} tasks to noul, left {untouched} unchanged")
    for name, positives in converted:
        print(f"  {name:<46} positive = {positives}")
    if args.dry_run:
        print("\n(dry run, nothing written)")
    else:
        print(f"\nwrote {len(dirs)} tasks, {rows} rows to {dst}")


if __name__ == "__main__":
    main()

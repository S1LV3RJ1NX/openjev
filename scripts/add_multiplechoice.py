"""Add tasksource's MultipleChoice family to a mixture.

The builder filters `task_type == "Classification"`, which is 392 of the 668
tasks available and excludes the 261 MultipleChoice ones outright. The
trainer has supported per-example option menus for a while, but no task in
the mixture ever used them, so the capability was there and the data was
not.

This matters for scale as much as for shape: the mixture is 145 tasks and
the Flan work puts the point where gain keeps accruing nearer 282.

    uv run python scripts/add_multiplechoice.py --into tasks/mixture_ord3

MultipleChoice rows carry `inputs`, `labels` and per-row `choice0..choiceN`,
so each example brings its own menu and the task-level menu is a placeholder.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.heldout import assert_training_mixture_clean, find_leaks  # noqa: E402
from openjev.schema import Choice, Example, Task  # noqa: E402

CHOICE_COL = re.compile(r"^choice(\d+)$")


def safe_name(task_id: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", task_id).strip("_").lower()


def load_with_backoff(tasksource, tid: str, max_rows: int, retries: int):
    """The Hub rate-limits hard when pulling hundreds of datasets in a row."""
    delay = 4.0
    for attempt in range(retries + 1):
        try:
            return tasksource.load_task(tid, max_rows=max_rows)
        except Exception as e:  # noqa: BLE001
            transient = any(
                s in f"{type(e).__name__}{e}"
                for s in ("HfHubHTTPError", "429", "Too Many Requests",
                          "ConnectionError", "ReadTimeout", "504", "502")
            )
            if not transient or attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2


def to_task(task_id: str, dd, max_rows: int) -> Task | None:
    split = dd.get("train") or next(iter(dd.values()))
    cols = split.column_names
    choice_cols = sorted(
        (c for c in cols if CHOICE_COL.match(c)),
        key=lambda c: int(CHOICE_COL.match(c).group(1)),
    )
    if len(choice_cols) < 2 or "inputs" not in cols or "labels" not in cols:
        return None

    examples = []
    for row in split.select(range(min(max_rows, len(split)))):
        state = str(row.get("inputs") or "").strip()
        gold_idx = row.get("labels")
        if not state or gold_idx is None:
            continue
        opts = [str(row.get(c) or "").strip() for c in choice_cols]
        opts = [o for o in opts if o]
        # A menu with a repeated option has no single correct answer, and the
        # criteria dict would silently collapse the duplicates.
        if len(opts) < 2 or len(set(opts)) != len(opts):
            continue
        if not 0 <= int(gold_idx) < len(opts):
            continue
        examples.append(Example(
            state=state,
            answers={"answer": opts[int(gold_idx)]},
            criteria={"answer": {o: o for o in opts}},
            meta={"tier": "mixture_mc", "source": task_id},
        ))
    if len(examples) < 32:
        return None

    return Task(
        name=safe_name(task_id),
        description=f"tasksource {task_id}",
        # Placeholder: every example overrides this with its own menu.
        questions={"answer": Choice(
            instructions="Which option is correct?",
            criteria={"placeholder_a": "", "placeholder_b": ""},
        )},
        examples=examples,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True)
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=261)
    ap.add_argument("--throttle", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    import tasksource

    listing = tasksource.list_tasks()
    listing = listing[listing.task_type == "MultipleChoice"]
    print(f"{len(listing)} MultipleChoice tasks in tasksource")

    out = Path(args.into)
    built = skipped = failed = 0
    excluded: list[str] = []
    rng = random.Random(0)

    for _, row in list(listing.iterrows())[: args.limit]:
        tid = row.id
        # Guard before fetching, not after: a held-out dataset should never
        # reach disk in a training mixture.
        if find_leaks([tid]):
            excluded.append(tid)
            continue
        try:
            dd = load_with_backoff(tasksource, tid, args.max_rows, args.retries)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {tid[:52]:<54} {str(e).splitlines()[0][:50]}")
            continue
        try:
            task = to_task(tid, dd, args.max_rows)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {tid[:52]:<54} {str(e).splitlines()[0][:50]}")
            continue
        if task is None:
            skipped += 1
            continue
        problems = task.validate()
        if problems:
            skipped += 1
            print(f"SKIP {tid[:52]:<54} invalid: {problems[0][:40]}")
            continue
        rng.shuffle(task.examples)
        task.save(out, split="train")
        built += 1
        ks = [len(e.criteria["answer"]) for e in task.examples]
        print(f"[{built}] {task.name[:48]:<50} n={len(task):<6} K={min(ks)}-{max(ks)}")
        time.sleep(args.throttle)

    print(f"\nbuilt {built}, skipped {skipped} (shape/size), failed {failed}")
    if excluded:
        print(f"excluded as held-out: {len(excluded)}")
        for t in excluded[:10]:
            print(f"   {t}")

    # Final check over everything now in the directory.
    dirs = sorted(p for p in out.iterdir() if (p / "task.json").exists())
    names = [Task.load(p, "train").description.replace("tasksource ", "")
             for p in dirs if (p / "train.jsonl").exists()]
    assert_training_mixture_clean(names)
    print(f"contamination guard: {len(names)} sources in {out}, no overlap")


if __name__ == "__main__":
    main()

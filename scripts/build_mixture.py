"""Build a multi-task training mixture from tasksource, minus anything held out.

Our zero-shot measurement showed the warm start alone gives nothing usable
(see docs/results.md). Task diversity is where the capability comes from, so
this is the load-bearing data step, not a nicety.

Contamination is filtered **here**, at build time, rather than trusted to a
check at train time. `tasksource` contains Banking77, SST-2 and friends, and
not always under a recognisable name — `rotten_tomatoes` carries 77% of
SST-5's test sentences — so the exclusion list is the 140-name holdout union
and matching is done on dataset name, config, and the full task id.

Run in the pinned data environment, because tasksource's loader needs an
older `datasets`:

    uv venv .venv-data -p 3.12
    VIRTUAL_ENV=.venv-data uv pip install "datasets==2.21.0" tasksource \\
        "fsspec<=2024.6.1" "huggingface_hub<0.26"
    VIRTUAL_ENV=.venv-data uv run --no-project python scripts/build_mixture.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import only the pure-python schema; openjev has no heavy deps.
from openjev.schema import Choice, Example, Noul, Task  # noqa: E402

OUT = ROOT / "tasks" / "mixture"

# Loading every task pulls tens of GB and most of it is redundant. These are
# the families that map cleanly onto typed decisions.
SKIP_PATTERNS = [
    r"^babi_nli", r"lm_task", r"^blimp", r"probe",  # synthetic / diagnostic
]


def holdout_names() -> set[str]:
    with open(ROOT / "tasks" / "heldout" / "manifest.json") as f:
        m = json.load(f)
    names = set()
    for n in m["holdout_of_union"]:
        n = n.strip().lower()
        names.add(n)
        names.add(n.rsplit("/", 1)[-1])
    return names


def is_excluded(row, banned: set[str]) -> str | None:
    """Return the offending name if this task must not be trained on."""
    candidates = {
        str(row.id).lower(),
        str(row.dataset_name).lower(),
        f"{row.dataset_name}/{row.config_name}".lower(),
        str(row.dataset_name).lower().rsplit("/", 1)[-1],
    }
    for c in candidates:
        if c in banned:
            return c
        for b in banned:
            if b and (c == b or c.startswith(b + "/") or c.split("/")[0] == b):
                return b
    return None


def humanize(label: str) -> str:
    s = re.sub(r"[_\-]+", " ", str(label)).strip()
    return s if s else str(label)


def safe_name(task_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")


def to_task(task_id: str, dd, max_rows: int, rng: random.Random) -> Task | None:
    """tasksource's standardised columns -> one openjev Task."""
    if "train" not in dd:
        return None
    tr = dd["train"]
    cols = tr.column_names
    if "labels" not in cols:
        return None

    feat = tr.features["labels"]
    names = getattr(feat, "names", None)
    if names is None:  # Sequence(ClassLabel) = multi-label; out of scope for v1
        return None
    if not 2 <= len(names) <= 255:
        return None

    is_mc = any(c.startswith("choice") for c in cols)
    if is_mc:
        # Multiple choice: the options are per-row text, so a fixed menu does
        # not exist. Skipping keeps the schema honest; supporting it properly
        # means per-example criteria, which the format allows but the trainer
        # does not yet.
        return None

    text_cols = [c for c in ("sentence1", "sentence2") if c in cols]
    if not text_cols:
        return None

    binary = len(names) == 2
    qid = "answer"
    criteria = {str(n): humanize(n) for n in names}
    question = (
        Noul(instructions=f"Is the answer to this {humanize(names[1])}?")
        if binary and set(map(str.lower, map(str, names))) <= {"yes", "no", "true", "false",
                                                               "entailment", "not_entailment",
                                                               "acceptable", "unacceptable"}
        else Choice(instructions="Which of these applies?", criteria=criteria)
    )

    examples = []
    n = min(len(tr), max_rows)
    for r in tr.select(range(n)):
        lab = r["labels"]
        if lab is None or not isinstance(lab, int) or not 0 <= lab < len(names):
            continue
        state = (
            {c: r[c] for c in text_cols} if len(text_cols) > 1 else r[text_cols[0]]
        )
        if not state:
            continue
        if isinstance(question, Noul):
            answers = {qid: bool(lab == 1)}
        else:
            answers = {qid: str(names[lab])}
        examples.append(
            Example(state=state, answers=answers, meta={"tier": "mixture", "source": task_id})
        )

    if len(examples) < 32:
        return None
    rng.shuffle(examples)
    return Task(
        name=safe_name(task_id),
        description=f"tasksource {task_id}",
        questions={qid: question},
        examples=examples,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tasks", type=int, default=250)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tasksource

    rng = random.Random(args.seed)
    banned = holdout_names()
    listing = tasksource.list_tasks()
    listing = listing[listing.task_type == "Classification"]
    print(f"{len(listing)} Classification tasks in tasksource")

    OUT.mkdir(parents=True, exist_ok=True)
    built, skipped_holdout, failed, rejected = 0, [], [], 0
    manifest = []

    for row in listing.itertuples():
        if built >= args.max_tasks:
            break
        tid = str(row.id)
        if any(re.search(p, tid) for p in SKIP_PATTERNS):
            continue
        hit = is_excluded(row, banned)
        if hit:
            skipped_holdout.append((tid, hit))
            continue
        try:
            dd = tasksource.load_task(tid, max_rows=args.max_rows)
            task = to_task(tid, dd, args.max_rows, rng)
        except Exception as e:  # noqa: BLE001
            failed.append((tid, f"{type(e).__name__}: {str(e)[:60]}"))
            continue
        if task is None:
            rejected += 1
            continue
        task.save(OUT.parent / "mixture_tmp", split="train")
        (OUT.parent / "mixture_tmp" / task.name).rename(OUT / task.name)
        built += 1
        k = len(next(iter(task.questions.values())).labels)
        manifest.append({"name": task.name, "source": tid, "n": len(task), "K": k})
        print(f"[{built:3d}] {task.name[:44]:<46} n={len(task):<5} K={k}")

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "tasks": manifest,
                "n_tasks": built,
                "n_rows": sum(m["n"] for m in manifest),
                "excluded_as_heldout": [{"task": t, "matched": h} for t, h in skipped_holdout],
                "failed": [{"task": t, "error": e} for t, e in failed],
            },
            indent=1,
        )
    )
    print(f"\nbuilt {built} tasks, {sum(m['n'] for m in manifest)} rows")
    print(f"excluded as held-out: {len(skipped_holdout)}")
    for t, h in skipped_holdout[:12]:
        print(f"   {t:<44} matched {h!r}")
    print(f"rejected (shape/size): {rejected}   load failures: {len(failed)}")


if __name__ == "__main__":
    main()

"""Loading the held-out suite, and refusing to train on it by accident.

The suite lives in `tasks/heldout/`. Every task there carries a `holdout_of`
list naming the datasets that must not appear in a training mixture used to
score it, and `tasks/heldout/manifest.json` carries the union.

    from openjev.heldout import load_suite, assert_training_mixture_clean

    assert_training_mixture_clean(mixture_names)   # call BEFORE training
    suite = load_suite()                           # call after

Why the matcher is not a set intersection
-----------------------------------------
A mixture drawn from `tasksource` names Banking77 `banking77`, while the Hub
calls it `PolyAI/banking77` and `mteb/banking77`. A plain
`set(holdout) & set(mixture)` misses two of those three and reports clean.
So this module matches on the lowercased full name, on the name with its Hub
owner stripped (but only against hold-out entries that are themselves bare
names, so `oasst2_dense_flat/toxicity` is not confused with
`civil_comments/toxicity`), and on explicit `*` patterns.

`Task.assert_disjoint` is still called for every task, so the schema's own
guard is exercised rather than replaced.
"""

from __future__ import annotations

import fnmatch
import json
from functools import lru_cache
from pathlib import Path

from .schema import Noul, Question, Score, Task

SUITE_ROOT = Path(__file__).resolve().parents[1] / "tasks" / "heldout"
REBUILD = "uv run python scripts/build_heldout.py --fetch"


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def manifest(root: str | Path | None = None) -> dict:
    path = Path(root or SUITE_ROOT) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no suite manifest at {path}. Build it with:\n  {REBUILD}")
    return json.loads(path.read_text())


def task_names(root: str | Path | None = None) -> list[str]:
    return list(manifest(root)["tasks"])


def holdout_union(root: str | Path | None = None) -> list[str]:
    """Every dataset name that is off-limits to training, across all tasks."""
    return list(manifest(root)["holdout_of_union"])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _rows_path(root: Path, name: str, split: str) -> Path:
    return root / name / f"{split}.jsonl"


def missing_rows(root: str | Path | None = None, split: str = "test") -> list[str]:
    """Tasks whose rows are not on disk.

    Only ever non-empty for the tasks whose licence blocks redistribution, so
    their rows are generated locally rather than committed.
    """
    root = Path(root or SUITE_ROOT)
    return [n for n in task_names(root) if not _rows_path(root, n, split).exists()]


def load_suite(
    root: str | Path | None = None, split: str = "test", strict: bool = True
) -> dict[str, Task]:
    """Load every task in the suite, keyed by name.

    `strict` controls what happens when a task's rows are absent because its
    licence blocked us from vendoring them: raise (default), or skip. Skipping
    silently is how a suite quietly shrinks, so it is opt-in.
    """
    root = Path(root or SUITE_ROOT)
    absent = missing_rows(root, split)
    if absent and strict:
        raise FileNotFoundError(
            f"rows missing for {absent} (split {split!r}). These datasets are not "
            f"redistributable so their rows are generated, not committed. Run:\n  {REBUILD}"
        )
    out: dict[str, Task] = {}
    for name in task_names(root):
        if name in absent:
            continue
        out[name] = Task.load(root / name, split=split)
    return out


def gold_label(question: Question, value: object) -> str:
    """The option-menu key for a gold answer, for scoring against `metrics.py`.

    Dispatches on the *question type*, never on the value. `Noul` gold is a
    bool and `Score` gold is an int, and in Python `True == 1` and
    `False == 0`, so the obvious `{True: "true", False: "false"}.get(v, str(v))`
    silently rewrites score levels 0 and 1 as "false" and "true". That costs
    two of five levels on both ordinal tasks and looks like a weak model
    rather than a bug, which is exactly the kind of error this suite exists
    to not make.
    """
    if isinstance(question, Noul):
        return "true" if value else "false"
    if isinstance(question, Score):
        return str(int(value))
    return str(value)


def gold_labels(task: Task, qid: str | None = None) -> list[str]:
    """Every example's gold answer for one question, as menu keys."""
    qid = qid or next(iter(task.questions))
    q = task.questions[qid]
    return [gold_label(q, e.answers[qid]) for e in task.examples if qid in e.answers]


def load_task(name: str, root: str | Path | None = None, split: str = "test") -> Task:
    root = Path(root or SUITE_ROOT)
    if name not in task_names(root):
        raise KeyError(f"{name!r} is not in the suite: {task_names(root)}")
    if not _rows_path(root, name, split).exists():
        raise FileNotFoundError(f"{name}/{split}.jsonl not built. Run:\n  {REBUILD}")
    return Task.load(root / name, split=split)


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def _schema_only(root: Path, name: str) -> Task:
    """The task's schema without paying to read its rows.

    `Task.load` skips a split file that does not exist, so naming a split that
    is never written gives us `questions` and `holdout_of` alone.
    """
    return Task.load(root / name, split="__schema_only__")


def _norm(name: str) -> str:
    return name.strip().lower().replace("\\", "/")


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def find_leaks(
    training_task_names: list[str], root: str | Path | None = None
) -> dict[str, list[str]]:
    """Map each offending mixture entry to the hold-out names it matches.

    Returned rather than raised so a mixture can be filtered programmatically.
    """
    root = Path(root or SUITE_ROOT)
    union = holdout_union(root)
    exact = {_norm(n): n for n in union if "*" not in n}
    bare = {k: v for k, v in exact.items() if "/" not in k}
    globs = [n for n in union if "*" in n]

    leaks: dict[str, list[str]] = {}
    for entry in training_task_names:
        n = _norm(entry)
        hits = set()
        if n in exact:
            hits.add(exact[n])
        if _basename(n) in bare:
            hits.add(bare[_basename(n)])
        for g in globs:
            if fnmatch.fnmatch(n, _norm(g)):
                hits.add(g)
        if hits:
            leaks[entry] = sorted(hits)
    return leaks


def which_tasks(holdout_names: list[str], root: str | Path | None = None) -> list[str]:
    """Which suite tasks are invalidated by these hold-out names appearing."""
    tasks = manifest(root)["tasks"]
    names = set(holdout_names)
    return sorted(t for t, meta in tasks.items() if names & set(meta["holdout_of"]))


def assert_training_mixture_clean(
    training_task_names: list[str], root: str | Path | None = None
) -> None:
    """Raise if any held-out dataset appears in the training mixture.

    Call this before every training run whose numbers you intend to report on
    the held-out suite. `tasksource` contains Banking77, AG News, Rotten
    Tomatoes, all seven Civil Comments configs and all five HelpSteer configs,
    so the default state of an unchecked mixture is contaminated.
    """
    root = Path(root or SUITE_ROOT)
    leaks = find_leaks(training_task_names, root)
    if leaks:
        invalidated = which_tasks(sorted({h for hs in leaks.values() for h in hs}), root)
        detail = "\n".join(
            f"    {entry!r} matches held-out {hits}" for entry, hits in sorted(leaks.items())
        )
        raise ValueError(
            f"Training mixture contains {len(leaks)} held-out dataset(s):\n{detail}\n"
            f"  This invalidates: {invalidated}\n"
            f"  Remove them from the mixture, or do not report held-out numbers for "
            f"those tasks. Generalization measured this way is meaningless."
        )
    # Defence in depth: also run the schema's own guard, so a task whose
    # holdout_of drifts out of sync with the manifest still fails.
    for name in task_names(root):
        _schema_only(root, name).assert_disjoint(training_task_names)


__all__ = [
    "SUITE_ROOT",
    "manifest",
    "task_names",
    "holdout_union",
    "missing_rows",
    "load_suite",
    "load_task",
    "gold_label",
    "gold_labels",
    "find_leaks",
    "which_tasks",
    "assert_training_mixture_clean",
]

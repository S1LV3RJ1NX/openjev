"""Drop the examples and tasks that cannot teach anything, before training.

Training is the expensive step, so faults that survive a run and surface as a
bad number hours later are the ones worth removing up front. Two classes,
both found by `scripts/audit_data.py` across every mixture we built:

  contradictory golds  the same state carrying different answers. Whichever
                       the model predicts, some of them are wrong, so the
                       gradient is pure noise. `ethics_deontology` had 195.
  degenerate tasks     one gold for every example. A constant answer scores
                       perfectly, so there is nothing to learn.

    uv run python scripts/clean_mixture.py --in tasks/mixture_mc \\
        --out tasks/mixture_clean
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.schema import Task  # noqa: E402

SPLITS = ("train", "dev", "test")


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--min-rows", type=int, default=32)
    ap.add_argument("--max-menu-chars", type=int, default=4000,
                    help="drop examples whose option menu alone is longer than this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dirs = sorted(p for p in src.iterdir() if (p / "task.json").exists())
    if not dirs:
        raise SystemExit(f"no tasks under {src}")

    kept = dropped_tasks = 0
    dropped_rows = 0
    notes: list[str] = []

    for d in dirs:
        splits = {s: Task.load(d, s) for s in SPLITS if (d / f"{s}.jsonl").exists()}
        if not splits:
            continue
        drop_task = False

        for s, t in splits.items():
            qid = next(iter(t.questions), None)
            if qid is None:
                drop_task = True
                break

            # --- contradictory golds ---------------------------------------
            # Keyed on state *and* menu: with per-example criteria the same
            # state under different options legitimately has a different
            # answer, and treating those as contradictions would delete
            # correct data from every MultipleChoice task.
            def key(e):
                menu = tuple(sorted((e.criteria or {}).get(qid, {}))) if e.criteria else ()
                return (norm(e.state), menu)

            groups: dict[tuple, set] = collections.defaultdict(set)
            for e in t.examples:
                groups[key(e)].add(str(e.answers.get(qid)))
            bad = {k for k, v in groups.items() if len(v) > 1}
            if bad:
                before = len(t.examples)
                t.examples = [e for e in t.examples if key(e) not in bad]
                removed = before - len(t.examples)
                dropped_rows += removed
                notes.append(f"{d.name}/{s}: dropped {removed} contradictory rows")

            # --- menus too long to pack ------------------------------------
            # Some MultipleChoice options are themselves multi-paragraph
            # passages, so the menu alone exceeds the context budget and the
            # trainer raises mid-run. Subsampling the menu does not help when
            # two options are 2,000 characters each, and truncating an option
            # can cut away the very text that distinguishes it, so these
            # examples are dropped instead.
            def menu_chars(e) -> int:
                # criteria is a dict for choice/noul and a list for score.
                menu = (e.criteria or {}).get(qid) or getattr(t.questions[qid], "criteria", None)
                if isinstance(menu, dict):
                    return sum(len(str(v)) for v in menu.values())
                if isinstance(menu, list):
                    return sum(len(str(v)) for v in menu)
                return 0

            over = [e for e in t.examples if menu_chars(e) > args.max_menu_chars]
            if over:
                before = len(t.examples)
                keep_ids = {id(e) for e in over}
                t.examples = [e for e in t.examples if id(e) not in keep_ids]
                dropped_rows += before - len(t.examples)
                notes.append(f"{d.name}/{s}: dropped {before - len(t.examples)} rows "
                             f"whose menu exceeds {args.max_menu_chars} chars")

            if len(t.examples) < args.min_rows:
                drop_task = True
                notes.append(f"{d.name}: only {len(t.examples)} rows left, dropping task")
                break

            # --- degenerate label set --------------------------------------
            golds = {str(e.answers.get(qid)) for e in t.examples if qid in e.answers}
            if len(golds) < 2:
                drop_task = True
                notes.append(f"{d.name}: one gold for every example, dropping task")
                break

        if drop_task:
            dropped_tasks += 1
            continue

        if not args.dry_run:
            for s, t in splits.items():
                problems = t.validate()
                if problems:
                    notes.append(f"{d.name}/{s}: still invalid, dropping: {problems[0][:50]}")
                    dropped_tasks += 1
                    break
            else:
                for s, t in splits.items():
                    t.save(dst, split=s)
                kept += 1
        else:
            kept += 1

    for n in notes[:30]:
        print(f"  {n}")
    if len(notes) > 30:
        print(f"  ... and {len(notes) - 30} more")
    print(f"\nkept {kept} tasks, dropped {dropped_tasks}, removed {dropped_rows} rows")
    if args.dry_run:
        print("(dry run, nothing written)")
    else:
        print(f"written to {dst}")


if __name__ == "__main__":
    main()

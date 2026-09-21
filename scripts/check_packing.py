"""Sweep every example through the packer before training touches a GPU.

Two runs died on "packed sequence is N tokens, over max_len" after twenty
minutes and after ten thousand steps. Both faults were findable in advance:
the packer is pure tokenisation, so the whole mixture can be swept on CPU in
a few minutes.

    uv run python scripts/check_packing.py tasks/mixture_final --max-len 2048

Exercises the same TaskDataset the trainer uses, augmentation included, so a
failure here is a failure there.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from openjev.data import TaskDataset  # noqa: E402
from openjev.encode import Packer  # noqa: E402
from openjev.schema import Task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--distractor-prob", type=float, default=0.5)
    ap.add_argument("--max-options", type=int, default=120)
    ap.add_argument("--noul-prob", type=float, default=0.15)
    ap.add_argument("--sample", type=int, default=0,
                    help="examples per task; 0 sweeps everything")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    root = Path(args.root)
    dirs = ([root] if (root / "task.json").exists()
            else sorted(p for p in root.iterdir() if (p / "task.json").exists()))

    tasks = [Task.load(p, "train") for p in dirs if (p / "train.jsonl").exists()]
    tasks = [t for t in tasks if len(t)]

    # The distractor pool is what makes augmented menus large, so the check
    # is only meaningful with the same pool the trainer builds.
    pool, seen = [], set()
    for t in tasks:
        for q in t.questions.values():
            # Only choice/noul carry a dict; a score's criteria is a list of
            # levels and contributes no distractor labels.
            crit = getattr(q, "criteria", None)
            if not isinstance(crit, dict):
                continue
            for lab, desc in crit.items():
                k = str(lab).strip().lower()
                if k not in seen:
                    seen.add(k)
                    pool.append((str(lab), str(desc) if desc else str(lab)))

    print(f"{len(tasks)} tasks, {sum(len(t) for t in tasks)} rows, "
          f"distractor pool {len(pool)}")

    packer = Packer(tok, max_len=args.max_len, max_state_len=args.max_len // 2)
    rng = random.Random(0)
    failures: list[str] = []
    checked = 0
    t0 = time.time()

    for i, t in enumerate(tasks):
        ds = TaskDataset(
            t, packer, shuffle_options=True, seed=i,
            distractors=pool, distractor_prob=args.distractor_prob,
            max_options=args.max_options, noul_prob=args.noul_prob,
        )
        idxs = range(len(ds))
        if args.sample and len(ds) > args.sample:
            idxs = rng.sample(range(len(ds)), args.sample)
        for j in idxs:
            try:
                ds[j]
            except Exception as e:  # noqa: BLE001
                failures.append(f"{t.name}[{j}]: {type(e).__name__}: {str(e)[:90]}")
            checked += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tasks)} tasks, {checked} examples, "
                  f"{len(failures)} failures, {time.time()-t0:.0f}s")

    print(f"\nchecked {checked} examples in {time.time()-t0:.0f}s")
    if failures:
        print(f"\n{len(failures)} FAILURES (the trainer would die on these):")
        for f in failures[:25]:
            print(f"  {f}")
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more")
        sys.exit(1)
    print("every example packs. safe to train.")


if __name__ == "__main__":
    main()

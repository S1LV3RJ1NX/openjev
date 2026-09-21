"""Baseline: one model per question, against one packed model for all of them.

This is the obvious alternative to packing and the first thing a careful
engineer would try. Separate classifiers have strictly more capacity per
question, so packing's advantage ought to be cost rather than accuracy.
That is a claim, and until this script runs it is an unmeasured one.

    uv run python scripts/baseline_separate.py --task tasks/healthcare_router \\
        --out /tmp/sep --epochs 6

Splits a multi-question task into one single-question task per question,
trains each independently with the same backbone and schedule the packed
model used, then reports per-question accuracy and the total latency of
asking all of them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.schema import Example, Task  # noqa: E402


def split_task(task_dir: Path, out: Path) -> list[str]:
    """One single-question task per question, same rows, same splits."""
    qids: list[str] = []
    for split in ("train", "dev", "test"):
        if not (task_dir / f"{split}.jsonl").exists():
            continue
        t = Task.load(task_dir, split)
        for qid, q in t.questions.items():
            rows = [
                Example(state=e.state, answers={qid: e.answers[qid]},
                        criteria={qid: e.criteria[qid]} if e.criteria and qid in e.criteria else {},
                        meta=dict(e.meta))
                for e in t.examples if qid in e.answers
            ]
            if not rows:
                continue
            sub = Task(name=f"{t.name}__{qid}",
                       description=f"single-question split of {t.name}",
                       questions={qid: q}, examples=rows)
            problems = sub.validate()
            if problems:
                raise SystemExit(f"{qid}/{split} invalid: {problems[:2]}")
            sub.save(out, split=split)
            if qid not in qids:
                qids.append(qid)
    return qids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tasks/healthcare_router")
    ap.add_argument("--out", default="/tmp/separate")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--python", default="./.venv/bin/python")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    tasks_dir = out / "tasks"
    qids = split_task(Path(args.task), tasks_dir)
    print(f"split into {len(qids)} single-question tasks: {', '.join(qids)}\n")

    ckpt_root = out / "ckpt"
    if not args.skip_train:
        t0 = time.time()
        for i, qid in enumerate(qids, 1):
            name = f"{Path(args.task).name}__{qid}"
            cmd = [args.python, "-u", "scripts/train.py",
                   "--task", str(tasks_dir / name),
                   "--backbone", args.backbone,
                   "--epochs", str(args.epochs), "--bs", str(args.bs),
                   "--max-len", str(args.max_len),
                   "--out", str(ckpt_root)]
            if args.init_from:
                cmd += ["--init-from", args.init_from]
            print(f"[{i}/{len(qids)}] training {qid} ...", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-1500:]); print(r.stderr[-1500:])
                raise SystemExit(f"training failed for {qid}")
        print(f"\ntrained {len(qids)} models in {time.time()-t0:.0f}s total")

    # ---- accuracy and latency of the separate-model system ---------------
    import torch
    from openjev.infer import DecisionModel

    task = Task.load(Path(args.task), "test")
    states = [e.state for e in task.examples]
    per_q_acc, models = {}, {}

    for qid in qids:
        name = f"{Path(args.task).name}__{qid}"
        m = DecisionModel.from_pretrained(ckpt_root / name, max_len=args.max_len)
        models[qid] = m
        sub = Task.load(tasks_dir / name, "test")
        outs = m.answer_batch([e.state for e in sub.examples], sub.questions, batch_size=16)
        gold = [e.answers[qid] for e in sub.examples]
        pred = [o[qid].label for o in outs]
        # noul golds are bools; labels are the strings "true"/"false".
        norm = [str(g).lower() for g in gold]
        per_q_acc[qid] = sum(p == g for p, g in zip(pred, norm)) / len(gold)

    print("\nper-question accuracy, one model each:")
    for qid, a in per_q_acc.items():
        print(f"  {qid:<24} {a:.4f}")

    # Latency: every question needs its own forward pass.
    dev = next(iter(models.values()))
    device = next(dev.model.parameters()).device
    probe = states[:40]
    for s in probe[:8]:
        for qid, m in models.items():
            m.answer(s, {qid: Task.load(tasks_dir / f"{Path(args.task).name}__{qid}",
                                        "test").questions[qid]})
    qmap = {qid: Task.load(tasks_dir / f"{Path(args.task).name}__{qid}", "test").questions[qid]
            for qid in qids}
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for s in probe[8:]:
        t = time.perf_counter()
        for qid, m in models.items():
            m.answer(s, {qid: qmap[qid]})
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()

    print(f"\nlatency for all {len(qids)} questions, one forward pass each:")
    print(f"  p50 {statistics.median(times):7.1f} ms   p95 {times[int(len(times)*.95)]:7.1f} ms")
    print(f"\nCompare against the packed model, which answers all {len(qids)} in one pass.")
    print("Packing's claim is cost, not accuracy; this is the number that tests it.")

    (out / "results.json").write_text(json.dumps(
        {"per_question_accuracy": per_q_acc,
         "latency_ms_p50": statistics.median(times),
         "latency_ms_p95": times[int(len(times) * .95)],
         "n_models": len(qids)}, indent=1))
    print(f"\nwritten to {out/'results.json'}")


if __name__ == "__main__":
    main()

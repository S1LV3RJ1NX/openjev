"""Wall-clock latency per decision, encoder against decoder.

Accuracy decides whether a router is usable; latency decides whether it is
affordable. The decoder is 1.7B parameters against the encoder's 149M and we
had compared only accuracy, which is the wrong basis for choosing a default.

    uv run python scripts/bench_latency.py --ckpt checkpoints_v2/healthcare_router
    uv run python scripts/bench_latency.py --ckpt checkpoints_dec2/mixture_ord2 --decoder

Reports the median rather than the mean: one slow first batch should not be
allowed to describe the steady state, and a router's p50 and p95 are what a
capacity plan is built from.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev import Task  # noqa: E402
from openjev.infer import DecisionModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="tasks/healthcare_router")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100, help="states to time")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bs", type=int, default=1, help="1 is the interactive case")
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    model = DecisionModel.from_pretrained(args.ckpt, max_len=args.max_len)
    device = next(model.model.parameters()).device
    task = Task.load(args.task, args.split)
    states = [e.state for e in task.examples][: args.n + args.warmup]
    questions = task.questions
    n_q = len(questions)
    n_opt = sum(len(getattr(q, "criteria", []) or [1, 2]) or 2 for q in questions.values())

    def run(batch: list) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model.answer_batch(batch, questions, batch_size=len(batch))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    # Warm up: the first calls pay for kernel autotuning and allocator growth,
    # and reporting those as latency would describe a state no user is in.
    for i in range(0, args.warmup, args.bs):
        run(states[i : i + args.bs])

    timed = states[args.warmup :]
    per_call = [run(timed[i : i + args.bs]) for i in range(0, len(timed), args.bs)
                if timed[i : i + args.bs]]
    per_state = [t / args.bs for t in per_call]
    per_state.sort()

    def pct(p: float) -> float:
        return per_state[min(len(per_state) - 1, int(len(per_state) * p))]

    kind = "decoder" if model.meta.get("decoder") else "encoder"
    params = sum(p.numel() for p in model.model.parameters())
    print(f"\n{args.ckpt}")
    print(f"  {kind}, {model.meta.get('backbone')}, {params/1e6:.0f}M params, {device}")
    print(f"  {n_q} questions and {n_opt} options per state, batch size {args.bs}, "
          f"{len(per_state)} timed calls")
    print(f"\n  per state   p50 {statistics.median(per_state):8.1f} ms"
          f"   p95 {pct(0.95):8.1f} ms   min {per_state[0]:8.1f} ms")
    print(f"  per decision p50 {statistics.median(per_state)/n_q:8.2f} ms"
          f"   ({n_q} questions answered in one forward pass)")
    print(f"  throughput   {1000/statistics.median(per_state)*args.bs:8.1f} states/s")


if __name__ == "__main__":
    main()

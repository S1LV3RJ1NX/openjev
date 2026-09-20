"""Is the shared state prefix actually worth it?

Two ways to answer N questions about one state:

  packed  one sequence: [state][Q1 block][Q2 block]... with a block-diagonal
          mask. State encoded once. Sequence length grows with sum(questions).
  naive   N sequences, one per question, each carrying its own copy of the
          state, run as a batch. State encoded N times.

Naive is the obvious implementation and it is what you get for free from any
cross-encoder. Packed is only worth its complexity if it wins, and the win
should grow with state length, since that is the thing being duplicated.

Also measures whether packed latency is flat in question count, which is the
property the whole design is sold on.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev.encode import Packer  # noqa: E402
from openjev.model import OpenJev  # noqa: E402
from openjev.schema import Noul  # noqa: E402

PARAGRAPH = (
    "The member opened a plan in March 2023 and has since raised four support "
    "tickets, two about delayed delivery and two about unrecognised charges. "
    "Their card was declined last night. They are still waiting on a refund "
    "from a purchase cancelled on Tuesday. Balance is 412.80 with no overdraft. "
)


def questions(n: int) -> dict:
    return {
        f"q{i}": Noul(instructions=f"Does this message concern signal number {i}?")
        for i in range(n)
    }


def timed(fn, warmup: int = 3, iters: int = 10) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    pk = Packer(tok, max_len=8192, max_state_len=7000)
    model = OpenJev(backbone=args.backbone, vocab_size=len(tok)).to(dev, dtype).eval()

    print(f"{args.backbone}  {args.dtype}  {torch.cuda.get_device_name(0)}\n")

    for reps, label in [(1, "~90 tok"), (8, "~700 tok"), (32, "~2.8k tok")]:
        state = PARAGRAPH * reps
        print(f"=== state {label} ===")
        print(f"{'n_q':>4} {'packed_tok':>11} {'naive_tok':>10} {'packed_ms':>10} "
              f"{'naive_ms':>9} {'speedup':>8}")

        base = None
        for n in (1, 5, 10, 25, 50):
            qs = questions(n)

            p = pk.pack(state, qs)
            b = pk.collate([p])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            packed_tok = len(p)

            # naive: one sequence per question, batched together
            singles = [pk.pack(state, {k: v}) for k, v in qs.items()]
            nb = pk.collate(singles)
            nb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in nb.items()}
            naive_tok = sum(len(s) for s in singles)

            with torch.no_grad():
                pm, _ = timed(lambda: model(b))
                nm, _ = timed(lambda: model(nb))

            if base is None:
                base = pm
            print(f"{n:>4} {packed_tok:>11} {naive_tok:>10} {pm:>10.1f} "
                  f"{nm:>9.1f} {nm / pm:>7.2f}x")

        print(f"     packed latency at 50q / at 1q = {pm / base:.2f}x "
              f"(flat would be 1.00x)\n")


if __name__ == "__main__":
    main()

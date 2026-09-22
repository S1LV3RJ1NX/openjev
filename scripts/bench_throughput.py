"""Sustained throughput under concurrent load, which batch-1 latency does not tell you.

Every latency number elsewhere in this project is batch size 1, the
interactive case. That measurement says nothing about how many requests per
second a single GPU serves, and it is misleading to divide one by the other:
batch 1 leaves an H100 almost entirely idle, so the reciprocal of latency
badly understates what the hardware will do.

This matters for the comparison we care about. The reference API scales to
roughly 47 requests per second at 32 in-flight requests, with its median flat
in the 400 to 500 ms range, and degrades past that. Comparing our 22 ms
single-request median against that number is not a like-for-like comparison,
because theirs includes a round trip and ours has no concurrency in it at all.

Reports the sustained rate at a range of batch sizes, and the latency a
caller would see at each, so the throughput and latency trade is visible
rather than assumed.

    uv run python scripts/bench_throughput.py --ckpt <model.pt> --decoder
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev import Task  # noqa: E402
from openjev.ckpt import load_into, lora_rank, wants_head  # noqa: E402
from openjev.data import TaskDataset, collate  # noqa: E402
from openjev.encode import Packer  # noqa: E402
from openjev.infer import DEFAULT_DECODER_TEMPLATE  # noqa: E402
from openjev.model import OpenJev  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="tasks/healthcare_router")
    ap.add_argument("--split", default="test")
    ap.add_argument("--decoder", action="store_true")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batches", default="1,2,4,8,16,32",
                    help="batch sizes to sweep, as a proxy for in-flight requests")
    ap.add_argument("--states", type=int, default=256, help="states per measurement")
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    is_decoder = bool(ck.get("decoder")) or args.decoder
    backbone = args.backbone or ck.get("backbone", "answerdotai/ModernBERT-base")
    tok = AutoTokenizer.from_pretrained(backbone)
    packer = Packer(
        tok, max_len=args.max_len, max_state_len=args.max_len // 2,
        marker=":" if is_decoder else None, marker_after=is_decoder,
        option_template=(ck.get("option_template")
                         or (DEFAULT_DECODER_TEMPLATE if is_decoder else None)),
        preamble=ck.get("preamble"),
    )

    if is_decoder:
        from openjev.decoder import OpenJevDecoder
        model = OpenJevDecoder(backbone=backbone, tokenizer=tok,
                               learned_head=wants_head(ck), lora_r=lora_rank(ck)).to(device)
    else:
        model = OpenJev(backbone=backbone, vocab_size=len(tok)).to(device)
    load_into(model, ck)
    model.eval()

    task = Task.load(args.task, args.split)
    n_q = len(task.questions)
    print(f"{args.ckpt}  {'decoder' if is_decoder else 'encoder'}  "
          f"{n_q} questions per state  device={device}")
    print(f"\n{'batch':>6}{'states/s':>11}{'questions/s':>13}"
          f"{'p50 batch':>11}{'p95 batch':>11}{'p50/state':>11}")

    for bs in [int(x) for x in args.batches.split(",")]:
        ds = TaskDataset(task, packer, shuffle_options=False)
        dl = DataLoader(ds, batch_size=bs, shuffle=False,
                        collate_fn=lambda s: collate(s, packer=packer))
        times, done = [], 0
        with torch.no_grad():
            for i, batch in enumerate(dl):
                b = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    model(b)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if i >= args.warmup:
                    times.append(dt)
                    done += bs
                if done >= args.states:
                    break
        if not times:
            continue
        times.sort()
        total = sum(times)
        p50 = statistics.median(times)
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
        print(f"{bs:>6}{done / total:>11.1f}{done * n_q / total:>13.0f}"
              f"{p50 * 1000:>10.1f}ms{p95 * 1000:>10.1f}ms{p50 / bs * 1000:>10.1f}ms")

    print("\nBatch size stands in for in-flight requests: a server batching "
          "arrivals\nsees this trade. Larger batches raise throughput and raise "
          "the latency of\nany single request, which is the same trade the "
          "reference API makes when its\nmedian rises past 32 concurrent.")
    print("Single GPU, one process, no server overhead, so read these as an "
          "upper bound\non what a deployment achieves rather than as a "
          "deployment measurement.")


if __name__ == "__main__":
    main()

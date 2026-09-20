"""Score a trained checkpoint on the held-out suite, with a held-in control.

The control is the point. "Held-out accuracy is at chance" means two very
different things depending on whether the same checkpoint still scores well on
a task it *was* trained on:

  held-in high, held-out chance  -> genuine failure to transfer
  held-in also chance            -> a bug in loading, packing or scoring

Reporting the first without checking the second is how a broken eval gets
written up as a research finding.

    uv run python scripts/eval_heldout.py --ckpt checkpoints/mixture/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev import Task  # noqa: E402
from openjev.data import TaskDataset, collate  # noqa: E402
from openjev.encode import Packer  # noqa: E402
from openjev.heldout import load_suite  # noqa: E402
from openjev.metrics import accuracy, bootstrap_ci, macro_f1  # noqa: E402
from openjev.model import OpenJev  # noqa: E402


@torch.no_grad()
def score(model, task, packer, device, bs, limit=None):
    ex = task.examples[:limit] if limit else task.examples
    sub = Task(name=task.name, questions=task.questions, examples=ex)
    ds = TaskDataset(sub, packer, shuffle_options=False)

    def c(s):
        b = collate(s, packer)
        b["_samples"] = s
        return b

    gold, pred = [], []
    for batch in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=c):
        samples = batch.pop("_samples")
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lp = model(b).log_probs.float().cpu()
        cur = 0
        for s in samples:
            for qid, k in zip(s.packed.question_ids, s.packed.n_options):
                p = lp[cur : cur + k]
                cur += k
                if qid not in s.targets:
                    continue
                labels = s.packed.labels[qid]  # type: ignore[attr-defined]
                gold.append(labels[s.targets[qid]])
                pred.append(labels[int(p.argmax())])
    return gold, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mixture", default="tasks/mixture",
                    help="for the held-in control")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone", "answerdotai/ModernBERT-base")
    tok = AutoTokenizer.from_pretrained(backbone)
    packer = Packer(tok, max_len=args.max_len, max_state_len=args.max_len // 2)

    model = OpenJev(backbone=backbone, vocab_size=len(tok)).to(device)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    print(f"{args.ckpt}  backbone={backbone}  trained_on={ck.get('trained_on')}")
    if missing or unexpected:
        print(f"!! state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
        print(f"   first missing: {missing[:3]}")

    # ---- control: tasks this checkpoint WAS trained on --------------------
    mix = Path(args.mixture)
    if mix.exists():
        dirs = sorted(p for p in mix.iterdir() if (p / "task.json").exists())[:5]
        print(f"\n== HELD-IN CONTROL (trained on these) ==")
        print(f"{'task':<40}{'K':>5}{'n':>6}{'acc':>8}{'x chance':>10}")
        for p in dirs:
            t = Task.load(p, "train")
            if not len(t):
                continue
            K = len(next(iter(t.questions.values())).labels)
            g, pr = score(model, t, packer, device, args.bs, limit=args.limit or 200)
            a = accuracy(pr, g)
            print(f"{p.name[:38]:<40}{K:>5}{len(g):>6}{a:>8.4f}{a * K:>9.1f}x")

    # ---- the actual question ---------------------------------------------
    print(f"\n== HELD-OUT SUITE (schemas never trained on) ==")
    print(f"{'task':<26}{'K':>5}{'n':>6}{'acc':>8}{'95% CI':>16}{'x chance':>10}{'macroF1':>9}")
    mult = []
    for name, task in load_suite().items():
        qid = next(iter(task.questions))
        K = len(task.questions[qid].labels)
        try:
            g, pr = score(model, task, packer, device, args.bs, args.limit)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<26}{K:>5}  failed: {str(e)[:50]}")
            continue
        a, lo, hi = bootstrap_ci(accuracy, pr, g, n_boot=500)
        mult.append(a * K)
        print(f"{name:<26}{K:>5}{len(g):>6}{a:>8.4f}{f'[{lo:.3f},{hi:.3f}]':>16}"
              f"{a * K:>9.1f}x{macro_f1(pr, g):>9.4f}")
    if mult:
        print(f"\nmean multiple of chance: {sum(mult) / len(mult):.1f}x")


if __name__ == "__main__":
    main()

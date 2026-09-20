"""Zero-shot evaluation on the held-out suite, with no training at all.

Scores each option by asking the backbone's own masked-LM head what it would
predict at the option's marker: `logit(yes) - logit(no)`. No new parameters,
so this measures what the warm start gives us for free, which is the only
honest way to compare candidate backbones before committing to a data
pipeline.

    uv run python scripts/eval_zeroshot.py --backbone answerdotai/ModernBERT-base
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev.data import TaskDataset, collate  # noqa: E402
from openjev.encode import Packer  # noqa: E402
from openjev.heldout import load_suite  # noqa: E402
from openjev.metrics import accuracy, bootstrap_ci, brier, ece, macro_f1  # noqa: E402
from openjev.model import OpenJev  # noqa: E402


@torch.no_grad()
def run_task(model, task, packer, device, bs: int, limit: int | None):
    if limit:
        task.examples = task.examples[:limit]
    ds = TaskDataset(task, packer, shuffle_options=False)

    def c(samples):
        b = collate(samples, packer)
        b["_samples"] = samples
        return b

    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=c)
    probs, gold, pred = [], [], []
    for batch in dl:
        samples = batch.pop("_samples")
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lp = model(b).log_probs.float().cpu()
        cur = 0
        for s in samples:
            for qid, k in zip(s.packed.question_ids, s.packed.n_options):
                p = lp[cur : cur + k].exp()
                cur += k
                if qid not in s.targets:
                    continue
                labels = s.packed.labels[qid]  # type: ignore[attr-defined]
                probs.append(dict(zip(labels, p.tolist())))
                gold.append(labels[s.targets[qid]])
                pred.append(labels[int(p.argmax())])
    return probs, gold, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="rows per task, for smoke runs")
    ap.add_argument("--decoder", action="store_true", help="use a causal LM backbone")
    ap.add_argument("--preamble", default=None, help="instruction prepended to every state")
    ap.add_argument("--template", default=None, help="override the per-option rendering")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.backbone)
    marker = tok.mask_token if not args.decoder else ":"
    template = args.template or (
        "\nOption: {opt}\nIs this the correct answer to the question? answer"
        if args.decoder else None
    )
    packer = Packer(tok, max_len=args.max_len, max_state_len=args.max_len // 2,
                    marker=marker, marker_after=args.decoder,
                    option_template=template, preamble=args.preamble)
    if args.decoder:
        from openjev.decoder import OpenJevDecoder

        model = OpenJevDecoder(backbone=args.backbone, tokenizer=tok).to(device).eval()
        kind = "decoder/yes-no"
    else:
        model = OpenJev(backbone=args.backbone, scorer="mlm", tokenizer=tok).to(device).eval()
        kind = "encoder/mlm"

    print(f"{args.backbone}  {kind}  marker={packer.marker!r}  device={device}")
    print(f"yes/no token ids: {model.yes_id} / {model.no_id}\n")
    print(f"{'task':<26}{'prim':>6}{'K':>5}{'n':>6}{'acc':>8}{'1/K':>7}"
          f"{'x chance':>9}{'macroF1':>9}{'ECE':>7}{'Brier':>8}")

    suite = load_suite()
    rows = []
    for name, task in suite.items():
        qid = next(iter(task.questions))
        K = len(task.questions[qid].labels)
        t0 = time.time()
        try:
            probs, gold, pred = run_task(model, task, packer, device, args.bs, args.limit)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<26}{'':>6}{K:>5}  FAILED: {str(e)[:60]}")
            continue
        acc, lo, hi = bootstrap_ci(accuracy, pred, gold, n_boot=500)
        chance = 1 / K
        rows.append((name, acc, chance))
        print(
            f"{name:<26}{task.questions[qid].type:>6}{K:>5}{len(gold):>6}"
            f"{acc:>8.4f}{chance:>7.3f}{acc / chance:>8.1f}x"
            f"{macro_f1(pred, gold):>9.4f}"
            f"{ece([max(p.values()) for p in probs], [int(a == b) for a, b in zip(pred, gold)]):>7.3f}"
            f"{brier(probs, gold):>8.4f}   {time.time() - t0:.0f}s"
        )

    if rows:
        print(f"\nmean accuracy {sum(r[1] for r in rows) / len(rows):.4f}   "
              f"mean multiple of chance {sum(r[1] / r[2] for r in rows) / len(rows):.1f}x")
    print("\nNo training of any kind. This is what the warm start gives for free.")


if __name__ == "__main__":
    main()

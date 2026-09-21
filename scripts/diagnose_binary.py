"""Does a chance-level binary task have no signal, or the wrong threshold?

Accuracy at 0.500 with macro-F1 at 0.333 means one class is never predicted.
That has two very different causes and the fix differs completely:

  AUROC ~= 0.5   the scores carry no information. A data problem.
  AUROC >> 0.5   the ranking is fine and the decision boundary is misplaced.
                 A calibration problem, fixable without more data.

Reporting "at chance" without separating these is how a threshold bug gets
written up as a failure to generalize.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev import Task  # noqa: E402
from openjev.infer import DecisionModel  # noqa: E402


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC, ties averaged."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return (sum(r for r, l in zip(ranks, labels) if l) - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="tasks/heldout/civil_comments_toxicity")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--strip-descriptions", action="store_true",
                    help="score the bare question, to test whether the option "
                         "rubric is fighting the labels")
    args = ap.parse_args()

    model = DecisionModel.from_pretrained(args.ckpt)
    task = Task.load(args.task, args.split)
    examples = task.examples[: args.limit]
    qid = next(iter(task.questions))

    questions = task.questions
    if args.strip_descriptions:
        from openjev.schema import Noul

        q = questions[qid]
        questions = {qid: Noul(instructions=q.instructions)}

    outs = model.answer_batch([e.state for e in examples], questions,
                              batch_size=args.bs)

    scores, labels = [], []
    for e, o in zip(examples, outs):
        gold = e.answers.get(qid)
        if gold is None:
            continue
        a = o[qid]
        p = a.probabilities.get("true")
        if p is None:  # a choice menu: score the second label
            keys = list(a.probabilities)
            p = a.probabilities[keys[-1]]
            gold = str(gold) == keys[-1]
        scores.append(float(p))
        labels.append(int(bool(gold)))

    n, pos = len(labels), sum(labels)
    at_half = sum((s >= 0.5) == bool(l) for s, l in zip(scores, labels)) / n
    au = auroc(scores, labels)

    # Best achievable accuracy over all thresholds: the headroom a correct
    # decision boundary would unlock, with no retraining at all.
    cand = sorted(set(scores))
    best_t, best_a = 0.5, at_half
    for t in cand:
        a = sum((s >= t) == bool(l) for s, l in zip(scores, labels)) / n
        if a > best_a:
            best_t, best_a = t, a

    print(f"\n{args.task}  n={n}, {pos} positive ({pos/n:.1%})")
    print(f"  accuracy at 0.50   {at_half:.4f}")
    print(f"  predicted positive {sum(s >= 0.5 for s in scores)/n:.1%} of the time")
    print(f"  score range        {min(scores):.4f} to {max(scores):.4f}"
          f"   (mean {sum(scores)/n:.4f})")
    print(f"  AUROC              {au:.4f}")
    print(f"  best threshold     {best_t:.4f} -> accuracy {best_a:.4f}")
    print()
    if au != au:  # nan
        print("  single-class set, AUROC undefined.")
    elif au < 0.55:
        print("  VERDICT: no signal. The scores do not rank the classes, so a")
        print("  threshold cannot rescue this. It needs data, not calibration.")
    else:
        print(f"  VERDICT: signal present (AUROC {au:.3f}) and the boundary is")
        print(f"  misplaced. Accuracy goes {at_half:.3f} -> {best_a:.3f} on a")
        print( "  threshold alone. Report this as calibration, not as a failure")
        print( "  to transfer.")


if __name__ == "__main__":
    main()

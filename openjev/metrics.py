"""The metric suite, written once so every comparison uses the same code.

Design notes, all of them earned from measuring Jev:

* **Bootstrap confidence intervals on everything.** Our tiers are small and
  the differences we care about are a few points. A number without an interval
  invites reading noise as signal.
* **Brier over log loss.** Jev quantizes probabilities to 0.01, so the gold
  label gets a literal 0.00 on a few percent of items and log loss becomes
  arbitrary. Brier degrades gracefully; log loss does not.
* **Expected cost per decision is the ranking metric.** Accuracy, ECE and
  latency are gates. This is the one number that absorbs accuracy, sharpness,
  calibration and class imbalance together, weighted the way the business
  weights them.
* **Per-class thresholds, not one global one.** A misrouted adverse-event
  report and a misrouted store-hours question are not the same error.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


# --------------------------------------------------------------------------
# single-label
# --------------------------------------------------------------------------


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else 0.0


def macro_f1(pred: list[str], gold: list[str], labels: list[str] | None = None) -> float:
    labels = labels or sorted(set(gold) | set(pred))
    total = 0.0
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in zip(pred, gold))
        fp = sum(p == lab and g != lab for p, g in zip(pred, gold))
        fn = sum(p != lab and g == lab for p, g in zip(pred, gold))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / len(labels) if labels else 0.0


def ece(scores: list[float], correct: list[int], bins: int = 15) -> float:
    """Expected calibration error, equal-width bins."""
    if not scores:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if lo < s <= hi or (b == 0 and s <= lo)]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        total += len(idx) / len(scores) * abs(conf - acc)
    return total


def brier(probs: list[dict[str, float]], gold: list[str]) -> float:
    """Multiclass Brier: mean over examples of sum_k (p_k - y_k)^2."""
    if not gold:
        return 0.0
    total = 0.0
    for p, g in zip(probs, gold):
        total += sum((v - (1.0 if k == g else 0.0)) ** 2 for k, v in p.items())
    return total / len(gold)


# --------------------------------------------------------------------------
# multi-label (the compound-utterance case)
# --------------------------------------------------------------------------


@dataclass
class MultiLabelScores:
    precision: float
    recall: float
    f1: float
    exact_set: float
    hamming: float


def multilabel(
    pred: list[set[str]], gold: list[set[str]]
) -> MultiLabelScores:
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    fp = sum(len(p - g) for p, g in zip(pred, gold))
    fn = sum(len(g - p) for p, g in zip(pred, gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    exact = sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else 0.0
    ham = sum(len(p ^ g) for p, g in zip(pred, gold)) / len(gold) if gold else 0.0
    return MultiLabelScores(prec, rec, f1, exact, ham)


# --------------------------------------------------------------------------
# selective prediction and cost
# --------------------------------------------------------------------------


def risk_coverage(scores: list[float], correct: list[int], steps: int = 21) -> list[tuple]:
    """(coverage, accuracy-on-covered) as the threshold sweeps.

    Sorting by score and sweeping coverage avoids the trap where quantized
    probabilities make certain coverage levels unreachable by thresholding.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    out = []
    for s in range(1, steps + 1):
        n = max(1, round(len(order) * s / steps))
        idx = order[:n]
        out.append((n / len(order), sum(correct[i] for i in idx) / n))
    return out


def expected_cost(
    probs: list[dict[str, float]],
    gold: list[str],
    cost_wrong: dict[str, float],
    cost_escalate: float,
    default_wrong: float = 1.0,
) -> tuple[float, float]:
    """Bayes-optimal per-item escalation, and the resulting mean cost.

    For each item, auto-handling the argmax has expected cost
    `sum_{k != argmax} p_k * cost_wrong[k]`. Escalate when that exceeds
    `cost_escalate`. This yields per-class thresholds for free rather than one
    global cutoff, which is where most of the money is on imbalanced traffic.

    Returns (mean cost per decision, coverage).
    """
    total, auto = 0.0, 0
    for p, g in zip(probs, gold):
        j = max(p, key=p.get)
        risk = sum(v * cost_wrong.get(k, default_wrong) for k, v in p.items() if k != j)
        if risk > cost_escalate:
            total += cost_escalate
        else:
            auto += 1
            if j != g:
                total += cost_wrong.get(g, default_wrong)
    n = len(gold) or 1
    return total / n, auto / n


def recall_for(pred: list[str], gold: list[str], label: str) -> float:
    tp = sum(p == label and g == label for p, g in zip(pred, gold))
    fn = sum(p != label and g == label for p, g in zip(pred, gold))
    return tp / (tp + fn) if tp + fn else float("nan")


# --------------------------------------------------------------------------
# uncertainty on the estimates themselves
# --------------------------------------------------------------------------


def bootstrap_ci(
    fn, *arrays, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) by paired bootstrap over examples."""
    rng = random.Random(seed)
    n = len(arrays[0])
    point = fn(*arrays)
    if n == 0:
        return point, float("nan"), float("nan")
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(fn(*[[a[i] for i in idx] for a in arrays]))
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def mcnemar(a_correct: list[int], b_correct: list[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar, for paired model comparisons on the same items."""
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)

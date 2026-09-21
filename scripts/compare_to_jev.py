"""Paired comparison of an OpenJev run against a Jev run on the same items.

Aggregate deltas do not say whether two systems actually differ: two models
can score the same and disagree on a third of the set. This pairs them item
by item and runs exact McNemar on the disagreements.

    uv run python scripts/eval_router.py --ckpt <ckpt> --dump ours.json
    uv run python scripts/compare_to_jev.py --ours ours.json --jev jev.json

The Jev dump is keyed by state text, with `A_pred` and a `nouls` map of
probabilities. Ours is keyed the same way, with `intent_pred` and thresholded
`nouls`. Both are produced against `tasks/healthcare_router` test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev import Task  # noqa: E402
from openjev.metrics import mcnemar  # noqa: E402

GATES = ("G_clinical", "G_abusive", "G_injection", "G_pharmacy")


def verdict(b10: int, b01: int, p: float) -> str:
    if p >= 0.05:
        return "level"
    return "OpenJev" if b10 > b01 else "Jev"


def report(name: str, ours: list[int], theirs: list[int]) -> None:
    # b01 is items Jev got and we missed; b10 is the reverse.
    b01, b10, p = mcnemar(ours, theirs)
    a_acc = sum(ours) / len(ours)
    b_acc = sum(theirs) / len(theirs)
    print(f"{name:<24}{a_acc:>8.3f}{b_acc:>8.3f}{a_acc - b_acc:>+9.3f}"
          f"{b10:>6}{b01:>6}{p:>11.2e}  {verdict(b10, b01, p)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--jev", required=True)
    ap.add_argument("--task", default="tasks/healthcare_router")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    ours = json.loads(Path(args.ours).read_text())
    jev = json.loads(Path(args.jev).read_text())
    task = Task.load(args.task, args.split)

    shared = [e for e in task.examples if e.state in ours and e.state in jev]
    if len(shared) < len(task.examples):
        print(f"!! pairing on {len(shared)} of {len(task.examples)} items; "
              f"the rest are missing from one dump")
    if not shared:
        raise SystemExit("no overlap between the two dumps")

    print(f"paired on {len(shared)} items\n")
    print(f"{'':<24}{'ours':>8}{'Jev':>8}{'delta':>9}"
          f"{'b10':>6}{'b01':>6}{'McNemar p':>11}  winner")

    # ---- intent, lenient: correct if the prediction is in the acceptable set
    o_int, j_int = [], []
    for e in shared:
        gold = e.answers.get("A_intent")
        ok = set(e.meta.get("acceptable") or ([gold] if gold is not None else []))
        mine, theirs = ours[e.state].get("intent_pred"), jev[e.state].get("A_pred")
        # Items with no acceptable set, or that one side never answered, are
        # not paired observations and must not be scored as agreement.
        if not ok or mine is None or theirs is None:
            continue
        o_int.append(int(mine in ok))
        j_int.append(int(theirs in ok))
    report("intent (lenient)", o_int, j_int)

    # ---- multi-label exact set over the five intent nouls
    labels = [q for q in task.questions if q.startswith("C_")]
    o_ml, j_ml = [], []
    for e in shared:
        gold = {c: bool(e.answers[c]) for c in labels if c in e.answers}
        if not gold:
            continue
        o_ml.append(int(all(bool(ours[e.state]["nouls"].get(c)) == v
                            for c, v in gold.items())))
        j_ml.append(int(all((jev[e.state]["nouls"].get(c, 0.0) >= args.thresh) == v
                            for c, v in gold.items())))
    report("multi-label exact set", o_ml, j_ml)

    # ---- each safety gate, scored as plain correctness
    for g in GATES:
        o_g, j_g = [], []
        for e in shared:
            if g not in e.answers:
                continue
            v = bool(e.answers[g])
            o_g.append(int(bool(ours[e.state]["nouls"].get(g)) == v))
            j_g.append(int((jev[e.state]["nouls"].get(g, 0.0) >= args.thresh) == v))
        if o_g:
            report(g, o_g, j_g)

    # ---- oblique clinical recall, the tier Jev was weakest on
    obl = [e for e in shared if e.meta.get("tier") == "clinical_oblique"
           and bool(e.answers.get("G_clinical"))]
    if obl:
        o_o = [int(bool(ours[e.state]["nouls"].get("G_clinical"))) for e in obl]
        j_o = [int(jev[e.state]["nouls"].get("G_clinical", 0.0) >= args.thresh) for e in obl]
        print()
        report("clinical_oblique recall", o_o, j_o)
        print(f"\nn={len(obl)} on that tier. With zero losses the smallest "
              f"reachable p is {0.5 ** len(obl):.3g}, so read it accordingly.")


if __name__ == "__main__":
    main()

"""Score a checkpoint on the healthcare router, on the same slices as the reference.

Reports exactly what we measured Jev on, so the comparison is line-for-line:
intent choice scored leniently against the acceptable set, multi-label nouls
with per-tier breakdowns, and the four safety gates with recall *and*
false-positive rate — a gate that fires on ordinary traffic is worse than no
gate.

    uv run python scripts/eval_router.py --ckpt checkpoints/router/model.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev import Task  # noqa: E402
from openjev.infer import DEFAULT_DECODER_TEMPLATE  # noqa: E402
from openjev.data import TaskDataset, collate  # noqa: E402
from openjev.encode import Packer  # noqa: E402
from openjev.ckpt import load_into, lora_rank, wants_head  # noqa: E402
from openjev.metrics import bootstrap_ci, multilabel  # noqa: E402
from openjev.model import OpenJev  # noqa: E402

INTENT_NOULS = [
    "C_refill", "C_order_status", "C_drug_availability",
    "C_store_hours", "C_vaccine_appointment",
]
GATES = ["G_clinical", "G_abusive", "G_injection", "G_pharmacy"]

# Measured on jev-1.13.0, 20 Sep 2026, same 450 test items.
#
# The two intent figures use different denominators and are not
# interchangeable. 0.909 is over all 450 items, including the 112 whose gold
# is an acceptable *set* rather than one label. 0.941 is over the 338 with a
# single gold, where lenient and strict coincide. Our harness only asks the
# intent question when the example carries a gold for it, so it scores the
# 338 — compare against 0.941, not 0.909.
JEV = {
    "intent_gold338": 0.941, "intent_all450": 0.909,
    "ml_exact": 0.822, "ml_f1": 0.890,
    "compound_3_intent": 0.581, "compound_3_exact": 0.645,
    "ambiguous_exact": 0.250,
    "G_clinical_recall": 0.926, "G_clinical_fpr": 0.006,
    "clinical_oblique_recall": 0.793,
}


@torch.no_grad()
def predict(model, task, packer, device, bs):
    ds = TaskDataset(task, packer, shuffle_options=False)

    def c(s):
        b = collate(s, packer)
        b["_samples"] = s
        return b

    out = {}
    for batch in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=c):
        samples = batch.pop("_samples")
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lp = model(b).log_probs.float().cpu()
        cur = 0
        for s in samples:
            rec = {}
            for qid, k in zip(s.packed.question_ids, s.packed.n_options):
                p = lp[cur : cur + k].exp()
                cur += k
                labels = s.packed.labels[qid]  # type: ignore[attr-defined]
                rec[qid] = dict(zip(labels, p.tolist()))
            key = s.example.state if isinstance(s.example.state, str) else str(s.example.state)
            out[key] = rec
    return out


def delta(ours: float, theirs: float) -> str:
    d = ours - theirs
    return f"{d:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="tasks/healthcare_router")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dump", help="write per-item predictions for paired comparison")
    ap.add_argument("--decoder", action="store_true",
                    help="causal backbone; usually inferred from the checkpoint")
    ap.add_argument("--preamble", default=None,
                    help="overrides the preamble stored in the checkpoint")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone", "answerdotai/ModernBERT-base")
    tok = AutoTokenizer.from_pretrained(backbone)
    # A decoder checkpoint needs the causal packing it was trained with: the
    # marker sits after the option text and the task preamble is part of the
    # format. Reading those off the checkpoint keeps eval and training in step.
    is_decoder = bool(ck.get("decoder")) or args.decoder
    # The packer formats this with `opt` alone, so a `{q}` placeholder raises
    # KeyError on every call. Keep it identical to what training used.
    template = ck.get("option_template") or (
        DEFAULT_DECODER_TEMPLATE if is_decoder else None
    )
    packer = Packer(
        tok, max_len=args.max_len, max_state_len=args.max_len // 2,
        marker=":" if is_decoder else None, marker_after=is_decoder,
        option_template=template,
        preamble=args.preamble if args.preamble is not None else ck.get("preamble"),
    )

    if is_decoder:
        from openjev.decoder import OpenJevDecoder

        model = OpenJevDecoder(
            backbone=backbone, tokenizer=tok, learned_head=wants_head(ck), lora_r=lora_rank(ck),
        ).to(device)
    else:
        model = OpenJev(backbone=backbone, vocab_size=len(tok)).to(device)
    load_into(model, ck)
    model.eval()

    task = Task.load(args.task, args.split)
    print(f"{args.ckpt}  {len(task)} items  device={device}")
    preds = predict(model, task, packer, device, args.bs)
    key = lambda e: e.state if isinstance(e.state, str) else str(e.state)  # noqa: E731

    if args.dump:
        import json

        out = {}
        for e in task.examples:
            rec = preds[key(e)]
            row = {"tier": e.tier}
            if "A_intent" in rec:
                p = rec["A_intent"]
                row["intent_pred"] = max(p, key=p.get)
                row["intent_ok"] = int(row["intent_pred"] in (e.meta.get("acceptable") or []))
            row["nouls"] = {
                q: int(v.get("true", 0.0) >= args.thresh)
                for q, v in rec.items()
                if set(v) == {"false", "true"}
            }
            out[key(e)] = row
        with open(args.dump, "w") as f:
            json.dump(out, f)
        print(f"per-item predictions -> {args.dump}")

    by_tier = defaultdict(list)
    for e in task.examples:
        by_tier[e.tier].append(e)

    # ---------------- intent ----------------
    print("\n== INTENT CHOICE (lenient: prediction is in the acceptable set)")
    print(f"{'tier':<22}{'n':>5}{'ours':>8}{'95% CI':>16}{'Jev':>8}{'delta':>8}")
    all_len, all_str = [], []
    tier_scores = {}
    for tier, exs in sorted(by_tier.items(), key=lambda kv: -len(kv[1])):
        lenient, strict = [], []
        for e in exs:
            p = preds[key(e)].get("A_intent")
            if not p:
                continue
            pred = max(p, key=p.get)
            lenient.append(int(pred in (e.meta.get("acceptable") or [])))
            if "A_intent" in e.answers:
                strict.append(int(pred == e.answers["A_intent"]))
        if not lenient:
            continue
        all_len += lenient
        all_str += strict
        a = sum(lenient) / len(lenient)
        tier_scores[tier] = a
        lo, hi = bootstrap_ci(lambda v: sum(v) / len(v), lenient, n_boot=800)[1:]
        ref = JEV.get(f"{tier}_intent")
        print(f"{tier:<22}{len(lenient):>5}{a:>8.3f}{f'[{lo:.2f},{hi:.2f}]':>16}"
              f"{(f'{ref:.3f}' if ref else '-'):>8}{(delta(a, ref) if ref else '-'):>8}")
    ov, lo, hi = bootstrap_ci(lambda v: sum(v) / len(v), all_len, n_boot=2000)
    print(f"\n  OVERALL lenient {ov:.3f} [{lo:.3f},{hi:.3f}]  vs Jev "
          f"{JEV['intent_gold338']:.3f}  ({delta(ov, JEV['intent_gold338'])})"
          f"   on the {len(all_len)} items carrying a gold intent")
    if all_str:
        s = sum(all_str) / len(all_str)
        print(f"  OVERALL strict  {s:.3f}  vs Jev {JEV['intent_gold338']:.3f} "
              f"({delta(s, JEV['intent_gold338'])})")
    print(f"  Jev scores {JEV['intent_all450']:.3f} over all 450 including the "
          f"ambiguous tier, which this harness does not ask. Not comparable to "
          f"the line above.")

    # ---------------- multi-label nouls ----------------
    print("\n== MULTI-LABEL INTENT NOULS @ %.2f" % args.thresh)
    print(f"{'tier':<22}{'n':>5}{'exact':>8}{'prec':>7}{'rec':>7}{'F1':>7}{'Jev exact':>11}")
    gp, gg = [], []
    for tier, exs in sorted(by_tier.items(), key=lambda kv: -len(kv[1])):
        pr, go = [], []
        for e in exs:
            if not any(q in e.answers for q in INTENT_NOULS):
                continue
            rec = preds[key(e)]
            pr.append({q for q in INTENT_NOULS
                       if rec.get(q, {}).get("true", 0.0) >= args.thresh})
            go.append({q for q in INTENT_NOULS if e.answers.get(q)})
        if not go:
            continue
        gp += pr
        gg += go
        m = multilabel(pr, go)
        ref = JEV.get(f"{tier}_exact")
        print(f"{tier:<22}{len(go):>5}{m.exact_set:>8.3f}{m.precision:>7.3f}"
              f"{m.recall:>7.3f}{m.f1:>7.3f}{(f'{ref:.3f}' if ref else '-'):>11}")
    m = multilabel(gp, gg)
    print(f"\n  OVERALL exact {m.exact_set:.3f} vs Jev {JEV['ml_exact']:.3f} "
          f"({delta(m.exact_set, JEV['ml_exact'])})   "
          f"F1 {m.f1:.3f} vs {JEV['ml_f1']:.3f} ({delta(m.f1, JEV['ml_f1'])})")

    # ---------------- safety gates ----------------
    print("\n== SAFETY GATES")
    for g in GATES:
        pos = [e for e in task.examples if e.answers.get(g) is True]
        neg = [e for e in task.examples if e.answers.get(g) is False]
        if not pos:
            continue
        fire = lambda e: preds[key(e)].get(g, {}).get("true", 0.0) >= args.thresh  # noqa: E731
        rec = sum(map(fire, pos)) / len(pos)
        fpr = sum(map(fire, neg)) / len(neg) if neg else 0.0
        jr, jf = JEV.get(f"{g}_recall"), JEV.get(f"{g}_fpr")
        extra = f"   Jev recall {jr:.3f} ({delta(rec, jr)})  fpr {jf:.3f}" if jr else ""
        print(f"\n  {g}: recall {rec:.3f} (n={len(pos)})  FPR {fpr:.3f} (n={len(neg)}){extra}")
        per = defaultdict(list)
        for e in pos:
            per[e.tier].append(fire(e))
        for tier, v in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            r = sum(v) / len(v)
            ref = JEV.get(f"{tier}_recall") if g == "G_clinical" else None
            print(f"      {tier:<24}{sum(v)}/{len(v)} = {r:.3f}"
                  + (f"   Jev {ref:.3f} ({delta(r, ref)})" if ref else ""))


if __name__ == "__main__":
    main()

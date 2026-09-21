"""Fine-tune OpenJev on one task, then evaluate and calibrate it.

    uv run python scripts/train.py --task tasks/banking77_specialist

Deliberately small and readable rather than configurable: the point of the
project is that someone can read the training loop, not that it has every
knob. Temperature scaling is fitted on dev afterwards, per (question type,
option count), because a confidence threshold is not portable across menus of
different sizes.

If the task being trained on is a member of the held-out suite, this refuses
to run unless `--allow-heldout` is passed, and records the override in the
checkpoint. Generalization claims are the easiest thing in this project to
break by accident.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openjev import Choice, Task  # noqa: E402
from openjev.data import TaskDataset, collate, option_labels  # noqa: E402
from openjev.encode import Packer  # noqa: E402
from openjev.metrics import accuracy, bootstrap_ci, brier, ece, macro_f1  # noqa: E402
from openjev.model import OpenJev  # noqa: E402


def heldout_check(task_name: str, allow: bool) -> None:
    try:
        from openjev.heldout import task_names
        names = task_names()
    except Exception:
        return
    if task_name in names or any(task_name.startswith(n) for n in names):
        if not allow:
            raise SystemExit(
                f"\n'{task_name}' is part of the held-out suite. Training on it means\n"
                f"any generalization number you later quote for it is meaningless.\n"
                f"If that is intended (e.g. an in-task specialist experiment), pass\n"
                f"--allow-heldout and say so wherever the number appears.\n"
            )
        print(f"!! training on held-out task '{task_name}' -- in-task numbers only\n")


@torch.no_grad()
def evaluate(model, loader, task, device, temps=None):
    model.eval()
    per_q = defaultdict(lambda: {"probs": [], "gold": [], "pred": []})
    for batch in loader:
        samples = batch.pop("_samples")
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(b).log_probs.float().cpu()
        cursor = 0
        for s in samples:
            for qid, k in zip(s.packed.question_ids, s.packed.n_options):
                lp = logits[cursor : cursor + k]
                cursor += k
                if qid not in s.targets:
                    continue
                if temps and qid in temps:
                    lp = torch.log_softmax(lp / temps[qid], dim=-1)
                p = lp.exp()
                labels = s.packed.labels[qid]  # type: ignore[attr-defined]
                per_q[qid]["probs"].append(dict(zip(labels, p.tolist())))
                per_q[qid]["gold"].append(labels[s.targets[qid]])
                per_q[qid]["pred"].append(labels[int(p.argmax())])
    model.train()
    return per_q


def report(per_q, title):
    print(f"\n{title}")
    print(f"{'question':<22}{'n':>6}{'acc':>8}{'95% CI':>16}{'macroF1':>9}{'ECE':>7}{'Brier':>8}")
    for qid, d in per_q.items():
        n = len(d["gold"])
        if not n:
            continue
        acc, lo, hi = bootstrap_ci(accuracy, d["pred"], d["gold"], n_boot=1000)
        top = [max(p.values()) for p in d["probs"]]
        corr = [int(a == b) for a, b in zip(d["pred"], d["gold"])]
        print(
            f"{qid:<22}{n:>6}{acc:>8.4f}{f'[{lo:.3f},{hi:.3f}]':>16}"
            f"{macro_f1(d['pred'], d['gold']):>9.4f}"
            f"{ece(top, corr):>7.3f}{brier(d['probs'], d['gold']):>8.4f}"
        )
    return per_q


def fit_temperature(per_q):
    """One temperature per question, on dev. Cannot reorder, so accuracy is safe."""
    temps = {}
    for qid, d in per_q.items():
        if not d["gold"]:
            continue
        labels = list(d["probs"][0])
        logp = torch.log(torch.tensor([[p[l] for l in labels] for p in d["probs"]]).clamp_min(1e-12))
        gold = torch.tensor([labels.index(g) for g in d["gold"]])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logp / log_t.exp(), gold)
            loss.backward()
            return loss

        opt.step(closure)
        temps[qid] = float(log_t.exp().detach())
    return temps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--mixture", help="directory of task dirs, trained jointly")
    ap.add_argument("--eval-heldout", action="store_true",
                    help="score the held-out suite after training")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--distractor-prob", type=float, default=0.0,
                    help="probability of padding a choice menu with borrowed labels")
    ap.add_argument("--max-options", type=int, default=128)
    ap.add_argument("--decoder", action="store_true",
                    help="causal LM backbone with yes/no readout")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the calibration head (use with --decoder)")
    ap.add_argument("--preamble", default=None)
    ap.add_argument("--init-from", default=None,
                    help="start from a checkpoint instead of the raw backbone")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    # Redirected to a file, Python block-buffers stdout and a long run looks
    # hung for minutes at a time. Line-buffer so `tail -f` is useful.
    sys.stdout.reconfigure(line_buffering=True)

    if not (args.task or args.mixture):
        raise SystemExit("pass --task or --mixture")
    name = Path(args.mixture or args.task).name
    if args.task:
        heldout_check(name, args.allow_heldout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.backbone)
    template = (
        "\nOption: {opt}\nIs this the correct answer to the question? answer"
        if args.decoder else None
    )
    packer = Packer(
        tok, max_len=args.max_len, max_state_len=args.max_len // 2,
        marker=":" if args.decoder else None,
        marker_after=args.decoder, option_template=template, preamble=args.preamble,
    )

    splits = {}
    if args.mixture:
        from torch.utils.data import ConcatDataset

        dirs = sorted(p for p in Path(args.mixture).iterdir() if (p / "task.json").exists())
        tasks = [Task.load(p, "train") for p in dirs]
        tasks = [t for t in tasks if len(t)]
        # Every task in a mixture must be disjoint from the held-out suite.
        # The builder already filtered, but a mixture can be assembled by
        # hand, so check again rather than trust it.
        from openjev.heldout import assert_training_mixture_clean
        assert_training_mixture_clean([t.description.replace("tasksource ", "") for t in tasks])
        ks = [len(next(iter(t.questions.values())).labels) for t in tasks]
        print(f"mixture {name}: {len(tasks)} tasks, {sum(len(t) for t in tasks)} rows")
        print(f"option counts: min {min(ks)} median {sorted(ks)[len(ks)//2]} max {max(ks)}")
        print(f"device: {device}")

        # Distractor pool for label-space augmentation, drawn from every
        # task's label set. Without it the model never sees a menu larger
        # than the biggest task in the mixture, and collapses onto one label
        # when a deployment hands it 77 or 151 options.
        pool: list[tuple[str, str]] = []
        for t in tasks:
            for q in t.questions.values():
                if isinstance(q, Choice):
                    for lab, desc in q.criteria.items():
                        pool.append((str(lab), str(desc) if desc else str(lab)))
        seen, uniq = set(), []
        for lab, desc in pool:
            if lab.strip().lower() not in seen:
                seen.add(lab.strip().lower())
                uniq.append((lab, desc))
        print(f"distractor pool: {len(uniq)} distinct labels  "
              f"(augmentation p={args.distractor_prob}, max options {args.max_options})")
        train_ds = ConcatDataset(
            [
                TaskDataset(
                    t, packer, shuffle_options=True, seed=i,
                    distractors=uniq, distractor_prob=args.distractor_prob,
                    max_options=args.max_options,
                )
                for i, t in enumerate(tasks)
            ]
        )
    else:
        for s in ("train", "dev", "test"):
            t = Task.load(args.task, s)
            if len(t):
                splits[s] = t
        print(f"task {name}: " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
        print(f"questions: {len(splits['train'].questions)}  device: {device}")
        train_ds = TaskDataset(splits["train"], packer, shuffle_options=True)
    fn = partial(collate, packer=packer)

    def wrap(ds, shuffle):
        def c(samples):
            b = fn(samples)
            b["_samples"] = samples
            return b
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle, collate_fn=c)

    train_dl = wrap(train_ds, True)
    eval_dls = {
        s: wrap(TaskDataset(splits[s], packer, shuffle_options=False), False)
        for s in splits
        if s != "train"
    }

    if args.decoder:
        from openjev.decoder import OpenJevDecoder

        model = OpenJevDecoder(
            backbone=args.backbone, tokenizer=tok, learned_head=True,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        if args.freeze_backbone:
            n = model.freeze_backbone()
            total = sum(p.numel() for p in model.parameters())
            print(f"frozen backbone: training {n / 1e6:.2f}M of {total / 1e6:.0f}M "
                  f"({100 * n / total:.3f}%)")
    else:
        model = OpenJev(backbone=args.backbone, vocab_size=len(tok)).to(device)

    if args.init_from:
        # The project's central claim is that a general checkpoint makes a
        # specialist cheap. That is only testable if a specialist can start
        # from one.
        prev = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(prev["state_dict"], strict=False)
        print(f"initialised from {args.init_from} "
              f"(trained on {prev.get('trained_on')}); "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

    head = [p for n, p in model.named_parameters()
            if n.startswith("scorer") and p.requires_grad]
    body = [p for n, p in model.named_parameters()
            if not n.startswith("scorer") and p.requires_grad]
    groups = [{"params": head, "lr": args.head_lr}]
    if body:
        groups.insert(0, {"params": body, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler_dtype = torch.bfloat16

    print(f"\ntraining: {steps} steps, {len(train_dl)} per epoch")
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        run = 0.0
        for batch in train_dl:
            batch.pop("_samples")
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device, dtype=scaler_dtype, enabled=device == "cuda"):
                out = model(b)
                loss = -out.log_probs[b["target_flat"]].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss)
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step {step}/{steps}  loss {run / 50:.4f}  "
                      f"{time.time() - t0:.0f}s")
                run = 0.0
        if "dev" in eval_dls:
            report(evaluate(model, eval_dls["dev"], splits["dev"], device),
                   f"-- dev after epoch {ep}")

    temps = {}
    if "dev" in eval_dls:
        temps = fit_temperature(evaluate(model, eval_dls["dev"], splits["dev"], device))
        print(f"\nfitted temperatures: { {k: round(v, 3) for k, v in temps.items()} }")

    if "test" in eval_dls:
        report(evaluate(model, eval_dls["test"], splits["test"], device), "== TEST (raw)")
        report(evaluate(model, eval_dls["test"], splits["test"], device, temps),
               "== TEST (temperature-scaled)")

    if args.eval_heldout:
        from openjev.heldout import load_suite

        print("\n== HELD-OUT SUITE (schemas never trained on)")
        print(f"{'task':<26}{'K':>5}{'n':>6}{'acc':>8}{'1/K':>7}{'x chance':>10}{'macroF1':>9}")
        accs = []
        for tname, task in load_suite().items():
            qid = next(iter(task.questions))
            K = len(task.questions[qid].labels)
            dl = wrap(TaskDataset(task, packer, shuffle_options=False), False)
            try:
                per_q = evaluate(model, dl, task, device)
            except Exception as e:  # noqa: BLE001
                print(f"{tname:<26}{K:>5}  failed: {str(e)[:48]}")
                continue
            d = per_q.get(qid) or next(iter(per_q.values()))
            a = accuracy(d["pred"], d["gold"])
            accs.append(a * K)
            print(f"{tname:<26}{K:>5}{len(d['gold']):>6}{a:>8.4f}{1 / K:>7.3f}"
                  f"{a * K:>9.1f}x{macro_f1(d['pred'], d['gold']):>9.4f}")
        if accs:
            print(f"\nmean multiple of chance: {sum(accs) / len(accs):.1f}x")

    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": args.backbone,
            "temperatures": temps,
            "trained_on": name,
            "heldout_override": bool(args.allow_heldout),
        },
        out_dir / "model.pt",
    )
    print(f"\nsaved to {out_dir/'model.pt'}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()

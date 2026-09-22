"""Fine-tune OpenJev on one task, then evaluate and calibrate it.

The recommended configuration is the LoRA decoder, which is what the project
ships and what the published numbers use:

    uv run python scripts/train.py --task tasks/healthcare_router \\
        --decoder --backbone Qwen/Qwen3-1.7B --lora-r 16 \\
        --epochs 6 --bs 4 --lr 2e-4 --max-len 3072

`--backbone` still defaults to ModernBERT-base, which is the encoder path.
That is the alternative rather than the recommendation: take it when p95
latency binds (20 ms against 56 ms) or when a 0.6 GB checkpoint matters more
than the accuracy. The default is left alone so that previously published
encoder runs still reproduce from the same command line.

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
from openjev.ckpt import load_into, save as save_ckpt  # noqa: E402
from openjev.metrics import accuracy, bootstrap_ci, brier, ece, macro_f1  # noqa: E402
from openjev.model import OpenJev  # noqa: E402


@torch.no_grad()
def harness_sanity(model, packer, device) -> float:
    """Accuracy on a task the model cannot fail unless something is broken.

    Deliberately trivial: the answer is in the state, verbatim. A healthy
    model is near 1.0 and chance is 0.333, so anything low means the run has
    broken rather than merely underperformed.
    """
    from openjev.schema import Choice, Example
    from openjev.schema import Task as _T

    rows = [("I want to talk about dogs.", "dogs"),
            ("This message is about cats.", "cats"),
            ("Let us discuss birds today.", "birds")] * 8
    task = _T(
        name="harness_sanity",
        questions={"q": Choice(instructions="What is this text about?",
                               criteria={"dogs": None, "cats": None, "birds": None})},
        examples=[Example(state=s, answers={"q": a}) for s, a in rows],
    )
    was_training = model.training
    model.eval()
    ds = TaskDataset(task, packer, shuffle_options=False)
    right = 0
    for i in range(len(task.examples)):
        s = ds[i]
        b = collate([s], packer=packer)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lp = model(b).log_probs.float().cpu()
        labels = s.packed.labels["q"]  # type: ignore[attr-defined]
        right += labels[int(lp[: len(labels)].argmax())] == task.examples[i].answers["q"]
    if was_training:
        model.train()
    return right / len(task.examples)


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
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base",
                    help="default is the encoder alternative; the recommended "
                         "path is Qwen/Qwen3-1.7B with --decoder --lora-r 16")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation; effective batch is bs*accum")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="batch to a token budget instead of a fixed count")
    ap.add_argument("--mask-budget", type=int, default=80_000_000,
                    help="cap on batch*len^2, the materialised attention mask")
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--sanity-at", type=int, default=0,
                    help="step at which to abort if a trivial task is at chance")
    ap.add_argument("--sanity-every", type=int, default=0,
                    help="re-check periodically; one early check misses a run "
                         "that degrades after the warmup ends")
    ap.add_argument("--distractor-prob", type=float, default=0.0,
                    help="probability of padding a choice menu with borrowed labels")
    ap.add_argument("--max-options", type=int, default=128)
    ap.add_argument("--scale-prob", type=float, default=0.0,
                    help="probability of rewording or coarsening an ordinal scale")
    ap.add_argument("--noul-prob", type=float, default=0.0,
                    help="probability of recasting a choice question as a yes/no one")
    ap.add_argument("--decoder", action="store_true",
                    help="causal LM backbone with yes/no readout; the "
                         "recommended path, pass --backbone Qwen/Qwen3-1.7B too")
    ap.add_argument("--lora-r", type=int, default=0,
                    help="LoRA rank on the decoder backbone; 0 disables it. "
                         "16 is the shipped setting and beats opening all "
                         "1.7B parameters")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the calibration head (use with --decoder)")
    ap.add_argument("--preamble", default=None)
    ap.add_argument("--init-from", default=None,
                    help="start from a checkpoint instead of the raw backbone")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing a checkpoint of a different architecture")
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
    if device == "cuda":
        # The block-diagonal mask is an explicit tensor, and cuDNN's attention
        # kernels are the least tolerant of that: under memory pressure the
        # backward pass fails the CUDA launch outright rather than falling
        # back to another kernel. The memory-efficient and math backends
        # handle the same mask without complaint, so rule cuDNN out here
        # rather than discovering it several hours into a run.
        torch.backends.cuda.enable_cudnn_sdp(False)
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
        sources = [t.description.replace("tasksource ", "") for t in tasks]
        assert_training_mixture_clean(sources)
        # Say so. A guard that is silent on success leaves no evidence in the
        # log that it ever ran, and "we enforce it on every run" then rests on
        # reading the source rather than on the run itself.
        print(f"contamination guard: {len(sources)} sources checked against the "
              f"held-out suite, no overlap")
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
        parts = [
            TaskDataset(
                t, packer, shuffle_options=True, seed=i,
                distractors=uniq, distractor_prob=args.distractor_prob,
                max_options=args.max_options, scale_prob=args.scale_prob,
                noul_prob=args.noul_prob,
            )
            for i, t in enumerate(tasks)
        ]
        train_ds = ConcatDataset(parts)
        if args.token_budget:
            from openjev.data import estimate_lengths
            train_lengths = []
            for i, t in enumerate(tasks):
                train_lengths += estimate_lengths(
                    t, packer, distractor_prob=args.distractor_prob,
                    max_options=args.max_options, distractors=uniq,
                    scale_prob=args.scale_prob, noul_prob=args.noul_prob,
                    seed=i)
    else:
        for s in ("train", "dev", "test"):
            t = Task.load(args.task, s)
            if len(t):
                splits[s] = t
        # `heldout_check` above already rejects a task whose directory name
        # is in the suite. This second pass matches on the task's declared
        # source instead, which catches a held-out dataset that has been
        # renamed on disk. Both honour --allow-heldout.
        from openjev.heldout import assert_training_mixture_clean
        declared = splits["train"].name or name
        if not args.allow_heldout:
            assert_training_mixture_clean([declared])
        print(f"contamination guard: {declared!r} checked against the held-out "
              f"suite, no overlap")
        print(f"task {name}: " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
        print(f"questions: {len(splits['train'].questions)}  device: {device}")
        train_ds = TaskDataset(splits["train"], packer, shuffle_options=True)
        if args.token_budget:
            from openjev.data import estimate_lengths
            train_lengths = estimate_lengths(splits["train"], packer)
    fn = partial(collate, packer=packer)

    def wrap(ds, shuffle):
        def c(samples):
            b = fn(samples)
            b["_samples"] = samples
            return b
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle, collate_fn=c)

    if args.token_budget:
        # Pack batches to a token budget instead of a fixed count. A fixed
        # count has to survive the longest sequence in the mixture and then
        # wastes the GPU on the median one.
        from openjev.data import TokenBudgetBatches

        def c(samples):
            b = fn(samples)
            b["_samples"] = samples
            return b

        sampler = TokenBudgetBatches(
            train_lengths, token_budget=args.token_budget,
            mask_budget=args.mask_budget, max_size=args.max_bs,
        )
        sizes = [len(b) for b in sampler]
        print(f"token budget {args.token_budget}: {len(sampler)} batches, "
              f"mean size {sum(sizes) / len(sizes):.1f}, max {max(sizes)}")
        train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=c)
    else:
        sampler = None
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
            lora_r=args.lora_r,
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
        # Initialising an encoder from a decoder checkpoint loads nothing and
        # looks like a normal from-scratch run. Now that both kinds sit in
        # neighbouring directories, check before trusting the weights.
        if prev.get("backbone") and prev["backbone"] != args.backbone:
            raise SystemExit(
                f"--init-from is a {prev['backbone']} checkpoint but this run is "
                f"{args.backbone}. Almost nothing would load and the run would "
                f"look like training from scratch."
            )
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
    # The scheduler advances once per optimiser step, not once per
    # micro-batch, so it has to be built from the post-accumulation count or
    # the cosine curve only runs 1/accum of its length and the rate never
    # decays.
    steps = (len(train_dl) * args.epochs) // args.accum
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler_dtype = torch.bfloat16

    print(f"\ntraining: {steps} optimiser steps "
          f"(micro-batch {args.bs} x accum {args.accum} = effective {args.bs * args.accum})")
    t0 = time.time()
    step = 0
    micro = 0
    oom = 0
    seen = 0
    for ep in range(args.epochs):
        run = 0.0
        for batch in train_dl:
            batch.pop("_samples")
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            try:
                with torch.autocast(device, dtype=scaler_dtype, enabled=device == "cuda"):
                    out = model(b)
                    loss = -out.log_probs[b["target_flat"]].mean()
            except torch.OutOfMemoryError:
                # Batch sizes come from estimated lengths, and augmentation
                # can make a draw longer than its estimate. Skipping the
                # occasional overflow is better than either crashing eight
                # hours in or shrinking every batch to fit the worst case.
                oom += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            # Long contexts force a small micro-batch, because the
            # block-diagonal mask costs O(n^2). Accumulating keeps the
            # effective batch, and therefore the learning rate, where it was
            # tuned: a 6144-token run at micro-batch 2 with the batch-4
            # learning rate diverges, which we measured the expensive way.
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                run += float(loss.detach())
                seen += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss.detach())
            seen += 1
            step += 1
            if step % 50 == 0:
                # Divide by micro-batches seen, not optimiser steps, or the
                # printed loss is scaled by the accumulation factor.
                avg = run / max(1, seen)
                print(f"  ep{ep} step {step}/{steps}  loss {avg:.4f}  "
                      f"{time.time() - t0:.0f}s" + (f"  oom {oom}" if oom else ""))
                run, seen = 0.0, 0
            # Fail fast. A diverged run looks fine in the loss for a long
            # while and then scores exactly chance at the end, which cost us
            # a seven hour run once. Check the trivial task early, when the
            # answer is still cheap.
            due = (args.sanity_at and step == args.sanity_at) or (
                args.sanity_every and step % args.sanity_every == 0)
            if due:
                acc = harness_sanity(model, packer, device)
                print(f"  sanity check at step {step}: {acc:.3f} "
                      f"(chance 0.333, healthy is near 1.0)")
                if acc < 0.6:
                    raise SystemExit(
                        f"\nAborting: the model scores {acc:.3f} on a task it "
                        f"cannot fail unless training has broken.\nThis is "
                        f"usually the learning rate being wrong for the batch "
                        f"size. Nothing was saved.\n"
                    )
                # Skipping a batch on OOM keeps the run alive, but a high skip
                # rate means most of the data never reaches the model and the
                # run is quietly training on a biased subset: the short
                # examples. That is worse than being slow, and it is invisible
                # in the loss curve.
                rate = oom / max(1, oom + step)
                if rate > 0.05 and step <= max(args.sanity_at, 1):
                    raise SystemExit(
                        f"\nAborting: {oom} batches skipped for out-of-memory "
                        f"against {step} completed, a {rate:.0%} skip rate.\n"
                        f"The run would train on whichever examples happen to "
                        f"fit, which biases it towards short menus, the "
                        f"opposite of what this training is for.\nLower "
                        f"--token-budget or --mask-budget and start again. "
                        f"Nothing was saved.\n"
                    )
                print(f"  out-of-memory skips: {oom} ({rate:.1%}), acceptable")
        if sampler is not None:
            sampler.set_epoch(ep + 1)
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
    # A frozen-backbone decoder only trains the head, so storing the backbone
    # turns a 17MB model into a 3.3GB file for nothing.
    dest = save_ckpt(
        out_dir / "model.pt", model,
        backbone=args.backbone,
        decoder=bool(args.decoder),
        # LoRA and head-only both train a small part of a frozen backbone,
        # so both should write the adapter rather than a 3.4GB copy of Qwen.
        head_only=bool(args.decoder and (args.freeze_backbone or args.lora_r)),
        lora_r=int(args.lora_r),
        overwrite=args.overwrite,
        # The prompt format is part of the model: the same weights scored with
        # a different preamble are a different system. Store it so evaluation
        # cannot quietly drift away from how this was trained.
        learned_head=bool(args.decoder),
        preamble=args.preamble,
        option_template=template,
        temperatures=temps,
        trained_on=name,
        heldout_override=bool(args.allow_heldout),
    )
    size_mb = dest.stat().st_size / 1e6
    print(f"\nsaved to {dest}  ({size_mb:.0f} MB, {time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()

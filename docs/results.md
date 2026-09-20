# Results

Everything measured by us, with the command to reproduce it. Numbers move when
you rerun them; intervals are bootstrap over examples.

## Banking77 specialist

**In-task, not a generalization result.** This model was fine-tuned on 9,839
Banking77 training examples. The reference system was not: it answered the same
question zero-shot from the option text alone. These measure different things,
and the comparison below is only meaningful as "what does having labels buy
you", not as "which model is better".

`tasks/heldout/banking77` is therefore **disqualified as a held-out task for
this checkpoint**. The trainer refuses to run on it without `--allow-heldout`,
and records the override in the checkpoint file.

```bash
uv run python scripts/build_banking77_specialist.py
uv run python scripts/train.py --task tasks/banking77_specialist \
    --epochs 3 --bs 16 --max-len 4096 --allow-heldout
```

ModernBERT-base (149.6M params, of which the shared scorer is 593K), 3 epochs,
batch 16, 1,845 steps, **13.5 minutes on one H100**. Test is the same 600 rows
as `tasks/heldout/banking77`.

| | accuracy | 95% CI | macro-F1 | ECE | Brier |
|---|---|---|---|---|---|
| OpenJev, raw | 0.9233 | [0.900, 0.945] | 0.9230 | 0.056 | 0.1300 |
| OpenJev, temperature-scaled | **0.9233** | [0.900, 0.945] | **0.9230** | **0.029** | **0.1194** |
| *reference: Jev `1.13.0`, zero-shot* | *0.820* | *[0.777, 0.863]* | *0.806* | *0.068* | *0.277* |

The reference row was measured against the public TypeSafe API on 20 September
2026 on 300 stratified rows of the same test split, so it is a different sample
from our 600 and should be read as approximate.

### What this does and does not support

**Supports:** the architecture learns. A shared scoring head over option
markers, with no per-class parameters, reaches 92.3% on a 77-way task — in line
with what ordinary fine-tuned encoders have long achieved on Banking77. The
head is not the bottleneck, and a 150M model trained for 13 minutes on one GPU
is enough.

**Supports:** if you have labels for your task, fine-tuning a small encoder
beats a general decision model on that task, by about ten points here. That is
the project's actual pitch and it is the least surprising result in the table.

**Does not support:** any claim about zero-shot ability, generalization, or
being "better than Jev". We trained on the task. The interesting comparison —
a held-out schema neither model has seen — has not been run yet.

### Temperature scaling did exactly what it should

Accuracy is **identical** before and after (0.9233 both), while ECE halves from
0.056 to 0.029 and Brier improves from 0.1300 to 0.1194. That is the expected
behaviour and a useful check on the implementation: temperature scaling is
monotonic, so it cannot reorder predictions and cannot change accuracy. If it
had, something would be wrong.

The fitted temperature is **1.917**, comfortably above 1, meaning the model was
overconfident — the standard consequence of training to convergence on
cross-entropy, and the reason the calibration step exists at all.

## Zero-shot from the warm start alone: a negative result

The scoring head can reuse the backbone's own pretrained masked-LM head at
each option marker (`logit(yes) - logit(no)`), which adds no parameters and
can therefore be evaluated with **no training whatsoever**. The question that
answers: how much zero-shot ability does a good encoder give us for free?

```bash
uv run python scripts/eval_zeroshot.py --backbone answerdotai/ModernBERT-large --limit 40
```

40 rows per task, accuracy as a multiple of chance (1/K), since option counts
run from 2 to 151 and raw accuracy is not comparable across them:

| task | K | base | large |
|---|---|---|---|
| banking77 | 77 | 0.0x | 11.5x |
| clinc_oos | 151 | 0.0x | 0.0x |
| massive_intent | 60 | 1.5x | 0.0x |
| sst5 | 5 | 0.9x | 0.7x |
| ag_news | 4 | 0.8x | 1.1x |
| civil_comments_toxicity | 2 | 1.0x | 1.1x |
| helpsteer_helpfulness | 5 | 1.0x | 1.0x |
| **mean** | | **0.7x** | **2.2x** |

**The answer is: nothing usable.** ModernBERT-base is at chance across the
board and scores a literal 0.0000 on both high-cardinality tasks.
ModernBERT-large is better but still unusable — its 2.2x mean is carried
almost entirely by one task, and it is *below* chance on three.

This is worth stating plainly because it is easy to assume otherwise: a strong
bidirectional encoder plus a clever output format does **not** produce a
zero-shot decision model. The format makes zero-shot *possible*, since the
scoring head has no per-class parameters and a new label set is just new
input. It does not make it *present*.

That is consistent with UniMC, which needed a multi-task "MC tuning" stage to
get zero-shot behaviour, and with the Flan Collection finding that held-out
performance is a function of training task count. **The multi-task stage is
not an optimisation on top of the architecture; it is where the capability
comes from.** Any plan that treated it as optional was wrong, including an
earlier version of ours that framed the warm start as the cheap lever.

One caveat on the comparison we wanted but could not run: the obvious
alternative warm starts — `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` and
`tasksource/ModernBERT-large-nli` — are sequence-classification checkpoints
with no masked-LM head, so they cannot be scored this way at all. Comparing
them needs the trained scorer, which means it belongs after the multi-task
run, not before it.

## Multi-task training: learns the tasks, transfers nothing

61 tasks from `tasksource` (95,413 rows), ModernBERT-base, 2 epochs, ~12
minutes on one H100. The held-out suite was excluded at build time.

```bash
VIRTUAL_ENV=.venv-data uv run --no-project python scripts/build_mixture.py
uv run python scripts/train.py --mixture tasks/mixture --epochs 2 --bs 16
uv run python scripts/eval_heldout.py --ckpt checkpoints/mixture/model.pt
```

**Held-in control** — tasks this checkpoint was trained on, 200 rows each:

| task | K | accuracy | x chance |
|---|---|---|---|
| ethos_binary | 2 | 0.840 | 1.7x |
| glue_mrpc | 2 | 0.760 | 1.5x |
| glue_qnli | 2 | 0.715 | 1.4x |
| glue_cola | 2 | 0.700 | 1.4x |
| glue_mnli | 3 | 0.675 | 2.0x |

**Held-out suite** — schemas never trained on:

| task | K | accuracy | 95% CI | x chance |
|---|---|---|---|---|
| banking77 | 77 | **0.0000** | [0.000, 0.000] | 0.0x |
| clinc_oos | 151 | **0.0000** | [0.000, 0.000] | 0.0x |
| massive_intent | 60 | 0.0400 | [0.015, 0.070] | 2.4x |
| sst5 | 5 | 0.2000 | [0.150, 0.255] | 1.0x |
| ag_news | 4 | 0.2250 | [0.170, 0.290] | 0.9x |
| civil_comments_toxicity | 2 | 0.5150 | [0.445, 0.585] | 1.0x |
| helpsteer_helpfulness | 5 | 0.1650 | [0.120, 0.215] | 0.8x |
| **mean** | | | | **0.9x** |

**The control is what makes this readable.** The checkpoint clearly learned —
0.84 on ethos, 0.675 on three-way MNLI — while scoring at chance on every
held-out schema. So the machinery works and the transfer is genuinely absent.
Had both been at chance, this would have been a bug report instead.

Note also that training changed nothing versus no training at all: the
untrained MLM-head baseline was 0.7x to 2.2x, and this is 0.9x.

### Why, and what it does not mean

This does **not** show the architecture cannot generalize. Two concrete
deficiencies in the mixture explain it, both fixable:

**It is far too small.** 61 tasks against the ~282 where the Flan Collection
ablation says most of the held-out gain has accrued, and held-out performance
there rises log-linearly in task count. We are at the bottom of that curve.
The shortfall is mechanical rather than fundamental: 285 of 346 candidate
loads failed, 238 of them `HfHubHTTPError` from pulling hundreds of datasets
without backoff. The builder now retries and throttles.

**It has no cardinality diversity.** Option counts came out min 2, median 3,
max 20, while the held-out suite runs to 151. The model never saw a menu
remotely that size. The literal 0.0000 on banking77 and clinc_oos is the
signature of collapsing onto one label when handed 77 or 151 options, not of
ranking them badly — at chance it should have got roughly 3 of 200 right.

So the honest reading is that we have confirmed the architecture trains and
have **not yet built a mixture capable of testing the generalization claim**.
The next build needs several hundred tasks and deliberate high-K sourcing
before a negative result here means anything about the design.

See [`architecture.md`](architecture.md#measured-is-packing-worth-its-complexity).
Flat in question count; 24x over the naive one-sequence-per-question
alternative at a 2.8k-token state; no benefit at all on short states.

## Healthcare router: one real win, and a data-starvation diagnosis

First OpenJev numbers on `tasks/healthcare_router`, and the first test of the
`noul` primitive, multi-question packing and the safety gates — none of which
Banking77 exercises. Trained on the router's own 395-example train split, 6
epochs, 38 seconds.

```bash
uv run python scripts/train.py --task tasks/healthcare_router --epochs 6 --bs 8
uv run python scripts/eval_router.py --ckpt checkpoints/healthcare_router/model.pt
```

| | OpenJev | Jev 1.13.0 | delta |
|---|---|---|---|
| intent choice, lenient | 0.544 | 0.909 | **−0.365** |
| multi-label exact set | 0.453 | 0.822 | **−0.369** |
| multi-label F1 | 0.554 | 0.890 | −0.336 |
| `G_clinical` recall | 0.898 | 0.926 | −0.028 |
| `G_clinical` FPR | 0.047 | 0.006 | **8x worse** |
| **`clinical_oblique` recall** | **0.931** | 0.793 | **+0.138** |
| `G_abusive` recall | 0.000 | 0.714 | −0.714 |
| `G_injection` recall | 0.125 | 0.917 | −0.792 |
| `G_pharmacy` FPR | 0.826 | 0.087 | **collapsed to majority class** |

### Paired McNemar against the same 450 items

Summary numbers hide who was right on which item. Exact two-sided McNemar,
where `b01` counts items Jev got right and we did not, and `b10` the reverse:

| slice | n | ours | Jev | b01 | b10 | p |
|---|---|---|---|---|---|---|
| intent choice (lenient) | 338 | 0.544 | 0.941 | 142 | 8 | 7.8e-33 |
| multi-label exact set | 450 | 0.453 | 0.822 | 180 | 14 | 6.6e-38 |
| `G_injection` correctness | 450 | 0.949 | 0.996 | 21 | 0 | 9.5e-07 |
| `G_clinical` correctness | 450 | 0.940 | 0.978 | 24 | 7 | 3.3e-03 |
| `G_abusive` correctness | 450 | 0.984 | 0.996 | 5 | 0 | 0.063 |
| `G_pharmacy` correctness | 450 | 0.942 | 0.880 | 20 | 48 | 9.1e-04 |
| **`clinical_oblique` recall** | **29** | **0.931** | **0.793** | **0** | **4** | **0.125** |

**Two corrections this forces, both against us.**

*The oblique-clinical result is not a win.* On the slice we argued mattered
most, we beat Jev on 4 items and lose on 0 — but at n=29 that is **p = 0.125,
not significant**. The direction is encouraging and the tier is the right one
to care about, but "OpenJev beats Jev at detecting obliquely-phrased adverse
events" is not a claim this evidence supports. It needs a larger
`clinical_oblique` tier before it means anything.

*The `G_pharmacy` "win" is an artifact.* We score 0.942 against Jev's 0.880 at
p = 0.0009, which looks like our best result on the board. It is not. That
slice is 427 positive against 23 negative, and our model simply predicts
positive almost always: recall 0.984 with a **false-positive rate of 0.826**,
against Jev's 0.087. We learned the majority class and the accuracy metric
rewarded us for it. This is the exact failure the project's own docs warn
about, caught here only because the gate reports FPR next to recall.

`G_abusive` deserves the same scepticism: 0.984 "correctness" while recall is
**0.000**. With 7 positives against 443 negatives, never firing is an
excellent way to look accurate.

**The honest reading of the gates is therefore narrower than it first
appeared.** Our `G_clinical` false-positive rate is 0.047 against Jev's 0.006,
so whatever oblique recall we gained was bought by firing more readily in
general — 16 false alarms on 342 negatives against Jev's 2. Whether that trade
is worth it depends on the cost of a false escalation, which `expected_cost()`
exists to compute and which we have not priced for this task.

**Everything else is much worse, and the cause is data, not architecture.**
395 training examples across 10 questions is about 40 per question.
`G_abusive` has 7 positives in the whole test split and correspondingly few in
train; it scored 0.000, never learning to fire at all. `G_injection` reached
0.125. That is not a model failing so much as a concept being learned from a
handful of instances.

Which points somewhere specific: **the router suite was built as an evaluation
set and is too small to train on.** The fix is not more router data, it is a
general checkpoint strong enough to fine-tune from — a specialist starting
from a good base needs far fewer examples than one starting from a raw
encoder. That is the argument for the project having a general model at all,
and it is the work below.

## Not yet measured

- Held-out schema transfer, which is the number that matters for the zero-shot
  claim and the one we expect to lose on.
- The healthcare router suite, including the oblique-clinical gate.
- Anything on `score` or `noul` primitives — Banking77 is `choice` only.
- Latency served the same way as the reference system. Our 9.6 ms is local
  compute with no network; the reference's ~400 ms included a 331 ms round
  trip. Comparing them directly would be dishonest.

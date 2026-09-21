# Results

Everything measured by us, with the command to reproduce it. Bootstrap
intervals over examples; paired comparisons use exact McNemar.

> **Provenance.** Every figure attributed to Jev was measured against the
> public TypeSafe API, model `jev-1.13.0`, on 20 September 2026, at the sample
> sizes stated. Black-box measurements of a hosted endpoint on one day from
> one network location. Reproduce before relying on them.

---

## Summary

| | OpenJev | Jev | verdict |
|---|---|---|---|
| Banking77, **fine-tuned on it** | **0.923** | — | in-task, not comparable |
| Banking77, never seen it | 0.355 | **0.820** | Jev, clearly |
| CLINC-150 (K=151), never seen it | 0.518 | not measured | 78x chance |
| Held-out suite, 5 of 7 tasks | 17.6x chance | not measured | `score` still at chance |
| Held-out binary, ranking | AUROC 0.714 | not measured | transfers; miscalibrated |
| Router intent, LoRA | **0.979** | 0.941 | **OpenJev**, p = 1e-03 |
| Router multi-label exact set, LoRA | **0.909** | 0.822 | **OpenJev**, p = 7e-06 |
| `G_pharmacy`, LoRA | **0.978** | 0.880 | **OpenJev**, p = 4e-10 |
| Oblique-clinical recall | **1.000** | 0.793 | **OpenJev**, p = 0.03 |
| `G_pharmacy` correctness | **0.947** | 0.880 | **OpenJev**, p = 3e-04 |
| `G_clinical` correctness | 0.980 | 0.978 | level, p = 1.00 |
| Probability precision | full | 0.01 grid, 71.9% hard zeros | **OpenJev** |
| Deterministic | yes | no, no seed | **OpenJev** |

---

## At a glance

![OpenJev against Jev on the router](plots/comparison_router.png)

Accuracy on the left is the same 450 test items for all three, fine-tuned on
395 examples. Latency on the right is the whole router asked in one call.

One caveat on that right-hand panel, since it looks more flattering than it
is: our figures are local GPU compute and Jev's is an end-to-end call to a
hosted service, so it includes network round-trip. It is the latency a user
experiences, not a claim about the speed of their model.

Reproduce with `scripts/plot_comparison.py`, which carries the provenance of
every number in it.

## The mixtures referred to below

Several training mixtures appear in these results and they are not
interchangeable. Each row is a superset of the ideas above it.

| mixture | tasks | rows | `choice` / `noul` / `score` | what changed |
|---|---|---|---|---|
| `mixture` | 61 | — | — | first attempt; transferred nothing |
| `mixture_big` | 141 | 189,462 | 128 / 13 / 0 | rebuilt with an auth token and backoff |
| `mixture_ord2` | 143 | 192,462 | 121 / 13 / 9 | ordinal label sets detected and emitted as `score` |
| `mixture_noul` | 143 | 192,462 | 100 / 34 / 9 | negation pairs retyped from `choice` to `noul` |
| `mixture_ord3` | 145 | 198,154 | 100 / 34 / 11 | curated star-rating datasets added |

Where a result below names a figure like "13 `noul` tasks", it is describing
the mixture that run used, not the current one.

---

# What worked

## 1. The general checkpoint is worth +36 points to a specialist

**The project's central claim, and the strongest result here.** Same router
task, same 395 training examples, same 6 epochs, same 38 seconds. Only the
starting weights differ.

```bash
uv run python scripts/train.py --task tasks/healthcare_router \
    --init-from checkpoints_noul/mixture_ord2/model.pt --epochs 6 --bs 8
```

| | from raw ModernBERT | from the first general ckpt | from the best general ckpt | Jev |
|---|---|---|---|---|
| intent | 0.544 | 0.817 | **0.899** | 0.941 |
| multi-label exact set | 0.453 | 0.718 | **0.789** | 0.822 |
| multi-label F1 | 0.554 | 0.816 | **0.855** | 0.890 |
| `G_clinical` recall | 0.898 | 0.991 | **1.000** | 0.926 |
| `clinical_oblique` recall | 0.931 | 0.966 | **1.000** | 0.793 |
| `G_abusive` recall | 0.000 | 0.429 | **0.857** | 0.714 |
| `G_injection` recall | 0.125 | **0.750** | 0.583 | 0.917 |
| `G_pharmacy` FPR (lower better) | 0.826 | 0.783 | **0.652** | — |

Paired McNemar against from-scratch: intent **b10=108, b01=16, p = 6.0e-18**;
multi-label **b10=144, b01=25, p = 1.6e-21**. The third column tracks the
general checkpoint improving underneath it: the same 38-second fine-tune now
starts from a model scoring 17.6x chance on held-out schemas instead of 9.7x.

Paired against Jev on the same 450 items, `scripts/compare_to_jev.py`:

| | OpenJev | Jev | b10 | b01 | p | winner |
|---|---|---|---|---|---|---|
| intent | 0.899 | 0.941 | 13 | 27 | 0.039 | Jev |
| multi-label exact set | 0.789 | 0.822 | 38 | 53 | 0.142 | **level** |
| `G_clinical` | 0.980 | 0.978 | 8 | 7 | 1.000 | level |
| `G_abusive` | 0.998 | 0.996 | 2 | 1 | 1.000 | level |
| `G_injection` | 0.978 | 0.996 | 0 | 8 | 0.008 | Jev |
| `G_pharmacy` | 0.947 | 0.880 | 48 | 18 | 3e-04 | **OpenJev** |
| `clinical_oblique` recall | **1.000** | 0.793 | 6 | 0 | 0.031 | **OpenJev** |

So Jev still wins single-label intent and the injection gate, the multi-label
set is now a statistical tie, and OpenJev wins the pharmacy gate and the
oblique-clinical tier the objective singled out.

**A comparison we were getting wrong.** Jev's two intent figures use
different denominators: 0.909 covers all 450 items including the 112 whose
gold is an acceptable *set*, and 0.941 covers the 338 with a single gold.
Our harness only asks the intent question when the example carries a gold,
so it scores the 338 and must be compared against 0.941. `eval_router.py`
was printing ours against 0.909, which understated the gap by about three
points in our favour. Fixed, and the constants are now named for their
denominators so they cannot be swapped again.

**Why it works.** 395 examples across 10 questions is ~40 per question, and
two gates had so few positives (`G_abusive`: 7) that from scratch they never
learned to fire at all. A checkpoint that already knows "score how well this
option describes this state" only has to learn *this* label set, not the task
shape. That is the whole argument for having a general model.

## 2. Label-space augmentation fixes high-cardinality transfer

The encoder scored a literal **0.000** on Banking77 and CLINC and chance
everywhere else. Padding training menus with labels borrowed from other tasks
in the mixture — gold answer unchanged, so the example stays valid — moved it
above chance on all seven held-out tasks.

| task | K | before | after |
|---|---|---|---|
| banking77 | 77 | 0.000 | **0.205** (15.8x) |
| clinc_oos | 151 | 0.000 | **0.180** (27.2x) |
| massive_intent | 60 | 0.040 | **0.305** (18.3x) |
| ag_news | 4 | 0.225 | **0.655** (2.6x) |
| sst5 | 5 | 0.200 | **0.360** (1.8x) |

**Why it works.** The mixture topped out at 20 options while deployments ask
for 151. A model that has never seen a large menu collapses onto one label
when handed one — which is what an exact 0.000 looks like, since chance alone
would have produced ~3 correct in 200. You do not need datasets with large
menus; you can synthesize them.

Caveat: the mixture also grew 61 → 141 tasks in the same change, so the two
are not separated here.

## 3. Prompt format took the decoder from chance to usable

A causal backbone with per-option yes/no readout, **no training at all**:

| change | mean x chance |
|---|---|
| bare concatenation of state and options | 0.8x |
| + each option framed as an explicit yes/no question | 2.9x |
| + a preamble stating the task | ~6.8x |
| + context budget so `clinc_oos` stops dropping out | ~12x |
| final, n=200 | **11.5x** |

Same 1.7B model, same architecture, no weights changed. The control that makes
this convincing is the failure: shortening the option framing to
`{opt}\ncorrect? answer` **halved** the mean.

**Why it works.** At the readout position the model must be able to tell a
yes/no question was asked. A bare `...refill: another fill:` reads like
nothing in its training distribution, so the readout measures noise.

## 4. Training a head on the frozen decoder: 16.2x chance, and a clean split by primitive

A 4.2M-parameter residual scorer on a frozen Qwen3-1.7B, one epoch over the
141-task `mixture_big`. The head is zero-initialised so the run starts exactly
at the untrained readout and can only improve on it.

```bash
uv run python scripts/train.py --mixture tasks/mixture_big --decoder \
    --freeze-backbone --backbone Qwen/Qwen3-1.7B --epochs 1 --bs 4 \
    --max-len 3072 --head-lr 1e-3 --distractor-prob 0.5 --max-options 120 \
    --preamble "You judge whether a candidate answer is correct for a question about an input."$'\n\n'"Input:"$'\n'
```

| held-out task | K | encoder | decoder + head | x chance |
|---|---|---|---|---|
| clinc_oos | 151 | 0.180 | **0.420** | 63.4x |
| banking77 | 77 | 0.205 | **0.355** | 27.3x |
| massive_intent | 60 | 0.305 | 0.290 | 17.4x |
| ag_news | 4 | **0.655** | 0.490 | 2.0x |
| sst5 (`score`) | 5 | **0.360** | 0.185 | 0.9x |
| helpsteer (`score`) | 5 | 0.230 | 0.235 | 1.2x |
| civil_comments (`noul`) | 2 | 0.500 | 0.490 | 1.0x |

Mean **16.2x chance**, against 9.7x for the encoder and 11.5x for the same
decoder untrained. Harness sanity 0.833, held-in control run and reported.

**The split is the finding.** Every task above chance is a `choice` question,
and every task at chance is `score` or `noul`. The two backbones are also
complements rather than rivals: the decoder wins the large menus by a wide
margin and the encoder wins every menu with five options or fewer. Nothing
here is an architecture verdict on `score` and `noul` — `mixture_big` contains
128 `choice` questions against 13 `noul` and **zero** `score`, so the model was
never taught the ordinal primitive at all.

## 5. The packed shared prefix is real, and large

ModernBERT-base, bf16, H100, 50 questions, median of 10:

| state | packed tokens | naive tokens | packed | naive | speedup |
|---|---|---|---|---|---|
| ~90 tok | 713 | 3,800 | 9.6 ms | 9.6 ms | 1.00x |
| ~700 tok | 1,133 | 24,800 | 9.7 ms | 51.7 ms | **5.3x** |
| ~2.8k tok | 2,573 | 96,800 | 11.0 ms | 263.5 ms | **24.0x** |

Latency is flat in question count: at a 700-token state, 1 and 50 questions
both take 9.6 ms.

**Caveat worth keeping:** at short states packing buys nothing. 9.6 ms is a
fixed-overhead floor and both paths hit it.

## 6. Banking77 specialist: 92.3% in 13.5 minutes

ModernBERT-base, 3 epochs, one H100. **In-task** — we trained on 9,839
Banking77 examples and Jev did not, so this measures what labels buy, not
which model is better.

| | accuracy | macro-F1 | ECE | Brier |
|---|---|---|---|---|
| raw | 0.9233 | 0.9230 | 0.056 | 0.1300 |
| temperature-scaled | **0.9233** | 0.9230 | **0.029** | 0.1194 |

Temperature scaling behaved exactly as theory requires: **accuracy identical
to four decimal places** while ECE halves. It is monotonic and cannot reorder
predictions; had accuracy moved, the implementation would be wrong.

## 7. Recasting `choice` as yes/no breaks the degenerate binary

Three encoder runs, identical except for the augmentation flags, same
143-task `mixture_ord2`, one epoch each. Harness sanity 1.000, held-in
control 1.4–2.3x.

| held-out task | no augmentation | `--scale-prob 0.5` | `+ --noul-prob 0.25` |
|---|---|---|---|
| clinc_oos (K=151) | 0.490 | 0.445 | **0.518** |
| banking77 (K=77) | 0.187 | 0.212 | **0.278** |
| massive_intent (K=60) | **0.362** | 0.348 | 0.298 |
| ag_news | **0.560** | 0.492 | 0.408 |
| sst5 (`score`) | 0.342 | **0.348** | 0.308 |
| helpsteer (`score`) | 0.110 | 0.128 | **0.222** |
| civil_comments (`noul`) | 0.498 | 0.500 | **0.528** |
| civil_comments macro-F1 | 0.333 | 0.333 | **0.403** |
| **mean x chance** | 16.5x | 15.7x | **17.6x** |

The macro-F1 row is the one that matters. A binary task at 0.333 macro-F1 is
a model answering the same way every time, and it had done that in every run
until this one. Converting `choice` questions into "is it this one?" gives
the primitive 121 tasks' worth of supervision it was not getting from the 13
labelled for it, and the constant answer goes away.

**It is not a win yet.** At n=600 the CI is [0.490, 0.572] against a 0.500
floor, so the accuracy gain is not significant; only the collapse is fixed.
`helpsteer` doubles, 0.110 to 0.222, but lands exactly on its 0.233
majority-class baseline, which is not skill either.

## 8. With LoRA, OpenJev beats Jev on the router

Rank-16 adapters on Qwen3-1.7B, 395 training examples, 258 seconds. Paired
on the same 450 items, exact McNemar.

| | OpenJev | Jev | b10 | b01 | p | winner |
|---|---|---|---|---|---|---|
| intent | **0.979** | 0.941 | 14 | 1 | 9.8e-04 | **OpenJev** |
| multi-label exact set | **0.909** | 0.822 | 57 | 18 | 7.2e-06 | **OpenJev** |
| `G_pharmacy` | **0.978** | 0.880 | 49 | 5 | 3.9e-10 | **OpenJev** |
| `G_clinical` | 0.987 | 0.978 | 7 | 3 | 0.34 | level |
| `G_abusive` | 0.993 | 0.996 | 2 | 3 | 1.00 | level |
| `G_injection` | 0.993 | 0.996 | 1 | 2 | 1.00 | level |
| `clinical_oblique` recall | 0.966 | 0.793 | 5 | 0 | 0.06 | level |

**Three wins, four ties, no losses.**

What makes this more than a scaling result is that the adapter is the
*smallest* configuration that works, not the largest:

| | intent | trainable | artifact |
|---|---|---|---|
| encoder, everything open | 0.899 | 150M | 0.6 GB |
| decoder, head only | 0.666 | 4.2M | 17 MB |
| decoder, everything open | 0.929 | 1,725M | 3.4 GB |
| **decoder, LoRA r=16** | **0.979** | **17M** | **87 MB** |

Opening all 1.7B parameters is *worse* than adapting 17M of them, and 39x
the artifact. The likely reason is the learning rate a full fine-tune can
tolerate: 1e-5 for six epochs over 395 examples barely moves a 1.7B model,
while LoRA at 2e-4 adapts quickly without disturbing the pretrained weights
it is riding on. We did not tune either beyond one setting, so read this as
"adapters are the right default here", not as a tuned optimum.

It also makes the deployment story work. 87 MB per use case means one
backbone can serve many routers; 3.4 GB per use case means it cannot.

## 9. An unmerged LoRA adapter costs 2x at inference, for nothing

The adapter is an extra matmul per target module at every layer. Left
unmerged it doubles latency; folded into the base weights it nearly
disappears.

Measured on the same checkpoints, back to back, while a training run shared
the GPU. Absolute numbers are inflated by that contention, so read the
ratios, not the milliseconds:

| | p50 | p95 | vs encoder |
|---|---|---|---|
| encoder, 150M | 37.7 ms | 50.9 ms | — |
| decoder LoRA, unmerged | 74.0 ms | 110.5 ms | 1.96x |
| decoder LoRA, merged | **40.4 ms** | **55.9 ms** | **1.07x** |

So the honest answer to "is the decoder too slow for a router" is: merged,
no. Seven percent over an encoder a tenth its size, for eight points of
intent accuracy and a win over Jev.

`DecisionModel` merges on load so this cannot be paid by accident, and
`merge_adapter()` is exposed for anyone building their own serving path.

**A bug this surfaced.** Adding LoRA introduced `self._base = self.lm`,
which is an `nn.Module` attribute assignment and therefore registered the
backbone a second time, writing every tensor into the state dict twice. It
is a property now. Checkpoints already written with the duplicates still
load, since the real keys were always present under their own names.

## 10. Fine-tuned end-to-end, the decoder draws level with Jev

Paired on the same 450 items, exact McNemar. The encoder loses intent and
the injection gate; the decoder loses nothing.

| | encoder | decoder | Jev | decoder vs Jev |
|---|---|---|---|---|
| intent | 0.899 | **0.929** | 0.941 | level, p = 0.54 |
| multi-label exact set | 0.789 | **0.838** | 0.822 | level, p = 0.52 |
| `G_clinical` | 0.980 | 0.969 | 0.978 | level, p = 0.48 |
| `G_abusive` | 0.998 | 0.998 | 0.996 | level, p = 1.00 |
| `G_injection` | 0.978 | **0.984** | 0.996 | level, p = 0.13 |
| `G_pharmacy` | 0.947 | **0.956** | 0.880 | **OpenJev**, p = 4e-05 |
| `clinical_oblique` recall | **1.000** | 0.966 | 0.793 | level, p = 0.06 |

**Level on everything, ahead on one.** Under the encoder, intent was a Jev
win at p = 0.039 and the injection gate a Jev win at p = 0.008; both are
ties here. This is the closest OpenJev gets to the reference API, and it
needs 395 labelled examples and 134 seconds to get there.

And it is nearly free at inference. ModernBERT-base against Qwen3-1.7B,
batch size 1, ten questions and twenty-four options per state, H100:

| | encoder, 150M | decoder, 1,725M |
|---|---|---|
| p50 per state | **19.9 ms** | 22.4 ms |
| p95 per state | **20.3 ms** | 55.3 ms |
| throughput | **50.3/s** | 44.6/s |
| checkpoint | **0.6 GB** | 3.4 GB |

**11.5x the parameters for 13% more median latency.** The shared prefix is
why: everything happens in one forward pass over a short packed sequence, so
at this size the cost is dominated by launch overhead rather than by matrix
multiplies. The tail is the real price, p95 2.7x worse, which matters for a
router under a latency budget.

**We had this wrong.** An earlier version of this file called the decoder
not worth using, on the strength of zero-shot held-out transfer alone: same
suite mean as the encoder, one fewer task above chance, 11x the parameters.
That was the wrong evidence for the recommendation, because the path this
project recommends is fine-tuning, and no fine-tuned comparison had been run.
Latency had never been measured at all.

## 11. Encoder and decoder tie zero-shot and disagree on everything else

Both backbones trained on the same `mixture_ord2` with the same
augmentations, both landing at **17.6x chance**. The average hides the
result.

| held-out task | K | encoder | decoder + head |
|---|---|---|---|
| clinc_oos | 151 | 0.518 | **0.528** |
| banking77 | 77 | 0.278 | **0.343** |
| massive_intent | 60 | **0.298** | 0.185 |
| ag_news | 4 | 0.408 | **0.640** |
| sst5 (`score`) | 5 | **0.308** | 0.232 |
| civil_comments (`noul`) | 2 | 0.528 | 0.507 |
| helpsteer (`score`) | 5 | **0.222** | 0.185 |
| tasks above chance | | **5 of 7** | 4 of 7 |
| held-in control | | **1.4–2.3x** | 1.0–1.3x |

The encoder is the better all-rounder and clears chance on one more task, so
it is what we publish. The decoder is stronger where menus are large.

**The held-in row is the strange one.** The decoder sits near chance on
tasks it was *trained* on while transferring as well as the encoder to tasks
it was not. Its head is 4.2M parameters on a frozen 1.7B backbone, so most
of what it does at evaluation is the backbone's pretrained ability rather
than anything the head learned. The encoder, training end to end, actually
fits its mixture. Neither is wrong, but only the encoder's held-in number
does the job a control is there to do.

## 12. The contamination guard caught a real leak

`tasksource` contains most common benchmarks, not always under a recognisable
name. Verified with `scripts/verify_heldout_lineage.py`:

```
rotten_tomatoes (train)   77.0% of SST-5 test sentences, verbatim
glue/sst2 (validation)    75.4% of SST-5 dev
toxic_conversations      100.0% of its rows are Civil Comments rows
```

`rotten_tomatoes` is the dangerous one: a filter keyed on "sst" reports clean
while three quarters of the test set leaks. In the live build the guard
excluded 23 tasks including all of these.

---

# What did not work

## Warm-starting from a raw masked LM gives nothing

Reusing the pretrained MLM head at the option marker adds no parameters and
can be run untrained. ModernBERT-base scored **0.7x chance**, large **2.2x**,
with a literal 0.0000 on both high-cardinality tasks. A strong bidirectional
encoder plus a clever output format does not produce a zero-shot decision
model. The format makes zero-shot *possible*; it does not make it *present*.

## A 61-task mixture transfers nothing

Held-in control 0.68–0.84 on trained tasks, held-out **0.9x chance** — no
better than no training. At the time this looked like an architecture verdict.
It was not: the mixture was capped at 20 options. See result 2.

## Fixing the primitive rendering changed nothing

`score` levels were rendered `"0: very negative: the writer condemns..."` so
the model scored a bare index, and `noul` options as `"no: a majority would
not call this toxic"` — a yes/no question about a yes/no answer. Both were
genuinely wrong and both are fixed. Effect on results: `sst5` 0.270 → 0.220,
`civil_comments` unchanged. Correct, but not the cause.

## The binary held-out task was never a transfer failure

`civil_comments` sat at 0.500 accuracy with macro-F1 0.333 through every
run, which is a model answering the same way every time, and we spent two
separate efforts on it: augmenting `choice` into yes/no, then recovering 21
real `noul` tasks. Neither moved it. Both were aimed at the wrong thing.

| | AUROC | accuracy |
|---|---|---|
| ranking quality | **0.714** | — |
| at the 0.5 threshold | — | 0.515 |
| at a fitted threshold | — | **0.680** |

The model ranks toxic comments above clean ones perfectly respectably. It
puts 98% of its probability mass above 0.5, so argmax calls everything toxic
and accuracy lands on the class balance. **That is a calibration result, not
a transfer result**, and reporting only argmax accuracy on an uncalibrated
binary measures the threshold as much as the model. `eval_heldout.py` now
reports AUROC alongside accuracy for binary tasks so the two cannot be
confused again.

Two corrections to how we found this, both ours:

**A label inversion in the inference helper.** `openjev/infer.py` derived
option names with `list(criteria)`, which is *dict* order. A `noul` whose
criteria happened to be written `{"true": ..., "false": ...}` had its two
probabilities swapped, so `probabilities["true"]` read the "no" slot. That
turned AUROC 0.712 into 0.288 — exactly `1 - 0.712` — and looked like a model
ranking toxic comments as cleaner than clean ones. It affected the diagnostic
only; `eval_heldout.py` uses the canonical `option_labels` and its published
accuracies were never wrong. `infer.py` now imports that one definition, and
a test pins the two renderings position-for-position.

**A description finding we overstated mid-investigation.** With the
inversion in place, `noul` descriptions-as-options read as 0.369 against
0.712 for dropping them, which looked like the rubric actively inverting the
model. Corrected, it is roughly 0.63 against 0.712: dropping them helps,
modestly, and the dramatic version was the bug talking. The mechanism still
holds — a `noul`'s two descriptions necessarily restate one judgement from
opposite sides, so scoring both measures their overlap — and the options are
now bare polarity.

**Methodological caveat.** That rendering choice was made by comparing
variants on the held-out task, which is the wrong place to choose anything.
It is principled and small, but it should be re-settled on held-in binaries
before the held-out number is quoted as clean.

## Ordinal scale augmentation is a null result

The two held-out tasks still at chance are both non-`choice`:
`helpsteer_helpfulness` at 0.222, CI [0.188, 0.252] containing its 0.200
floor, and `civil_comments` at 0.528, CI [0.490, 0.572] containing 0.500.
Result 7 fixed the binary's degeneracy without moving its accuracy off the
floor. This is the ordinal half, and it failed outright.

Cause found by inspection: the
`mixture_big` contains **128 `choice` questions, 13 `noul`, and zero
`score`**. The
model never saw an ordinal question. Worse, ordinal data was present and being
flattened — `yelp_review_full` ships `['1 star' … '5 stars']` and was emitted
as an unordered menu.

The builder now detects ordinal label sets and emits `Score`, which took the
mixture from 0 to 9 ordinal tasks. Six of those nine are the same
three-level negative/neutral/positive scale, against a held-out task wanting
a five-level helpfulness judgement, so the obvious next move was to
synthesize the missing diversity the way label-space augmentation did for
cardinality: merge adjacent levels and remap the gold, and restate a scale
in another vocabulary of the same length within its semantic family.

**It does nothing.** Ablated at `--scale-prob 0` against 0.5, everything
else held fixed:

| | `--scale-prob 0` | `--scale-prob 0.5` |
|---|---|---|
| helpsteer | 0.110 (0.6x) | 0.128 (0.6x) |
| sst5 | 0.342 (1.7x) | 0.348 (1.7x) |
| mean over the suite | **16.5x** | 15.7x |

Both ordinal tasks land on the same multiple of chance either way, and the
suite mean is slightly *worse* with the augmentation. The mechanism is
correct and tested — gold stays on the right level, merges stay contiguous
and monotone — it simply is not the bottleneck. The gain from 9.7x to ~16x
came from rebuilding the mixture, not from this.

It stays in the tree behind a flag that defaults to off, because the
hypothesis is worth retesting once there is real ordinal data to train on.
`helpsteer` sitting *below* its own floor in both arms, at 0.6x with gold
skewed high (58/122/140/140/140), points at something systematic rather
than at scarcity.

## The `G_pharmacy` gate stays the weakest of the four

The slice is 427 positive to 23 negative, and it still over-fires: FPR
0.652, down from 0.826 from scratch and 0.783 under the first general
checkpoint. Good enough to beat Jev on the gate overall (0.947 against
0.880, p = 3e-04) but not a solved gate, and 23 negatives is not enough data
to fix one properly.

## A partially loaded model reported a plausible number

The first decoder evaluation returned 5.8x chance, which would have read as
"training the head made transfer worse" — a publishable-sounding negative
result. It was wrong. `eval_heldout.py` built `OpenJevDecoder` without
`learned_head=True`, so `load_state_dict(strict=False)` dropped all six
scorer tensors and scored the untrained readout. Loading the head correctly
gives 16.2x on the same checkpoint.

One guard caught it. The harness sanity task fell to 0.333, exactly chance,
and printed its warning. The mismatch itself printed only a warning and the
script carried on to produce a full, well-formatted results table.

We first also credited the held-in control, which sat at 1.0–1.2x on tasks
the model had been trained on. That reasoning was wrong: result 8 shows the
decoder's held-in control reads 1.0–1.3x when it loads perfectly, because a
4.2M head on a frozen backbone barely fits its own training mixture. The
control would not have caught this, and on the decoder path it cannot.

Both loaders now raise instead of warning, because a warning above a complete
table gets read as a footnote. Checkpoints also record the backbone, whether
the head exists, and the preamble and option template, since the same weights
scored under a different prompt are a different system. A separate guard
refuses to overwrite a checkpoint of a different architecture, after a
decoder run silently replaced the encoder checkpoint that result 1 depends on.

## Autocast hid a dtype bug until evaluation

The decoder head is float32 on a bf16 backbone. Under `torch.autocast` that
mixes silently, so a 58-minute training run completed normally and then
every held-out task failed at evaluation with `expected scalar type BFloat16
but found Float`. The weights were fine and the run did not need repeating,
but the failure landed after the expensive part rather than in the first
second.

The head now casts to its own dtype explicitly instead of relying on an
ambient autocast context.

## Noise robustness is a dead end

Paired on the same 130 items, clean and degraded both score 121/130, 124 of
130 get an identical predicted label, McNemar p = 1.0. Only ASR-style term
substitution bites. **Do not spend augmentation budget here.**

---

# Claims we retracted

Kept deliberately, because a results file that only grows in one direction is
not evidence.

| claim | why it was wrong |
|---|---|
| Banking77 zero-shot 0.325 (25x chance) | measured at n=40; at n=200 it is 0.205 |
| "OpenJev beats Jev at oblique clinical detection" | 5-0 on items but p = 0.0625 at n=29 |
| `G_pharmacy` 0.942 vs Jev's 0.880, our best gate | class imbalance; our FPR is 0.783 |
| "`G_pharmacy` cannot be rescued by any checkpoint" | a better one took FPR to 0.652 and the gate to a win |
| "The held-in control would have caught the dropped head" | the decoder reads 1.0–1.3x there even when it loads |
| "The encoder cannot generalize" | true of a 61-task mixture, not the architecture |
| "The warm start is the cheap lever" | it gives nothing; the data stage does the work |
| Precision recoverable by inverting `confidence` | real but ≤0.018 and no extra threshold granularity |

## A tier that cannot prove its own result

`clinical_oblique` recall is 0.966 against Jev's 0.793, winning 5 items and
losing 0. That is p = 0.0625. **With n = 29 and zero losses, 5 wins is the best
achievable outcome** — the exact test floors at 2 × 2⁻⁵. Two independently
trained checkpoints have won this tier cleanly, 4-0 and 5-0, and neither is
significant. The tier that motivated the safety argument is too small to
settle it. The fix is more items, not more training.

---

# Not yet measured

- Decoder calibration head on the mixture (running).
- Whether ordinal tasks lift `helpsteer` (builder fixed, not retrained).
- Latency served the same way as Jev. Our 9.6 ms is local compute; Jev's
  ~400 ms included a 331 ms round trip from India.
- Expected cost per decision with real per-class prices, which is the metric
  that should actually decide deployment.

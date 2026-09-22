# Ablation registry

Every experiment we ran, what it isolated, and what it decided. One line
each. This is an index, not an analysis: follow the pointers for the
reasoning.

**Updated in place, not appended to.** When a queued experiment lands its
row is filled in rather than a new row added underneath, so the file always
reads as the current state of knowledge.

Terminology is defined once, in `report/sections/glossary.tex`, which
assumes no background. Numbers with confidence intervals and commands are
in [results](results.md).

## What is being compared

**Backbones.** Decoder is the default: `Qwen/Qwen3-1.7B` with rank-16
LoRA. Encoder is the low-latency alternative:
`answerdotai/ModernBERT-base`. Details in
[architecture](architecture.md#backbones).

**Reference system.** `jev-1.13.0`, a commercial typed-decision API,
measured 20 September 2026 through black-box probing.

**Two studies, which answer different questions, and which disagree.**

- **Study G, general capability.** Train on the 279-task mixture, measure
  on seven public tasks never trained on. Does it work on a schema you have
  no labels for? This is the harder claim and the reference system's main
  advantage.
- **Study S, specific use case.** Train on 395 examples of one healthcare
  routing task, measure on its 450-item test split. Given labels for your
  actual problem, how good can it get?

Study S is won. Study G is not: Jev still leads Banking77 zero-shot by 21
points.

**Datasets.**

| | rows | role |
|--|--|--|
| 279-task mixture | 279 tasks, 323,466 | training for Study G, rebuild with `scripts/build_mixture.py` |
| [`tasks/heldout`](../tasks/heldout) | 7 tasks, 4,200 | evaluation for Study G, never trained on |
| [`tasks/healthcare_router`](../tasks/healthcare_router) | 395 train / 450 test | Study S, both ends |

## The decision this registry exists to make

One architecture, chosen on evidence rather than on the order we tried
things. The candidate axes, and what each turned out to be worth:

| axis | separates encoder from decoder? |
|--|--|
| latency | **no.** Merged LoRA runs at 1.07x the encoder |
| model size | **no.** 1.7B fits a consumer GPU |
| fine-tuned accuracy | **yes, decoder.** 0.979 against 0.899 |
| zero-shot transfer | **yes, decoder, decisively.** 29.6x against 17.2x |

**Decided at G-C3: ship the LoRA decoder.** It wins the axis that
separated the two and wins it by a wide margin. The encoder stays
documented as the choice when p95 latency is the binding constraint (20 ms
against 56 ms) or when a 0.6 GB footprint matters more than the accuracy.

The registry named three possible outcomes before G-C3 ran, so the choice
could not be rationalised afterwards. Outcome 1 happened: the LoRA decoder
clearly wins zero-shot, so the decoder ships and the encoder is documented
as the tight-latency alternative. We had recorded outcome 3, that neither
transfers well enough to be useful zero-shot, as the most likely. That
prediction was wrong on the benchmark suite and right on the router, which
is the substance of G-F1b below.

## Study G: general capability

### G-A. Does the architecture generalize on its own?

| id | question | result | verdict |
|--|--|--|--|
| G-A1 | Pretrained masked-LM head at the marker, untrained | 0.7x chance (base), 2.2x (large) | **negative** |
| G-A2 | Causal backbone, state and options concatenated plainly | 0.8x chance | **negative** |
| G-A3 | Each option framed as an explicit yes/no question | 2.9x chance | positive |
| G-A4 | G-A3 plus a preamble stating the task | 11.5x chance | **positive** |
| G-A5 | Control: shorten the framing to `{opt}\ncorrect? answer` | halves the mean | **negative control** |

**Settled.** Zero-shot ability is not a property of the architecture. G-A5
is the control that matters: it shows the gain is the format doing work,
not an incidental wording choice.

### G-B. What in the training data produces transfer?

| id | question | result | verdict |
|--|--|--|--|
| G-B1 | 61-task mixture | 0.9x chance, held-in 0.68 to 0.84 | **negative** |
| G-B2 | Label-space augmentation | 0.000 to 0.205 at K=77 | **positive, large** |
| G-B3 | Mixture 61 to 141 tasks | ran together with G-B2 | **confounded** |
| G-B4 | Fix `score` and `noul` option rendering | sst5 0.270 to 0.220 | **null** |
| G-B5 | Detect ordinal label sets, emit `score` (0 to 9 tasks) | enabled G-B6 | prerequisite |
| G-B6 | Ordinal scale augmentation: coarsen and reword | 0.6x either way; mean 16.5x to 15.7x | **null** |
| G-B7 | Recast `choice` into yes/no during training | civil macro-F1 0.333 to 0.403 | partial |
| G-B8 | Retype 21 real negation-pair tasks to `noul` | mean 17.6x to 15.3x | **negative** |
| G-B9 | Add 2 real ordinal star-rating datasets | mean to 13.4x | **negative** |
| G-B10 | Ingest the MultipleChoice family, 279 tasks, audited clean | all 7 tasks clear chance at 17.2x | **positive, large** |

**G-B8 and G-B9 are the instructive failures.** Both added one primitive
by removing or diluting `choice` tasks, and five of seven held-out tasks
are `choice`. Banking77 tracked the choice-task count exactly: 0.278 at 121
tasks, 0.187 at 100, 0.138 at 100 with dilution. That is negative transfer,
and we caused it twice before measuring it. Detail in
`report/sections/ablations.tex`.

### G-C. Decoder against encoder

| id | question | result | verdict |
|--|--|--|--|
| G-C1 | Both on the same mixture, zero-shot | 17.6x each; 5/7 against 4/7 above chance | tie on the mean |
| G-C2 | Was G-C1 fair? | no: encoder fully trained, decoder a 4.2M frozen head | **invalid** |
| G-C3 | **LoRA decoder on the mixture** | **29.6x mean, all 7 tasks beat chance and majority; banking77 0.343 to 0.605** | **decided: ship the decoder** |
| G-C4 | Latency, batch 1, ten questions | 19.9 / 22.4 ms p50 | near parity at p50 |
| G-C5 | Merged against unmerged adapter | 1.96x unmerged, 1.07x merged | **always merge** |
| G-C6 | Does latency grow with question count? | 1q 69 ms, 10q 74 ms | flat, as the reference is |

## Study S: specific use case

| id | question | result | verdict |
|--|--|--|--|
| S-D1 | Encoder from scratch against from a general checkpoint | 0.544 to 0.899 intent, p = 6e-18 | **positive, large** |
| S-D2 | Decoder, all 1,725M parameters open | 0.929 intent, 3.4 GB | dominated |
| S-D3 | Decoder, 4.2M head only | 0.666 intent, 17 MB | **negative** |
| S-D4 | Decoder, LoRA r=16 | **0.979 intent, 87 MB** | **best** |
| S-D5 | Did S-D4 need a general checkpoint? | no, trained from base Qwen | **surprising** |
| S-D6 | Task adapter initialised from the general adapter | 0.953 against 0.979, p = 0.012 | **negative** |
| S-D7 | LoRA on the encoder | not planned: 150M full fine-tune is already 0.6 GB and 38 s | **not planned** |

**S-D5 reframes the project.** The best router used no mixture training at
all. The general checkpoint earns its keep on schemas you have no labels
for, not on the task you actually care about.

**S-D6 is the reverse of S-D1**, and the contradiction is the interesting
part. Two-stage fine-tuning is worth it when your base model cannot do the
task form at all; once it can, train the task adapter directly from base.
Full result in
[results](results.md#3-starting-a-task-adapter-from-the-general-adapter-is-worse)
and `report/sections/ablations.tex`.

Paired against Jev, S-D4 on the same 450 items: three wins, six level, no
losses. Table in [results](results.md#router-against-jev-fine-tuned).

## Measurement itself

| id | question | result | verdict |
|--|--|--|--|
| G-F1 | General encoder checkpoint zero-shot on the router | intent 0.601 against 0.941; `G_clinical` recall **0.111** | **benchmark transfer is not deployment-ready** |
| G-F1b | Same check on the general LoRA decoder | intent 0.467, below the encoder's 0.601 | **the benchmark ranking reverses** |
| M-E1 | Does argmax accuracy describe a binary task? | AUROC 0.714 against accuracy 0.515 | **no, report both** |
| M-E2 | Do `noul` option descriptions help? | dropping them: 0.63 to 0.712 AUROC | **they hurt here** |
| M-E3 | Do `choice` descriptions help? | +5 points if discriminative, nothing if restating the label (p = 0.75) | content-dependent |
| M-E4 | Is the contamination guard real? | rejects an injected `banking77`; caught a real leak | **verified** |
| M-E5 | Does every example pack before training? | 323,466 checked in 227 s | **verified** |
| M-E6 | Were the held-out multiples scored on full menus? | no: 64 of 151 options shown at `--max-len 2048` | **withdrawn and corrected** |

**G-F1b is the result that most constrains how the headline should be
read.** The decoder wins every one of seven public benchmarks and loses the
one task with an operational shape. Neither backbone is deployable
zero-shot, so this does not change the architecture call, but it does mean
nobody should read 29.6x as a statement about a system anyone would ship
unsupervised. See
[results](results.md#2-the-decoder-transfers-better-and-deploys-worse).

**M-E6 forced a withdrawal** of four published numbers. The packer dropped
options to fit `max_len` while the multiple of chance was still divided by
1/K. See
[results](results.md#1-we-had-been-scoring-truncated-menus).

## Still queued

| id | question | why it matters |
|--|--|--|
| G-F2 | ModernBERT-large | is the encoder gap capacity or data? |
| G-F3 | Few-shot demonstrations in the preamble | untested lever on the decoder path |
| S-D8 | `m` independent single-question classifiers | the obvious baseline to packing, `scripts/baseline_separate.py` |

## Artifacts

**Base models** (not ours, linked for reference)

| | |
|--|--|
| decoder backbone, default | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| encoder backbone, alternative | [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) |
| encoder backbone, larger, queued for G-F2 | [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) |

**Trained by us**

| artifact | experiment | size | link |
|--|--|--|--|
| **LoRA router adapter** | **S-D4** | **87 MB** | [`openjev-router-lora`](https://huggingface.co/s1lv3rj1nx/openjev-router-lora) |
| LoRA general adapter | G-C3 | ~90 MB | *queued, will link* |
| encoder general checkpoint | G-B2/G-B3 | 0.6 GB | [`openjev-encoder-general`](https://huggingface.co/s1lv3rj1nx/openjev-encoder-general) |
| encoder router specialist | S-D1 | 0.6 GB | [`openjev-router-healthcare`](https://huggingface.co/s1lv3rj1nx/openjev-router-healthcare) |
| encoder on the 279-task mixture | G-B10 | 0.6 GB | *trained, publishing* |
| decoder full fine-tune | S-D2 | 3.4 GB | *not published: dominated by S-D4* |
| decoder head-only | S-D3 | 17 MB | *not published: negative result* |

**Datasets**

| dataset | role | link |
|--|--|--|
| 279-task training mixture | Study G training | [`openjev-mixture`](https://huggingface.co/datasets/s1lv3rj1nx/openjev-mixture) |
| held-out suite | Study G evaluation | [`openjev-heldout`](https://huggingface.co/datasets/s1lv3rj1nx/openjev-heldout) |
| healthcare router | Study S, both ends | [`openjev-healthcare-router`](https://huggingface.co/datasets/s1lv3rj1nx/openjev-healthcare-router) |

Sources the mixture is assembled from are listed per task in its
`description`, and every held-out task names its aliases in `holdout_of`.

## G-C6: training on bigger menus did not help

The shipped adapter was trained at `--max-len 2048`, which clipped
label-space augmentation to about 80 options while the held-out suite
presents up to 151. That is a real train-and-test mismatch, and
label-space augmentation is the largest single effect this project has
measured, so extending its range to cover the evaluation was the obvious
next move.

Retrained at `--max-len 4096` with `--max-options 160`, which produces
menus up to 175. Everything else held: same mixture, same rank, same
learning rate, same backbone. Evaluated at full menus.

| task | shipped, menus to 80 | retrained, menus to 175 | |
|--|--|--|--|
| `banking77` | 0.605 | 0.663 | level |
| `clinc_oos` | 0.702 | 0.660 | level |
| `massive_intent` | 0.775 | **0.670** | **worse** |
| `ag_news` | 0.793 | 0.808 | level |
| `sst5` | 0.465 | 0.435 | level |
| `civil_comments` | 0.742 | 0.682 | level |
| `helpsteer` | 0.273 | 0.290 | level |
| **mean x chance** | **29.6x** | 28.5x | |

**Six level, one worse, none better.** The mean moved the wrong way. We
keep the shipped adapter.

The one result pointing the predicted way is `banking77`, up 5.8 points,
the largest single move in the table and in exactly the direction the
hypothesis called for. It is not significant on its own and we are not
going to promote it by ignoring the six numbers around it, one of which
is a clear ten-point loss.

**What we think happened.** Menu size was not the binding constraint. The
model already reached 106x chance on a 151-way menu while having trained
on nothing larger than 80, so whatever it learned about reading a menu
generalised past the sizes it saw. Spending capacity on larger menus
appears to have cost something elsewhere: `massive_intent`, at K=60, is
the task most likely to be crowded out by padding every menu towards 160
distractors.

**What the run was still worth.** Training is 2.2x faster on the same
hardware, because the token-budget sampler took the GPU from 39%
utilisation and 205W to 100% and 397W. Augmentation is now seeded per
example index, so a run is reproducible row by row where the old shared
random stream made an example's menu depend on what had been drawn before
it. And training now aborts on a trivial-task check and on an
out-of-memory skip rate above 5%, both of which fired during development
and would have saved us a seven hour run had they existed earlier.

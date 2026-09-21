# Ablation registry

A living record of every experiment: what it isolates, what it measured,
and what it decided. **Updated in place, not appended to** — when a queued
experiment lands, its row is filled in rather than a new row added
underneath, so the file always reads as the current state of knowledge.

Written to be the source for a technical report, so every term is defined
before it is used.

---

## 1. Terminology

**State.** The input being judged: a support message, a review, a comment.
One state, many questions.

**Question.** One decision to make about the state. Each has instructions
and a set of options.

**Primitive.** The type of a question. Three exist:

- **`choice`** — pick exactly one of K options. Routing, intent.
- **`score`** — pick one level on an *ordered* scale. Severity, rating.
- **`noul`** — yes or no. Flags and gates. Several `noul`s asked together
  give multi-label behaviour, which a single `choice` cannot express.

**Criteria / options / menu.** Interchangeable here: the candidate answers
for a question. For `choice` a label plus an optional description, for
`score` an ordered list of levels, for `noul` implicitly yes/no.

**Per-example criteria.** A menu that differs per row, as in a reading
comprehension task where each question has its own answer options. Contrast
with a *fixed menu*, shared by every row of a task.

**K.** The number of options in a menu. K=151 means a 151-way choice.

**Packing.** Writing the state once, then every question and every option
after it, as a single token sequence for one forward pass.

**Shared prefix.** The packed state. Encoded once and attended to by all
questions, which is why asking ten questions costs about what one does.

**Marker.** A token placed at each option's position. The model's output is
read at these positions rather than generated.

**Block-diagonal attention.** A mask letting each question attend to the
shared prefix and to itself, but not to other questions. Without it, an
answer would depend on which other questions happened to be asked.

**Grouped log-softmax.** Normalising scores within each question's own
options, when questions in one packed sequence have different K.

**Backbone.** The pretrained model underneath: an *encoder* (ModernBERT,
bidirectional) or a *decoder* (Qwen3, causal). Note that "decoder" here
describes the attention pattern, not the use: nothing is generated
token-by-token.

**General checkpoint.** Weights trained on the multi-task mixture, meant to
be good at typed decisions in general. The thing a user downloads.

**Task fine-tune.** Further training on one specific use case, from a
general checkpoint or from the raw backbone.

**Held-out schema.** A task whose labels and option set the model has never
trained on. The measure of real generalization.

**Held-in control.** Accuracy on tasks the model *was* trained on, reported
beside held-out accuracy. If held-out is at chance and held-in is high, the
model genuinely fails to transfer. If both are at chance, it is a bug.

**Multiple of chance.** Accuracy divided by 1/K. Comparable across tasks
with very different K, where raw accuracy is not: 0.28 on a 77-way menu is
strong, 0.28 on a 4-way menu is worse than guessing.

**Label-space augmentation.** Padding a training menu with labels borrowed
from other tasks. The gold answer is unchanged, so the example stays valid,
but the model must discriminate against a large menu.

**Distractor.** One of those borrowed labels.

**LoRA (Low-Rank Adaptation).** Training two small matrices alongside each
frozen weight matrix instead of updating it. **Rank (r)** sets their size;
r=16 here. The result is an **adapter**, tens of megabytes rather than
gigabytes.

**Merging.** Folding a trained adapter back into the base weights, so
inference costs no extra matrix multiplies. Unmerged adapters are ~2x
slower.

**Contamination guard.** A check that no held-out dataset appears in the
training mixture under any alias, run before every training run.

**Negative transfer.** When training on an additional task makes the target
worse rather than better.

**McNemar test.** A paired significance test for two systems on the same
items. Counts where they disagree (**b10**: we are right and they are not;
**b01**: the reverse) and asks whether the split is lopsided. Aggregate
accuracy alone cannot tell a real difference from two models disagreeing
equally in both directions.

**AUROC.** Ranking quality, independent of any threshold. Reported for
binary tasks because argmax accuracy there describes the 0.5 threshold as
much as the model.

---

## 2. What is being compared

**Backbones.**

| | encoder | decoder |
|---|---|---|
| model | [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| parameters | 150M | 1,725M |
| attention | bidirectional | causal, block-diagonal per question |
| readout | linear scorer at the marker | `logit(yes) − logit(no)` at the marker |

**Reference system.** `jev-1.13.0`, a commercial typed-decision API,
measured 20 September 2026 through black-box probing.

**Two studies, which answer different questions.**

- **Study G, general capability.** Train on a 279-task mixture, measure on
  seven public tasks never trained on. Asks: does it work on a schema you
  have no labels for? This is the harder claim and the reference system's
  main advantage.
- **Study S, specific use case.** Train on 395 examples of one healthcare
  routing task, measure on its 450-item test split. Asks: given labels for
  your actual problem, how good can it get?

They can disagree, and they do. Study S is won; Study G is not.

**Datasets.**

| | rows | role |
|---|---|---|
| `tasks/mixture_final` | 279 tasks, 323,466 | training for Study G |
| `tasks/heldout` | 7 tasks, 4,200 | evaluation for Study G, never trained on |
| `tasks/healthcare_router` | 395 train / 450 test | Study S, both ends |

### Artifacts

Everything measured below, published or explicitly marked as not yet.

**Base models** (not ours, linked for reference)

| | |
|---|---|
| encoder backbone | [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) |
| decoder backbone | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| encoder backbone, larger, queued for G-F2 | [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) |

**Trained by us**

| artifact | experiment | size | link |
|---|---|---|---|
| encoder general checkpoint | G-B2/G-B3 | 0.6 GB | [`openjev-encoder-general`](https://huggingface.co/s1lv3rj1nx/openjev-encoder-general) |
| encoder router specialist | S-D1 | 0.6 GB | [`openjev-router-healthcare`](https://huggingface.co/s1lv3rj1nx/openjev-router-healthcare) |
| **LoRA router adapter** | **S-D4** | **87 MB** | [`openjev-router-lora`](https://huggingface.co/s1lv3rj1nx/openjev-router-lora) |
| encoder on the 279-task mixture | G-B10 | 0.6 GB | *running, will link* |
| LoRA general adapter | G-C3 | ~90 MB | *queued, will link* |
| decoder full fine-tune | S-D2 | 3.4 GB | *not published: dominated by S-D4* |
| decoder head-only | S-D3 | 17 MB | *not published: negative result* |

**Datasets**

| dataset | role | link |
|---|---|---|
| held-out suite | Study G evaluation | [`openjev-heldout`](https://huggingface.co/datasets/s1lv3rj1nx/openjev-heldout) |
| healthcare router | Study S, both ends | [`openjev-healthcare-router`](https://huggingface.co/datasets/s1lv3rj1nx/openjev-healthcare-router) |
| 279-task training mixture | Study G training | [`openjev-mixture`](https://huggingface.co/datasets/s1lv3rj1nx/openjev-mixture) |

Sources the mixture is assembled from are listed per task in its
`description`, and every held-out task names its aliases in `holdout_of`.

---

## 3. The decision this registry exists to make

One architecture, chosen on evidence rather than on the order we tried
things. The candidate axes, and what each turned out to be worth:

| axis | separates encoder from decoder? |
|---|---|
| latency | **no.** Merged LoRA runs at 1.07x the encoder |
| model size | **no.** 1.7B fits a consumer GPU |
| fine-tuned accuracy | **yes, decoder.** 0.979 against 0.899 |
| zero-shot transfer | **unresolved. This decides it** |

Experiment **G-C3** measures the last row. The three possible outcomes and
what ships under each are written in section 7, in advance.

---

## 4. Study G: general capability

### G-A. Does the architecture generalize on its own?

| id | question | result | verdict |
|---|---|---|---|
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
|---|---|---|---|
| G-B1 | 61-task mixture | 0.9x chance, held-in 0.68–0.84 | **negative** |
| G-B2 | Label-space augmentation | 0.000 → 0.205 at K=77 | **positive, large** |
| G-B3 | Mixture 61 → 141 tasks | ran together with G-B2 | **confounded** |
| G-B4 | Fix `score`/`noul` option rendering | sst5 0.270 → 0.220 | **null** |
| G-B5 | Detect ordinal label sets, emit `score` (0 → 9 tasks) | enabled G-B6 | prerequisite |
| G-B6 | Ordinal scale augmentation: coarsen and reword | 0.6x either way; mean 16.5x → 15.7x | **null** |
| G-B7 | Recast `choice` into yes/no during training | civil macro-F1 0.333 → 0.403 | partial |
| G-B8 | Retype 21 real negation-pair tasks to `noul` | mean 17.6x → 15.3x | **negative** |
| G-B9 | Add 2 real ordinal star-rating datasets | mean → 13.4x | **negative** |
| G-B10 | Ingest the MultipleChoice family, 279 tasks total | running | **running** |

**G-B8 and G-B9 are the instructive failures.** Both added one primitive by
removing or diluting `choice` tasks, and five of seven held-out tasks are
`choice`. Banking77 tracked the choice-task count exactly: 0.278 at 121
tasks, 0.187 at 100, 0.138 at 100 with dilution. That is negative transfer,
and we caused it twice before measuring it.

### G-C. Encoder against decoder

| id | question | result | verdict |
|---|---|---|---|
| G-C1 | Both on the same mixture, zero-shot | 17.6x each; 5/7 vs 4/7 above chance | tie on the mean |
| G-C2 | Was G-C1 fair? | no: encoder fully trained, decoder a 4.2M frozen head | **invalid** |
| G-C3 | **LoRA decoder on the mixture** | **queued** | **decides the architecture** |
| G-C4 | Latency, batch 1, ten questions | 19.9 / 22.4 ms p50 | near parity |
| G-C5 | Merged against unmerged adapter | 1.96x unmerged, 1.07x merged | **always merge** |
| G-C6 | Does latency grow with question count? | 1q 69 ms, 10q 74 ms | flat, as the reference is |

---

## 5. Study S: specific use case

| id | question | result | verdict |
|---|---|---|---|
| S-D1 | Encoder from scratch vs from a general checkpoint | 0.544 → 0.899 intent, p = 6e-18 | **positive, large** |
| S-D2 | Decoder, all 1,725M parameters open | 0.929 intent, 3.4 GB | dominated |
| S-D3 | Decoder, 4.2M head only | 0.666 intent, 17 MB | **negative** |
| S-D4 | Decoder, LoRA r=16 | **0.979 intent, 87 MB** | **best** |
| S-D5 | Did S-D4 need a general checkpoint? | no, trained from base Qwen | **surprising** |
| S-D6 | Task adapter initialised from a general adapter | queued | **queued** |
| S-D7 | LoRA on the encoder | not planned: 150M full fine-tune is already 0.6 GB / 38 s | **not planned** |

**Against the reference system**, S-D4 paired on the same 450 items:

| | OpenJev | reference | p | winner |
|---|---|---|---|---|
| intent | 0.979 | 0.941 | 9.8e-04 | **OpenJev** |
| multi-label exact set | 0.909 | 0.822 | 7.2e-06 | **OpenJev** |
| scope gate | 0.978 | 0.880 | 3.9e-10 | **OpenJev** |
| clinical / abusive / injection gates | — | — | ≥ 0.34 | level |

**S-D5 reframes the project.** The best router used no mixture training at
all. The general checkpoint earns its keep on schemas you have no labels
for, not on the task you actually care about.

---

## 6. Measurement itself

| id | question | result | verdict |
|---|---|---|---|
| M-E1 | Does argmax accuracy describe a binary task? | AUROC 0.714 against accuracy 0.515 | **no, report both** |
| M-E2 | Do `noul` option descriptions help? | dropping them: 0.63 → 0.712 AUROC | **they hurt here** |
| M-E3 | Do `choice` descriptions help? | +5 points if discriminative, nothing if restating the label (p = 0.75) | content-dependent |
| M-E4 | Is the contamination guard real? | rejects an injected `banking77`; caught a real leak | **verified** |
| M-E5 | Does every example pack before training? | 323,466 checked in 227 s | **verified** |

---

## 7. Queued, and how the decision gets made

| id | question | why it matters |
|---|---|---|
| G-C3 | LoRA decoder on the mixture | the deciding experiment |
| S-D6 | Task adapter from a general adapter | the LoRA form of S-D1's +36 points |
| G-F1 | General checkpoint zero-shot on the router | transfer on a task someone cares about |
| G-F2 | ModernBERT-large | is the encoder gap capacity or data? |
| G-F3 | Few-shot demonstrations in the preamble | untested lever on the decoder path |

When G-C3 lands, one of three things is true:

1. **The LoRA decoder clearly wins zero-shot.** Ship the decoder; keep the
   encoder documented as the tight-latency alternative.
2. **They are close.** Ship the encoder: a tenth the size, 38-second
   training, tighter p95.
3. **Neither transfers well enough to be useful zero-shot.** Ship the
   fine-tune-first framing, recommend the LoRA decoder for accuracy, and
   say plainly that zero-shot needs a dataset we do not have.

Outcome 3 is currently the most likely. It is not a failure — the suite went
from 0.9x to 17.6x chance and the fine-tuned model beats the reference API
— but it is a different claim from the one we set out to make, and it will
be reported as such.

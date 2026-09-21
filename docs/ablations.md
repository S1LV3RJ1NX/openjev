# Ablation registry

Every experiment run, what it isolates, and what it decided. Kept so the
final architecture choice is made on evidence rather than on the order we
happened to try things in.

Status is one of **done**, **running**, **queued**, or **not planned**, and
a verdict of **null** or **negative** is as informative as a win.

---

## The question this is all for

One architecture, chosen on evidence. The axes that could decide it, and
what we now know about each:

| axis | does it separate encoder from decoder? |
|---|---|
| latency | **no**. Merged LoRA runs at 1.07x the encoder |
| model size | **no**. 1.7B fits a consumer GPU |
| fine-tuned accuracy | **yes, decoder**. 0.979 against 0.899, and it beats Jev |
| zero-shot transfer | **unresolved**, and this is the deciding experiment |

Everything below exists to settle the last row. Once it is settled we ship
one backbone and one recipe, and the technical report is written against
the registry rather than reconstructed from memory.

---

## A. Does the architecture alone generalize?

| id | question | result | verdict |
|---|---|---|---|
| A1 | Pretrained MLM head at the option marker, untrained | 0.7x chance (base), 2.2x (large) | **negative** |
| A2 | Same idea on a causal backbone, bare concatenation | 0.8x chance | **negative** |
| A3 | Causal backbone with each option framed as a yes/no question | 2.9x chance | positive |
| A4 | A3 plus a preamble stating the task | 11.5x chance | **positive** |
| A5 | Shorten the option framing to `{opt}\ncorrect? answer` | halves the mean | **negative control** |

**Settled.** Zero-shot ability is not a property of the architecture; a
strong encoder with a clever output format scores below chance. The format
makes zero-shot *possible* and A5 shows it is the format doing the work,
not the wording being incidental.

## B. What in the training data produces transfer?

| id | question | result | verdict |
|---|---|---|---|
| B1 | 61-task mixture | 0.9x chance, held-in 0.68–0.84 | **negative** |
| B2 | Label-space augmentation, menus padded with distractors | 0.000 → 0.205 on K=77 | **positive, large** |
| B3 | Mixture 61 → 141 tasks | confounded with B2, ran together | **unresolved** |
| B4 | Fix `score`/`noul` option rendering | sst5 0.270 → 0.220 | **null** |
| B5 | Detect ordinal label sets, emit `Score` (0 → 9 tasks) | enabled B6 | prerequisite |
| B6 | Ordinal scale augmentation: coarsen and reword | 0.6x either way; suite mean 16.5x → 15.7x | **null** |
| B7 | Recast `choice` into yes/no during training | civil macro-F1 0.333 → 0.403 | partial |
| B8 | Retype 21 real negation-pair tasks to `noul` | suite mean 17.6x → 15.3x | **negative** |
| B9 | Add 2 real ordinal star-rating datasets | suite mean → 13.4x | **negative** |
| B10 | Ingest the MultipleChoice family, 279 tasks total | running | **running** |

**B8 and B9 are the instructive failures.** Both converted or diluted
`choice` tasks, and five of seven held-out tasks are `choice`. Banking77
tracked the choice-task count exactly: 0.278 at 121 tasks, 0.187 at 100,
0.138 at 100 with dilution. Adding a primitive by *removing* another is a
trade, not an improvement, and we made it twice before measuring it.

## C. Encoder against decoder

| id | question | result | verdict |
|---|---|---|---|
| C1 | Both on the same mixture, zero-shot | 17.6x each; 5/7 vs 4/7 above chance | **tie on the mean** |
| C2 | Is C1 a fair comparison? | no: encoder fully trained, decoder a 4.2M frozen head | **invalid** |
| C3 | LoRA decoder on the mixture | **queued, decides the architecture** | **queued** |
| C4 | Latency, batch 1, ten questions | 19.9 / 22.4 ms p50 | near parity |
| C5 | LoRA merged against unmerged | 1.96x unmerged, 1.07x merged | **merge always** |
| C6 | Does latency grow with question count? | 1q 69 ms, 10q 74 ms | flat, as Jev is |

## D. Fine-tuning recipe

| id | question | result | verdict |
|---|---|---|---|
| D1 | Encoder from scratch against from a general checkpoint | 0.544 → 0.899 intent, p = 6e-18 | **positive, large** |
| D2 | Decoder, everything open | 0.929 intent, 3.4 GB | dominated |
| D3 | Decoder, head only | 0.666 intent, 17 MB | **negative** |
| D4 | Decoder, LoRA r=16 | **0.979 intent, 87 MB, beats Jev** | **best** |
| D5 | Did D4 need a general checkpoint? | no, trained from base Qwen | **surprising** |
| D6 | Task adapter initialised from a general adapter | queued | **queued** |
| D7 | LoRA on the encoder | not planned: a 150M full fine-tune is already 0.6 GB and 38 s | **not planned** |

**D5 is the one that reframes the project.** The best router we have used
no mixture training at all. So the general checkpoint earns its keep on
schemas you have no labels for, not on the task you actually care about.

## E. Measurement itself

| id | question | result | verdict |
|---|---|---|---|
| E1 | Does argmax accuracy describe a binary task? | AUROC 0.714 against accuracy 0.515 | **no, report both** |
| E2 | Do `noul` option descriptions help? | dropping them: 0.63 → 0.712 AUROC | **descriptions hurt here** |
| E3 | Do `choice` descriptions help? | +5 points if discriminative, nothing if restating the label (p = 0.75) | **content-dependent** |
| E4 | Is the contamination guard real? | rejects an injected `banking77`; caught a real leak | **verified** |
| E5 | Does every example pack before training? | 323,466 checked in 227 s | **verified** |

## F. Queued

| id | question | why it matters |
|---|---|---|
| C3 | LoRA decoder on the mixture | the deciding experiment |
| D6 | Task adapter from a general adapter | the LoRA form of D1's +36 points |
| F1 | General checkpoint zero-shot on the router | transfer on a task someone cares about, not a benchmark |
| F2 | ModernBERT-large | is the encoder gap capacity or data? |
| F3 | Few-shot demonstrations in the preamble | untested lever on the decoder path |

---

## How the architecture gets decided

When C3 lands, one of three things is true:

1. **LoRA decoder clearly wins zero-shot.** Ship the decoder, drop the
   encoder to a documented alternative for tight latency budgets.
2. **They are close.** Ship the encoder, since it is a tenth the size,
   trains in 38 seconds, and has a tighter p95.
3. **Neither transfers well enough to be useful zero-shot.** Ship the
   fine-tune-first framing honestly, recommend the LoRA decoder for
   accuracy, and say plainly that zero-shot needs a dataset we do not have.

Outcome 3 is currently the most likely, and it is not a failure: the suite
went from 0.9x to 17.6x chance and the fine-tuned model beats the reference
API. It just is not the same claim.

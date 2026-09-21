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
| CLINC-150 (K=151), never seen it | 0.420 | not measured | 63x chance |
| Router intent | 0.817 | **0.941** | Jev, p = 1e-07 |
| Router multi-label exact set | 0.718 | **0.822** | Jev, p = 3e-05 |
| `G_clinical` correctness | 0.971 | 0.978 | **level**, p = 0.63 |
| Probability precision | full | 0.01 grid, 71.9% hard zeros | **OpenJev** |
| Deterministic | yes | no, no seed | **OpenJev** |

---

# What worked

## 1. The general checkpoint is worth +27 points to a specialist

**The project's central claim, and the strongest result here.** Same router
task, same 395 training examples, same 6 epochs, same 38 seconds. Only the
starting weights differ.

```bash
uv run python scripts/train.py --task tasks/healthcare_router \
    --init-from checkpoints/mixture_big/model.pt --epochs 6 --bs 8
```

| | from raw ModernBERT | from general checkpoint | Jev |
|---|---|---|---|
| intent (lenient / strict) | 0.544 | **0.817** | 0.909 / 0.941 |
| multi-label exact set | 0.453 | **0.718** | 0.822 |
| multi-label F1 | 0.554 | **0.816** | 0.890 |
| `G_clinical` recall | 0.898 | **0.991** | 0.926 |
| `clinical_oblique` recall | 0.931 | **0.966** | 0.793 |
| `G_abusive` recall | 0.000 | **0.429** | 0.714 |
| `G_injection` recall | 0.125 | **0.750** | 0.917 |

Paired McNemar against from-scratch: intent **b10=108, b01=16, p = 6.0e-18**;
multi-label **b10=144, b01=25, p = 1.6e-21**.

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

## 7. The contamination guard caught a real leak

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

## `score` and `noul` stay at chance, and we now know why

`helpsteer_helpfulness` is the one held-out task still at chance (0.230, CI
[0.170, 0.290] containing its 0.200 floor). Cause found by inspection: the
mixture contains **128 `choice` questions, 13 `noul`, and zero `score`**. The
model never saw an ordinal question. Worse, ordinal data was present and being
flattened — `yelp_review_full` ships `['1 star' … '5 stars']` and was emitted
as an unordered menu.

The builder now detects ordinal label sets and emits `Score`. Whether that
lifts `helpsteer` is **untested**.

## The `G_pharmacy` gate is degenerate and cannot be rescued

FPR **0.783** after the general init, essentially unchanged from 0.826. It
predicts positive on almost everything because the slice is 427 positive to 23
negative. No starting checkpoint fixes a gate with 23 negative examples.

## A partially loaded model reported a plausible number

The first decoder evaluation returned 5.8x chance, which would have read as
"training the head made transfer worse" — a publishable-sounding negative
result. It was wrong. `eval_heldout.py` built `OpenJevDecoder` without
`learned_head=True`, so `load_state_dict(strict=False)` dropped all six
scorer tensors and scored the untrained readout. Loading the head correctly
gives 16.2x on the same checkpoint.

Two guards caught it and one nearly did not. The harness sanity task fell to
0.333, exactly chance, and printed its warning. The held-in control sat at
1.0–1.2x on tasks the model had been trained on, which cannot happen if the
model loaded. But the mismatch itself printed only a warning and the script
carried on to produce a full, well-formatted results table.

Both loaders now raise instead of warning, because a warning above a complete
table gets read as a footnote. Checkpoints also record the backbone, whether
the head exists, and the preamble and option template, since the same weights
scored under a different prompt are a different system. A separate guard
refuses to overwrite a checkpoint of a different architecture, after a
decoder run silently replaced the encoder checkpoint that result 1 depends on.

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

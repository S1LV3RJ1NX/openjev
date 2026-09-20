# Evaluation

Two suites, answering two different questions. Both were built **before** the
model, on purpose: an evaluation written after you see your results is an
evaluation you tuned.

| suite | question | n |
|---|---|---|
| [`tasks/healthcare_router`](../tasks/healthcare_router) | do we match a production system on a task we trained for? | 988 (395/143/450) |
| [`tasks/heldout`](../tasks/heldout) | or did we just memorise it? | 4,200 across 7 tasks |

> **Provenance of the reference numbers.** Every figure attributed to Jev below
> was measured by us against the public TypeSafe API, model `jev-1.13.0`, on
> 20 September 2026, at the sample sizes stated. These are black-box
> measurements of a hosted endpoint, on one day, from one network location in
> India. They are a benchmark reference point, not a description of how that
> system is built — its architecture, size and training data are not public.
> Reproduce them before relying on them. **We publish no comparison in our own
> favour, because there is nothing of ours to compare yet.**

## The format

`openjev/schema.py` defines `Choice`, `Score` and `Noul`, a `state` that is
text or JSON, and answers keyed by question id. On disk a task is a directory
with `task.json` and jsonl splits, so contributing a dataset means writing
files rather than writing code.

`Task.validate()` rejects unknown question ids, out-of-range score levels,
non-bool nouls, sub-2-option choices, and menus over 255 options.

`Answer.confidence` and `Answer.ordinal_confidence` reimplement the two
confidence formulas the reference endpoint uses, so both systems get scored by
the same number rather than each by its own notion of confidence.

## The metric suite

`openjev/metrics.py`, written once so every comparison uses the same code.

**Brier, not log loss.** The reference endpoint quantizes probabilities to
0.01, so the gold label receives a literal `0.00` on a few percent of items and
log loss becomes arbitrary. Brier degrades gracefully. This will matter less
for our own model, which will not quantize, but the comparison has to be run on
a metric both can take.

**Expected cost per decision is the ranking metric**; accuracy, ECE and latency
are gates. `expected_cost()` escalates per item when
`Σ_{k≠argmax} p_k · cost_wrong[k]` exceeds the cost of a human review, which
yields per-class thresholds for free rather than one global cutoff. On
imbalanced traffic that difference is usually where the money is.

**`risk_coverage()` sweeps by rank, not by probability threshold.** With
quantized outputs, 94 of 300 items tied at exactly 1.00 in one of our runs,
making coverage below 31% unreachable by thresholding at all. Ranking
sidesteps that.

**Bootstrap intervals and exact McNemar on everything**, because tiers are
small and the differences that matter are a few points.

## Suite 1: the healthcare router

Five pharmacy intents (`refill`, `order_status`, `drug_availability`,
`store_hours`, `vaccine_appointment`), a `choice` with `none_of_the_above`, one
`noul` per intent for multi-label, and four independent safety gates.

All items are synthetic and handwritten, so labels are definitional. No real
user text. 260 items with more than one defensible reading omit the single gold
and carry the acceptable set in `meta["acceptable"]`, because forcing a primary
intent onto a two-intent message would mark a correct model wrong. Splits are
disjoint by *content*, not by row: paraphrases and noisy variants inherit their
source's group, so a rephrasing cannot straddle train and test.

### Reference baseline (test, n=450)

Intent choice **0.909**, 95% CI [0.880, 0.933] scored leniently (the prediction
is in the item's acceptable set); **0.941** strict on the 338 single-gold
items. Multi-label nouls at threshold 0.5: **exact set 0.822, F1 0.890**.

Seven tiers sit at 1.000 and are **regression floors, not scoring targets**:
`single`, `out_of_scope`, `near_oos`, `clinical_urgent`, `abuse`, `chitchat`,
and the intent question on `clinical_oblique`. A model cannot win there; it can
only break.

The tiers with headroom:

| tier | n | intent | multi-label exact set |
|---|---|---|---|
| `compound_3` | 31 | **0.581** | **0.645** |
| `ambiguous_scored` | 20 | 0.950 | **0.250** |
| `negation_conditional` | 35 | 0.886 | 0.743 |
| `clinical_embedded` | 40 | 0.900 | 0.800 |
| `rambling` | 29 | 0.862 | 0.828 |
| `noisy` | 130 | 0.931 | 0.846 |

`compound_3` is the largest gap: three simultaneous intents drop a single
`choice` question to 0.581, against 0.909 overall. That is the structural limit
of picking one winner from a softmax, and it is the clearest thing a
multi-label-trained model should improve.

### The safety result

The four gates are `noul` questions evaluated independently of the intent
question. `G_clinical` scores 0.926 recall at a 0.006 false-positive rate.
Broken out by tier:

```
clinical_embedded        40/40 = 1.000    red flag buried in a normal request
clinical_urgent            5/5 = 1.000    explicit emergency
noisy                    32/34 = 0.941
clinical_oblique         23/29 = 0.793    no medical vocabulary
```

The tier we most expected to fail does not: a clinical red flag buried inside
an ordinary refill request is caught 40 out of 40. The failure is oblique
phrasing — adverse events described the way people actually describe them:

```
0.12  I'm not right in myself since you changed the supplier
0.19  Something is off since the new one, I can't put my finger on it.
0.30  Honestly I've not been myself since the switch.
```

Recall falls twenty points once the medical vocabulary is gone. With a 0.006
false-positive rate there is budget to trade, so this is a target rather than a
ceiling.

One gate is badly worded and that is our fault, not the model's. `G_pharmacy`
asks whether a message "concerns pharmacy or health business", and *"Are you
open on Sunday?"* concerns **store** business — so it misses 16 of 110 in-scope
messages. The general lesson is worth more than the fix: **a gate cannot be
validated only on the tier it is meant to catch.** Run it against ordinary
in-scope traffic too, or ship one that silently rejects a seventh of your
volume.

### Negative result: skip noise robustness

The `noisy` tier retains `meta["clean_text"]`, giving a paired measurement over
the same 130 items.

```
clean     121/130 = 0.931
degraded  121/130 = 0.931
identical predicted label on both versions: 124/130
McNemar: 3 vs 3, p = 1.0
```

Degradation reshuffles *which* items fail without changing how many. By type,
only ASR-style term substitution bites (0.800); lowercasing, dropped
punctuation, fillers, run-ons and self-correction are free. **Augmentation
budget should not go here.**

## Suite 2: held-out generalization

Seven public tasks, capped at 600 test rows each, covering all three primitives
with option counts from 2 to 151.

| task | primitive | K | why |
|---|---|---|---|
| `banking77` | choice | 77 | anchor, with a reference number |
| `clinc_oos` | choice | 151 | the only public set with a real out-of-scope class |
| `massive_intent` | choice | 60 | different domain, high cardinality |
| `sst5` | score | 5 | does ordinal transfer at all |
| `ag_news` | choice | 4 | easy control: separates "model broken" from "task hard" |
| `civil_comments_toxicity` | noul | 2 | binary, the least-covered primitive |
| `helpsteer_helpfulness` | score | 5 | ordinal that is not sentiment |

A random baseline lands on 1/K for all seven. That check is cheap and it earned
its keep: `Noul` gold is a `bool` and `Score` gold is an `int`, and because
`True == 1` in Python the obvious conversion rewrote SST-5 levels 0 and 1 as
`"false"`/`"true"`, corrupting 240 of 600 rows in a way that would have read as
"the ordinal head does not transfer".

### Contamination is the default, not the exception

The obvious training source is `tasksource`, which contains most common
benchmarks — and not always under a recognisable name. Verified with
[`scripts/verify_heldout_lineage.py`](../scripts/verify_heldout_lineage.py):

```
rotten_tomatoes (train)   77.0% of SST-5 test sentences, verbatim
glue/sst2 (validation)    75.4% of SST-5 dev
glue/sst2 (train)          0.3%   -- GLUE respected the split
toxic_conversations      100.0% of its rows are Civil Comments rows
```

The `rotten_tomatoes` path is the dangerous one. Its name contains no hint of
SST, so a filter keyed on the obvious string reports clean while three quarters
of the test set leaks.

`holdout_of` therefore totals 140 names, and the matcher is deliberately not a
set intersection: a mixture may call Banking77 `banking77`, `PolyAI/banking77`
or `mteb/banking77`, and two of those three slip past a naive check. It matches
full name, owner-stripped basename, and glob patterns.

```python
from openjev.heldout import assert_training_mixture_clean
assert_training_mixture_clean(mixture_names)   # before every training run
```

### Caveats that travel with these numbers

- 600 rows over CLINC's 151 labels is 3 to 4 per label, so macro-F1 there is
  noise. Only accuracy and out-of-scope detection rate are interpretable.
- The toxicity task is sampled 50/50 against a natural prevalence around 8%, so
  any ECE or expected-cost figure on it reflects the resampled prior.
- **Held out from training is not held out from pretraining.** Every dataset
  here is public and old. We can guarantee these schemas were absent from our
  mixture, not from any backbone's.
- `costs` is deliberately empty for these academic tasks. Inventing per-class
  error prices would make the ranking metric arbitrary while looking rigorous.
- SST-5 and AG News have no usable licence, so their schemas and provenance are
  committed here but the rows are not redistributed; regenerate them with
  `scripts/build_heldout.py --fetch`.
- One MASSIVE label (`cooking_query`) has zero test rows, so that menu offers
  60 options of which only 59 can ever be correct. Left in rather than quietly
  making the task easier.

## What a model has to do here

1. **Match** the seven saturated router tiers. Pass/fail.
2. **Beat** `compound_3` (0.581 intent, 0.645 exact set) and `ambiguous_scored`
   (0.250 exact set). Both are multi-label representation failures.
3. **Beat** oblique-clinical gate recall (0.793) without spending the 0.006
   false-positive rate.
4. **Not collapse** on the held-out suite.
5. **Ignore** noise robustness.

# The held-out generalization suite

Seven typed-decision tasks that no openjev training mixture may ever contain.
They exist to answer one question honestly: **does a model that was never
trained on a question schema still answer it well?**

[`../../docs/evaluation.md`](../../docs/evaluation.md) commits us to holding
out *entire question schemas*,
never random rows, and names zero-shot transfer to unseen schemas as the thing
we are least likely to get for free. This is the instrument for finding out.
`docs/04-open-questions.md` records the risk that makes it fragile: the
training mixture comes from `tasksource`, which contains Banking77, AG News,
Rotten Tomatoes, all seven Civil Comments configs and all five HelpSteer
configs. Without enforcement the default state of a mixture is contaminated,
and the resulting numbers look fine.

So every task carries a `holdout_of` list, `manifest.json` carries the union of
all of them, and `openjev.heldout.assert_training_mixture_clean()` refuses a
mixture that intersects it.

## The suite

| task | primitive | K | n test | n dev | source | licence | rows |
|---|---|---|---|---|---|---|---|
| `banking77` | choice | 77 | 600 | – | `mteb/banking77` | CC-BY-4.0 | vendored |
| `clinc_oos` | choice | 151 | 600 | 600 | `clinc/clinc_oos` (`plus`) | CC-BY-3.0 | vendored |
| `massive_intent` | choice | 60 | 600 | 600 | `mteb/amazon_massive_intent` (`en`) | CC-BY-4.0 | vendored |
| `sst5` | score | 5 | 600 | 600 | `SetFit/sst5` | **unknown** | generated |
| `ag_news` | choice | 4 | 600 | – | `fancyzhx/ag_news` | **unknown** | generated |
| `civil_comments_toxicity` | noul | 2 | 600 | 600 | `google/civil_comments` | CC0-1.0 | vendored |
| `helpsteer_helpfulness` | score | 5 | 600 | – | `nvidia/HelpSteer` | CC-BY-4.0 | vendored |

4,200 test rows and 2,400 dev rows in total, every task capped at 600 and
stratified with `seed=0`, so the whole suite is cheap to run. Option counts
span 2, 4, 5, 60, 77 and 151, which is the range needed to see whether
accuracy degrades with cardinality. All three primitives are covered, and
`score` is measured on two rubrics rather than only on sentiment.

## Using it

```python
from openjev.heldout import load_suite, assert_training_mixture_clean, gold_labels

assert_training_mixture_clean(mixture_names)   # BEFORE training. raises on leak
suite = load_suite()                           # {name: Task}

task = suite["sst5"]
gold = gold_labels(task)                       # menu keys, ready for metrics.py
```

**Use `gold_labels`, do not convert gold answers by hand.** `Noul` gold is a
`bool` and `Score` gold is an `int`, and in Python `True == 1` and
`False == 0`. The obvious `{True: "true", False: "false"}.get(v, str(v))`
therefore rewrites score levels 0 and 1 as `"false"` and `"true"`: it mangles
240 of 600 `sst5` rows and 180 of 600 `helpsteer_helpfulness` rows, and the
result looks like an ordinal head that does not transfer rather than a bug.
`gold_label` dispatches on the question type instead of the value. A random
baseline lands on 1/K for all seven tasks, which is the cheapest check that
your scoring is wired up correctly.

To rebuild from source (network, a few minutes, ~250 MB of parquet cached in
`/tmp/ojcache`):

```
uv run python scripts/build_heldout.py --fetch
```

## Where the option descriptions came from

Our own measurement says this matters more than it looks. Discriminative
option text is worth **+5.1 accuracy points on Banking77** (95% CI [+2.2,
+8.3], p = 0.0011), while mechanical restatements of the label name are worth
nothing (p = 0.75 at n = 300). A description contaminated by a test label
would therefore inflate a held-out number by more than most effects this
project cares about.

**No test split was read while writing any description in this suite.** Not
the text, and not the labels. Per task:

| task | descriptions derived from |
|---|---|
| `banking77` | **train split**, verbatim from the measured description set (see below) |
| `clinc_oos` | **train split**, via the per-label dumps |
| `massive_intent` | **train split**, via the per-label dumps |
| `sst5` | **documentation**: the SST five-way scale and its level names |
| `ag_news` | **documentation** (the card's ClassLabel names) plus the **train split** |
| `civil_comments_toxicity` | **documentation**: the Jigsaw crowd-rating question and its definition of toxic |
| `helpsteer_helpfulness` | **documentation** (card definition, 0–4 Likert, higher better) plus the **train split** for what each anchor means in practice |

The train dumps are reproducible with
`uv run python scripts/build_heldout.py --dump-train`, which writes four
sampled utterances per label to `tasks/heldout/_train_dumps/`. That directory
is the complete set of rows anyone consulted.

Banking77 is a special case worth stating plainly: its descriptions are not
rewritten here, they are lifted **verbatim** from the description set used in
our option-description ablation, and are vendored in
`scripts/heldout_criteria_banking77.py` rather than retyped, so a typo cannot
creep in. That exact text is the arm the +5.1 points was measured
on. Rewording it would quietly invalidate the comparison.

Label names are the source dataset's own strings, verbatim, including the
misleading ones (`get_physical_card` is about PINs, not cards) and the oddly
cased ones (`Refund_not_showing_up`). Our Banking77 work found the raw label
surface form carries real semantic weight, so renaming would change results.
**There is exactly one rename in the suite**: CLINC's `oos` is surfaced as
`none_of_the_above`, which the escape-option design asks for. The builder
asserts it is the only one.

## The tasks

### `banking77` — choice, K=77, the anchor

Customer-support intent routing for a digital bank. Chosen because it is the
one task where we already have Jev reference numbers: **0.820 accuracy, 0.806
macro-F1, ECE 0.068**.

Those numbers were measured on 300 stratified rows of this exact parquet
(sha256 `9575f63…`). Rather than draw a fresh 600 and caveat the comparison,
the builder **pins all 300 reference rows into our sample** and fills the rest
stratified, so the reference set is a strict subset of ours. Those rows carry
`meta.jev_reference_300 = true`. All 300 are present; per-class counts run 6–8.

No official dev split exists, so there is none here.

### `clinc_oos` — choice, K=151, the escape option

150 assistant intents plus a genuine out-of-scope class. The only public set
that tests the escape option directly, which
[`../../docs/architecture.md`](../../docs/architecture.md) calls
out as a capability an enumerated output space removes and which we measured
Jev failing at (escapes only 45.0% of the time in the near-miss regime, ECE
0.374).

The `oos` class is mapped to a `none_of_the_above` option and its instruction
text tells the model the option is real and frequently correct. OOS rows carry
`meta.out_of_scope = true` and `meta.source_label = "oos"`.

Sampling departs from even stratification here, deliberately. Even
stratification over 151 labels would leave the escape option with about 4
rows, which measures nothing. Instead the benchmark's own OOS prevalence is
preserved: **109 out-of-scope and 491 in-scope** in test, the in-scope portion
stratified over the 150 intents at 3–4 rows each.

### `massive_intent` — choice, K=60, a second high-cardinality domain

Amazon MASSIVE smart-speaker commands, English only. High cardinality in a
domain unlike Banking77, which is how we find out whether a cardinality effect
is about the number or about the banking vocabulary.

One real wrinkle: **`cooking_query` has zero rows in the English test split**
(it has only 4 in train). The menu offers 60 options but only 59 can ever be
correct, so macro-F1 over the menu is not the same quantity as macro-F1 over
the labels present. It is left in the menu because removing it would be
changing the task to make it easier. `source.json` records this under
`test_label_coverage`.

### `sst5` — score, K=5, does ordinal transfer at all

Fine-grained sentiment on the Stanford Sentiment Treebank's own five-way
scale. Included because we measured that Jev's `score` head is not genuinely
ordinal: 11.8% of its score distributions are multimodal over ordered levels,
and on inputs that state a 50/50 split between extremes it puts 51% of its
mass in the middle third at confidence 0.85. A real ordinal head is somewhere
we expect to win, so we need a clean ordinal task to show it on.

`Score.criteria` is an ordered list, so the gold answer is a level index and
there is no label string to preserve. Each level therefore quotes the source
dataset's own `label_text` verbatim at the front of its description, so the
source surface form still reaches the model.

### `ag_news` — choice, K=4, the control

Four-way news topic classification. Deliberately easy, and there purely so
that a bad number elsewhere can be read as "this task is hard" rather than
"the model is broken". If a model cannot do AG News, nothing else in the suite
is interpretable.

### `civil_comments_toxicity` — noul, K=2, the binary primitive

**Why this one.** The brief allowed prompt injection, toxicity or spam.
Civil Comments wins on all four criteria that matter here:

- *Real human labels.* Each comment was rated by up to ten crowd workers
  against a published question, and the released `toxicity` field is the
  fraction who called it toxic. Not model-generated, not heuristic.
- *Licence.* CC0-1.0, a public domain dedication. It is the only candidate we
  can vendor without reservation.
- *Not saturated.* The realistic alternatives are. `deepset/prompt-injections`
  has 116 test rows and is close to solved; SMS spam is at ceiling. A binary
  task that everything scores 0.99 on measures nothing.
- *It is the shape of the real product.* A `noul` question in production is a
  gate on a decision, which is exactly what content moderation is.

The cost is that toxicity is the fuzziest of the three. Labels are a rater
fraction binarised at the standard `>= 0.5`, and agreement near the boundary
is genuinely poor. The raw fraction is kept in `meta.toxicity` so the
threshold can be revisited and the sample reweighted.

**The sample is 300 toxic / 300 not, which is not the deployment prevalence**
of roughly 8%. Balanced sampling is right for measuring the primitive and
wrong for reading calibration off directly: any ECE or expected-cost number
computed on this task reflects the resampled prior, not production. Reweight
with `meta.toxicity` before quoting one.

### `helpsteer_helpfulness` — score, K=5, ordinal that is not sentiment

Response helpfulness on the annotators' 0–4 Likert scale. Included so `score`
is not evaluated purely on sentiment: if a model only ever learned that
ordinal means "how positive", this is where it shows.

It is a genuine rubric with real human labels — about 200 trained US-based
annotators working from written per-level criteria, with two rounds of QA.
The state is a JSON object (`{"prompt": ..., "response": ...}`), which also
exercises the structured-state path that the string-only tasks do not.

Two caveats. The **responses are model-generated**, so the input distribution
is LLM text rather than natural text, and the task measures agreement with
human raters about model output. And the validation split is used as our test
split, because it is the only held-out split the dataset ships; there is
therefore no dev split, and level 0 supplies only 58 of the 600 rows because
that is all it has.

## Licences, and what we did not vendor

Verified against the Hub API and each dataset card.

| dataset | licence | redistributable | action |
|---|---|---|---|
| `mteb/banking77` / `PolyAI/banking77` | CC-BY-4.0 (mirror: MIT) | yes | rows vendored |
| `clinc/clinc_oos` | CC-BY-3.0 | yes | rows vendored |
| `mteb/amazon_massive_intent` | CC-BY-4.0 (mirror: Apache-2.0) | yes | rows vendored |
| `google/civil_comments` | CC0-1.0 | yes | rows vendored |
| `nvidia/HelpSteer` | CC-BY-4.0 | yes | rows vendored |
| `SetFit/sst5` | **none declared** | no | **loader only** |
| `fancyzhx/ag_news` | **`unknown`** | no | **loader only** |

`SetFit/sst5` carries no licence metadata at all and upstream
`stanfordnlp/sst` is tagged `unknown`. `fancyzhx/ag_news` declares
`license: unknown`; the original AG corpus terms permit non-commercial
academic use but do not grant redistribution rights we can rely on.

**Absence of a licence is not permission.** For both, the task directory holds
`task.json` and `source.json` — the schema, the instructions, the option
descriptions, the provenance — and the rows are *generated locally* by
`scripts/build_heldout.py` and git-ignored. `source.json` records the sha256
of the file the builder produced, so a collaborator who rebuilds can confirm
they got byte-identical rows without anyone redistributing them.
`load_suite()` raises a message naming the rebuild command if they are absent.

## The contamination lineage, measured

`holdout_of` is a claim that other datasets contain these rows. Two claims
were worth checking, and one of them changed what we believe. Reproduce with
`uv run python scripts/verify_heldout_lineage.py`.

**Civil Comments → `toxic_conversations`.** The card says the data is "an
exact replica" of the Jigsaw Unintended Bias release. Measured against all
1,999,514 Civil Comments rows: **100.0%** of both
`mteb/toxic_conversations_50k` and `SetFit/toxic_conversations` appear
verbatim in Civil Comments. Not a subset relation to argue about — those
datasets *are* these comments under another name, and neither name contains
the string "civil".

**SST-5 → `rotten_tomatoes`, which was not the suspected path.** The suspected
leak was GLUE SST-2, and GLUE turns out to respect the split: only 0.3% of
SST-5 test sentences occur in `glue/sst2` train. The real one:

| pool | rows | % of SST-5 **test** present | % of SST-5 **dev** present |
|---|---|---|---|
| `glue/sst2` train | 67,349 | 0.3% | 0.2% |
| `glue/sst2` validation | 872 | 0.0% | **75.4%** |
| `rotten_tomatoes` train | 8,530 | **77.0%** | 76.7% |
| `rotten_tomatoes` test | 1,066 | 0.0% | 0.0% |

`rotten_tomatoes` is task **#378 in `tasksource`**. Its train split contains
77% of our SST-5 test sentences verbatim, with binary labels over the same
text, and nothing about its name would warn you. This is the exact failure the
suite exists to prevent, and it would have survived any check that only
excluded names containing "sst".

`source.json` and `manifest.json` therefore record `tasksource_leak_paths` per
task: the task ids actually present in `tasksource` that reach this test set.

## What `holdout_of` includes, and what it does not

A name goes in if it is (a) the dataset itself under any alias, (b) a dataset
sharing source documents with it, or (c) a subsample or relabelling of it. The
asymmetry is deliberate: a false positive costs one training task, a false
negative costs the entire claim. Hence HelpSteer2 and HelpSteer3 are excluded
from `helpsteer_helpfulness` even though they are separate collections.

A name does **not** go in merely for being the same topic. `tweet_eval/hate`
is not held out from Civil Comments, and `imdb` is not held out from SST-5 —
excluding those would gut the mixture without protecting anything. Each task's
`source.json` lists these under `related_not_excluded` so the judgement is
auditable rather than invisible.

The matcher in `openjev/heldout.py` is not a set intersection, because a
mixture calls Banking77 `banking77` while the Hub calls it
`PolyAI/banking77`. It matches lowercased full names, names with the Hub owner
stripped (only against hold-out entries that are themselves bare names, so
`oasst2_dense_flat/toxicity` is not confused with `civil_comments/toxicity`),
and explicit `*` patterns.

## What would make a generalization claim from this suite misleading

Read this before quoting any number off it.

1. **600 rows across 151 labels is 3–4 per label.** Per-class and macro
   metrics on `clinc_oos` are not interpretable, and they are thin on
   `banking77` (6–8) and `massive_intent` (0–11). Quote accuracy, OOS
   detection rate and ECE; treat macro-F1 as indicative at best. Every
   difference needs the bootstrap CIs in `openjev/metrics.py` — at n=300 our
   own ECE estimate moved ±0.02, the size of most effects we care about.
2. **The `civil_comments_toxicity` prevalence is artificial.** 50/50 against a
   natural ~8%. Calibration and expected-cost numbers on it are about the
   resampled prior unless reweighted.
3. **`massive_intent` has a dead option.** 60 in the menu, 59 reachable.
4. **`helpsteer_helpfulness` inputs are model-generated**, and its level 0 has
   58 rows. It measures rater agreement on LLM output, not on natural text.
5. **Four of seven tasks are intent classification**, and three of those are
   short single-sentence utterances. The suite is broader in primitive and
   cardinality than it is in genre. It says little about long documents, and
   the longest states in it are HelpSteer's (~3.1k characters at the median).
6. **`costs` and `cost_escalate` are deliberately empty.** `openjev/metrics.py`
   treats expected cost per decision as the ranking metric, but these are
   academic datasets with no real price on an error. Inventing per-class costs
   would make that metric arbitrary while looking rigorous. Supply real ones
   per deployment, or do not report expected cost on this suite.
7. **Held out from training is not held out from pretraining.** Every dataset
   here is public and old enough to be in any web-scale pretraining corpus. We
   can guarantee these schemas were absent from *our* mixture. We cannot
   guarantee the backbone has never seen the text, and no one publishing on
   these benchmarks can.
8. **This measures transfer to unseen schemas, not to unseen domains.** The
   schemas are new to the model; intent classification as a genre is not.

## Files

```
tasks/heldout/
  manifest.json          every task, and the union of all holdout_of names
  <task>/task.json       schema: questions, criteria, holdout_of
  <task>/test.jsonl      rows (absent for sst5 and ag_news: see licences)
  <task>/dev.jsonl       where a natural dev split exists
  <task>/source.json     provenance, licence, label coverage, sha256
```

Built by `scripts/build_heldout.py` from `scripts/heldout_criteria.py` and
`scripts/heldout_criteria_banking77.py`. Loaded by `openjev/heldout.py`.

<!-- BEGIN GENERATED holdout_of -->

## Appendix: the full `holdout_of` lists

Generated from `manifest.json` by `scripts/build_heldout.py`. 140 distinct names across 7 tasks.

### `banking77` — 14 names

In `tasksource` and reaching this test set: `banking77`

```
banking77
PolyAI/banking77
mteb/banking77
legacy-datasets/banking77
DeepPavlov/banking77
FastFit/banking_77
mteb/BankingClassification
mteb/Banking77Classification
gtfintechlab/banking77
haizelabs/banking77
willcb/banking77
AutoIntent/banking77_aug
DeepPavlov/banking77-translated
Donoe/Multilingual_Banking77
```

### `clinc_oos` — 16 names

No `tasksource` task is known to reach this test set.

```
clinc_oos
clinc/clinc_oos
clinc150
CLINC150
DeepPavlov/clinc_oos
DeepPavlov/clinc150
DeepPavlov/clinc150_subset
contemmcm/clinc150
FastFit/clinc_150
Syedtahaali94/clinc_oos
cmaldona/Generalization-MultiClass-CLINC150-ROSTD
cmaldona/All-Generalization-OOD-CLINC150
AutoIntent/clinc150_aug_qwen2.5-7b-awq
OpenVoiceOS/ovos-intent-bench-clinc150
OpenVoiceOS/ovos-intent-bench-clinc150-train
mteb/CLINC150Classification
```

### `massive_intent` — 13 names

No `tasksource` task is known to reach this test set.

```
massive
AmazonScience/massive
amazon_massive_intent
mteb/amazon_massive_intent
mteb/MassiveIntentClassification
MassiveIntentClassification
SetFit/amazon_massive_intent_en-US
amazon_massive_scenario
mteb/amazon_massive_scenario
mteb/MassiveScenarioClassification
SetFit/amazon_massive_scenario_en-US
SetFit/amazon_massive_intent_*
SetFit/amazon_massive_scenario_*
```

### `sst5` — 19 names

In `tasksource` and reaching this test set: `rotten_tomatoes`, `glue/sst2`

```
sst5
SetFit/sst5
sst-5
Realgon/sst5
Samsoup/SST5
VirtualRoyalty/SST5
gimmaru/SetFit-sst5
sst2
glue/sst2
stanfordnlp/sst2
SetFit/sst2
gpt3mix/sst2
sst
stanfordnlp/sst
sst/default
rotten_tomatoes
cornell-movie-review-data/rotten_tomatoes
mattymchen/mr
SetFit/rotten_tomatoes
```

### `ag_news` — 13 names

In `tasksource` and reaching this test set: `ag_news`

```
ag_news
fancyzhx/ag_news
SetFit/ag_news
sh0416/ag_news
pietrolesci/agnews
sentence-transformers/agnews
contemmcm/ag_news
r-three/ag_news_subset
ag_news_subset
Recognai/ag_news_corrected_labels
Recognai/corrected_labels_ag_news
Lots-of-LoRAs/task379_agnews_topic_classification
Lots-of-LoRAs/task1541_agnews_classification
```

### `civil_comments_toxicity` — 35 names

In `tasksource` and reaching this test set: `civil_comments/toxicity`, `civil_comments/severe_toxicity`, `civil_comments/obscene`, `civil_comments/threat`, `civil_comments/insult`, `civil_comments/identity_attack`, `civil_comments/sexual_explicit`, `jigsaw_toxicity`, `toxic_conversations`

```
civil_comments
google/civil_comments
civil_comments/toxicity
civil_comments/severe_toxicity
civil_comments/obscene
civil_comments/threat
civil_comments/insult
civil_comments/identity_attack
civil_comments/sexual_explicit
civil_comments__toxicity
civil_comments__severe_toxicity
civil_comments__obscene
civil_comments__threat
civil_comments__insult
civil_comments__identity_attack
civil_comments__sexual_explicit
jigsaw_unintended_bias
google/jigsaw_unintended_bias
jigsaw_toxicity
tasksource/jigsaw_toxicity
toxic_conversations
SetFit/toxic_conversations
SetFit/toxic_conversations_50k
mteb/toxic_conversations_50k
mteb/ToxicConversationsClassification
civilcomments_wilds
pietrolesci/civilcomments-wilds
shlomihod/civil-comments-wilds
lighteval/civil_comments_helm
Lots-of-LoRAs/task1720_civil_comments_toxicity_classification
Lots-of-LoRAs/task1721_civil_comments_obscenity_classification
Lots-of-LoRAs/task1722_civil_comments_threat_classification
Lots-of-LoRAs/task1723_civil_comments_sexuallyexplicit_classification
Lots-of-LoRAs/task1724_civil_comments_insult_classification
Lots-of-LoRAs/task1725_civil_comments_severtoxicity_classification
```

### `helpsteer_helpfulness` — 30 names

In `tasksource` and reaching this test set: `HelpSteer/helpfulness`, `HelpSteer/correctness`, `HelpSteer/coherence`, `HelpSteer/complexity`, `HelpSteer/verbosity`, `HelpSteer2/helpfulness`, `HelpSteer2/correctness`, `HelpSteer2/coherence`, `HelpSteer2/complexity`, `HelpSteer2/verbosity`

```
HelpSteer
nvidia/HelpSteer
helpsteer
HelpSteer/helpfulness
HelpSteer/correctness
HelpSteer/coherence
HelpSteer/complexity
HelpSteer/verbosity
helpsteer__helpfulness
helpsteer__correctness
helpsteer__coherence
helpsteer__complexity
helpsteer__verbosity
HelpSteer2
nvidia/HelpSteer2
HelpSteer2/helpfulness
HelpSteer2/correctness
HelpSteer2/coherence
HelpSteer2/complexity
HelpSteer2/verbosity
helpsteer_2__helpfulness
helpsteer_2__correctness
helpsteer_2__coherence
helpsteer_2__complexity
helpsteer_2__verbosity
HelpSteer3
nvidia/HelpSteer3
Weyaxi/HelpSteer-filtered
RLHFlow/Helpsteer-preference-standard
Columbia-NLP/DPO-HelpSteer
```

### Union

This is what `assert_training_mixture_clean()` checks against.

```
AmazonScience/massive
AutoIntent/banking77_aug
AutoIntent/clinc150_aug_qwen2.5-7b-awq
CLINC150
Columbia-NLP/DPO-HelpSteer
DeepPavlov/banking77
DeepPavlov/banking77-translated
DeepPavlov/clinc150
DeepPavlov/clinc150_subset
DeepPavlov/clinc_oos
Donoe/Multilingual_Banking77
FastFit/banking_77
FastFit/clinc_150
HelpSteer
HelpSteer/coherence
HelpSteer/complexity
HelpSteer/correctness
HelpSteer/helpfulness
HelpSteer/verbosity
HelpSteer2
HelpSteer2/coherence
HelpSteer2/complexity
HelpSteer2/correctness
HelpSteer2/helpfulness
HelpSteer2/verbosity
HelpSteer3
Lots-of-LoRAs/task1541_agnews_classification
Lots-of-LoRAs/task1720_civil_comments_toxicity_classification
Lots-of-LoRAs/task1721_civil_comments_obscenity_classification
Lots-of-LoRAs/task1722_civil_comments_threat_classification
Lots-of-LoRAs/task1723_civil_comments_sexuallyexplicit_classification
Lots-of-LoRAs/task1724_civil_comments_insult_classification
Lots-of-LoRAs/task1725_civil_comments_severtoxicity_classification
Lots-of-LoRAs/task379_agnews_topic_classification
MassiveIntentClassification
OpenVoiceOS/ovos-intent-bench-clinc150
OpenVoiceOS/ovos-intent-bench-clinc150-train
PolyAI/banking77
RLHFlow/Helpsteer-preference-standard
Realgon/sst5
Recognai/ag_news_corrected_labels
Recognai/corrected_labels_ag_news
Samsoup/SST5
SetFit/ag_news
SetFit/amazon_massive_intent_*
SetFit/amazon_massive_intent_en-US
SetFit/amazon_massive_scenario_*
SetFit/amazon_massive_scenario_en-US
SetFit/rotten_tomatoes
SetFit/sst2
SetFit/sst5
SetFit/toxic_conversations
SetFit/toxic_conversations_50k
Syedtahaali94/clinc_oos
VirtualRoyalty/SST5
Weyaxi/HelpSteer-filtered
ag_news
ag_news_subset
amazon_massive_intent
amazon_massive_scenario
banking77
civil_comments
civil_comments/identity_attack
civil_comments/insult
civil_comments/obscene
civil_comments/severe_toxicity
civil_comments/sexual_explicit
civil_comments/threat
civil_comments/toxicity
civil_comments__identity_attack
civil_comments__insult
civil_comments__obscene
civil_comments__severe_toxicity
civil_comments__sexual_explicit
civil_comments__threat
civil_comments__toxicity
civilcomments_wilds
clinc/clinc_oos
clinc150
clinc_oos
cmaldona/All-Generalization-OOD-CLINC150
cmaldona/Generalization-MultiClass-CLINC150-ROSTD
contemmcm/ag_news
contemmcm/clinc150
cornell-movie-review-data/rotten_tomatoes
fancyzhx/ag_news
gimmaru/SetFit-sst5
glue/sst2
google/civil_comments
google/jigsaw_unintended_bias
gpt3mix/sst2
gtfintechlab/banking77
haizelabs/banking77
helpsteer
helpsteer_2__coherence
helpsteer_2__complexity
helpsteer_2__correctness
helpsteer_2__helpfulness
helpsteer_2__verbosity
helpsteer__coherence
helpsteer__complexity
helpsteer__correctness
helpsteer__helpfulness
helpsteer__verbosity
jigsaw_toxicity
jigsaw_unintended_bias
legacy-datasets/banking77
lighteval/civil_comments_helm
massive
mattymchen/mr
mteb/Banking77Classification
mteb/BankingClassification
mteb/CLINC150Classification
mteb/MassiveIntentClassification
mteb/MassiveScenarioClassification
mteb/ToxicConversationsClassification
mteb/amazon_massive_intent
mteb/amazon_massive_scenario
mteb/banking77
mteb/toxic_conversations_50k
nvidia/HelpSteer
nvidia/HelpSteer2
nvidia/HelpSteer3
pietrolesci/agnews
pietrolesci/civilcomments-wilds
r-three/ag_news_subset
rotten_tomatoes
sentence-transformers/agnews
sh0416/ag_news
shlomihod/civil-comments-wilds
sst
sst-5
sst/default
sst2
sst5
stanfordnlp/sst
stanfordnlp/sst2
tasksource/jigsaw_toxicity
toxic_conversations
willcb/banking77
```

<!-- END GENERATED holdout_of -->

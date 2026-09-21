---
license: apache-2.0
task_categories:
  - text-classification
  - zero-shot-classification
tags:
  - typed-decisions
  - generalization
  - openjev
size_categories:
  - 1K<n<10K
---

# OpenJev held-out generalization suite

Seven tasks for measuring whether a typed-decision model transfers to
schemas it has never been trained on. Built to answer one question honestly:
does the model generalize, or has it seen this before?

Used by [OpenJev](https://github.com/S1LV3RJ1NX/openjev), but the format is
plain JSONL and nothing here depends on that code.

## The suite

| task | primitive | K | rows | chance |
|---|---|---|---|---|
| `clinc_oos` | choice | 151 | 600 | 0.007 |
| `banking77` | choice | 77 | 600 | 0.013 |
| `massive_intent` | choice | 60 | 600 | 0.017 |
| `sst5` | score | 5 | 600 | 0.200 |
| `helpsteer_helpfulness` | score | 5 | 600 | 0.200 |
| `ag_news` | choice | 4 | 600 | 0.250 |
| `civil_comments_toxicity` | noul | 2 | 600 | 0.500 |

The high-cardinality end is the point. A 151-option menu is where models
that look fine on four-way classification collapse, and where we measured a
literal 0.000 before fixing it.

## The contamination guard is the main feature

A held-out suite is worthless if the training mixture quietly contains the
same data under another name. `tasksource` calls it `banking77`, the Hub
also has `PolyAI/banking77`, `mteb/banking77`, `legacy-datasets/banking77`
and more.

So every task carries a `holdout_of` list naming every alias we could find,
`manifest.json` carries the union, and the matcher checks full names,
owner-stripped basenames and glob patterns rather than doing a set
intersection.

```python
from openjev.heldout import assert_training_mixture_clean
assert_training_mixture_clean(mixture_names)   # raises on a leak
```

This caught a real leak during development. It is not decoration.

## Known limitations, stated up front

**Two tasks have classes too small to learn.** `clinc_oos` has 3 examples of
`translate` in 600, `massive_intent` has 1 of `general_greet`. Those classes
will be learned as "never predict this" and that is a property of the
sample, not the model.

**`civil_comments` needs AUROC, not accuracy.** The labels are crowd
majority votes on borderline political hostility, and a model can rank well
while putting every probability above 0.5, scoring exactly the class
balance. We measured AUROC 0.714 alongside accuracy 0.515 on the same
predictions. Report both or the task reads as a failure it is not.

**Option descriptions carry weight, and can hurt.** Descriptions that state
what distinguishes a label from its neighbours were worth about +5 accuracy
points on Banking77. Descriptions that restate the label name were worth
nothing (p = 0.75). On `civil_comments`, the original two `noul` descriptions
were worse than none at all, because they restate one judgement from
opposite sides and a model scoring both measures their overlap.

## Format

Each task directory holds `task.json` (the schema) and `train/dev/test.jsonl`
(one example per line). Fully specified in
[docs/dataset-format.md](https://github.com/S1LV3RJ1NX/openjev/blob/main/docs/dataset-format.md).

```json
{"state": "how do I top up my card?", "answers": {"intent": "top_up_by_card"}, "meta": {"tier": "default"}}
```

## Provenance and licences

Every task is a sample of a public dataset, named in its `source.json` and
in `holdout_of`. Built by `scripts/build_heldout.py`, which records source,
split, sampling seed and row lineage; verify with
`scripts/verify_heldout_lineage.py`.

| task | source |
|---|---|
| `banking77` | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) |
| `clinc_oos` | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) |
| `massive_intent` | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) |
| `ag_news` | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) |
| `sst5` | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) |
| `civil_comments` | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) |
| `helpsteer_helpfulness` | [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) |

**The GitHub repository withholds the `sst5` and `ag_news` rows** and ships
only their schemas plus a rebuild command, because those two sources are
the ones whose terms we read as not clearly permitting redistribution. They
are included here for convenience with attribution above. If you are a
rightsholder and would prefer they were not, open a discussion on this repo
and they will be removed.

The option descriptions, the sampling, the contamination manifest and the
assembly are ours, Apache 2.0. Every underlying dataset keeps its own
licence and its own terms govern the rows.

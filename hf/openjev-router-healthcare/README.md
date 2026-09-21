---
license: apache-2.0
library_name: pytorch
base_model: answerdotai/ModernBERT-base
pipeline_tag: text-classification
tags:
  - typed-decisions
  - intent-classification
  - routing
  - guardrails
  - openjev
---

# OpenJev healthcare router

A worked example you can run in three lines. It routes pharmacy messages:
one intent, five multi-label topic flags and four safety gates, **all
answered in a single forward pass** over the shared message.

It exists to show what 395 training examples and 38 seconds on one GPU buy
you, and it is the model behind the numbers in the
[results](https://github.com/S1LV3RJ1NX/openjev/blob/main/docs/results.md).

```python
from openjev import DecisionModel, Choice, Noul

model = DecisionModel.from_pretrained("s1lv3rj1nx/openjev-router-healthcare")

out = model.answer(
    "I need a refill on my thyroid tablets, and are you open on Sunday?",
    {
        "A_intent": Choice(
            instructions="Which single topic does this pharmacy message concern?",
            criteria={
                "refill": "asking us to dispense another fill of an existing prescription",
                "store_hours": "when a branch is open",
                "order_status": "where an existing order is",
            },
        ),
        "C_store_hours": Noul(instructions="Does this message ask about when a branch is open?"),
        "G_clinical": Noul(instructions="Does this describe a clinical symptom?"),
    },
)

out["A_intent"].label                         # 'refill'      p = 0.874
out["C_store_hours"].probabilities["true"]    # 1.000  — both topics caught
out["G_clinical"].probabilities["true"]       # 0.000
```

`pip install git+https://github.com/S1LV3RJ1NX/openjev` for the package. The
checkpoint carries its own backbone name, prompt format and fitted
temperatures, so there is nothing else to configure.

## Measured against a commercial typed-decision API

Paired on the same 450 held-out test items, exact McNemar. Reproducible with
`scripts/compare_to_jev.py`.

| | this model | commercial API | b10 | b01 | p | winner |
|---|---|---|---|---|---|---|
| intent | 0.899 | 0.941 | 13 | 27 | 0.039 | API |
| multi-label exact set | 0.789 | 0.822 | 38 | 53 | 0.142 | **level** |
| `G_clinical` | 0.980 | 0.978 | 8 | 7 | 1.000 | level |
| `G_abusive` | 0.998 | 0.996 | 2 | 1 | 1.000 | level |
| `G_injection` | 0.978 | 0.996 | 0 | 8 | 0.008 | API |
| `G_pharmacy` | 0.947 | 0.880 | 48 | 18 | 3e-04 | **this model** |
| obliquely-worded clinical risk | **1.000** | 0.793 | 6 | 0 | 0.031 | **this model** |

Per-question test accuracy, temperature-scaled, with bootstrap CIs:

| question | n | accuracy | 95% CI | ECE | Brier |
|---|---|---|---|---|---|
| `A_intent` | 338 | 0.899 | [0.870, 0.932] | 0.034 | 0.155 |
| `C_refill` | 450 | 0.953 | [0.933, 0.971] | 0.014 | 0.064 |
| `C_store_hours` | 450 | 0.960 | [0.940, 0.978] | 0.027 | 0.060 |
| `G_clinical` | 450 | 0.982 | [0.969, 0.993] | 0.015 | 0.033 |
| `G_abusive` | 450 | 0.998 | [0.993, 1.000] | 0.012 | 0.006 |
| `G_injection` | 450 | 0.978 | [0.964, 0.989] | 0.018 | 0.041 |

## Limitations, stated plainly

**It is a demo on synthetic data.** The task is synthetic pharmacy messages,
not a deployment. Treat the numbers as evidence that the recipe works, not
as a claim about your traffic.

**`G_pharmacy` over-fires.** False-positive rate 0.652. The slice is 427
positive to 23 negative, and 23 negatives is not enough to train a gate
properly. It still beats the commercial API overall on that gate, which says
more about the difficulty than about either model.

**`G_injection` is the weakest gate**, recall 0.583 against the API's 0.917,
and it regressed from 0.750 under an earlier general checkpoint.

**Do not use this for real clinical triage.** It flags whether a message
*mentions* clinical content so it can be escalated. It is not a medical
device and makes no clinical judgement.

## How it was made

```bash
hf download s1lv3rj1nx/openjev-encoder-general model.pt --local-dir checkpoints/general
uv run python scripts/train.py --task tasks/healthcare_router \
    --init-from checkpoints/general/model.pt --epochs 6 --bs 8
```

Starting from the [general
checkpoint](https://huggingface.co/s1lv3rj1nx/openjev-encoder-general)
instead of raw ModernBERT is worth **+36 points** for the same 38 seconds:
intent 0.544 → 0.899, multi-label 0.453 → 0.789. Two gates that never
learned to fire at all from scratch reach 0.857 and 0.583 recall from a
handful of positives.

## Licence

Apache 2.0.

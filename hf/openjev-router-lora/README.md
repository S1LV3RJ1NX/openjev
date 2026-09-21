---
license: apache-2.0
library_name: pytorch
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-classification
tags:
  - typed-decisions
  - routing
  - guardrails
  - lora
  - peft
  - openjev
---

# OpenJev healthcare router, LoRA adapter

An 87 MB rank-16 adapter on [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
that **beats the commercial typed-decision API it was benchmarked against**,
from 395 training examples and 258 seconds on one GPU.

Ten questions answered in a single forward pass over the shared message:
one intent `choice`, five multi-label `noul` topic flags, four safety gates.

```python
from openjev import DecisionModel, Choice, Noul

model = DecisionModel.from_pretrained("s1lv3rj1nx/openjev-router-lora")

out = model.answer(
    "I need a refill on my thyroid tablets, and are you open on Sunday?",
    {
        "A_intent": Choice(
            instructions="Which single topic does this pharmacy message concern?",
            criteria={"refill": "asking us to dispense another fill of an existing prescription",
                      "store_hours": "when a branch is open",
                      "order_status": "where an existing order is"}),
        "C_store_hours": Noul(instructions="Does this message ask about when a branch is open?"),
        "G_clinical": Noul(instructions="Does this describe a clinical symptom?"),
    },
)
out["A_intent"].label
```

`pip install git+https://github.com/S1LV3RJ1NX/openjev`. The adapter is
merged into the base weights on load, because leaving it unmerged costs
about 2x at inference for nothing.

## Measured against `jev-1.13.0`

Paired on the same 450 held-out test items, exact McNemar. Reproduce with
`scripts/compare_to_jev.py`.

| | this adapter | reference API | b10 | b01 | p | winner |
|---|---|---|---|---|---|---|
| intent | **0.979** | 0.941 | 14 | 1 | 9.8e-04 | **this** |
| multi-label exact set | **0.909** | 0.822 | 57 | 18 | 7.2e-06 | **this** |
| scope gate | **0.978** | 0.880 | 49 | 5 | 3.9e-10 | **this** |
| clinical gate | 0.987 | 0.978 | 7 | 3 | 0.34 | level |
| abusive gate | 0.993 | 0.996 | 2 | 3 | 1.00 | level |
| injection gate | 0.993 | 0.996 | 1 | 2 | 1.00 | level |
| oblique clinical recall | 0.966 | 0.793 | 5 | 0 | 0.06 | level |

Three wins, four ties, no losses.

## Why an adapter and not a fine-tune

Every configuration we tried on this task, same data, same 6 epochs:

| | intent | trainable | artifact |
|---|---|---|---|
| encoder (ModernBERT-base), all open | 0.899 | 150M | 0.6 GB |
| decoder, 4.2M head only | 0.666 | 4.2M | 17 MB |
| decoder, all 1.7B open | 0.929 | 1,725M | 3.4 GB |
| **decoder, LoRA r=16** | **0.979** | **17M** | **87 MB** |

Adapting 17M parameters beats opening 1,725M, at a thirty-ninth of the
size. The likely reason is the learning rate a full fine-tune tolerates:
1e-5 over six epochs of 395 examples barely moves a 1.7B model, while LoRA
at 2e-4 adapts quickly without disturbing the weights it rides on. Neither
was tuned past one setting.

It also means one loaded backbone can serve many routers, which a 3.4 GB
per-task artifact cannot.

## Limitations

**Synthetic data.** Handwritten and systematically composed pharmacy
messages, not real traffic. Evidence the recipe works, not a claim about
your users.

**The scope gate over-fires.** False-positive rate 0.826 on 23 negative
examples. It beats the reference API overall and is still the weakest of
the four.

**One tier cannot prove itself.** `clinical_oblique` has 29 items, so a
paired test with zero losses floors at p = 0.0625. The 0.966 against 0.793
is real and not statistically separable.

**Not a medical device.** The gates flag whether a message *mentions*
clinical content so it can be escalated to a human.

**No zero-shot claim.** This is fine-tuned on its task. On a schema it has
never seen, OpenJev scores well below the reference API, and that is
documented rather than hidden.

## Related

- [Code and full results](https://github.com/S1LV3RJ1NX/openjev)
- [Training data](https://huggingface.co/datasets/s1lv3rj1nx/openjev-healthcare-router)
- [General checkpoint](https://huggingface.co/s1lv3rj1nx/openjev-encoder-general) for tasks without labels
- [Held-out suite](https://huggingface.co/datasets/s1lv3rj1nx/openjev-heldout)

## Licence

Apache 2.0. Base model under its own licence.

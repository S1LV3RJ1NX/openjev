---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
  - text-classification
  - zero-shot-classification
  - intent-classification
  - lora
  - openjev
library_name: peft
datasets:
  - s1lv3rj1nx/openjev-mixture
language:
  - en
---

# OpenJev general adapter (LoRA, Qwen3-1.7B)

**This is the checkpoint to start from.** Rank-16 LoRA adapters over
Qwen3-1.7B, trained on 279 classification tasks. It answers typed
questions about text against a menu of options it has never seen, and it
is the recommended OpenJev artifact after the architecture ablation.

Use it two ways: zero-shot on a schema you invent today, or as the
initialisation for a task adapter, which is worth far more than starting
from the base model.

## What it does on schemas it was never trained on

Seven held-out tasks, none in the training mixture, contamination
enforced by an automated guard. 600 items each, 95% CIs over 1,000
bootstrap resamples.

| task | K | chance | accuracy | 95% CI | x chance |
|---|---|---|---|---|---|
| clinc_oos | 151 | 0.007 | **0.783** | [0.752, 0.817] | **118.3x** |
| banking77 | 77 | 0.013 | **0.728** | [0.693, 0.765] | **56.1x** |
| massive_intent | 60 | 0.017 | **0.773** | [0.738, 0.805] | **46.4x** |
| ag_news | 4 | 0.250 | 0.803 | [0.772, 0.838] | 3.2x |
| sst5 | 5 | 0.200 | 0.438 | [0.400, 0.475] | 2.2x |
| civil_comments | 2 | 0.500 | 0.688 | [0.655, 0.727] | 1.4x |
| helpsteer | 5 | 0.200 | 0.282 | [0.247, 0.320] | 1.4x |

**Mean 32.7x chance.** All seven beat chance, and all seven beat their
majority-class baseline, including the skewed ones: helpsteer 0.282
against 0.233 at p = 0.0025.

Read the three-digit menus first. Choosing correctly among 151 intents
having never seen the label set is the capability that makes this useful.
The four-way and five-way rows are honest about the limit: on short
sentiment and helpfulness judgements it is better than guessing and not
much more.

## Honest comparison

TypeSafe's Jev scores **0.820** on Banking77 zero-shot. This adapter
scores **0.728** on the same 600 items. They are ahead by nine points.

That gap used to be fifty-three, against our encoder's 0.290. If you want
the best zero-shot accuracy available and can send data to an API, their
number is still the better one.

## Quick start

```python
from openjev.infer import DecisionModel
from openjev.schema import Choice

m = DecisionModel.from_pretrained("s1lv3rj1nx/openjev-general-lora")

q = {"intent": Choice(
    instructions="Which banking intent is this?",
    criteria={"declined_transaction": None, "card_lost": None, "top_up_failed": None},
)}

a = m.answer("my card got declined at the grocery store", q)["intent"]
print(a.label, a.p_max)
```

Real output from this checkpoint, on a three-way menu invented for this
card and present in no training task:

```
my card got declined at the grocery store       -> declined_transaction  0.975
I left my wallet in a taxi and need to block... -> card_lost             0.978
the money I added has not shown up yet          -> top_up_failed         0.848
```

`criteria` maps each option to an optional description. Pass `None` when
the label speaks for itself.

The adapter merges into the base weights at load, so inference costs the
same as the unadapted model. An unmerged adapter costs roughly 2x for
nothing.

## Using it as an initialisation

This is the higher-value path. A task adapter started from this
checkpoint rather than from base Qwen3 inherits everything the mixture
taught it about reading a menu:

```bash
python scripts/train.py --task tasks/your_task --decoder --lora-r 16 \
  --init-from general_lora/model.pt --epochs 6
```

The equivalent transfer on the encoder path was worth **+36 points** over
training the same task from scratch. Two-stage fine-tuning is standard
practice, not a trick: see Phang et al.'s STILTs and Gururangan et al.'s
"Don't Stop Pretraining."

## Training

- 279 tasks, 224k examples, curated from tasksource
- Rank-16 LoRA, alpha 32, on all attention and MLP projections
- 1 epoch, batch 4, lr 2e-4, max_len 2048, bf16, one H100, 5.3 hours
- Label-space augmentation: menus padded with distractors up to K=24
- Contamination guard asserts no held-out task is in the mixture

Data was audited before training, which mattered more than any
architecture change: 3,781 contradictory rows removed, and an
ingestion bug that put the gold answer at index 0 in all 83
multiple-choice tasks found and fixed.

## Limitations

- English only
- Nine points behind the commercial reference zero-shot on Banking77
- Fine-grained sentiment (sst5) and helpfulness (helpsteer) barely clear
  their majority baselines
- `civil_comments` accuracy 0.688 against AUROC 0.766, so the ranking is
  better than the decision at a 0.5 threshold. Tune the threshold.
- Not a safety classifier. The safety gates in our router were trained,
  not zero-shot, and even then recall is 0.84 with a wide CI.

## Links

- Code, evaluation harness, full results: https://github.com/s1lv3rj1nx/openjev
- Training mixture: https://huggingface.co/datasets/s1lv3rj1nx/openjev-mixture
- Held-out suite: https://huggingface.co/datasets/s1lv3rj1nx/openjev-heldout
- Encoder alternative, 21.9x chance, 20 ms p95: https://huggingface.co/s1lv3rj1nx/openjev-encoder

## Which OpenJev checkpoint should I use?

| | this adapter | encoder |
|---|---|---|
| held-out mean | **32.7x chance** | 21.9x |
| Banking77 zero-shot | **0.728** | 0.290 |
| p95 latency, batch 1 | 56 ms | **20 ms** |
| footprint | 3.4 GB | **0.6 GB** |

Start here. Move to the encoder only if p95 latency binds or the
footprint does.

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

Use it zero-shot, on a schema you invent today. If you have a few
hundred labels for your task, train a task adapter from the base weights
instead of from this one: we measured intermediate initialisation to be
worse here, and the numbers are under "Using it as an initialisation".

## What it does on schemas it was never trained on

Seven held-out tasks, none in the training mixture, contamination
enforced by an automated guard. 600 items each, 95% CIs over 1,000
bootstrap resamples.

| task | K | chance | accuracy | 95% CI | x chance |
|---|---|---|---|---|---|
| clinc_oos | 151 | 0.007 | **0.702** | [0.667, 0.738] | **106.0x** |
| banking77 | 77 | 0.013 | **0.605** | [0.570, 0.642] | **46.6x** |
| massive_intent | 60 | 0.017 | **0.775** | [0.742, 0.807] | **46.5x** |
| ag_news | 4 | 0.250 | 0.793 | [0.760, 0.828] | 3.2x |
| sst5 | 5 | 0.200 | 0.465 | [0.425, 0.502] | 2.3x |
| civil_comments | 2 | 0.500 | 0.742 | [0.710, 0.777] | 1.5x |
| helpsteer | 5 | 0.200 | 0.273 | [0.240, 0.312] | 1.4x |

**Mean 29.6x chance, every task scored at its full advertised menu**,
which needs `--max-len 6144` for the 151-way one. All seven beat chance,
and all seven beat their majority-class baseline, including the skewed
ones, though helpsteer only just: 0.273 against 0.233 at p = 0.010.

Read the three-digit menus first. Choosing correctly among 151 intents
having never seen the label set is the capability that makes this useful.
The four-way and five-way rows are honest about the limit: on short
sentiment and helpfulness judgements it is better than guessing and not
much more.

## Honest comparison

We measured TypeSafe's Jev on all seven held-out tasks, 600 items each,
the same items this adapter is scored on.

| task | Jev | this adapter |
|---|---|---|
| clinc_oos | **0.938** | 0.702 |
| ag_news | **0.880** | 0.793 |
| banking77 | **0.863** | 0.605 |
| massive_intent | 0.838 | 0.775 (level) |
| civil_comments | 0.748 | 0.742 (level) |
| sst5 | **0.560** | 0.465 |
| helpsteer | **0.415** | 0.273 |
| mean x chance | **38.3x** | 29.6x |

**They win five and tie two, and lose none.** If you want the best
zero-shot accuracy available and you can send your data to an API, use
theirs. This adapter is the answer when the data cannot leave your
network, when you need determinism or full-precision probabilities, or
when you have a few hundred labels, which is the case where a task
adapter beats them outright.

Their per-item predictions ship in the repository under `baselines/` so
the comparison is checkable without an API key.

## Quick start

```python
from openjev import DecisionModel, Choice

m = DecisionModel.from_pretrained("s1lv3rj1nx/openjev-general-lora")

q = {"intent": Choice(
    instructions="Which banking intent is this?",
    criteria={"declined_transaction": None, "card_lost": None, "top_up_failed": None},
)}

a = m.answer("my card got declined at the grocery store", q)["intent"]
print(a.label, a.p_max)
```

`a.label` is the winning option and `a.p_max` its probability;
`a.probabilities` is the full distribution over the menu you passed.
`criteria` maps each option to an optional description, so pass `None`
when the label speaks for itself. The menu is model input rather than
weights, which is why it can change between calls.

Menus larger than a few dozen options need the sequence budget raised:
`DecisionModel.from_pretrained(..., max_len=6144)` is what the 151-way
held-out task was scored with.

The adapter is merged into the base weights at load, which costs 1.07x
an unadapted model. Left unmerged it costs 1.96x for nothing, so
`from_pretrained` always merges.

## Using it as an initialisation

Do not. We tried it and it is worse than starting from base weights.
Initialising the healthcare router adapter from this checkpoint instead
of from base Qwen3, trained identically, scores **0.953 against 0.979**:
paired McNemar p = 0.012, so the intermediate stage costs accuracy
rather than buying it.

```bash
# Train a task adapter from base weights, not from this adapter.
python scripts/train.py --task tasks/your_task \
  --decoder --lora-r 16 --backbone Qwen/Qwen3-1.7B --epochs 6
```

The same move on the encoder path was worth **+36 points**, so this is
architecture-specific rather than a general verdict on two-stage
fine-tuning (see Phang et al.'s STILTs and Gururangan et al.'s "Don't
Stop Pretraining"). The explanation that fits both results: the encoder
had to learn what a menu is from a few hundred examples, and base Qwen3
already reads menus from pretraining, so a 279-task specialisation is
net interference on one narrow problem.

**So: this adapter is for zero-shot use, where there is no task data to
train on. It is the wrong starting point when there is.**

## Training

- 279 tasks, 323,466 rows, curated from tasksource
- Rank-16 LoRA, alpha 32, on all attention and MLP projections
- 1 epoch, batch 8, lr 2e-4, max_len 2048, bf16, one H100, 5.3 hours
- Label-space augmentation: menus padded with distractors, requested up to
  K=120 from a pool of 670 labels. The context budget of 2048 clipped that to
  about 80 in practice, so this checkpoint never saw a menu as large as the
  151 it is evaluated on.
- Contamination guard asserts no held-out task is in the mixture

Data was audited before training, which mattered more than any
architecture change: 3,781 contradictory rows removed, and an
ingestion bug that put the gold answer at index 0 in all 83
multiple-choice tasks found and fixed.

## Limitations

- English only
- Twenty-one points behind the commercial reference zero-shot on
  Banking77
- Fine-grained sentiment (sst5) and helpfulness (helpsteer) barely clear
  their majority baselines
- `civil_comments` accuracy 0.742 against AUROC 0.828, so the ranking is
  better than the decision at a 0.5 threshold. Tune the threshold.
- Not a safety classifier. The safety gates in our router only work once
  trained on that task. Zero-shot, this adapter catches 38% of the
  clinical cases it should escalate, at a 12% false-positive rate, which
  is not a gate anyone should deploy.
- **Public benchmark scores did not predict readiness on a real task.**
  This adapter wins all seven held-out benchmarks and then loses to the
  encoder when both are run zero-shot on our healthcare router: intent
  0.467 against 0.601. If your task has packed multi-question states,
  compound multi-label answers or gates with recall floors, measure it
  yourself rather than reading across from the table above.

## Links

- Code, evaluation harness, full results: https://github.com/s1lv3rj1nx/openjev
- Fine-tuned router adapter: https://huggingface.co/s1lv3rj1nx/openjev-router-lora
- Training mixture: https://huggingface.co/datasets/s1lv3rj1nx/openjev-mixture
- Held-out suite: https://huggingface.co/datasets/s1lv3rj1nx/openjev-heldout
- Encoder alternative, 17.2x chance, 20 ms p95: https://huggingface.co/s1lv3rj1nx/openjev-encoder-general

## Which OpenJev checkpoint should I use?

| | this adapter | encoder |
|---|---|---|
| held-out mean | **29.6x chance** | 17.2x |
| Banking77 zero-shot | **0.605** | 0.343 |
| throughput, idle H100 | 84 states/s | **655 states/s** |
| p95 latency, batch 1 | 55.5 ms | **20.3 ms** |
| p95 latency, batch 1 | 56 ms | **20 ms** |
| footprint | 3.4 GB | **0.6 GB** |

Start here. Move to the encoder only if p95 latency binds or the
footprint does.

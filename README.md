<h1 align="center">OpenJev</h1>

<p align="center">
  <b>Typed decisions from a small model you can train yourself.</b><br>
  State in, typed answers with calibrated probabilities out. No text generation.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20preview-orange">
  <img alt="Backbone" src="https://img.shields.io/badge/backbone-ModernBERT%20%7C%20Qwen3-green">
  <a href="docs/results.md"><img alt="Results" src="https://img.shields.io/badge/results-measured%2C%20with%20CIs-brightgreen"></a>
  <a href="https://buymeacoffee.com/prathams"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-prathams-FFDD00?logo=buy-me-a-coffee&logoColor=black"></a>
</p>

---

## Quick start

```bash
git clone https://github.com/S1LV3RJ1NX/openjev && cd openjev
uv sync
```

Train a decision model on your own task in about a minute:

```bash
uv run python scripts/train.py --task tasks/healthcare_router --epochs 6 --bs 8
uv run python scripts/eval_router.py --ckpt checkpoints/healthcare_router/model.pt
```

Score any held-out schema with no training at all:

```bash
uv run python scripts/eval_heldout.py --decoder --backbone Qwen/Qwen3-1.7B --limit 200
```

## How to use it

Give it a state and some typed questions. Every question is answered in **one
forward pass over a shared state**, so asking twenty costs about the same as
asking one.

```python
from openjev import Choice, Score, Noul

result = model.predict(
    state="I need a refill on my thyroid tablets, and are you open on Sunday?",
    questions={
        "intent":      Choice(instructions="What is this about?",
                              criteria={"refill": "another fill of a prescription",
                                        "store_hours": "when a branch is open"}),
        "urgency":     Score(instructions="How time-critical?",
                             criteria=["can wait", "today", "immediate"]),
        "needs_human": Noul(instructions="Does this need a human?"),
    },
)

result["intent"].probabilities      # {"refill": 0.71, "store_hours": 0.24, ...}
result["intent"].confidence         # chance-corrected, comparable across menus
result["urgency"].score             # 1.30 — the expected level, not the argmax
result["urgency"].is_unimodal       # False means that 1.30 describes nothing
result["needs_human"].probabilities["true"]
```

| primitive | returns | use for |
|---|---|---|
| `choice` | one of N options, with probabilities | routing, intent, classification |
| `score` | a **number** on an ordered rubric | urgency, severity, quality |
| `noul` | P(true) | flags, gates, multi-label |

Multi-label is `noul` per label, not `choice`: a message can be about a refill
*and* opening hours, and a softmax cannot say so.

## Two recipes

**Fine-tune on your task.** 395 examples, 38 seconds, one GPU.

**Start from the general checkpoint first.** Same data, same time — worth
**+27 points**. This is the project's central claim and it is measured:
intent 0.544 → 0.817, multi-label 0.453 → 0.718, p = 6e-18.

```bash
uv run python scripts/train.py --task tasks/your_task \
    --init-from checkpoints/mixture_big/model.pt --epochs 6
```

## Where Jev wins

We measured TypeSafe's Jev against our own suite and it is ahead on the things
that matter most, so use it if those matter more than self-hosting:

- **Zero-shot accuracy.** 0.820 on Banking77 having never seen it; our best
  untrained number is 0.205.
- **Routing quality.** Router intent 0.941 against our 0.817 (p = 1e-07).
- **Gate precision.** `G_clinical` false-positive rate 0.006 against our 0.035.
- **Scale.** 255 options and a 32k context, out of the box.

Where we are level or ahead: `G_clinical` correctness (p = 0.63, no detectable
difference), full-precision probabilities (Jev quantizes to 0.01, putting
71.9% of values at a hard zero), determinism (Jev has none and no seed), and
cost — this runs on your own hardware with no data leaving it.

## Where this architecture fits

**Good fit:** high-volume routing, triage, guardrails and tool selection where
you have labels; long states asked many questions at once; regulated data that
cannot leave your network; anything needing a real probability to threshold on.

**Poor fit:** you have no labels and need it to work on arbitrary schemas
today — use Jev. Short states with one or two questions — the shared prefix
buys nothing and a plain classifier is simpler.

## Docs

| | |
|---|---|
| [Results](docs/results.md) | Every measurement, what worked and what did not |
| [Architecture](docs/architecture.md) | The design, with diagrams, and why each choice |
| [Evaluation](docs/evaluation.md) | The two suites and the contamination guard |
| [Prior art](docs/prior-art.md) | UniMC, PCW, CORN, ADB — most of this exists |

## Honesty policy

Every Jev figure was measured by us against the public API (`jev-1.13.0`,
20 Sep 2026) at stated sample sizes, with the caveats kept next to the
numbers. Negative results are documented as prominently as positive ones, and
[docs/results.md](docs/results.md) contains a list of claims we retracted after
better measurement. No claim of parity with Jev is made anywhere, because the
numbers do not show it.

## Support

Everything here stays free. If it saved you time,
[buy me a coffee](https://buymeacoffee.com/prathams).

## Licence

Apache 2.0. Dataset rows keep their upstream licences; see
[`tasks/heldout/README.md`](tasks/heldout/README.md).

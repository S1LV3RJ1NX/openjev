---
license: apache-2.0
task_categories:
  - text-classification
tags:
  - typed-decisions
  - routing
  - guardrails
  - synthetic
  - openjev
size_categories:
  - n<1K
---

# Healthcare router: a typed-decision benchmark with real headroom

Synthetic pharmacy messages, routed by ten questions answered together: one
intent `choice`, five multi-label `noul` topic flags, and four safety gates.
988 items across 18 tiers, 450 in test.

Built for [OpenJev](https://github.com/S1LV3RJ1NX/openjev) to compare
against a commercial typed-decision API, and deliberately built to be hard.

## Why it exists

The first version of this task was useless. A commercial API scored **1.000
on four of its tiers** — single-intent, obvious out-of-scope, pure clinical,
chitchat — which means those slices could not distinguish any two systems.
A benchmark at ceiling measures nothing.

So the tiers here target where that model actually failed:

| tier | n | what it tests |
|---|---|---|
| `clinical_oblique` | 29 | symptoms described without clinical vocabulary |
| `clinical_embedded` | 40 | a symptom buried inside a routine request |
| `compound_3` | 31 | three intents in one message |
| `negation_conditional` | 35 | "don't refill unless...", "if it's ready" |
| `near_oos_hard` | 30 | plausibly pharmacy-adjacent but out of scope |
| `injection_hard` | 20 | subtle prompt injection |
| `rambling` | 29 | long messages with the request buried |
| `noisy` | 130 | voice-transcript and mobile-typing degradation |
| `ambiguous_scored` | 20 | genuinely two-way, with an acceptable set |

`clinical_oblique` is the sharpest: the reference API catches 79.3% and
misses one in five obliquely-worded clinical risks.

## Reference numbers

Measured on `jev-1.13.0`, 20 September 2026, on the 450 test items.

| | score |
|---|---|
| intent, 338 items with a single gold | 0.941 |
| intent, all 450 including ambiguous | 0.909 |
| multi-label exact set | 0.822 |
| `compound_3` exact set | 0.645 |
| `G_clinical` recall / FPR | 0.926 / 0.006 |
| `clinical_oblique` recall | 0.793 |

Those two intent figures use **different denominators** and are not
interchangeable. 112 items have an acceptable *set* rather than one gold.
Comparing a 338-item score against 0.909 understates the gap by three
points, which we did before catching it.

## Known limitations

**It is synthetic.** Handwritten and systematically composed, not real
traffic. Treat it as evidence a method works, not as a claim about your
users.

**Two gates are badly imbalanced.** `G_abusive` has 7 positives in 450, and
`G_pharmacy` has 23 negatives against 427 positives. Both are learnable but
neither supports a confident false-positive-rate claim.

**Small tiers cannot reach significance.** `clinical_oblique` has 29 items,
so a paired test with zero losses floors at p = 0.0625. A perfect score
there still is not statistically separable.

**Not for clinical use.** The gates flag whether a message *mentions*
clinical content so it can be escalated to a human. Nothing here makes a
medical judgement.

## Format

`task.json` plus `train/dev/test.jsonl`, specified in
[docs/dataset-format.md](https://github.com/S1LV3RJ1NX/openjev/blob/main/docs/dataset-format.md).
Splits are disjoint by content, not by row: paraphrases and noisy variants
of an item all stay in one split, grouped at Jaccard >= 0.62.

Ambiguous items carry `meta.acceptable` instead of a single gold, so a
defensible second answer is not scored wrong.

## Licence

Apache 2.0.

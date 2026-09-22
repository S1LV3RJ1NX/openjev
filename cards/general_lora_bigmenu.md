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

# OpenJev general adapter, large-menu variant

**This is not the recommended checkpoint. Use
[`openjev-general-lora`](https://huggingface.co/s1lv3rj1nx/openjev-general-lora)
unless you have a specific reason not to.**

Published because it is a real artifact from a real experiment, and because
a repository that only ships the runs that worked is not evidence of
anything. The experiment did not work.

## The experiment

The recommended adapter trains at a 2048-token budget, which silently
clips label-space augmentation to about 80 options, while the held-out
suite presents menus of up to 151. That is a genuine train-and-test
mismatch, and label-space augmentation is the largest single effect we
have measured, so extending its range was the obvious thing to try.

This adapter is identical except for training at a 4096-token budget with
`--max-options 160`, which produces menus up to 175. Same 279-task
mixture, same rank, same learning rate, same backbone.

## The result

| task | recommended (menus to 80) | this one (menus to 175) | |
|---|---|---|---|
| banking77 | 0.605 | **0.663** | level |
| clinc_oos | **0.702** | 0.660 | level |
| massive_intent | **0.775** | 0.670 | **worse** |
| ag_news | 0.793 | 0.808 | level |
| sst5 | 0.465 | 0.435 | level |
| civil_comments | 0.742 | 0.682 | level |
| helpsteer | 0.273 | 0.290 | level |
| **mean x chance** | **29.6x** | 28.5x | |

**Six level, one worse, none better, and the mean moved the wrong way.**

Banking77 gained 5.8 points, the largest move in the table and exactly the
direction predicted. We are not promoting that, because doing so would
mean ignoring the ten-point loss on `massive_intent` beside it.

## Why it did not work, as best we can tell

Menu size was not the binding constraint. The recommended adapter already
reaches 106x chance on a 151-way menu having trained on nothing larger
than 80, so whatever it learned about reading a menu generalised past the
sizes it saw. Padding every menu towards 160 distractors appears to cost
something elsewhere, and `massive_intent` at K=60 is the size most likely
to be crowded out.

## When you might still want this one

If your menus are consistently very large and Banking77-like, this
checkpoint is better there. That is one task out of seven, so treat it as
a hypothesis about your data rather than a recommendation.

## Fine-tuned on a task

Used as a starting point for a task adapter it reaches 0.9615 intent on
our healthcare router, against 0.9615 for the recommended adapter's
equivalent and **0.979 from base weights**. As with every decoder result
we have, starting a task adapter from base beats starting it from any
general adapter. Train from base.

## Links

- Recommended checkpoint: https://huggingface.co/s1lv3rj1nx/openjev-general-lora
- Code and full results: https://github.com/S1LV3RJ1NX/openjev
- Training mixture: https://huggingface.co/datasets/s1lv3rj1nx/openjev-mixture

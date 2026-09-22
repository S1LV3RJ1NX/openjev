#!/usr/bin/env bash
# Retrain the decoder so training sees menus as large as the ones we evaluate
# on. Two things were wrong before.
#
# The option cap never bound. At --max-len 2048 the packer clipped augmented
# menus to about 80 options while --max-options asked for 120, so label-space
# augmentation, the largest lever we measured, ran at two thirds strength and
# the model never saw a menu near the 151 the held-out suite presents.
#
# The GPU sat idle. A fixed micro-batch has to survive the longest sequence in
# the mixture, so at batch 2 the median 560-token example wasted the card: 39%
# utilisation, 205W of 400W. Batching to a token budget instead fills it.
#
# 4096 is chosen from the data, not from caution. With augmentation on, the
# packed length is 167 tokens at the median and 3,312 at the maximum over a
# sample of 1,800, so 4096 never binds and anything larger only costs memory
# that batching can use instead.
#
# The learning rate stays at 2e-4, the value the shipped run used. Token
# budgeting changes how examples are grouped, not how many: the shipped run
# averaged 8 examples per step and this averages 8.7, so there is no larger
# batch to scale the rate for. Raising it to 5e-4 on that mistaken reasoning
# produced a run whose loss improved to 1.69 during warmup and then degraded
# to a plateau near 3.1 the moment the schedule reached its peak.
#
# The budget is set by memory, not by compute. The block-diagonal mask is
# handed to SDPA as an explicit tensor, which drops it off the flash path and
# materialises attention scores in every layer, so memory runs out while the
# card still has compute to spare. At 12288 tokens the run held 100%
# utilisation and 391W and still skipped half its batches. Making the mask
# flash-compatible is what would raise this ceiling.
#
# The budget assumes the sampler's length estimates are honest. They now are:
# augmentation is seeded per index, so the estimator runs the same draw
# training will. Estimating an upper bound instead gave batches of 1.4, and
# estimating the unaugmented length gave batches that overflowed 72% of the
# time. Training aborts if the skip rate passes 5%, because skipping batches
# quietly trains on whichever rows fit, which is the short-menu ones.
set -euo pipefail
cd "$(dirname "$0")/.."

LEN="${1:-4096}"
BUDGET="${2:-4096}"
OPTS="${3:-160}"
LR="${4:-2e-4}"

exec ./.venv/bin/python -u scripts/train.py \
  --mixture tasks/mixture_final \
  --decoder --lora-r 16 --backbone Qwen/Qwen3-1.7B \
  --epochs 1 --max-len "$LEN" --lr "$LR" \
  --token-budget "$BUDGET" --max-bs 24 --mask-budget 12000000 \
  --distractor-prob 0.5 --max-options "$OPTS" \
  --scale-prob 0.15 --noul-prob 0.15 \
  --sanity-at 300 --sanity-every 2000 \
  --out checkpoints_dec8k \
  --preamble "You judge whether a candidate answer is correct for a question about an input.

Input:
"

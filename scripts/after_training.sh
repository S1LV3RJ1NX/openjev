#!/usr/bin/env bash
# Everything that needs an idle GPU, run in sequence once training ends.
#
# These are deliberately not run alongside training. Two of the three are
# timing measurements, and the report already carries two sets of latency
# numbers that disagree because one pair was taken while a training run
# shared the card. A contended benchmark is not a cheap approximation of an
# uncontended one, it is a different measurement.
set -uo pipefail
cd "$(dirname "$0")/.."

NEW="${1:-checkpoints_dec8k/mixture_final/model.pt}"
OLD="${2:-checkpoints_declora_mix/mixture_final/model.pt}"
PRE="You judge whether a candidate answer is correct for a question about an input.

Input:
"

echo "waiting for training to finish"
while pgrep -f "[t]rain.py" >/dev/null; do sleep 60; done
sleep 30   # let the allocator hand memory back before timing anything

echo
echo "=============== 1/4  retrained decoder, full menus ==============="
# 6144 because that is what fits clinc_oos's 151 options on the decoder
# path. The harness warns if any menu is still clipped.
./.venv/bin/python scripts/eval_heldout.py --ckpt "$NEW" --decoder \
  --backbone Qwen/Qwen3-1.7B --max-len 6144 --bs 2 --limit 600 \
  --mixture tasks/mixture_final --preamble "$PRE" 2>&1 | grep -vE "it/s|%\|"

echo
echo "=============== 2/4  shipped decoder, same conditions ==============="
# Re-measured rather than quoted, so the two are comparable on the same
# items, the same harness and the same idle card.
./.venv/bin/python scripts/eval_heldout.py --ckpt "$OLD" --decoder \
  --backbone Qwen/Qwen3-1.7B --max-len 6144 --bs 2 --limit 600 \
  --preamble "$PRE" 2>&1 | grep -vE "it/s|%\|"

echo
echo "=============== 3/4  throughput against in-flight requests ==============="
./.venv/bin/python scripts/bench_throughput.py --ckpt "$NEW" --decoder \
  --backbone Qwen/Qwen3-1.7B --max-len 3072 2>&1 | grep -vE "it/s|%\|"

echo
echo "=============== 4/4  latency on an idle card ==============="
# Settles the disagreement in the report: 1.1x under contention against
# 2.8x uncontended for the merged decoder versus the encoder.
./.venv/bin/python scripts/bench_latency.py --ckpt "$NEW" 2>&1 | grep -vE "it/s|%\|"

echo
echo "ALL-DONE"

# Prior art

Most of this design already exists in the literature. We list it here because
a project like this is easy to present as more novel than it is, and because
these are the papers you should read before contributing.

## The architecture

**UniMC — "Zero-Shot Learners for Natural Language Understanding via a Unified
Multiple Choice Perspective"**,
[arXiv:2210.08590](https://arxiv.org/abs/2210.08590), EMNLP 2022,
[code](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/unimc).

This is essentially the design. Input format:

```
[CLS] { [O-MASK]ⁱ optionⁱ } [SEP] question [SEP] passage [SEP]
```

with custom segment IDs, position IDs and attention masking to stop options
interacting, labels treated as options rather than verbalizer maps, and 235M
parameters. Three things we take from it rather than re-derive: the masking
scheme, reusing the pretrained MLM head at the option-mask position (no new
parameters), and "MC tuning" as the multi-task recipe that buys zero-shot
transfer.

UniMC handles one question per sequence. The shared-state, many-question part
is the piece we are adding.

## Shared-prefix, block-diagonal attention

Published for the mirror problem (many contexts, one task) rather than ours
(one context, many questions), but the machinery is identical.

- **Parallel Context Windows**,
  [arXiv:2212.10947](https://arxiv.org/abs/2212.10947). Sparse masking so each
  window attends only within itself while task tokens attend across all.
  Requires no training.
- **APE — Adaptive Parallel Encoding**,
  [arXiv:2502.05431](https://arxiv.org/abs/2502.05431). Block-local attention
  with position reuse, and the finding that a shared prefix must be prepended
  to all blocks to avoid duplicating attention-sink states.
- **Block-Attention for efficient RAG** (Sun et al., 2024), which fine-tunes to
  recover what PCW loses.
- **CEPE**, [arXiv:2402.16617](https://arxiv.org/abs/2402.16617). A small
  bidirectional encoder over parallel chunks feeding a decoder by
  cross-attention. Different target, useful encoder-side detail.

## Ordinal heads

- **CORAL**, [arXiv:1901.07884](https://arxiv.org/abs/1901.07884).
  Rank-monotonic guarantees via shared output weights with ordered biases.
- **CORN**, [arXiv:2111.08851](https://arxiv.org/abs/2111.08851). Rank
  consistency *without* CORAL's weight-sharing constraint, by chaining
  conditional probabilities. [`coral-pytorch`](https://raschka-research-group.github.io/coral-pytorch/)
  makes it a three-line change. Prefer this one.

## Second-order uncertainty

For the case an ordinal head cannot represent: a genuinely bimodal posterior,
where the honest answer is "this scale is the wrong instrument".

- **Dirichlet Prior Networks**,
  [arXiv:1802.10501](https://arxiv.org/abs/1802.10501), NeurIPS 2018. Output a
  Dirichlet *over* categoricals, which separates confident knowledge (sharp,
  near a vertex) from ambiguous data (sharp, inside the simplex) from ignorance
  (flat). The precision `α₀` is the signal; predictive entropy alone cannot
  make the distinction.
- **Improving DPN for OOD detection**,
  [openreview](https://openreview.net/forum?id=Bye4iaEFwr). Replaces the
  KL-between-Dirichlets loss with cross-entropy plus a sharpness regulariser,
  motivated by large class counts.
- **"Is Epistemic Uncertainty Faithfully Represented by Evidential Deep
  Learning Methods?"**, [arXiv:2402.09056](https://arxiv.org/abs/2402.09056).
  The necessary caveat: second-order risk minimisation is **not**
  quantitatively faithful. Relative orderings hold, which is why these methods
  do well at OOD detection. Use `α₀` as a ranking signal with a fitted
  threshold, never as a calibrated probability.

## Rejection and escalation

- **ADB — "Deep Open Intent Classification with Adaptive Decision Boundary"**,
  AAAI 2021, [code](https://github.com/thuiar/Adaptive-Decision-Boundary).
  Learns a per-class spherical boundary post hoc from known-intent data only.
  No out-of-scope samples, no architecture change.
- **DA-ADB**, [arXiv:2203.05823](https://arxiv.org/abs/2203.05823), TASLP 2023.
  Distance-aware representations and adaptive radii, shipped inside the
  [TEXTOIR](https://github.com/thuiar/TEXTOIR) toolkit.
- **"Two-Stage Learning to Defer with Multiple Experts"**, NeurIPS 2023. Train
  the predictor with ordinary cross-entropy, then learn a deferral function.
  H-consistency bounds attached.
- **"Predictor-Rejector Multi-Class Abstention"**, Mao et al., ALT 2024.
  Surrogate losses for multi-class abstention with consistency guarantees.

## Option descriptions

- **REDEX**, \*SEM 2024. Effective label keywords need relevance to the task,
  **inter-class exclusivity**, and intra-class diversity, obtained by
  generate-then-rerank with maximal marginal relevance and no knowledge base.
  Inter-class exclusivity is the formal name for something we hit empirically:
  sharpening one label's description made several of its neighbours worse, so
  descriptions must be optimised jointly rather than one class at a time.
- **GEN-Z**, ICLR 2024. Scores input likelihood conditioned on natural-language
  label descriptions, with multiple paraphrases per label to cut variance.
- **Incubator**, [arXiv:2404.10877](https://arxiv.org/abs/2404.10877), EMNLP
  2024. Generates training data from class definitions, handling mutually
  dependent classes such as "X" versus "Other".

## Task diversity

**The Flan Collection**, [arXiv:2301.13688](https://arxiv.org/abs/2301.13688),
ICML 2023, ran the ablation a data pipeline like ours needs: subsets of 8, 25,
50, 100, 200, 400, 800 and all 1,873 tasks.

- Held-out performance rises **log-linearly** with task count, best at 1,836.
- Held-in performance **peaks around 200 tasks and then degrades**.

Chung et al., [Scaling Instruction-Finetuned LMs](https://www.jmlr.org/papers/volume25/23-0870/23-0870.pdf),
put most of the gain inside the first 282 tasks. Both caution that tasks are
not interchangeable units and that over-sampling one source saturates.

## Calibration

- **"On Calibration of Modern Neural Networks"**,
  [arXiv:1706.04599](https://arxiv.org/abs/1706.04599). Temperature scaling and
  ECE. Fit per question type and option count, on a held-out split, after
  training. It cannot reorder predictions, so accuracy is untouched.

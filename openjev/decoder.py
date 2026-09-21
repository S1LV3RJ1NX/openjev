"""A decoder backbone with the same packing, and no generation.

Why a decoder at all, when the whole point is escaping autoregressive
decoding: **we never decode.** One forward pass, zero tokens generated. We
read logits at chosen positions, exactly as the encoder path does. What a
decoder buys is decision semantics — an instruction-tuned model already knows
what "is this message about a refill" means, which a raw masked encoder does
not, and which our measurements showed is where zero-shot ability actually
comes from.

Why not letter slots (`A.`, `B.`, `C.`), which is the obvious readout and what
other open implementations use: there are only so many single-token letters.
It works to roughly 26 options and our own held-out suite runs to 151. So
instead each option gets a marker and we read the language model's own
`yes`/`no` preference at it:

    score(option) = logit("yes") - logit("no")   at that option's marker

which is UniMC's mechanism applied to an LM head, scales to any option count,
and adds no parameters — so it can be evaluated with no training.

The state is a shared causal prefix and question blocks are isolated from each
other, which is the same block-diagonal structure as the encoder path. On a
decoder that structure is cheaper still, because a causal prefix is precisely
what a KV cache stores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .model import OpenJevOutput, causal_block_and_mask_fn, grouped_log_softmax


class OpenJevDecoder(nn.Module):
    def __init__(
        self,
        backbone: str = "Qwen/Qwen3-1.7B",
        tokenizer=None,
        yes_token: str = " yes",
        no_token: str = " no",
        dtype: torch.dtype = torch.bfloat16,
        learned_head: bool = False,
        lora_r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("a tokenizer is required to resolve the yes/no token ids")
        cfg = AutoConfig.from_pretrained(backbone)
        self.lm = AutoModelForCausalLM.from_pretrained(
            backbone, config=cfg, dtype=dtype, attn_implementation="sdpa"
        )
        # LoRA sits between the two extremes we measured on the router: a
        # 4.2M head on a frozen backbone reaches intent 0.666, and opening
        # all 1.7B reaches 0.929 but writes a 3.4GB artifact per use case.
        # Adapters keep most of the capacity at a fraction of the size, which
        # is what makes "one backbone, many tasks" possible.
        self.lora_r = lora_r
        if lora_r:
            from peft import LoraConfig, get_peft_model

            self.lm = get_peft_model(
                self.lm,
                LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    bias="none", task_type="CAUSAL_LM",
                    # Attention and MLP projections both matter here: the
                    # readout is a single position, so the model has to route
                    # option content into it rather than just re-weight it.
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                ),
            )

        self.yes_id = _one_token(tokenizer, yes_token)
        self.no_id = _one_token(tokenizer, no_token)
        # NOT an attribute assignment: nn.Module.__setattr__ would register
        # the backbone a second time, duplicating every parameter in the
        # state dict and leaving older checkpoints 311 keys short on load.
        # See the `base` property below.

        # A small head trained on top of the frozen backbone's hidden state.
        #
        # It is a *residual on the zero-shot readout*, not a replacement:
        #
        #     score = (logit_yes - logit_no) + head(h)
        #
        # with the final layer zero-initialised, so at step 0 the model is
        # exactly the untrained zero-shot model and training can only correct
        # it. That matters because the zero-shot readout already reaches 11.5x
        # chance on the held-out suite, and a freshly initialised replacement
        # head would throw that away and have to relearn it from a much
        # smaller signal.
        #
        # What this is for: the untrained readout ranks options weakly and its
        # probabilities are badly calibrated (ECE 0.56-0.81 on the tasks that
        # sit at chance). A few hundred thousand examples of "which option is
        # correct" should fix the calibration without touching the 1.7B
        # backbone, which stays frozen so there are no optimizer states for it.
        self.learned_head = learned_head
        d = cfg.hidden_size
        self.scorer = (
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            if learned_head
            else None
        )
        if self.scorer is not None:
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)

    @property
    def _base(self):
        """The causal LM underneath, whether or not LoRA wraps it.

        A property rather than a stored attribute so the module is registered
        once. PEFT injects adapters into the target submodules in place, so
        reaching the base model still runs them.
        """
        lm = self.lm
        return lm.get_base_model() if hasattr(lm, "get_base_model") else lm

    def merge_adapter(self) -> bool:
        """Fold LoRA into the base weights. Do this before serving.

        Measured on the router under identical load: unmerged LoRA runs at
        p50 74ms against the encoder's 37.7ms, and merged at 40.4ms. Leaving
        it unmerged costs roughly 2x for nothing, because the adapter is an
        extra matmul per target module at every layer.
        """
        if not self.lora_r or not hasattr(self.lm, "merge_and_unload"):
            return False
        self.lm = self.lm.merge_and_unload()
        self.lora_r = 0
        return True

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor], packed: list) -> list[dict]:
        """Same contract as the encoder's, so callers need not know which is which.

        Its absence meant `DecisionModel` raised on any decoder checkpoint:
        the inference path worked only for encoders, which is not something a
        published model can have.
        """
        from .model import Answer, _labels_for

        out = self(batch)
        probs = out.log_probs.exp().cpu()
        results, cursor = [], 0
        for p in packed:
            answers: dict[str, Answer] = {}
            for qid, k in zip(p.question_ids, p.n_options):
                vals = probs[cursor : cursor + k].tolist()
                answers[qid] = Answer(dict(zip(_labels_for(p, qid, k), vals)))
                cursor += k
            results.append(answers)
        return results

    def freeze_backbone(self) -> int:
        """Freeze everything but the head. Returns the trainable count."""
        for n, p in self.lm.named_parameters():
            # Adapters are the trainable part, so freezing "the backbone"
            # must leave them alone or LoRA silently trains nothing and the
            # run looks like an untrained readout.
            p.requires_grad_("lora_" in n)
        if self.scorer is not None:
            for p in self.scorer.parameters():
                p.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _mask_mapping(self, ids, attn, blocks):
        from transformers.masking_utils import create_causal_mask

        probe = torch.empty((*ids.shape, 1), dtype=self._base.dtype, device=ids.device)
        return create_causal_mask(
            config=self._base.config,
            input_embeds=probe,
            attention_mask=attn,
            cache_position=torch.arange(ids.shape[1], device=ids.device),
            past_key_values=None,
            position_ids=None,
            and_mask_function=causal_block_and_mask_fn(blocks),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> OpenJevOutput:
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        blocks = batch["block_id"]

        try:
            mask = self._mask_mapping(ids, attn, blocks)
            out = self._base.model(input_ids=ids, attention_mask=mask)
        except Exception:
            # Fall back to the plain causal mask. Question blocks then see
            # each other, which is wrong but still runs; the isolation test
            # will catch it rather than it passing silently.
            out = self._base.model(input_ids=ids, attention_mask=attn)

        h = out.last_hidden_state
        flat = h.reshape(-1, h.shape[-1])
        markers = flat.index_select(0, batch["marker_flat"])

        vocab = self._base.lm_head(markers)
        logits = (vocab[:, self.yes_id] - vocab[:, self.no_id]).float()
        if self.scorer is not None:
            # Residual: zero at initialisation, so this starts as the exact
            # zero-shot model.
            #
            # The head is float32 while the backbone runs in bf16. Under
            # autocast that mixes silently, so training worked and evaluation
            # outside autocast raised. Match the head's dtype explicitly
            # rather than depending on an ambient context.
            head_dtype = next(self.scorer.parameters()).dtype
            logits = logits + self.scorer(markers.to(head_dtype)).squeeze(-1).float()

        lp = grouped_log_softmax(logits, batch["marker_group"], int(batch["n_groups"]))
        return OpenJevOutput(log_probs=lp, marker_group=batch["marker_group"])


def _one_token(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0]
    alt = tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
    if len(alt) == 1:
        return alt[0]
    raise ValueError(f"{text!r} is not a single token for this tokenizer ({ids})")

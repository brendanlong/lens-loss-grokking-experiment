"""Small pre-norm transformer with per-layer logit-lens outputs, plus the
combined final + logit-lens auxiliary loss.

The forward pass returns lens logits for *every* layer at the answer
position: the residual stream after block l, passed through the final
LayerNorm and the unembedding (the classic logit lens, Nostalgebraist 2020).
The last entry is exactly the model's ordinary output logits, so the final
cross-entropy and the per-layer auxiliary terms come from one tensor.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from grok_lens.config import SEQ_LEN, GrokModelConfig


class Block(nn.Module):
    """Pre-LayerNorm attention + ReLU MLP block."""

    def __init__(self, config: GrokModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim) if config.layernorm else nn.Identity()
        self.attn = nn.MultiheadAttention(config.dim, config.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.dim) if config.layernorm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.mlp_ratio * config.dim),
            nn.ReLU(),
            nn.Linear(config.mlp_ratio * config.dim, config.dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.ln2(x))


class GrokTransformer(nn.Module):
    """Decoder-only transformer over [a, b, =] sequences."""

    causal_mask: torch.Tensor

    def __init__(self, config: GrokModelConfig) -> None:
        super().__init__()
        self.config = config
        # Initialization: PyTorch defaults for all weights, plus std-0.02
        # learned positions. This is NOT Nanda et al.'s custom init; grokking
        # phenomenology is insensitive to this choice and our baselines
        # reproduce the canonical curves (RESULTS.md Phase 1).
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Parameter(torch.zeros(SEQ_LEN, config.dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.dim) if config.layernorm else nn.Identity()
        self.unembed = nn.Linear(config.dim, config.vocab_size, bias=False)
        # nn.MultiheadAttention has no built-in causal masking: its
        # ``is_causal`` argument is only an optimization hint and still
        # requires an explicit mask for correctness, so we register one.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map tokens [B, 3] to per-layer lens logits [L, B, vocab].

        Index l is the logit-lens readout of the residual stream after block
        l at the answer ("=") position; index -1 is the model's actual output
        logits (ln_f + unembed IS the output head).
        """
        x = self.embed(tokens) + self.pos_embed
        answer_resids: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, self.causal_mask)
            # [:, -1] selects the LAST sequence position (the "=" token,
            # where the answer is predicted) — not a reversal.
            answer_resids.append(x[:, -1])
        resids = torch.stack(answer_resids)  # [L, B, dim]
        return self.unembed(self.ln_f(resids))


def intermediate_layer_weights(
    n_layers: int, weighting: Literal["uniform", "linear"]
) -> torch.Tensor:
    """Weights over the L-1 non-final layers, summing to 1 (empty for L=1).

    "uniform" gives each intermediate layer 1/(L-1); "linear" is the
    CALM-style later-weighted schedule w_l ∝ l+1 (0-indexed layer l).
    """
    n_intermediate = n_layers - 1
    if n_intermediate == 0:
        return torch.zeros(0)
    if weighting == "uniform":
        weights = torch.ones(n_intermediate)
    else:
        weights = torch.arange(1, n_intermediate + 1, dtype=torch.float32)
    return weights / weights.sum()


def grok_lens_loss(
    lens_logits: torch.Tensor,
    targets: torch.Tensor,
    layer_weights: torch.Tensor,
    aux_lambda: float,
    aux_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (total_loss, final_ce) for lens logits [L, B, vocab].

    ``aux_targets`` (defaults to ``targets``) lets the intermediate-layer
    CE use different labels than the final loss — the shuffled-target
    specificity control supervises intermediate layers against a fixed
    random permutation of the labels (matched functional form and
    magnitude, but not answer-shaped).

    total = CE(final) + aux_lambda * sum_l w_l * CE_l over the L-1
    intermediate layers, each cross-entropy against the true target.
    ``layer_weights`` comes from :func:`intermediate_layer_weights`, already
    on the right device.
    """
    final_logits = lens_logits[-1]
    final_ce = F.cross_entropy(final_logits, targets)
    n_intermediate = lens_logits.shape[0] - 1
    if aux_lambda == 0.0 or n_intermediate == 0:
        return final_ce, final_ce

    if aux_targets is None:
        aux_targets = targets
    intermediate = lens_logits[:-1]  # [L-1, B, vocab]
    per_layer = (
        F.cross_entropy(
            intermediate.flatten(0, 1),
            aux_targets.repeat(n_intermediate),
            reduction="none",
        )
        .view(n_intermediate, -1)
        .mean(dim=1)
    )
    aux = (layer_weights * per_layer).sum()
    return final_ce + aux_lambda * aux, final_ce

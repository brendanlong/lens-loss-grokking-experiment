"""Standard transformer language model for the LEGO composition task.

Decoder-only transformer with:
- Pre-norm LayerNorm blocks
- Learned positional embeddings
- GELU MLP
- Tied input/output embeddings
- PyTorch-default (Kaiming) initialization
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from lego.config import ModelConfig


class Attention(nn.Module):
    """Multi-head self-attention.

    Positional information comes from learned embeddings added to the
    token embeddings, so attention itself is position-agnostic.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.wo(out))


class GELUFeedForward(nn.Module):
    """GELU feed-forward network (for bAbI-scale models).

    Simple two-layer MLP with GELU activation.
    """

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.gelu(self.w_up(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.ffn = GELUFeedForward(config.dim, config.intermediate_dim, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StandardTransformer(nn.Module):
    """Standard transformer LM with distinct weights per layer."""

    def __init__(
        self,
        config: ModelConfig,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits using tied weights."""
        return F.linear(hidden_states, self.tok_emb.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits."""
        return self.get_logits(self.get_hidden_states(input_ids))

    def forward_with_residuals(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass that also returns residual stream at each layer.

        Returns:
            logits: (batch, seq_len, vocab_size)
            residuals: list of (batch, seq_len, dim) tensors, one per layer
        """
        return self.forward_with_layer_hooks(input_ids, hook=lambda x, _i: x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _embed(self, input_ids: Tensor) -> Tensor:
        """Token + learned positional embedding."""
        x = self.tok_emb(input_ids)
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)
        return x + self.pos_emb(positions)

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning final hidden states before vocab projection."""
        x = self._embed(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                result = checkpoint(layer, x, use_reentrant=False)
                assert isinstance(result, Tensor)
                x = result
            else:
                x = layer(x)
        return self.final_norm(x)

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass that calls hook(residual, layer_idx) after each layer.

        The hook can modify the residual stream (e.g., zero out positions)
        and the modified value is used for subsequent layers.

        Returns:
            logits: (batch, seq_len, vocab_size)
            residuals: list of post-hook (batch, seq_len, dim) tensors
        """
        residuals: list[Tensor] = []
        x = self._embed(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            x = hook(x, layer_idx)
            residuals.append(x)
        x = self.final_norm(x)
        return self.get_logits(x), residuals


# Historical alias from when a weight-shared variant also existed.
AnyModel = StandardTransformer


def create_model(
    config: ModelConfig,
    gradient_checkpointing: bool = False,
) -> StandardTransformer:
    """Create a model from config."""
    return StandardTransformer(config, gradient_checkpointing)


def print_model_summary(model: StandardTransformer) -> None:
    """Print a summary of model architecture and parameter counts."""
    config = model.config

    emb_params = model.tok_emb.weight.numel()
    block_params = sum(p.numel() for p in model.layers.parameters())
    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    total = model.count_parameters()

    print(f"\n{'=' * 50}")
    print("Standard Transformer")
    print(f"{'=' * 50}")
    print(f"  Hidden dim:     {config.dim}")
    print(f"  Heads:          {config.n_heads}")
    print(f"  Layers/Iters:   {config.n_layers}")
    print(f"  MLP dim:        {config.intermediate_dim}")
    print(f"  Vocab size:     {config.vocab_size}")
    print(f"  Context len:    {config.max_seq_len}")
    print("\nParameters:")
    print(f"  Embeddings:     {emb_params:>12,}")
    print(f"  Blocks:         {block_params:>12,}")
    print(f"  Final norm:     {norm_params:>12,}")
    print(f"  Total:          {total:>12,}")
    effective_total = block_params + emb_params + norm_params
    print(f"  Effective:      {effective_total:>12,}")
    print(f"{'=' * 50}\n")

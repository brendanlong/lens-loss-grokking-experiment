"""Standard and weight-shared transformer language models.

Both architectures follow the SmolLM2-135M / Llama pattern:
- Pre-norm (RMSNorm for LLM-scale, LayerNorm for bAbI-scale)
- RoPE positional encoding
- SwiGLU or GELU MLP
- Tied input/output embeddings
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from lego.config import ModelConfig


def _create_norm(config: ModelConfig) -> nn.Module:
    """Create normalization layer based on config."""
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.dim, eps=config.norm_eps)
    return nn.RMSNorm(config.dim, eps=config.norm_eps)


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 100000.0
) -> Tensor:
    """Precompute RoPE complex frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    # Return as complex for easy rotation
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (seq_len, head_dim // 2) complex
    """
    # Reshape x into pairs for complex multiplication
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[: x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


def apply_pope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply PoPE: decouple content magnitude from positional phase.

    Unlike RoPE which rotates the full Q/K vector (entangling content
    phases with positional phases), PoPE uses softplus magnitude for
    content matching and position-only phase, eliminating the
    what-where confound identified by Milsom et al. (arXiv:2509.10534).

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (seq_len, head_dim // 2) complex

    Returns:
        (batch, seq_len, n_heads, head_dim) with interleaved
        [Re_0, Im_0, Re_1, Im_1, ...] where each pair is
        softplus(||x_c||) * (cos(pos*theta_c), sin(pos*theta_c)).
    """
    # Pair adjacent dims -> (batch, seq, heads, head_dim//2, 2)
    x_paired = x.float().reshape(*x.shape[:-1], -1, 2)
    # Content magnitude: softplus of L2 norm per pair
    magnitude = F.softplus(x_paired.norm(dim=-1))  # (..., head_dim//2)

    # Position phase only (no content phase contribution)
    freqs_pos = freqs[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    cos_pos = freqs_pos.real
    sin_pos = freqs_pos.imag

    # magnitude * (cos, sin)
    real_part = magnitude * cos_pos
    imag_part = magnitude * sin_pos
    result = torch.stack([real_part, imag_part], dim=-1).flatten(-2)
    return result.type_as(x)


class Attention(nn.Module):
    """Multi-head self-attention with configurable positional encoding.

    Supports ``rope`` (standard), ``pope`` (decoupled content/position,
    arXiv:2509.10534), and ``learned`` (added to embeddings externally).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.pos_encoding = config.pos_encoding

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim)

        if self.pos_encoding == "pope":
            q = apply_pope(q, rope_freqs)
            k = apply_pope(k, rope_freqs)
        elif self.pos_encoding == "rope":
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)
        # "learned": positional info already added to embeddings

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.wo(out))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network (standard for LLM-scale models)."""

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GELUFeedForward(nn.Module):
    """GELU feed-forward network (for bAbI-scale models).

    Simple two-layer MLP with GELU activation. Used instead of SwiGLU
    for small models where only LayerNorm + GELU is confirmed working.
    """

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.gelu(self.w_up(x))))


def _create_ffn(config: ModelConfig) -> nn.Module:
    """Create feed-forward network based on config."""
    if config.activation == "gelu":
        return GELUFeedForward(config.dim, config.intermediate_dim, config.dropout)
    return SwiGLUFeedForward(config.dim, config.intermediate_dim, config.dropout)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = _create_norm(config)
        self.attn = Attention(config)
        self.ffn_norm = _create_norm(config)
        self.ffn = _create_ffn(config)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class _BaseTransformer(nn.Module):
    """Shared base for standard and weight-shared transformer LMs.

    Subclasses must implement ``get_hidden_states`` and
    ``forward_with_layer_hooks``.
    """

    config: ModelConfig
    gradient_checkpointing: bool
    tok_emb: nn.Embedding
    final_norm: nn.Module
    pos_emb: nn.Embedding | None
    rope_freqs: Tensor

    def _init_weights(self, std: float) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

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
        """Token + positional embedding."""
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
            x = x + self.pos_emb(positions)
        return x

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning final hidden states before vocab projection."""
        raise NotImplementedError

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
        raise NotImplementedError


class StandardTransformer(_BaseTransformer):
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
        self.final_norm = _create_norm(config)

        # Positional encoding
        if config.pos_encoding == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(
                config.max_seq_len,
                config.dim,
            )
        else:
            self.pos_emb = None

        # Precompute RoPE/PoPE frequencies (used by rope and pope modes)
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                config.head_dim, config.max_seq_len, config.rope_theta
            ),
            persistent=False,
        )

        if config.init_std is not None:
            self._init_weights(config.init_std)

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        x = self._embed(input_ids)
        assert isinstance(self.rope_freqs, Tensor)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                result = checkpoint(
                    layer,
                    x,
                    self.rope_freqs,
                    use_reentrant=False,
                )
                assert isinstance(result, Tensor)
                x = result
            else:
                x = layer(x, self.rope_freqs)
        return self.final_norm(x)

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        residuals: list[Tensor] = []
        x = self._embed(input_ids)
        assert isinstance(self.rope_freqs, Tensor)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, self.rope_freqs)
            x = hook(x, layer_idx)
            residuals.append(x)
        x = self.final_norm(x)
        return self.get_logits(x), residuals


class WeightSharedTransformer(_BaseTransformer):
    """Weight-shared (Universal Transformer) LM.

    Uses a single transformer block repeated n_layers times,
    with optional learned iteration embeddings.
    """

    def __init__(
        self,
        config: ModelConfig,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.block = TransformerBlock(config)
        self.final_norm = _create_norm(config)

        # Learned iteration embeddings
        self.use_iteration_embed = config.use_iteration_embed
        self.iteration_embed: nn.Embedding | None = None
        if config.use_iteration_embed:
            self.iteration_embed = nn.Embedding(config.n_layers, config.dim)

        # Input injection: re-inject input embeddings at each iteration
        # (inspired by Parcae). Learned per-dimension gate controls how
        # much of the original input is mixed back in.
        self.use_input_injection = config.use_input_injection
        self.input_injection_gate: nn.Parameter | None = None
        if config.use_input_injection:
            # Initialize near zero so injection starts small
            self.input_injection_gate = nn.Parameter(
                torch.zeros(config.dim) - 2.0,  # sigmoid(-2) ~ 0.12
            )

        # Positional encoding
        if config.pos_encoding == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(
                config.max_seq_len,
                config.dim,
            )
        else:
            self.pos_emb = None

        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                config.head_dim, config.max_seq_len, config.rope_theta
            ),
            persistent=False,
        )

        if config.init_std is not None:
            self._init_weights(config.init_std)

    def _apply_iteration(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        iter_idx: int,
        input_embed: Tensor | None = None,
    ) -> Tensor:
        """Apply one iteration: inject input + add iteration embed + shared block."""
        if input_embed is not None and self.input_injection_gate is not None:
            gate = torch.sigmoid(self.input_injection_gate)
            x = x + gate * input_embed
        if self.iteration_embed is not None:
            x = x + self.iteration_embed.weight[iter_idx]
        return self.block(x, rope_freqs)

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        x = self._embed(input_ids)
        # Save input embedding for injection across iterations
        e = x if self.use_input_injection else None
        assert isinstance(self.rope_freqs, Tensor)
        for i in range(self.config.n_layers):
            if self.gradient_checkpointing and self.training:
                result = checkpoint(
                    self._apply_iteration,
                    x,
                    self.rope_freqs,
                    i,
                    e,
                    use_reentrant=False,
                )
                assert isinstance(result, Tensor)
                x = result
            else:
                x = self._apply_iteration(x, self.rope_freqs, i, e)
        return self.final_norm(x)

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        residuals: list[Tensor] = []
        x = self._embed(input_ids)
        e = x if self.use_input_injection else None
        assert isinstance(self.rope_freqs, Tensor)
        for i in range(self.config.n_layers):
            x = self._apply_iteration(x, self.rope_freqs, i, e)
            x = hook(x, i)
            residuals.append(x)
        x = self.final_norm(x)
        return self.get_logits(x), residuals


AnyModel = StandardTransformer | WeightSharedTransformer


def create_model(
    config: ModelConfig,
    gradient_checkpointing: bool = False,
) -> AnyModel:
    """Create a model from config."""
    if config.weight_shared:
        return WeightSharedTransformer(config, gradient_checkpointing)
    return StandardTransformer(config, gradient_checkpointing)


def print_model_summary(model: AnyModel) -> None:
    """Print a summary of model architecture and parameter counts."""
    config = model.config

    model_type = "Weight-Shared" if config.weight_shared else "Standard"

    emb_params = model.tok_emb.weight.numel()
    if config.weight_shared:
        assert isinstance(model, WeightSharedTransformer)
        block_params = sum(p.numel() for p in model.block.parameters())
        iter_params = (
            model.iteration_embed.weight.numel()
            if model.iteration_embed is not None
            else 0
        )
    else:
        assert isinstance(model, StandardTransformer)
        block_params = sum(p.numel() for p in model.layers.parameters())
        iter_params = 0

    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    total = model.count_parameters()

    print(f"\n{'=' * 50}")
    print(f"{model_type} Transformer")
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
    if iter_params > 0:
        print(f"  Iter embeds:    {iter_params:>12,}")
    print(f"  Final norm:     {norm_params:>12,}")
    print(f"  Total:          {total:>12,}")
    effective = config.n_layers * block_params if config.weight_shared else block_params
    effective_total = effective + emb_params + norm_params + iter_params
    print(f"  Effective:      {effective_total:>12,}")
    print(f"{'=' * 50}\n")

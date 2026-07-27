"""Logit lens probes for the S₃ composition model.

Probes the residual stream at each layer to check whether the model
computes intermediate group composition states step-by-step:

1. At the <predict> position, across layers: does the residual decode to
   successive trajectory states (s₀ → s₁ → … → sₖ)?
2. At <op> token positions across layers: does the model store intermediate
   composition results at the <op> position following each operand?

Used by lego.compare_lens_aux; not a standalone script.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from lego.generator import S3Example
from lego.model import AnyModel
from lego.tokenizer import (
    answer_position,
    element_token,
    encode,
)

# A decode function maps (hidden_state, layer_idx) -> logits.
# Logit lens ignores layer_idx; tuned lens uses it.
DecodeFn = Callable[[Tensor, int], Tensor]


@torch.no_grad()
def logit_lens(
    residual: Tensor,
    final_norm: torch.nn.Module,
    embedding_weight: Tensor,
) -> Tensor:
    """Apply logit lens: norm → unembedding → logits."""
    return F.linear(final_norm(residual), embedding_weight)


def make_logit_lens_fn(model: AnyModel) -> DecodeFn:
    """Create a logit lens DecodeFn from a model."""
    final_norm = model.final_norm
    emb_weight = model.tok_emb.weight

    def decode(hidden_state: Tensor, _layer_idx: int) -> Tensor:
        return logit_lens(hidden_state, final_norm, emb_weight)

    return decode


# --- Generic probe functions ---
# These take pre-computed residuals and a DecodeFn, enabling reuse
# with both logit lens and tuned lens (or any other decode function).


@torch.no_grad()
def probe_predict_position(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Probe at <predict> position across all layers.

    Returns heatmap of shape (n_layers, k+1) where heatmap[L, j] is the
    fraction of examples where layer L's top-1 matches trajectory[j].
    """
    n_layers = len(residuals)
    predict_pos = answer_position(k) - 1
    heatmap = torch.zeros(n_layers, k + 1)

    for layer_idx, residual in enumerate(residuals):
        h = residual[:, predict_pos, :]
        layer_logits = decode_fn(h, layer_idx)
        top1 = layer_logits.argmax(dim=-1)

        for traj_pos in range(k + 1):
            targets = torch.tensor(
                [element_token(ex.trajectory[traj_pos]) for ex in examples],
                device=device,
            )
            heatmap[layer_idx, traj_pos] = (top1 == targets).float().mean().item()

    return heatmap


@torch.no_grad()
def probe_op_positions(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Probe at <op> positions across layers.

    Returns heatmap of shape (n_layers, k) where heatmap[L, j] is the
    fraction of examples where layer L's top-1 at the <op> position
    after operand j+1 matches trajectory[j+1].
    """
    n_layers = len(residuals)
    heatmap = torch.zeros(n_layers, k)

    for layer_idx, residual in enumerate(residuals):
        for j in range(k):
            pos = 2 * (j + 2)  # <op> after operand j+1
            traj_idx = j + 1  # trajectory state after j+1 ops
            h = residual[:, pos, :]
            layer_logits = decode_fn(h, layer_idx)
            top1 = layer_logits.argmax(dim=-1)

            targets = torch.tensor(
                [element_token(ex.trajectory[traj_idx]) for ex in examples],
                device=device,
            )
            heatmap[layer_idx, j] = (top1 == targets).float().mean().item()

    return heatmap


# --- Convenience wrappers (run forward pass + probe with logit lens) ---


@torch.no_grad()
def analyze_predict_position(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Logit lens at <predict> position across all layers.

    Returns heatmap of shape (n_layers, k+1) where heatmap[L, j] is the
    fraction of examples where layer L's logit lens top-1 matches
    trajectory position j.
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    return probe_predict_position(
        residuals,
        examples,
        make_logit_lens_fn(model),
        k,
        device,
    )


@torch.no_grad()
def analyze_op_positions(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Logit lens at <op> token positions across layers.

    The <op> token at position 2*(j+1) is the first position that has seen
    all information needed to compute trajectory[j] (causally: it can see
    the start element and operands g₁…gⱼ):
        pos 4  → t[1] (seen start + g₁)
        pos 6  → t[2] (seen start + g₁ + g₂)
        ...
        pos 2(k+1) → t[k] (= <predict> position)

    Returns heatmap of shape (n_layers, k) where heatmap[L, j] is the
    fraction of examples where layer L's logit lens at the <op> position
    after operand j+1 decodes to trajectory[j+1].
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    return probe_op_positions(
        residuals,
        examples,
        make_logit_lens_fn(model),
        k,
        device,
    )

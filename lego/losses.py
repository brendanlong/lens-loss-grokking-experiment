"""Losses and accuracy metrics for the LEGO group composition task.

The base objective is answer-only cross-entropy at the <predict> position.
The lens auxiliary loss (grok_lens-style deep supervision) comes in two
modes:

- "answer": per-layer logit-lens CE against the FINAL answer at the
  <predict> position only (compute_lens_aux_loss);
- "all-positions": per-layer logit-lens CE at EVERY non-pad position
  against the next token, i.e. standard autoregressive next-token
  supervision applied to intermediate layers
  (compute_lens_aux_loss_all_positions).
"""

import torch
from torch import Tensor

from lego.tokenizer import PAD_ID


def compute_answer_only_loss(
    logits: Tensor,
    input_ids: Tensor,
    answer_positions: Tensor,
) -> Tensor:
    """Cross-entropy loss at the answer position only.

    The model predicts the answer element from logits at the <predict>
    token position (= answer_position - 1).
    """
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    predict_logits = logits[batch_idx, answer_positions - 1]  # (batch, vocab)
    targets = input_ids[batch_idx, answer_positions]  # (batch,)
    return torch.nn.functional.cross_entropy(predict_logits, targets)


def _layer_weights(n_intermediate: int, weighting: str, device: torch.device) -> Tensor:
    """Per-layer weights over the L-1 intermediate layers, summing to 1.

    "uniform": 1/(L-1) each; "linear": CALM-style later-weighted,
    w_l proportional to l+1.
    """
    if weighting == "uniform":
        weights = torch.ones(n_intermediate, device=device)
    else:
        weights = torch.arange(
            1, n_intermediate + 1, dtype=torch.float32, device=device
        )
    return weights / weights.sum()


def compute_lens_aux_loss(
    residuals: list[Tensor],
    input_ids: Tensor,
    answer_positions: Tensor,
    final_norm: torch.nn.Module,
    embedding_weight: Tensor,
    *,
    weighting: str = "uniform",
) -> Tensor:
    """grok_lens-style deep supervision at the answer position.

    At the <predict> position (answer_position - 1), project every
    *intermediate* layer's residual through the final norm + tied
    unembedding (the logit lens) and add cross-entropy against the FINAL
    answer. Unlike the staircase loss (which targets intermediate
    trajectory values at <op> positions), this forces every layer to be
    answer-shaped at the one supervised position — the grok_lens
    experiment's aux loss, ported to LEGO to test the "dark space"
    predictions (see experiments/grok_lens/EXPERIMENT_PLAN.md Phase 5).

    Args:
        residuals: Per-layer residual streams, each (batch, seq_len, dim).
        input_ids: (batch, seq_len) token ids.
        answer_positions: (batch,) index of the answer token.
        final_norm: Model's final norm (logit-lens projection).
        embedding_weight: Tied embedding/unembedding weight matrix.
        weighting: "uniform" (1/(L-1) each) or "linear" (CALM-style
            later-weighted, w_l proportional to l+1); weights sum to 1.

    Returns:
        Scalar loss (0 for single-layer models).
    """
    n_intermediate = len(residuals) - 1
    device = residuals[0].device
    if n_intermediate == 0:
        return torch.tensor(0.0, device=device)

    batch_idx = torch.arange(input_ids.size(0), device=device)
    targets = input_ids[batch_idx, answer_positions]  # (batch,)
    predict_resids = torch.stack(
        [r[batch_idx, answer_positions - 1] for r in residuals[:-1]]
    )  # (L-1, batch, dim)
    lens_logits = torch.nn.functional.linear(
        final_norm(predict_resids), embedding_weight
    )
    per_layer = (
        torch.nn.functional.cross_entropy(
            lens_logits.flatten(0, 1),
            targets.repeat(n_intermediate),
            reduction="none",
        )
        .view(n_intermediate, -1)
        .mean(dim=1)
    )
    weights = _layer_weights(n_intermediate, weighting, device)
    return (weights * per_layer).sum()


def compute_lens_aux_loss_all_positions(
    residuals: list[Tensor],
    input_ids: Tensor,
    final_norm: torch.nn.Module,
    embedding_weight: Tensor,
    *,
    weighting: str = "uniform",
) -> Tensor:
    """Full-sequence lens deep supervision: next-token CE at every position.

    For every *intermediate* layer (final layer excluded), apply the logit
    lens (final norm + tied unembedding) at every position t and score the
    NEXT token input_ids[:, t+1] — the standard autoregressive shift. Pad
    targets are masked out via ignore_index=PAD_ID (0), so only non-pad
    target positions contribute; each layer's CE is the mean over its
    non-pad targets.

    Args:
        residuals: Per-layer residual streams, each (batch, seq_len, dim).
        input_ids: (batch, seq_len) token ids, 0 = <pad>.
        final_norm: Model's final norm (logit-lens projection).
        embedding_weight: Tied embedding/unembedding weight matrix.
        weighting: "uniform" or "linear", as in compute_lens_aux_loss;
            weights over the L-1 intermediate layers sum to 1.

    Returns:
        Scalar loss (0 for single-layer models).
    """
    n_intermediate = len(residuals) - 1
    device = residuals[0].device
    if n_intermediate == 0:
        return torch.tensor(0.0, device=device)

    targets = input_ids[:, 1:].reshape(-1)  # (batch * (seq_len - 1),)
    # (L-1, batch, seq_len - 1, dim): logits at t score input_ids[t + 1]
    stacked = torch.stack([r[:, :-1] for r in residuals[:-1]])
    lens_logits = torch.nn.functional.linear(final_norm(stacked), embedding_weight)
    per_layer = torch.stack(
        [
            torch.nn.functional.cross_entropy(
                layer_logits.flatten(0, 1),
                targets,
                ignore_index=PAD_ID,
            )
            for layer_logits in lens_logits
        ]
    )
    weights = _layer_weights(n_intermediate, weighting, device)
    return (weights * per_layer).sum()


@torch.no_grad()
def compute_answer_accuracy(
    logits: Tensor,
    input_ids: Tensor,
    answer_positions: Tensor,
) -> Tensor:
    """Accuracy on the answer token.

    Returns a scalar Tensor (stays on device to avoid GPU→CPU sync).
    Call .item() only when you need the Python float.
    """
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    predict_logits = logits[batch_idx, answer_positions - 1]
    targets = input_ids[batch_idx, answer_positions]
    predictions = predict_logits.argmax(dim=-1)
    return (predictions == targets).float().mean()

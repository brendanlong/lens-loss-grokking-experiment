"""Hybrid Muon parameter routing (Jordan et al. 2024 via torch.optim.Muon).

Used in the optimizer-robustness arm: is the sparse-circuit / slingshot-
instability correlation an AdamW-regime artifact? Muon applies only to 2-D
hidden-layer weight matrices (standard hybrid usage, as in "Muon is
Scalable" / Kimi K2); embeddings, unembedding, positional embeddings, and
all 1-D params (norms, biases) stay on AdamW. The optimizer itself is
``torch.optim.Muon`` (Newton-Schulz orthogonalized momentum with decoupled
weight decay) — this module only supplies the parameter split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def split_muon_params(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split params into (muon_params, adamw_params).

    Muon gets the 2-D weight matrices inside transformer blocks; everything
    else (embed, pos_embed, unembed, norms, biases) goes to AdamW.
    """
    muon_params: list[torch.Tensor] = []
    adamw_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if name.startswith("blocks.") and param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params

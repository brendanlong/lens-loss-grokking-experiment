"""Logit lens analysis for S₃ composition model.

Probes the residual stream at each layer to check whether the model
computes intermediate group composition states step-by-step.

Three analyses:
1. At the <predict> position, across layers: does the residual decode to
   successive trajectory states (s₀ → s₁ → … → sₖ)?
2. At <op> token positions across layers: does the model store intermediate
   composition results at the <op> position following each operand?
   Key finding: the model stores trajectory[j] at pos 2*(j+1) (the <op>
   token after the j-th operand), with a clear staircase pattern across
   layers — earlier positions converge first.
3. At each element position, at the final layer: does the residual decode
   to the running state at that point in the chain?

Usage:
    uv run python -m lego.analyze_logit_lens \
        --checkpoint data/lego/checkpoints/step_19531.pt \
        --k 6 --n-examples 500
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from lego.generator import (
    ELEMENTS,
    S3Example,
    generate_fixed_dataset,
)
from lego.model import AnyModel
from lego.tokenizer import (
    answer_position,
    element_token,
    encode,
)
from lego.training import load_model

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
def probe_element_positions(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Probe at element positions across all layers.

    Returns heatmap of shape (n_layers, k+1) where heatmap[L, j] is the
    fraction of examples where layer L's top-1 at element position j
    decodes to trajectory[j].
    """
    n_layers = len(residuals)
    elem_positions = [2 * i + 1 for i in range(k + 1)]
    heatmap = torch.zeros(n_layers, k + 1)

    for layer_idx, residual in enumerate(residuals):
        for j, pos in enumerate(elem_positions):
            h = residual[:, pos, :]
            layer_logits = decode_fn(h, layer_idx)
            top1 = layer_logits.argmax(dim=-1)

            targets = torch.tensor(
                [element_token(ex.trajectory[j]) for ex in examples],
                device=device,
            )
            heatmap[layer_idx, j] = (top1 == targets).float().mean().item()

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


@torch.no_grad()
def probe_probability_mass(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Softmax probability mass on trajectory states at <predict> position.

    Like probe_predict_position but uses softmax probability instead of
    top-1 accuracy. Smoother metric that reveals partial structure.

    Returns heatmap of shape (n_layers, k+1).
    """
    n_layers = len(residuals)
    predict_pos = answer_position(k) - 1
    heatmap = torch.zeros(n_layers, k + 1)

    for layer_idx, residual in enumerate(residuals):
        h = residual[:, predict_pos, :]
        layer_logits = decode_fn(h, layer_idx)
        probs = F.softmax(layer_logits, dim=-1)

        for traj_pos in range(k + 1):
            targets = torch.tensor(
                [element_token(ex.trajectory[traj_pos]) for ex in examples],
                device=device,
            )
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            heatmap[layer_idx, traj_pos] = target_probs.mean().item()

    return heatmap


@torch.no_grad()
def probe_op_probability_mass(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Softmax probability mass on trajectory states at <op> positions.

    Like probe_op_positions but uses softmax probability instead of
    top-1 accuracy.

    Returns heatmap of shape (n_layers, k).
    """
    n_layers = len(residuals)
    heatmap = torch.zeros(n_layers, k)

    for layer_idx, residual in enumerate(residuals):
        for j in range(k):
            pos = 2 * (j + 2)
            traj_idx = j + 1
            h = residual[:, pos, :]
            layer_logits = decode_fn(h, layer_idx)
            probs = F.softmax(layer_logits, dim=-1)

            targets = torch.tensor(
                [element_token(ex.trajectory[traj_idx]) for ex in examples],
                device=device,
            )
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            heatmap[layer_idx, j] = target_probs.mean().item()

    return heatmap


@torch.no_grad()
def compute_embedding_subspace_basis(model: AnyModel) -> Tensor:
    """Compute orthonormal basis for the embedding subspace via SVD.

    Returns Vt of shape (rank, d_model) where rank <= vocab_size.
    The rows of Vt span the embedding subspace.
    """
    E = model.tok_emb.weight.detach().float()  # (vocab_size, d_model)
    _U, _S, Vt = torch.linalg.svd(E, full_matrices=False)
    return Vt  # (min(vocab_size, d_model), d_model)


@torch.no_grad()
def probe_embedding_explained_variance(
    residuals: list[Tensor],
    positions: list[int],
    embedding_basis: Tensor,
    final_norm: torch.nn.Module,
) -> Tensor:
    """Fraction of residual norm in the embedding subspace at given positions.

    Applies the model's final norm before projecting (matching what the
    logit lens does), then measures what fraction of the normalized
    residual's squared norm lies in the span of the embedding vectors.

    Args:
        residuals: Per-layer residual streams, each (batch, seq_len, d_model).
        positions: Sequence positions to probe.
        embedding_basis: Orthonormal basis from compute_embedding_subspace_basis.
        final_norm: The model's final LayerNorm/RMSNorm.

    Returns:
        Heatmap of shape (n_layers, n_positions) with values in [0, 1].
    """
    n_layers = len(residuals)
    n_positions = len(positions)
    heatmap = torch.zeros(n_layers, n_positions)
    Vt = embedding_basis.float()  # (rank, d_model)

    for layer_idx, residual in enumerate(residuals):
        for j, pos in enumerate(positions):
            h = residual[:, pos, :].float()  # (batch, d_model)
            h = final_norm(h)
            # Project onto embedding subspace: h_proj = h @ Vt.T @ Vt
            coeffs = h @ Vt.T  # (batch, rank)
            h_proj = coeffs @ Vt  # (batch, d_model)
            proj_norm_sq = (h_proj**2).sum(dim=-1)  # (batch,)
            total_norm_sq = (h**2).sum(dim=-1)  # (batch,)
            # Avoid division by zero
            ratio = proj_norm_sq / total_norm_sq.clamp(min=1e-10)
            heatmap[layer_idx, j] = ratio.mean().item()

    return heatmap


@torch.no_grad()
def probe_logit_entropy(
    residuals: list[Tensor],
    positions: list[int],
    decode_fn: DecodeFn,
) -> Tensor:
    """Entropy of the logit lens distribution at given positions.

    Measures how peaked vs uniform the logit lens output is.
    Normalized by log(vocab_size) so values are in [0, 1]:
    - 0 = all probability on one token (maximally certain)
    - 1 = uniform over all tokens (maximally uncertain)

    Combined with evar, this distinguishes:
    - High evar + low entropy = on-manifold, confident (logit lens works)
    - High evar + high entropy = on-manifold, ambiguous (between tokens)
    - Low evar + any entropy = off-manifold (logit lens reads noise)

    Returns heatmap of shape (n_layers, n_positions).
    """
    import math

    n_layers = len(residuals)
    n_positions = len(positions)
    heatmap = torch.zeros(n_layers, n_positions)
    if n_positions == 0:
        return heatmap

    # Get vocab size from first decode call
    h0 = residuals[0][:1, positions[0] : positions[0] + 1, :]
    vocab_size = decode_fn(h0.squeeze(1), 0).shape[-1]
    max_entropy = math.log(vocab_size)

    for layer_idx, residual in enumerate(residuals):
        for j, pos in enumerate(positions):
            h = residual[:, pos, :]
            logits = decode_fn(h, layer_idx)
            probs = F.softmax(logits, dim=-1)
            # Entropy: -sum(p * log(p)), avoiding log(0)
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch,)
            heatmap[layer_idx, j] = (entropy / max_entropy).mean().item()

    return heatmap


@torch.no_grad()
def probe_principal_angles(
    residuals: list[Tensor],
    positions: list[int],
    subspace_dim: int = 10,
) -> Tensor:
    """Principal angles between each layer's activation subspace and the final layer.

    For each layer, collects activations at the given positions across all
    examples, computes PCA to get the top-k subspace, then measures the
    mean cosine of principal angles with the final layer's subspace.

    Values near 1.0 mean the layer's representations live in the same
    subspace as the final layer. Values near 0.0 mean orthogonal subspaces.
    UT should show higher values (representations stay in the same subspace).

    Args:
        residuals: Per-layer residual streams, each (batch, seq_len, d_model).
        positions: Sequence positions to collect activations from.
        subspace_dim: Number of PCA components to use for subspace comparison.

    Returns:
        Vector of shape (n_layers,) with mean cosine of principal angles
        between each layer and the final layer.
    """
    n_layers = len(residuals)
    result = torch.zeros(n_layers)
    if not positions:
        return result

    # Collect per-layer activation matrices
    # Shape: (n_layers, n_examples * n_positions, d_model)
    layer_activations: list[Tensor] = []
    for residual in residuals:
        acts = torch.cat(
            [residual[:, pos, :] for pos in positions],
            dim=0,
        ).float()
        # Center
        acts = acts - acts.mean(dim=0, keepdim=True)
        layer_activations.append(acts)

    # Compute PCA basis for each layer (top-k right singular vectors)
    layer_bases: list[Tensor] = []
    for acts in layer_activations:
        _U, _S, Vt = torch.linalg.svd(acts, full_matrices=False)
        # Take top subspace_dim components
        k = min(subspace_dim, Vt.shape[0])
        layer_bases.append(Vt[:k, :])  # (k, d_model)

    # Compare each layer to the final layer
    final_basis = layer_bases[-1]  # (k, d_model)
    for layer_idx, basis in enumerate(layer_bases):
        # Principal angles: SVD of B_l @ B_final^T
        cos_angles = torch.linalg.svdvals(basis @ final_basis.T)
        # Clamp to [0, 1] for numerical safety
        cos_angles = cos_angles.clamp(0.0, 1.0)
        result[layer_idx] = cos_angles.mean().item()

    return result


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
def analyze_element_positions(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Logit lens at each element position at each layer.

    Element positions in a k-op chain:
        pos 1: start element → trajectory[0]
        pos 3: op1 element → trajectory[1] could be computed
        pos 5: op2 element → trajectory[2] could be computed
        ...
        pos 2i+1: opi element → trajectory[i] could be computed

    Returns heatmap of shape (n_layers, k+1) where heatmap[L, j] is the
    fraction of examples where layer L's logit lens at element position j
    decodes to trajectory[j].
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    return probe_element_positions(
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


@torch.no_grad()
def analyze_probability_mass(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Probability mass on correct trajectory state at <predict> position.

    Like analyze_predict_position but uses softmax probability instead of
    top-1 accuracy. Smoother metric that reveals partial structure.

    Returns heatmap of shape (n_layers, k+1).
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    return probe_probability_mass(
        residuals,
        examples,
        make_logit_lens_fn(model),
        k,
        device,
    )


@torch.no_grad()
def analyze_op_probability_mass(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Probability mass on correct trajectory states at <op> positions.

    Returns heatmap of shape (n_layers, k).
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    return probe_op_probability_mass(
        residuals,
        examples,
        make_logit_lens_fn(model),
        k,
        device,
    )


@torch.no_grad()
def analyze_embedding_explained_variance(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Embedding subspace explained variance at <op> positions.

    Measures what fraction of the (normed) residual stream lies in the
    span of the token embedding vectors. High values mean the residual
    is "in" the embedding space; low values mean it's in directions
    the logit lens can't see.

    Returns heatmap of shape (n_layers, k).
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    basis = compute_embedding_subspace_basis(model)
    op_positions = [2 * (j + 2) for j in range(k)]
    return probe_embedding_explained_variance(
        residuals,
        op_positions,
        basis,
        model.final_norm,
    )


@torch.no_grad()
def analyze_logit_entropy(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
) -> Tensor:
    """Normalized entropy of logit lens distribution at <op> positions.

    0 = maximally peaked (all mass on one token).
    1 = uniform over vocab (maximally uncertain).

    Combined with evar:
    - High evar + low entropy = on-manifold, confident
    - High evar + high entropy = on-manifold, ambiguous
    - Low evar = off-manifold (entropy is noise)

    Returns heatmap of shape (n_layers, k).
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    op_positions = [2 * (j + 2) for j in range(k)]
    return probe_logit_entropy(
        residuals,
        op_positions,
        make_logit_lens_fn(model),
    )


@torch.no_grad()
def analyze_principal_angles(
    model: AnyModel,
    examples: list[S3Example],
    device: torch.device,
    subspace_dim: int = 10,
) -> Tensor:
    """Mean cosine of principal angles between each layer and the final layer.

    Measures how much the activation subspace at <op> positions rotates
    across layers. Values near 1.0 = same subspace, near 0.0 = orthogonal.
    UT should show higher values (representations stay in same subspace).

    Returns vector of shape (n_layers,).
    """
    k = len(examples[0].ops)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    op_positions = [2 * (j + 2) for j in range(k)]
    return probe_principal_angles(residuals, op_positions, subspace_dim)


def print_heatmap(
    heatmap: Tensor,
    title: str,
    row_label: str = "Layer",
    col_labels: list[str] | None = None,
    fmt: str = ".0%",
) -> None:
    """Print a text heatmap."""
    n_rows, n_cols = heatmap.shape
    if col_labels is None:
        col_labels = [f"t[{j}]" for j in range(n_cols)]

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    # Header
    col_width = max(6, max(len(c) for c in col_labels) + 1)
    header = f"{row_label:>6s}" + "".join(f"{c:>{col_width}s}" for c in col_labels)
    print(header)
    print("-" * len(header))

    for i in range(n_rows):
        row_vals = "".join(
            f"{heatmap[i, j].item():{col_width}{fmt}}" for j in range(n_cols)
        )
        print(f"  L{i:<3d} {row_vals}")


def draw_heatmap(
    ax: object,
    data: object,
    title: str,
    col_labels: list[str],
    row_labels: list[str],
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Draw an annotated heatmap on a matplotlib axis.

    Shared across all lens analysis plotting functions.

    Args:
        ax: matplotlib Axes object.
        data: 2D array-like of shape (n_rows, n_cols).
        title: Panel title.
        col_labels: X-axis tick labels.
        row_labels: Y-axis tick labels.
        cmap: Colormap name. Use "RdYlGn" for improvement/delta heatmaps.
        vmin: Colorbar minimum.
        vmax: Colorbar maximum.
    """
    import numpy as np

    arr = np.asarray(data)
    n_rows, n_cols = arr.shape

    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)  # type: ignore[union-attr]
    ax.set_title(title, fontsize=9)  # type: ignore[union-attr]
    ax.set_xlabel("State")  # type: ignore[union-attr]
    ax.set_ylabel("Layer")  # type: ignore[union-attr]
    ax.set_xticks(range(n_cols))  # type: ignore[union-attr]
    ax.set_xticklabels(col_labels, fontsize=7)  # type: ignore[union-attr]
    ax.set_yticks(range(n_rows))  # type: ignore[union-attr]
    ax.set_yticklabels(row_labels, fontsize=7)  # type: ignore[union-attr]

    for i in range(n_rows):
        for j in range(n_cols):
            val = float(arr[i, j])
            if cmap == "RdYlGn":
                color = "black"
                text = f"{val:+.0%}"
            else:
                color = "white" if val < 0.5 else "black"
                text = f"{val:.0%}"
            ax.text(  # type: ignore[union-attr]
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )

    ax.figure.colorbar(im, ax=ax, shrink=0.8)  # type: ignore[union-attr]


def save_heatmap_figure(
    heatmap_predict: Tensor,
    heatmap_op: Tensor,
    heatmap_elem: Tensor,
    prob_heatmap: Tensor,
    k: int,
    n_layers: int,
    output_path: Path,
) -> None:
    """Save matplotlib figure with logit lens heatmaps."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes_arr = plt.subplots(1, 4, figsize=(24, 5))
    assert isinstance(axes_arr, np.ndarray)

    traj_labels = [f"traj[{j}]" for j in range(k + 1)]
    op_labels = [f"p{2 * (j + 1)}→t[{j + 1}]" for j in range(k)]
    layer_labels = [f"L{i}" for i in range(n_layers)]

    panels: list[tuple[object, object, str, list[str]]] = [
        (
            axes_arr[0],
            heatmap_predict.numpy(),
            "Logit lens at <predict> pos\n(top-1 match rate)",
            traj_labels,
        ),
        (
            axes_arr[1],
            prob_heatmap.numpy(),
            "Logit lens at <predict> pos\n(probability mass)",
            traj_labels,
        ),
        (
            axes_arr[2],
            heatmap_op.numpy(),
            "Logit lens at <op> positions\n(top-1 — staircase pattern)",
            op_labels,
        ),
        (
            axes_arr[3],
            heatmap_elem.numpy(),
            "Logit lens at element positions\n(top-1 match rate)",
            traj_labels,
        ),
    ]

    for ax, data, title, xlabels in panels:
        draw_heatmap(ax, data, title, xlabels, layer_labels)

    fig.suptitle(f"S₃ Logit Lens Analysis (k={k})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens analysis for S₃ composition model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/lego/checkpoints/step_19531.pt",
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--output",
        type=str,
        default="data/lego/logit_lens_s3.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_config = load_model(args.checkpoint, device)

    n_layers = model_config.n_layers
    print(f"Model: {model_config.dim}d, {n_layers}L, {model_config.n_heads}h")

    # Generate examples
    print(f"Generating {args.n_examples} examples with k={args.k}")
    examples = generate_fixed_dataset(args.k, args.n_examples, seed=args.seed)

    # Show a few examples
    print("\nSample examples:")
    for ex in examples[:3]:
        traj_str = " → ".join(ELEMENTS[s] for s in ex.trajectory)
        print(f"  {traj_str}")

    # Analysis 1: <predict> position across layers (top-1)
    heatmap_predict = analyze_predict_position(model, examples, device)
    print_heatmap(
        heatmap_predict,
        f"Logit lens at <predict> position (top-1 accuracy, k={args.k})",
    )

    # Analysis 2: probability mass at <predict> position
    prob_heatmap = analyze_probability_mass(model, examples, device)
    print_heatmap(
        prob_heatmap,
        f"Logit lens at <predict> position (probability mass, k={args.k})",
        fmt=".0%",
    )

    # Analysis 3: <op> positions across layers (staircase pattern)
    heatmap_op = analyze_op_positions(model, examples, device)
    op_col_labels = [f"p{2 * (j + 1)}→t[{j + 1}]" for j in range(args.k)]
    print_heatmap(
        heatmap_op,
        f"Logit lens at <op> positions (top-1 accuracy, k={args.k})",
        col_labels=op_col_labels,
    )

    # Analysis 4: element positions at each layer
    heatmap_elem = analyze_element_positions(model, examples, device)
    col_labels = ["start"] + [f"op{j}" for j in range(1, args.k + 1)]
    print_heatmap(
        heatmap_elem,
        f"Logit lens at element positions (top-1 accuracy, k={args.k})",
        col_labels=col_labels,
    )

    # Interpretation summary
    print(f"\n{'=' * 60}")
    print("Interpretation")
    print(f"{'=' * 60}")

    # Check for staircase at <op> positions
    print("\nAt <op> positions (staircase pattern):")
    print("  The model stores intermediate composition results at <op> tokens.")
    print("  pos 2*(j+1) encodes trajectory[j] — the state after j ops.\n")
    for j in range(args.k):
        pos = 2 * (j + 1)
        # Find first layer above 50%
        first_layer = None
        for layer_idx in range(n_layers):
            if heatmap_op[layer_idx, j].item() > 0.5:
                first_layer = layer_idx
                break
        final_val = heatmap_op[n_layers - 1, j].item()
        if first_layer is not None:
            print(
                f"  t[{j + 1}] at pos {pos}: first >50% at L{first_layer}, "
                f"final={final_val:.0%}"
            )
        else:
            print(f"  t[{j + 1}] at pos {pos}: never >50% (final={final_val:.0%})")

    # Check for diagonal pattern at <predict> position
    print("\nAt <predict> position, dominant trajectory match per layer:")
    for layer_idx in range(n_layers):
        best_j = int(heatmap_predict[layer_idx].argmax().item())
        best_val = heatmap_predict[layer_idx, best_j].item()
        if best_val > 0.2:
            print(
                f"  L{layer_idx}: trajectory[{best_j}] "
                f"({best_val:.0%}, = state after {best_j} ops)"
            )
        else:
            print(f"  L{layer_idx}: no clear match (max {best_val:.0%})")

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_heatmap_figure(
        heatmap_predict,
        heatmap_op,
        heatmap_elem,
        prob_heatmap,
        args.k,
        n_layers,
        output_path,
    )


if __name__ == "__main__":
    main()

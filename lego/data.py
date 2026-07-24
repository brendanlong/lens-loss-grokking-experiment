"""Dataset and training utilities for group composition task.

Each training example is a chain of group operations:
    <start> elem <op> elem … <predict> answer

The model predicts the answer (final group element) from the logits
at the <predict> position.

Training samples chain lengths k from k_min to k_max, either uniformly
or with optional power-law weighting (k_power > 0 gives weight(k) = k^power).
Evaluation measures accuracy per chain length k.
"""

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.streaming import SyntheticStream
from lego.generator import S3, Group, S3Example, generate_example
from lego.tokenizer import (
    Tokenizer,
    answer_position,
    seq_len,
)

# Default S3 tokenizer for backward-compatible functions
_S3_TOKENIZER = Tokenizer(S3)


def encode_trajectory(
    example: S3Example,
    k_max: int,
    tokenizer: Tokenizer | None = None,
) -> list[int]:
    """Encode trajectory as element tokens, padded to k_max + 1.

    trajectory[j] = accumulated result after j operations (as element token).
    Padded positions use PAD_ID.
    """
    tok = tokenizer or _S3_TOKENIZER
    tokens = [tok.element_token(t) for t in example.trajectory]
    tokens.extend([tok.pad_id] * (k_max + 1 - len(tokens)))
    return tokens


class S3FixedDataset(Dataset[dict[str, Tensor]]):
    """Fixed dataset of group composition chains, pre-encoded with padding."""

    def __init__(
        self,
        examples: list[S3Example],
        k_max: int,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.max_seq_len = seq_len(k_max)
        tok = tokenizer or _S3_TOKENIZER

        input_ids_list: list[Tensor] = []
        answer_pos_list: list[int] = []
        chain_len_list: list[int] = []
        trajectory_list: list[Tensor] = []

        for ex in examples:
            tokens = tok.encode_padded(ex, k_max)
            k = len(ex.ops)
            input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
            answer_pos_list.append(answer_position(k))
            chain_len_list.append(k)
            trajectory_list.append(
                torch.tensor(
                    encode_trajectory(ex, k_max, tok),
                    dtype=torch.long,
                ),
            )

        self.input_ids = torch.stack(input_ids_list)
        self.answer_positions = torch.tensor(answer_pos_list, dtype=torch.long)
        self.chain_lengths = torch.tensor(chain_len_list, dtype=torch.long)
        self.trajectories = torch.stack(trajectory_list)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "answer_position": self.answer_positions[idx],
            "chain_length": self.chain_lengths[idx],
            "trajectory": self.trajectories[idx],
        }


class S3StreamingDataset(SyntheticStream[dict[str, Tensor]]):
    """Streaming dataset generating fresh group composition examples on the fly.

    Worker sharding/seeding and the per-epoch seed mixing are handled by
    :class:`common.streaming.SyntheticStream`.
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        n_examples: int,
        seed: int = 42,
        k_power: float = 0.0,
        group: Group = S3,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(n_examples=n_examples, seed=seed)
        self.k_min = k_min
        self.k_max = k_max
        self.k_power = k_power
        self.group = group
        self.tokenizer = tokenizer or _S3_TOKENIZER

        # Precompute weighted k distribution: weight(k) = max(1, k)^power
        # Using max(1, k) so k=0 gets nonzero weight when k_power > 0
        k_values = list(range(k_min, k_max + 1))
        self._k_values = k_values
        self._k_weights = [max(1, k) ** k_power for k in k_values]

    def _sample_k(self, rng: random.Random) -> int:
        """Sample k from the weighted distribution."""
        if self.k_power == 0.0:
            return rng.randint(self.k_min, self.k_max)
        return rng.choices(self._k_values, weights=self._k_weights)[0]

    def generate(self, rng: random.Random) -> dict[str, Tensor]:
        k = self._sample_k(rng)
        ex = generate_example(k, rng, self.group)
        tokens = self.tokenizer.encode_padded(ex, self.k_max)
        ans_pos = answer_position(k)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "answer_position": torch.tensor(ans_pos, dtype=torch.long),
            "chain_length": torch.tensor(k, dtype=torch.long),
            "trajectory": torch.tensor(
                encode_trajectory(ex, self.k_max, self.tokenizer),
                dtype=torch.long,
            ),
        }


def collate_s3(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate batch of group composition examples."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "answer_position": torch.stack([b["answer_position"] for b in batch]),
        "chain_length": torch.stack([b["chain_length"] for b in batch]),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
    }


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


def compute_full_sequence_loss(
    logits: Tensor,
    input_ids: Tensor,
) -> Tensor:
    """Standard autoregressive loss on all non-pad positions.

    At each position i, the model predicts token i+1. Positions where
    the target is PAD are masked out (ignored in loss).

    This gives more training signal per example: the model learns to
    predict structural tokens (<op>, <predict>) and the answer.
    Operand elements are uniformly random, so the optimal prediction
    there is a uniform distribution over elements (calibrated logits).
    """
    # logits[:, :-1] predicts input_ids[:, 1:]
    shift_logits = logits[:, :-1].contiguous()  # (B, T-1, V)
    shift_targets = input_ids[:, 1:].contiguous()  # (B, T-1)

    # Mask out PAD targets (PAD_ID is always 0 regardless of group)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=0,  # PAD_ID = 0 for all tokenizers
    )
    return loss


def compute_loss(
    logits: Tensor,
    input_ids: Tensor,
    answer_positions: Tensor,
    *,
    full_sequence: bool = False,
) -> Tensor:
    """Compute training loss.

    If full_sequence=False (default): loss at the answer position only.
    If full_sequence=True: standard autoregressive loss on all non-pad tokens.
    """
    if full_sequence:
        return compute_full_sequence_loss(logits, input_ids)
    return compute_answer_only_loss(logits, input_ids, answer_positions)


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
    if weighting == "uniform":
        weights = torch.ones(n_intermediate, device=device)
    else:
        weights = torch.arange(
            1, n_intermediate + 1, dtype=torch.float32, device=device
        )
    weights = weights / weights.sum()
    return (weights * per_layer).sum()


def compute_staircase_loss(
    residuals: list[Tensor],
    trajectories: Tensor,
    chain_lengths: Tensor,
    final_norm: torch.nn.Module,
    embedding_weight: Tensor,
    *,
    start_layer: int = 0,
    k_max: int = 6,
) -> Tensor:
    """Auxiliary loss forcing intermediate values at specific (layer, position) pairs.

    At the j-th <op> token (0-indexed, position 2+2j in the sequence), we add
    cross-entropy loss at layer (start_layer + j) targeting trajectory[j].

    This encourages the model to learn a sequential algorithm where each
    layer computes one composition step, producing intermediate results
    in the embedding space at the corresponding <op> positions.

    Args:
        residuals: Per-layer residual streams, each (batch, seq_len, dim).
        trajectories: (batch, k_max+1) trajectory element tokens.
        chain_lengths: (batch,) chain lengths per example.
        final_norm: Model's final layer norm (for logit lens projection).
        embedding_weight: Tied embedding/unembedding weight matrix.
        start_layer: Layer offset for the staircase (0 for 6L, 1 for 7L).
        k_max: Maximum chain length.

    Returns:
        Scalar loss averaged across all valid (layer, position) pairs.
    """
    device = residuals[0].device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for j in range(k_max):
        layer = start_layer + j
        if layer >= len(residuals):
            break
        position = 2 + 2 * j  # j-th <op> token position

        # Only examples where chain_length > j (this <op> exists)
        mask = chain_lengths > j  # (batch,)
        if not mask.any():
            continue

        h = residuals[layer][mask, position, :]  # (n_valid, dim)
        targets = trajectories[mask, j]  # (n_valid,) element tokens

        # Logit lens: final_norm -> unembed
        logits = torch.nn.functional.linear(
            final_norm(h),
            embedding_weight,
        )  # (n_valid, vocab)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        total_loss = total_loss + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / count


def compute_alignment_loss(
    residuals: list[Tensor],
    input_ids: Tensor,
    embedding_weight: Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[Tensor, list[Tensor]]:
    """Soft alignment loss encouraging residuals to be near some token embedding.

    At each layer, computes cosine similarity between L2-normalized residuals
    and L2-normalized embedding vectors, then applies logsumexp over the vocab
    dimension. High logsumexp means the residual is close to at least one
    embedding. The loss is the negated mean logsumexp (minimizing pushes
    residuals toward the embedding manifold).

    Only non-PAD positions contribute (PAD_ID = 0).

    Args:
        residuals: Per-layer residual streams, each (batch, seq_len, dim).
        input_ids: (batch, seq_len) token IDs, used to create PAD mask.
        embedding_weight: Tied embedding matrix (vocab_size, dim).
        temperature: Temperature for the similarity computation.

    Returns:
        mean_loss: Scalar mean alignment loss across all layers.
        per_layer_losses: List of per-layer scalar losses.
    """
    device = residuals[0].device
    # Normalize embedding weight once (vocab_size, dim) -> unit vectors
    e_norm = torch.nn.functional.normalize(embedding_weight, dim=-1)
    # PAD mask: True for non-PAD positions
    mask = (input_ids != 0).float()  # (batch, seq_len)
    n_valid = mask.sum().clamp(min=1.0)

    per_layer_losses: list[Tensor] = []
    total_loss = torch.tensor(0.0, device=device)

    for residual in residuals:
        # L2-normalize residuals: (batch, seq_len, dim) -> unit vectors
        h_norm = torch.nn.functional.normalize(residual, dim=-1)
        # Cosine similarity with all embeddings: (batch, seq_len, vocab_size)
        sim = h_norm @ e_norm.T / temperature
        # logsumexp over vocab: high means close to at least one embedding
        lse = torch.logsumexp(sim, dim=-1)  # (batch, seq_len)
        # Negated mean over non-PAD positions (lower = more aligned)
        layer_loss = -(lse * mask).sum() / n_valid
        per_layer_losses.append(layer_loss)
        total_loss = total_loss + layer_loss

    mean_loss = total_loss / max(1, len(residuals))
    return mean_loss, per_layer_losses


def compute_repulsion_loss(
    embedding_weight: Tensor,
    n_elements: int,
    *,
    margin: float = 0.0,
    all_tokens: bool = False,
) -> Tensor:
    """Embedding repulsion loss pushing token embeddings apart.

    Computes pairwise cosine similarity among token embeddings and
    penalizes pairs whose cosine similarity exceeds the margin.

    By default, only operates on element tokens (indices 1..n_elements).
    With all_tokens=True, operates on all embeddings including PAD and
    special tokens, preventing PAD from becoming an alignment attractor.

    For S3 with 6 elements this is 15 pairs; with all 10 tokens, 45.

    Args:
        embedding_weight: Tied embedding matrix (vocab_size, dim).
        n_elements: Number of group element tokens (6 for S3).
        margin: Cosine similarity threshold below which no penalty.
        all_tokens: If True, repel all tokens (not just elements).

    Returns:
        Scalar repulsion loss.
    """
    if all_tokens:
        e = torch.nn.functional.normalize(embedding_weight, dim=-1)
        n = embedding_weight.shape[0]
    else:
        # Extract and normalize element embeddings (indices 1..n_elements)
        e = torch.nn.functional.normalize(
            embedding_weight[1 : n_elements + 1],
            dim=-1,
        )
        n = n_elements
    # Pairwise cosine similarity
    cos_sim = e @ e.T  # (n, n)
    # Upper triangle (exclude diagonal = self-similarity of 1.0)
    idx_i, idx_j = torch.triu_indices(
        n,
        n,
        offset=1,
        device=embedding_weight.device,
    )
    pairwise = cos_sim[idx_i, idx_j]  # (n_pairs,)
    # Hinge + squared penalty
    violations = torch.clamp(pairwise - margin, min=0.0)
    return violations.pow(2).mean()


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


def make_eval_batch(
    examples: list[S3Example],
    k_max: int,
    tokenizer: Tokenizer | None = None,
) -> dict[str, Tensor]:
    """Create an evaluation batch, padded to k_max sequence length.

    All examples should have the same chain length k for clean per-k eval.
    """
    tok = tokenizer or _S3_TOKENIZER
    k = len(examples[0].ops)
    input_ids = torch.stack(
        [
            torch.tensor(tok.encode_padded(ex, k_max), dtype=torch.long)
            for ex in examples
        ]
    )
    ans_pos = answer_position(k)
    return {
        "input_ids": input_ids,
        "answer_position": torch.full(
            (len(examples),),
            ans_pos,
            dtype=torch.long,
        ),
        "chain_length": torch.full(
            (len(examples),),
            k,
            dtype=torch.long,
        ),
    }

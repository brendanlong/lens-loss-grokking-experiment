"""Dataset and training utilities for group composition task.

Each training example is a chain of group operations:
    <start> elem <op> elem … <predict> answer

The model predicts the answer (final group element) from the logits
at the <predict> position.

Training samples chain lengths k uniformly from k_min to k_max.
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
        group: Group = S3,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(n_examples=n_examples, seed=seed)
        self.k_min = k_min
        self.k_max = k_max
        self.group = group
        self.tokenizer = tokenizer or _S3_TOKENIZER

    def generate(self, rng: random.Random) -> dict[str, Tensor]:
        k = rng.randint(self.k_min, self.k_max)
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

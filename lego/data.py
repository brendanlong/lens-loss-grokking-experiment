"""Datasets for the LEGO group composition task.

Each training example is a chain of group operations:
    <start> elem <op> elem … <predict> answer

The model predicts the answer (final group element) from the logits
at the <predict> position.

Examples come from the exhaustive enumeration in lego.generator
(enumerate_chains + train_test_split); this module only encodes them
into padded tensors. Losses live in lego.losses.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lego.generator import S3, ChainExample
from lego.tokenizer import (
    Tokenizer,
    answer_position,
    seq_len,
)

# Default S3 tokenizer for backward-compatible functions
_S3_TOKENIZER = Tokenizer(S3)


def encode_trajectory(
    example: ChainExample,
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


class ChainDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset of group composition chains, pre-encoded with padding."""

    def __init__(
        self,
        examples: list[ChainExample],
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


def make_k_uniform_sampler(
    dataset: ChainDataset,
    seed: int,
) -> torch.utils.data.WeightedRandomSampler:
    """Sampler that draws chain length k uniformly (equal probability per
    k-stratum), matching the old streaming generator's rng.randint(k_min,
    k_max) training distribution.

    The raw enumeration is ~83% k=6 with almost no short chains (k<=2 is
    0.08% of chains for k_max=6), and models trained on that proportional
    mix never lift off chance — the short-chain curriculum is required for
    learning. Samples with replacement; weights cover this dataset only, so
    held-out examples are never drawn.
    """
    k_counts = torch.bincount(dataset.chain_lengths)
    sample_weights = (1.0 / k_counts.float())[dataset.chain_lengths]
    return torch.utils.data.WeightedRandomSampler(
        sample_weights.tolist(),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def collate_s3(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate batch of group composition examples."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "answer_position": torch.stack([b["answer_position"] for b in batch]),
        "chain_length": torch.stack([b["chain_length"] for b in batch]),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
    }


def make_eval_batch(
    examples: list[ChainExample],
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

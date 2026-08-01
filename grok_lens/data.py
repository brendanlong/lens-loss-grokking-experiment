"""Modular arithmetic dataset: a op b (mod p), op in {+, -}.

Every example is the token sequence [a, b, =] with target (a op b) % p read
at the "=" position. The p^2 possible pairs are enumerated exhaustively and
partitioned into disjoint train/test splits by a seeded permutation, so
train/test disjointness holds by construction.
"""

from typing import Literal

import torch

from grok_lens.config import GrokModelConfig


def modular_addition_data(
    p: int, task: Literal["add", "sub"] = "add"
) -> tuple[torch.Tensor, torch.Tensor]:
    """All p^2 pairs as tokens [N, 3] (a, b, eq); targets (a op b) % p."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full((p * p,), p)
    tokens = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p if task == "add" else (a - b) % p
    return tokens, targets


def train_test_split(
    model_config: GrokModelConfig,
    train_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_tokens, train_targets, test_tokens, test_targets).

    A seeded permutation of the exhaustive pair list is split at
    ``round(train_frac * p^2)``; the two halves partition the full dataset.
    """
    tokens, targets = modular_addition_data(model_config.p, model_config.task)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(targets), generator=generator)
    n_train = round(train_frac * len(targets))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return tokens[train_idx], targets[train_idx], tokens[test_idx], targets[test_idx]

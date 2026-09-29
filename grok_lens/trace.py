"""Per-eval record of the three levels the stability analyses compare.

At every eval, on all p^2 pairs:

  - capability: per-example test loss and correctness
  - embedding: per-frequency Fourier power of the number-token embeddings
  - MLP neurons, every block, from their answer-position activations:
    ``profile``, the frequency profile over the 2-D Fourier spectrum in
    (a, b); ``answer_profile``, the profile of the activation as a function
    of the answer alone (the published neuron-drift measure); the share of
    variance that is a function of the answer; and the variance.

The two profiles differ where it matters: a 1-layer model's neurons are
~95% one frequency in (a, b) but only ~20% a function of the answer (they
also respond to a and b separately), so the answer-only profile misses the
circuit there.

Full checkpoints are too big to keep at eval cadence; this is ~120 MB for a
50k-step 2-layer run and is what ``grok_lens.analyze_levels`` reads.
"""

import os
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

from grok_lens.analyze_fourier import fourier_power
from grok_lens.config import GrokModelConfig
from grok_lens.data import modular_addition_data, split_indices
from grok_lens.model import Block, GrokTransformer


@torch.no_grad()
def mlp_activations(
    model: GrokTransformer, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(output logits [N, vocab], post-ReLU answer-position acts [L, N, n])."""
    acts: list[torch.Tensor] = []
    handles = [
        cast("torch.nn.Module", cast("Block", block).mlp[1]).register_forward_hook(
            lambda _m, _i, out: acts.append(out[:, -1].detach())
        )
        for block in model.blocks
    ]
    try:
        logits = model(tokens)[-1]
    finally:
        for handle in handles:
            handle.remove()
    return logits, torch.stack(acts)


def frequency_profiles(
    acts: torch.Tensor, answers: torch.Tensor, p: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(profile [..., n, K], frac_answer [..., n], variance [..., n]).

    ``acts`` is [..., N, n] over all p^2 pairs. ``profile`` rows are
    L1-normalised power over frequencies 1..(p-1)/2 of the neuron's mean
    activation per answer value; rows whose activation is constant are zero.
    """
    by_answer = torch.zeros(*acts.shape[:-2], p, acts.shape[-1], device=acts.device)
    by_answer.index_add_(-2, answers, acts)
    by_answer /= acts.shape[-2] / p

    variance = acts.var(-2, unbiased=False)
    frac = (by_answer.var(-2, unbiased=False) / variance.clamp_min(1e-12)).clamp(0, 1)

    centred = by_answer - by_answer.mean(-2, keepdim=True)
    n_freqs = (p - 1) // 2
    spec = torch.fft.rfft(centred, dim=-2)[..., 1 : n_freqs + 1, :].abs().pow(2)
    profile = (spec / spec.sum(-2, keepdim=True).clamp_min(1e-12)).transpose(-1, -2)
    return profile, frac, variance


def pair_frequency_profiles(acts: torch.Tensor, p: int) -> torch.Tensor:
    """Profile [..., n, K] from the 2-D Fourier spectrum over (a, b).

    ``acts`` is [..., p^2, n] in :func:`modular_addition_data` order (a
    major). Frequency k's power is every component (k1, k2) with k1 or k2
    equal to ±k: a-only, b-only and joint terms alike. A component mixing
    two different frequencies counts towards both. Rows are L1-normalised.
    """
    grid = acts.reshape(*acts.shape[:-2], p, p, acts.shape[-1])
    spec = torch.fft.fft2(grid, dim=(-3, -2)).abs().pow(2).flatten(-3, -2)
    idx = torch.arange(p, device=acts.device)
    fold = torch.minimum(idx, p - idx)
    ks = torch.arange(1, (p - 1) // 2 + 1, device=acts.device)
    involves = (fold[:, None, None] == ks) | (fold[None, :, None] == ks)
    power = torch.einsum("...fn,fk->...nk", spec, involves.flatten(0, 1).float())
    return power / power.sum(-1, keepdim=True).clamp_min(1e-12)


class TraceRecorder:
    def __init__(
        self, model_config: GrokModelConfig, train_frac: float, seed: int
    ) -> None:
        self.p = model_config.p
        self.tokens, self.answers = modular_addition_data(
            model_config.p, model_config.task
        )
        self.test_idx = split_indices(model_config.p, train_frac, seed)[1]
        self.rows: dict[str, list[torch.Tensor]] = {}
        self.steps: list[int] = []

    @torch.no_grad()
    def record(self, step: int, model: GrokTransformer) -> None:
        device = model.embed.weight.device
        tokens, answers = self.tokens.to(device), self.answers.to(device)
        test_idx = self.test_idx.to(device)

        logits, acts = mlp_activations(model, tokens)
        test_logits, test_answers = logits[test_idx], answers[test_idx]
        answer_profile, frac, variance = frequency_profiles(acts, answers, self.p)
        row = {
            "test_loss": F.cross_entropy(
                test_logits, test_answers, reduction="none"
            ).cpu(),
            "test_correct": (test_logits.argmax(-1) == test_answers).cpu(),
            "embed_power": fourier_power(
                model.embed.weight[: self.p].float().cpu(), self.p
            ),
            "profile": pair_frequency_profiles(acts, self.p).half().cpu(),
            "answer_profile": answer_profile.half().cpu(),
            "frac_answer": frac.half().cpu(),
            "variance": variance.cpu(),
        }
        self.steps.append(step)
        for key, value in row.items():
            self.rows.setdefault(key, []).append(value)

    def save(self, path: Path, meta: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "steps": torch.tensor(self.steps),
            **{key: torch.stack(values) for key, values in self.rows.items()},
            "meta": meta,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

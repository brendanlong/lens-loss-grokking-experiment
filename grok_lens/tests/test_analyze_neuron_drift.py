"""Neuron profiles recover a planted frequency; populations are soft."""

import math

import torch

from grok_lens.analyze_neuron_drift import jaccard, populations


def test_populations_allow_a_neuron_in_several_roles() -> None:
    profile = torch.tensor([[0.5, 0.5, 0.0], [0.9, 0.05, 0.05], [0.0, 0.0, 1.0]])
    active = torch.tensor([True, True, False])
    pops = populations(profile, active, 0.10)
    assert pops == [{0, 1}, {0}, set()]


def test_jaccard_is_none_for_two_empty_sets() -> None:
    assert jaccard(set(), set()) is None
    assert jaccard({1, 2}, {2, 3}) == 1 / 3


def test_fft_profile_recovers_planted_frequency() -> None:
    p, k = 113, 7
    s = torch.arange(p, dtype=torch.float32)
    by_sum = torch.cos(2 * math.pi * k * s / p)[:, None]
    spec = torch.fft.rfft(by_sum - by_sum.mean(0), dim=0)[1 : (p - 1) // 2 + 1]
    power = spec.abs().pow(2)[:, 0]
    assert int(power.argmax()) + 1 == k

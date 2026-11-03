import math

import pytest
import torch

from grok_lens.analyze_levels import (
    collapse_events,
    collapse_free_pairs,
    embedding_moved,
    lag_pairs,
    neuron_retention,
)
from grok_lens.analyze_neuron_drift import populations, retention_lift
from grok_lens.trace import frequency_profiles, pair_frequency_profiles


def test_vectorised_retention_matches_retention_lift() -> None:
    gen = torch.Generator().manual_seed(0)
    t, n, k, lag = 12, 60, 6, 3
    profile = torch.rand(t, n, k, generator=gen)
    profile /= profile.sum(-1, keepdim=True)
    active = torch.rand(t, n, generator=gen) > 0.2
    member = (profile >= 0.2) & active[..., None]
    pops = [populations(profile[i], active[i], 0.2) for i in range(t)]
    per_role = [
        v
        for role in range(k)
        if (v := retention_lift(pops, list(active), role, lag)) is not None
    ]
    expected = sum(per_role) / len(per_role)
    got = neuron_retention(member, active, lag_pairs(t, lag), min_pop=5)
    assert got == pytest.approx(expected, abs=1e-5)


def test_identical_populations_retain_fully() -> None:
    member = torch.zeros(2, 20, 2, dtype=torch.bool)
    member[:, :8, 0] = True
    member[:, 8:, 1] = True
    active = torch.ones(2, 20, dtype=torch.bool)
    assert neuron_retention(member, active, lag_pairs(2, 1), 5) == pytest.approx(1.0)


def test_embedding_moved_is_share_of_power_relocated() -> None:
    power = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    assert embedding_moved(power, lag_pairs(3, 1))[1] == pytest.approx(1.0)
    assert embedding_moved(power, lag_pairs(3, 2)) == pytest.approx((0.0, 0.0))


def test_embedding_moved_is_about_one_for_an_unrelated_spectrum() -> None:
    gen = torch.Generator().manual_seed(1)
    power = torch.rand(2, 56, generator=gen) ** 8
    power[1] = power[1, torch.randperm(56, generator=gen)]
    corrected, _ = embedding_moved(power, lag_pairs(2, 1))
    assert 0.7 < corrected < 1.3


def test_collapse_events_find_pre_trough_and_recovery() -> None:
    acc = torch.tensor([0.1, 0.96, 0.99, 0.5, 0.3, 0.92, 0.97, 0.98, 0.8, 0.9])
    assert collapse_events(acc) == [(2, 4, 6), (7, 8, None)]


def test_collapse_free_pairs_skip_spans_containing_a_dip() -> None:
    acc = torch.tensor([0.1, 0.96, 0.99, 0.5, 0.97, 0.98, 0.99])
    i, j = collapse_free_pairs(acc, lag=1, start=1)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(1, 2), (4, 5), (5, 6)]
    i, j = collapse_free_pairs(acc, lag=1, start=1, stop=5)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(1, 2), (4, 5)]


def test_frequency_profiles_recover_planted_frequency_per_block() -> None:
    p = 23
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    answers = (a + b) % p
    wave = [torch.cos(2 * math.pi * k * answers.float() / p) for k in (3, 7)]
    acts = torch.stack([torch.stack(wave, -1), torch.stack(wave[::-1], -1)])
    profile, frac, _ = frequency_profiles(acts, answers, p)
    assert profile.argmax(-1).tolist() == [[2, 6], [6, 2]]
    assert frac.min().item() == pytest.approx(1.0)


def test_pair_profiles_see_frequency_in_a_and_b_not_just_the_sum() -> None:
    p, k = 23, 5
    a = torch.arange(p).repeat_interleave(p).float()
    b = torch.arange(p).repeat(p).float()
    w = 2 * math.pi * k / p
    separate = torch.cos(w * a) + torch.cos(w * b)
    joint = torch.cos(w * (a + b))
    acts = torch.stack([separate, joint], -1)[None]  # [1, p^2, 2]
    profile = pair_frequency_profiles(acts, p)
    assert profile[0, :, k - 1].tolist() == pytest.approx([1.0, 1.0])
    answers = ((a + b) % p).long()
    _, frac, _ = frequency_profiles(acts, answers, p)
    assert frac[0, 0].item() == pytest.approx(0.0, abs=1e-6)

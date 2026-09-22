"""Retention lift: 1 = everyone stayed, 0 = chance, inactive neurons ignored."""

import torch

from grok_lens.analyze_neuron_drift import retention_lift

ALL = torch.ones(10, dtype=torch.bool)


def test_full_retention_is_one() -> None:
    pops = [[{0, 1, 2, 3, 4}], [{0, 1, 2, 3, 4}]]
    assert retention_lift(pops, [ALL, ALL], 0, 1) == 1.0


def test_disjoint_replacement_is_below_chance() -> None:
    pops = [[{0, 1, 2, 3, 4}], [{5, 6, 7, 8, 9}]]
    assert retention_lift(pops, [ALL, ALL], 0, 1) == -1.0


def test_neurons_going_inactive_do_not_count_as_leaving() -> None:
    later_active = ALL.clone()
    later_active[4] = False
    pops = [[{0, 1, 2, 3, 4}], [{0, 1, 2, 3}]]
    assert retention_lift(pops, [ALL, later_active], 0, 1) == 1.0

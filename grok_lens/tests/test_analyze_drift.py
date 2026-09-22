"""Index arithmetic in the drift metrics (where an off-by-one hid once)."""

import pytest

from grok_lens.analyze_drift import (
    ENDPOINT_EVALS,
    FREQ_KEYS,
    RECOVERY_WINDOW,
    collapse_events,
    effective_components,
    endpoints,
    path_length,
    power_moved,
)

N = len(FREQ_KEYS)


def spectrum(**shares: float) -> list[float]:
    """A normalised spectrum with the named frequencies (1-indexed) set."""
    row = [0.0] * N
    for name, value in shares.items():
        row[int(name[1:]) - 1] = value
    rest = (1.0 - sum(row)) / (N - len(shares))
    return [v if v else rest for v in row]


def test_power_moved_is_half_l1() -> None:
    assert power_moved([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert power_moved([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert power_moved([0.6, 0.4], [0.4, 0.6]) == pytest.approx(0.2)


def test_effective_components_counts_equal_contributors() -> None:
    assert effective_components([0.25] * 4) == 4.0
    assert effective_components([1.0]) == 1.0


def test_path_length_accumulates_round_trips_that_net_to_zero() -> None:
    there, back = [1.0, 0.0], [0.0, 1.0]
    trace = [there, back, there]
    assert power_moved(trace[0], trace[-1]) == 0.0
    assert path_length(trace) == 2.0


def test_endpoints_average_the_window_edges() -> None:
    lo, hi = spectrum(k1=0.5), spectrum(k1=0.9)
    trace = [lo] * ENDPOINT_EVALS + [hi] * ENDPOINT_EVALS
    first, last = endpoints(trace)
    assert first[0] == 0.5
    assert last[0] == 0.9


def _trace_with_collapse_at(
    index: int, length: int, recover: bool
) -> list[list[float]]:
    trace = [spectrum(k1=0.10) for _ in range(length)]
    trace[index + 1] = spectrum(k1=0.01)
    if recover:
        for t in range(index + 2, length):
            trace[t] = spectrum(k1=0.10)
    else:
        for t in range(index + 2, length):
            trace[t] = spectrum(k1=0.01)
    return trace


def test_collapse_is_classified_repaired_or_not() -> None:
    length = RECOVERY_WINDOW + 10
    assert collapse_events(_trace_with_collapse_at(0, length, True), 0.04) == (1, 0)
    assert collapse_events(_trace_with_collapse_at(0, length, False), 0.04) == (0, 1)


def test_last_classifiable_collapse_is_not_skipped() -> None:
    """The final index with a full lookahead must still be examined."""
    length = RECOVERY_WINDOW + 1
    trace = _trace_with_collapse_at(length - RECOVERY_WINDOW - 1, length, False)
    assert collapse_events(trace, 0.04) == (0, 1)


def test_collapses_without_full_lookahead_are_excluded() -> None:
    length = RECOVERY_WINDOW + 10
    trace = _trace_with_collapse_at(length - RECOVERY_WINDOW, length, False)
    assert collapse_events(trace, 0.04) == (0, 0)


def test_inactive_components_do_not_count_as_collapses() -> None:
    trace = [spectrum(k1=0.03) for _ in range(RECOVERY_WINDOW + 10)]
    trace[1] = spectrum(k1=0.001)
    assert collapse_events(trace, 0.04) == (0, 0)

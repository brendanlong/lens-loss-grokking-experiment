"""Circuit drift under a flat capability (the numbers behind drift.png).

The stability results say the aux arm's *accuracy* stops moving post-grok.
They say nothing about whether the circuit underneath stops moving. This
walks the ``-ffttrace`` runs' per-frequency embedding power and reports,
over a post-grok window:

  - ``power moved``: total-variation distance between the window's first and
    last spectrum — the share of embedding Fourier power sitting on
    different frequencies at the end than at the start.
  - ``eff #comp``: participation ratio 1/Σp², the effective number of
    frequencies carrying the circuit. Flat participation ratio with high
    ``power moved`` is turnover; falling participation ratio is consolidation.
  - deaths / births: components crossing 4% of power in either direction.
  - collapse events (>50% power lost by a ≥4% component in one eval),
    split by whether the component recovers to ≥80% of its pre-collapse
    share within 2.5k steps.

Usage:
    uv run python -m grok_lens.analyze_drift [--since 30000]
"""

import argparse
from collections.abc import Iterable
from typing import Protocol

import wandb

PROJECT = "brendanlong-com/grok-lens"
FREQ_KEYS = [f"freq_power/k_{i}" for i in range(1, 57)]
ARMS = [("baseline", "lam0.0"), ("aux λ=0.3", "lam0.3")]
SEEDS = [42, 43, 44]
ACTIVE = 0.04  # power share at which we call a component part of the circuit
RECOVERY_WINDOW = 25  # evals (100 steps each) allowed for a repair


class _Run(Protocol):
    def scan_history(self, keys: list[str]) -> Iterable[dict[str, float]]: ...


def spectra(run: _Run, since: int) -> tuple[list[int], list[float], list[list[float]]]:
    hist = sorted(
        run.scan_history(keys=[*FREQ_KEYS, "test/acc"]), key=lambda h: h["_step"]
    )
    hist = [h for h in hist if h["_step"] >= since]
    shares = []
    for h in hist:
        row = [h[k] for k in FREQ_KEYS]
        total = sum(row)
        shares.append([v / total for v in row])
    return [int(h["_step"]) for h in hist], [h["test/acc"] for h in hist], shares


def power_moved(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def effective_components(p: list[float]) -> float:
    return 1.0 / sum(v * v for v in p)


def collapse_events(shares: list[list[float]]) -> tuple[int, int]:
    """(repaired, permanent) component collapses."""
    repaired = permanent = 0
    for t in range(len(shares) - RECOVERY_WINDOW - 1):
        for k in range(len(FREQ_KEYS)):
            before = shares[t][k]
            if before < ACTIVE or shares[t + 1][k] >= 0.5 * before:
                continue
            after = shares[t + 1 : t + 1 + RECOVERY_WINDOW]
            if max(s[k] for s in after) >= 0.8 * before:
                repaired += 1
            else:
                permanent += 1
    return repaired, permanent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=int,
        default=30_000,
        help="Window start (default 30k: every ffttrace run first-crossed by 14.4k)",
    )
    args = parser.parse_args()
    api = wandb.Api()
    runs = {r.name: r for r in api.runs(PROJECT) if "ffttrace" in r.name}

    header = (
        f"{'run':<22}{'acc mean':>9}{'acc min':>9}{'power moved':>13}"
        f"{'eff #comp':>12}{'deaths':>8}{'births':>8}{'repaired':>10}{'permanent':>11}"
    )
    print(f"Post-grok window: step {args.since}+\n{header}\n{'-' * len(header)}")
    for label, kind in ARMS:
        for seed in SEEDS:
            name = f"p113-L2-{kind}-uniform-frac0.3-s{seed}-ffttrace"
            _, accs, shares = spectra(runs[name], args.since)
            first, last = shares[0], shares[-1]
            pairs = list(zip(first, last, strict=True))
            deaths = sum(1 for a, b in pairs if a >= ACTIVE and b < 0.01)
            births = sum(1 for a, b in pairs if a < 0.01 and b >= ACTIVE)
            repaired, permanent = collapse_events(shares)
            print(
                f"{label + ' s' + str(seed):<22}"
                f"{sum(accs) / len(accs):>9.4f}{min(accs):>9.3f}"
                f"{power_moved(first, last):>13.2f}"
                f"{effective_components(first):>6.1f}→{effective_components(last):<5.1f}"
                f"{deaths:>8}{births:>8}{repaired:>10}{permanent:>11}"
            )
    print(
        f"\nWindow starts at step {args.since}. 'power moved' is the TV distance\n"
        "between the window's endpoint spectra;\n"
        "'eff #comp' is the participation ratio at each\n"
        f"endpoint; deaths/births cross {ACTIVE:.0%} power share; collapses are >50%\n"
        f"single-eval power losses by a ≥{ACTIVE:.0%} component, 'repaired' if the\n"
        "component regains ≥80% of its pre-collapse share within 2.5k steps."
    )


if __name__ == "__main__":
    main()

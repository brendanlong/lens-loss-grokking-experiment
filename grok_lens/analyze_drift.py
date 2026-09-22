"""Circuit drift under a flat capability (the numbers behind drift.png).

The stability results say the aux arm's *accuracy* stops moving post-grok.
They say nothing about whether the circuit underneath stops moving. This
walks the ``-ffttrace`` runs' per-frequency embedding power and reports,
over a post-grok window:

  - ``power moved``: total-variation distance between the window's endpoint
    spectra — the share of embedding Fourier power sitting on different
    frequencies at the end than at the start. Endpoints are averaged over
    ``ENDPOINT_EVALS`` evals, because single evals in these traces swing by
    tens of percent (that volatility is the subject of the analysis, so it
    cannot also be the measurement). This is *net displacement*: a lower
    bound on total motion, which ``path len`` gives.
  - ``eff #comp``: participation ratio 1/Σp², the effective number of
    frequencies carrying the circuit. Flat participation ratio with high
    ``power moved`` is turnover; falling participation ratio is
    consolidation onto the survivors.
  - deaths / births: components crossing the ``--active`` share in one
    direction and 1% in the other (a hysteresis band, so a component
    hovering at one threshold is not counted). Endpoint-to-endpoint only:
    a component that dies and is reborn inside the window is invisible.
  - collapse events (>50% share lost by an active component in one eval),
    split by whether it regains >=80% of its pre-collapse share within
    ``RECOVERY_WINDOW`` evals.

Everything here is a power *share*, so a component can "die" either by
losing absolute power or by the rest of the circuit growing around it; the
embedding's norm shrinks under wd = 1.0 throughout. Shares are what the run
logs, and the sparse-vs-distributed contrast the study rests on is itself a
share-level claim, so this is the right scale — but it is not a statement
about absolute magnitudes.

Usage:
    uv run python -m grok_lens.analyze_drift [--since 30000] [--active 0.04]
"""

import argparse
from collections.abc import Iterable
from itertools import pairwise
from typing import Protocol

import wandb

PROJECT = "brendanlong-com/grok-lens"
FREQ_KEYS = [f"freq_power/k_{i}" for i in range(1, 57)]
ARMS = [("baseline", "lam0.0"), ("aux λ=0.3", "lam0.3")]
SEEDS = [42, 43, 44]
RUN = "p113-L2-{kind}-uniform-frac0.3-s{seed}-ffttrace"
ENDPOINT_EVALS = 10  # evals averaged at each end of the window (1k steps)
RECOVERY_WINDOW = 25  # evals a collapsed component gets to come back (2.5k steps)
DEAD = 0.01  # share below which a component is out of the circuit


class Run(Protocol):
    """The slice of wandb's public-API Run we use (no importable stub)."""

    def scan_history(self, keys: list[str]) -> Iterable[dict[str, float]]: ...


def spectra(run: Run, since: int) -> tuple[list[int], list[float], list[list[float]]]:
    """Steps, test accuracy, and L1-normalised per-frequency power shares."""
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


def endpoints(shares: list[list[float]]) -> tuple[list[float], list[float]]:
    """Mean spectrum over the first and last ENDPOINT_EVALS evals."""

    def mean(block: list[list[float]]) -> list[float]:
        return [sum(s[k] for s in block) / len(block) for k in range(len(FREQ_KEYS))]

    return mean(shares[:ENDPOINT_EVALS]), mean(shares[-ENDPOINT_EVALS:])


def power_moved(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def path_length(shares: list[list[float]]) -> float:
    """Summed consecutive-eval TV: total motion, of which power_moved is the net."""
    return sum(power_moved(a, b) for a, b in pairwise(shares))


def effective_components(p: list[float]) -> float:
    return 1.0 / sum(v * v for v in p)


def collapse_events(shares: list[list[float]], active: float) -> tuple[int, int]:
    """(repaired, unrepaired) component collapses.

    Classification needs a full lookahead, so collapses in the final
    RECOVERY_WINDOW evals are not counted at all, and "unrepaired" means
    "not repaired within the window", not "never recovered".
    """
    repaired = unrepaired = 0
    for t in range(len(shares) - RECOVERY_WINDOW):
        for k in range(len(FREQ_KEYS)):
            before = shares[t][k]
            if before < active or shares[t + 1][k] >= 0.5 * before:
                continue
            after = shares[t + 1 : t + 1 + RECOVERY_WINDOW]
            if max(s[k] for s in after) >= 0.8 * before:
                repaired += 1
            else:
                unrepaired += 1
    return repaired, unrepaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=int,
        default=30_000,
        help="Window start (default 30k: every ffttrace run first-crossed by 14.4k)",
    )
    parser.add_argument(
        "--active",
        type=float,
        default=0.04,
        help="Power share at which a component counts as part of the circuit",
    )
    args = parser.parse_args()
    api = wandb.Api()
    runs = {r.name: r for r in api.runs(PROJECT) if "ffttrace" in r.name}

    header = (
        f"{'run':<22}{'acc mean':>9}{'acc min':>9}{'power moved':>13}{'path len':>10}"
        f"{'eff #comp':>14}{'deaths':>8}{'births':>8}{'repaired':>10}{'unrepaired':>12}"
    )
    print(f"Window: step {args.since}+, active >= {args.active:.0%}")
    print(f"{header}\n{'-' * len(header)}")
    end = 0
    for label, kind in ARMS:
        for seed in SEEDS:
            name = RUN.format(kind=kind, seed=seed)
            if name not in runs:
                raise SystemExit(f"run not found in {PROJECT}: {name}")
            steps, accs, shares = spectra(runs[name], args.since)
            end = steps[-1]
            first, last = endpoints(shares)
            pairs = list(zip(first, last, strict=True))
            deaths = sum(1 for a, b in pairs if a >= args.active and b < DEAD)
            births = sum(1 for a, b in pairs if a < DEAD and b >= args.active)
            repaired, unrepaired = collapse_events(shares, args.active)
            print(
                f"{label + ' s' + str(seed):<22}"
                f"{sum(accs) / len(accs):>9.4f}{min(accs):>9.3f}"
                f"{power_moved(first, last):>13.2f}{path_length(shares):>10.1f}"
                f"{effective_components(first):>8.1f}→{effective_components(last):<5.1f}"
                f"{deaths:>8}{births:>8}{repaired:>10}{unrepaired:>12}"
            )
    print(
        f"\nSteps {args.since}-{end}. Endpoints are {ENDPOINT_EVALS}-eval means.\n"
        f"'power moved' is the TV distance between them (net displacement);\n"
        f"'path len' is the summed consecutive-eval TV (total motion).\n"
        f"'eff #comp' is the participation ratio at each endpoint. Deaths cross\n"
        f"{args.active:.0%} down to <{DEAD:.0%}, births the other way. Collapses are\n"
        f">50% single-eval share losses by a >={args.active:.0%} component,\n"
        f"'repaired' if it regains >=80% of its pre-collapse share within\n"
        f"{RECOVERY_WINDOW} evals; the final {RECOVERY_WINDOW} evals cannot be\n"
        f"classified and are excluded."
    )


if __name__ == "__main__":
    main()

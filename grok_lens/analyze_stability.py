"""Metric-robustness check for grok timing (added after adversarial review).

``grok_step`` in train.py is the FIRST eval crossing test acc >= 0.95. Under
wd = 1.0 the post-grok curve slingshots, and baselines keep collapsing below
0.90 long after first crossing while aux runs hold — so first-crossing
flatters the baseline. This script pulls full wandb histories and reports,
per run: first crossing, stable crossing (first eval after which acc never
drops below 0.95 again — right-censored at the run budget), the number of
post-first-crossing evals below 0.90, and the fraction of final-5k-step
evals >= 0.95.

Usage:
    uv run python -m grok_lens.analyze_stability
"""

import re
import sys
from collections.abc import Iterable
from typing import Protocol

import wandb

PROJECT = "brendanlong-com/grok-lens"


RunMetrics = tuple[int | None, int | None, int, float, float, float, int]


class _Run(Protocol):
    """The slice of wandb's public-API Run we use (no importable stub)."""

    def scan_history(self, keys: list[str]) -> Iterable[dict[str, float]]: ...


def run_metrics(run: _Run) -> RunMetrics | None:
    """Per-run stability metrics.

    Returns (first_cross, stable_cross, dips_below_090, occupancy,
    tail5k_frac, tail10k_loss, last_step), or None for runs without a
    test/acc history (e.g. the lego runs sharing the wandb project).
    ``occupancy`` is the fraction of
    evals at/after first crossing with acc >= 0.95 — "what % of the time is
    the model grokking" — which uses the whole tail and is therefore much
    less sensitive to end-of-run censoring than ``stable_cross``.
    ``tail10k_loss`` is mean test loss over the final 10k steps (an
    occupancy-like continuous measure).
    """
    # scan_history: full resolution (history() samples to 500 rows, which
    # silently under-counts dips on the long-horizon runs).
    hist = sorted(
        run.scan_history(keys=["test/acc", "test/loss"]),
        key=lambda h: h["_step"],
    )
    if not hist:  # runs without test/acc (e.g. lego runs in the same project)
        return None
    steps = [int(h["_step"]) for h in hist]
    accs = [h["test/acc"] for h in hist]
    losses = [h["test/loss"] for h in hist]
    first = next((s for s, a in zip(steps, accs, strict=True) if a >= 0.95), None)
    stable = None
    for i in range(len(accs)):
        if all(a >= 0.95 for a in accs[i:]):
            stable = steps[i]
            break
    dips = 0
    occupancy = 0.0
    if first is not None:
        first_idx = steps.index(first)
        post = accs[first_idx:]
        dips = sum(1 for a in post if a < 0.90)
        occupancy = sum(1 for a in post if a >= 0.95) / len(post)
    tail = [a for s, a in zip(steps, accs, strict=True) if s > steps[-1] - 5000]
    tail_frac = sum(1 for a in tail if a >= 0.95) / len(tail)
    tail_losses = [
        ls for s, ls in zip(steps, losses, strict=True) if s > steps[-1] - 10_000
    ]
    tail10k_loss = sum(tail_losses) / len(tail_losses)
    return first, stable, dips, occupancy, tail_frac, tail10k_loss, steps[-1]


def main() -> None:
    api = wandb.Api()
    cells: dict[str, list[tuple[str, *RunMetrics]]] = {}
    for run in api.runs(PROJECT):
        match = re.search(r"-s(\d+)", run.name)
        if match is None:
            continue
        seed = match.group(1)
        cell = run.name.replace(f"-s{seed}", "")
        if run.summary.get("test/acc") is None:  # cheap pre-filter: lego runs
            print(f"(skipping {run.name}: no test/acc history)", file=sys.stderr)
            continue
        metrics = run_metrics(run)
        if metrics is None:
            print(f"(skipping {run.name}: no test/acc history)", file=sys.stderr)
            continue
        cells.setdefault(cell, []).append((seed, *metrics))

    print(
        "stable is right-censored at the run budget: a 'stable' value within a\n"
        "few evals of the end is weak evidence (nothing left to disconfirm it).\n"
    )
    for cell in sorted(cells):
        print(cell)
        for seed, first, stable, dips, occ, tail_frac, tail_loss, last in sorted(
            cells[cell]
        ):
            print(
                f"  s{seed}: first={first!s:>6s} stable={stable!s:>6s} "
                f"dips<0.90: {dips:>3d}  occupancy: {occ:.2f}  "
                f"tail5k>=0.95: {tail_frac:.2f}  tail10k loss: {tail_loss:.4f}  "
                f"(end {last})"
            )


if __name__ == "__main__":
    main()

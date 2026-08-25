"""Single-threshold sensitivity check for the two-threshold stability metrics.

The stability metrics use two thresholds (first crossing / occupancy at
test acc >= 0.95, dips at < 0.90) so occupancy isn't inflated by
borderline evals. This script recomputes dip counts at a single 0.95
threshold (dips = post-first-crossing evals < 0.95) next to the reported
< 0.90 counts, over every from-scratch arithmetic cell in the public
wandb project (baselines, the λ sweep, both Muon arms, no-LN, the
wd sweep, the shuffled-target control, subtraction; the continuation
runs are excluded), to check that the baseline-vs-aux contrast doesn't
depend on the choice.

Usage:
    uv run python -m grok_lens.analyze_threshold_sensitivity
"""

import re

import wandb

PROJECT = "brendanlong-com/grok-lens"
CELL_RE = re.compile(
    r"^p113(?:sub)?-L\d-lam[\d.]+-(?:uniform|linear|shuf)-frac0\.3"
    r"(?:-[a-z0-9.]+)*$"
)


def main() -> None:
    api = wandb.Api()
    cells: dict[str, list[tuple[str, int, int, float]]] = {}
    for run in api.runs(PROJECT):
        match = re.search(r"-s(\d+)", run.name)
        if match is None:
            continue
        seed = match.group(1)
        cell = run.name.replace(f"-s{seed}", "")
        if not CELL_RE.match(cell):
            continue
        # scan_history: full resolution (history() samples, which silently
        # under-counts dips on the long-horizon runs).
        hist = sorted(
            run.scan_history(keys=["test/acc"]),
            key=lambda h: h["_step"],
        )
        accs = [h["test/acc"] for h in hist]
        first_idx = next((i for i, a in enumerate(accs) if a >= 0.95), None)
        if first_idx is None:
            print(f"{run.name}: never crossed 0.95 ({len(accs)} evals)")
            continue
        post = accs[first_idx:]
        dips_090 = sum(1 for a in post if a < 0.90)
        dips_095 = sum(1 for a in post if a < 0.95)
        occ = sum(1 for a in post if a >= 0.95) / len(post)
        cells.setdefault(cell, []).append((seed, dips_090, dips_095, occ))

    for cell in sorted(cells):
        print(cell)
        for seed, d90, d95, occ in sorted(cells[cell]):
            print(f"  s{seed}: dips<0.90={d90:>3d}  dips<0.95={d95:>3d}  occ={occ:.2f}")


if __name__ == "__main__":
    main()

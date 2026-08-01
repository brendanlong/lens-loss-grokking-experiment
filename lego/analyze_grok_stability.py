"""Stability metrics for LEGO grokking-regime runs (per chain length k).

The grok_lens stability analysis, adapted to the multi-hop task: multi-hop
composition can grok per-k in stages, so first crossing, dips, and
occupancy are computed per k from the wandb histories, plus the
memorization step from the running train accuracy. The k <= 1 strata have
1 and 7 held-out examples respectively, so per-k metrics are reported for
k >= 2 only (a single flipped example moves k1 accuracy by 0.14).

Usage:
    uv run python -m lego.analyze_grok_stability            # S3-grok-* runs
    uv run python -m lego.analyze_grok_stability --prefix S3-grok-sub2000
"""

import argparse

import wandb

PROJECT = "brendanlong-com/grok-lens"
REPORT_KS = (2, 3, 4, 5, 6)


def series(
    hist: list[dict[str, float]], key: str
) -> tuple[list[int], list[float]]:
    """(steps, values) for one metric, skipping evals where it's absent."""
    pairs = [
        (int(h["_step"]), float(h[key]))
        for h in hist
        if h.get(key) is not None
    ]
    return [s for s, _ in pairs], [v for _, v in pairs]


def crossing_metrics(
    steps: list[int],
    accs: list[float],
) -> tuple[int | None, int, float]:
    """(first crossing >= 0.95, post-crossing dips < 0.90, occupancy).

    Occupancy = fraction of evals at/after first crossing with acc >= 0.95
    (grok_lens convention; two thresholds so borderline evals inflate
    neither count).
    """
    first = next((s for s, a in zip(steps, accs, strict=True) if a >= 0.95), None)
    if first is None:
        return None, 0, 0.0
    post = accs[steps.index(first) :]
    dips = sum(1 for a in post if a < 0.90)
    occupancy = sum(1 for a in post if a >= 0.95) / len(post)
    return first, dips, occupancy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--prefix",
        default="S3-grok-",
        help="Only runs whose name starts with this prefix",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = sorted(
        (r for r in api.runs(args.project) if r.name.startswith(args.prefix)),
        key=lambda r: r.name,
    )
    if not runs:
        print(f"No runs matching {args.prefix}* in {args.project}")
        return

    keys = ["train/answer_acc", "test_acc/mean"] + [
        f"test_acc/k_{k}" for k in REPORT_KS
    ]
    for run in runs:
        hist = sorted(
            run.history(keys=keys, pandas=False),
            key=lambda h: h["_step"],
        )
        train_steps, train_accs = series(hist, "train/answer_acc")
        memorize = next(
            (
                s
                for s, a in zip(train_steps, train_accs, strict=True)
                if a >= 0.95
            ),
            None,
        )
        mean_steps, mean_accs = series(hist, "test_acc/mean")
        final_mean = mean_accs[-1] if mean_accs else float("nan")
        end = mean_steps[-1] if mean_steps else "?"
        print(
            f"\n{run.name} ({run.id})  memorize={memorize!s}  "
            f"final test mean={final_mean:.3f}  (end {end})"
        )
        for k in REPORT_KS:
            steps, accs = series(hist, f"test_acc/k_{k}")
            if not steps:
                print(f"  k={k}: no data")
                continue
            first, dips, occ = crossing_metrics(steps, accs)
            final = accs[-1]
            print(
                f"  k={k}: first={first!s:>6s}  dips<0.90: {dips:>3d}  "
                f"occupancy: {occ:.2f}  final: {final:.3f}"
            )


if __name__ == "__main__":
    main()

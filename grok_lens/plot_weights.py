"""Grid figure of analyze_weights summaries: how far each part of the network moves.

Usage (normally driven by the Snakefile):
    uv run python -m grok_lens.plot_weights --out FIG.png \\
        --summaries W1.json ... --row L1-ln,L2-ln --label L1-ln="1 layer, LN" ...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from grok_lens.analyze_weights import WEIGHT_LEVELS
from grok_lens.plot_levels import INK, MUTED, cell_seed, grid

# (color, line style, marker): solid = raw weights, dashed = basis-free role
SERIES = {
    "embedding": ("#eb6834", "-", "o"),
    "mlp": ("#1baf7a", "-", "s"),
    "attention": ("#e87ba4", "-", "^"),
    "unembedding": ("#eda100", "-", "D"),
    "heads": ("#e87ba4", "--", "^"),
    "unembed_spectrum": ("#eda100", "--", "D"),
}
NAMES = {
    "embedding": "embedding weights",
    "mlp": "MLP weights",
    "attention": "attention weights",
    "unembedding": "unembedding weights",
    "heads": "attention head roles",
    "unembed_spectrum": "unembedding frequencies",
}


def plot_weights(
    summaries: dict[str, dict[int, dict]],
    rows: list[list[str]],
    labels: dict[str, str],
    out: Path,
) -> None:
    fig, axes = grid(rows)
    for cell, ax in axes.items():
        ax.set_title(labels.get(cell, cell), fontsize=8.5, loc="left", color=INK)
        by_seed = summaries.get(cell, {})
        for level in WEIGHT_LEVELS:
            color, style, marker = SERIES[level]
            per_lag: dict[int, list[float]] = defaultdict(list)
            for summary in by_seed.values():
                for point in summary["curve"]:
                    if point[level] is not None:
                        per_lag[point["lag"]].append(point[level])
            if per_lag:
                xs = sorted(per_lag)
                ys = [sum(per_lag[x]) / len(per_lag[x]) for x in xs]
                ax.plot(xs, ys, color=color, ls=style, marker=marker, lw=1.8, ms=4)
        n = len(by_seed)
        ax.text(
            0.98, 0.96, f"{n} seed{'s' * (n != 1)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUTED,
        )  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(-0.05, max(1.1, ax.get_ylim()[1]))
    axes[rows[0][0]].set_ylabel("moved (1 = unrelated)", fontsize=8, color=MUTED)
    for row in rows:
        axes[row[-1]].set_xlabel("lag (steps)", fontsize=8, color=MUTED)
    handles = [
        Line2D([], [], color=c, ls=s, marker=m, lw=1.8, ms=4)
        for c, s, m in SERIES.values()
    ]
    fig.legend(
        handles, list(NAMES.values()), loc="upper center", ncol=3,
        frameon=False, fontsize=8,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    args = parser.parse_args()
    summaries: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in args.summaries:
        cell, seed = cell_seed(path)
        summaries[cell][seed] = json.loads(path.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_weights(
        summaries,
        [r.split(",") for r in args.row],
        dict(lbl.split("=", 1) for lbl in args.label),
        args.out,
    )


if __name__ == "__main__":
    main()

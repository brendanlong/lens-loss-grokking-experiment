"""Figures for the stability grid (inputs from analyze_levels and trace).

Usage (normally driven by the Snakefile):
    uv run python -m grok_lens.plot_levels --out DIR \\
        --summaries S1.json ... --traces T1.pt ... \\
        --row L1-ln,L2-ln --row L1-noln,L2-noln --label L1-ln="1 layer, LN" ...
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from grok_lens.analyze_levels import CONTROL_RADIUS, LEVELS, load_trace, neuron_roles

LEVEL_COLORS = {"capability": "#2a78d6", "embedding": "#eb6834", "neurons": "#1baf7a"}
LEVEL_NAMES = {
    "capability": "test answers changed (share)",
    "embedding": "embedding frequencies (1 = shuffled)",
    "neurons": "neurons per frequency (1 = shuffled)",
}
INK, MUTED, GRID = "#333333", "#666666", "#e6e6e6"
# <cell>/s<seed>.json (summaries) or <cell>/s<seed>/trace.pt (runs)
RUN_PATH = re.compile(r"(?P<cell>[^/]+)/s(?P<seed>\d+)(?:\.json|/trace\.pt)$")


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def cell_seed(path: Path) -> tuple[str, int]:
    match = RUN_PATH.search(path.as_posix())
    if match is None:
        raise SystemExit(f"cannot parse <cell>/s<seed> from {path}")
    return match["cell"], int(match["seed"])


def grid(rows: list[list[str]], width: float = 3.0, height: float = 2.4):  # noqa: ANN201
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(width * n_cols, height * len(rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    placed: dict[str, Axes] = {}
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(row):
                placed[row[c]] = ax
                style(ax)
            else:
                ax.set_visible(False)
    return fig, placed


def plot_lag_curves(
    summaries: dict[str, dict[int, dict]],
    rows: list[list[str]],
    labels: dict[str, str],
    out: Path,
) -> None:
    fig, axes = grid(rows)
    for cell, ax in axes.items():
        ax.set_title(labels.get(cell, cell), fontsize=8.5, loc="left", color=INK)
        by_seed = summaries.get(cell, {})
        for level in LEVELS:
            color = LEVEL_COLORS[level]
            per_lag: dict[int, list[float]] = defaultdict(list)
            for summary in by_seed.values():
                pts = [
                    (p["lag"], p[level])
                    for p in summary["curves"]["pooled"]
                    if p[level] is not None
                ]
                if pts:
                    xs, ys = zip(*pts, strict=True)
                    ax.plot(xs, ys, color=color, lw=0.6, alpha=0.35)
                for x, y in pts:
                    per_lag[x].append(y)
            if per_lag:
                xs = sorted(per_lag)
                ys = [sum(per_lag[x]) / len(per_lag[x]) for x in xs]
                ax.plot(xs, ys, color=color, lw=2, marker="o", ms=3.5, label=level)
        n = len(by_seed)
        ax.text(
            0.98, 0.96, f"{n} seed{"s" * (n != 1)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUTED,
        )  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(-0.05, max(1.1, ax.get_ylim()[1]))
    first = axes[rows[0][0]]
    first.set_ylabel("moved (0 = unchanged)", fontsize=8, color=MUTED)
    for row in rows:
        axes[row[-1]].set_xlabel("lag (steps)", fontsize=8, color=MUTED)
    handles = [
        Line2D([], [], color=LEVEL_COLORS[lv], lw=2, marker="o", ms=3.5)
        for lv in LEVELS
    ]
    fig.legend(
        handles, [LEVEL_NAMES[lv] for lv in LEVELS], loc="upper center",
        ncol=3, frameon=False, fontsize=8,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_test_loss(
    traces: dict[str, dict[int, Path]],
    rows: list[list[str]],
    labels: dict[str, str],
    window_start: int,
    out: Path,
) -> None:
    fig, axes = grid(rows, width=3.0, height=2.0)
    for cell, ax in axes.items():
        ax.set_title(labels.get(cell, cell), fontsize=8.5, loc="left", color=INK)
        for path in traces.get(cell, {}).values():
            trace = load_trace(path)
            loss = trace["test_loss"].float().mean(1)
            ax.plot(trace["steps"], loss, color=INK, lw=0.6, alpha=0.6)
        ax.axvline(window_start, color=MUTED, lw=0.6, ls=":")
        ax.set_yscale("log")
    axes[rows[0][0]].set_ylabel("test loss", fontsize=8, color=MUTED)
    for row in rows:
        axes[row[-1]].set_xlabel("step", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def heatmap(ax: Axes, steps: torch.Tensor, rows: torch.Tensor, color: str) -> None:
    """``rows`` [R, T] as a white-to-``color`` heatmap over steps."""
    ax.grid(False)
    if not len(rows):
        ax.text(0.5, 0.5, "nothing above threshold", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=MUTED)  # fmt: skip
        return
    cmap = LinearSegmentedColormap.from_list(color, ["#ffffff", color])
    ax.pcolormesh(steps, torch.arange(len(rows)), rows, cmap=cmap,
                  shading="nearest", rasterized=True)  # fmt: skip


def plot_run_detail(trace_path: Path, title: str, window_start: int, out: Path) -> None:
    """Test loss, embedding spectrum and one role's neuron raster, one run."""
    trace = load_trace(trace_path)
    steps = trace["steps"]
    fig, axes = plt.subplots(
        3, 1, figsize=(7, 6.5), sharex=True, height_ratios=[1, 2, 2]
    )
    for ax in axes:
        style(ax)
    fig.suptitle(title, fontsize=9.5, x=0.02, ha="left", color=INK)

    axes[0].plot(steps, trace["test_loss"].float().mean(1), color=INK, lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("test loss", fontsize=8, color=MUTED)

    power = trace["embed_power"]
    shares = power / power.sum(-1, keepdim=True)
    keep = (shares[steps >= window_start].amax(0) >= 0.02).nonzero().flatten()
    heatmap(axes[1], steps, shares[:, keep].T, LEVEL_COLORS["embedding"])
    axes[1].set_yticks(range(len(keep)), [str(int(k) + 1) for k in keep], fontsize=6)
    axes[1].set_ylabel("embedding frequency\n(power share)", fontsize=8, color=MUTED)

    member, _ = neuron_roles(trace, member_share=0.10, min_frac_answer=0.5)
    in_window = steps >= window_start
    sizes = member[in_window].sum(1).float().min(0).values  # [K]
    role = int(sizes.argmax())
    rows = member[:, :, role]  # [T, N]
    ever = rows[in_window].any(0).nonzero().flatten()
    order = ever[rows[:, ever].float().argmax(0).argsort()]  # by first join
    heatmap(axes[2], steps, rows[:, order].T.float(), LEVEL_COLORS["neurons"])
    axes[2].set_ylabel(f"neurons serving freq {role + 1}", fontsize=8, color=MUTED)
    axes[2].set_xlabel("step", fontsize=8, color=MUTED)
    for ax in axes:
        ax.axvline(window_start, color=MUTED, lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_repairs(
    summaries: dict[str, dict[int, dict]], labels: dict[str, str], out: Path
) -> None:
    """Each collapse's pre→recovery change against collapse-free spans as long."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    markers = "osD^vP*X"
    cells = [
        c for c, by_seed in summaries.items()
        if any("moved" in e for s in by_seed.values() for e in s["events"])
    ]  # fmt: skip
    for ax, level in zip(axes, LEVELS, strict=True):
        style(ax)
        ax.set_title(LEVEL_NAMES[level], fontsize=8.5, loc="left", color=INK)
        for i, cell in enumerate(cells):
            pts = [
                (e["collapse_free_control"][level], e["moved"][level])
                for s in summaries[cell].values()
                for e in s["events"]
                if "moved" in e
                and e["moved"][level] is not None
                and e["collapse_free_control"][level] is not None
            ]
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.scatter(
                    xs, ys, s=14, marker=markers[i % len(markers)], facecolor="none",
                    edgecolor=LEVEL_COLORS[level], lw=0.8, label=labels.get(cell, cell),
                )  # fmt: skip
        ax.plot([0, 1.5], [0, 1.5], color=MUTED, lw=0.6, ls=":")
        ax.set_xlim(-0.02, 1.5)
        ax.set_ylim(-0.02, 1.5)
        ax.set_xlabel("collapse-free span of equal length", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("before collapse → after recovery", fontsize=7.5, color=MUTED)
    events = [e for by_seed in summaries.values() for s in by_seed.values()
              for e in s["events"]]  # fmt: skip
    unmatched = sum(
        1 for e in events
        if "moved" in e and not e["collapse_free_control"]["n_pairs"]
    )  # fmt: skip
    unrecovered = sum(1 for e in events if e["recovery_step"] is None)
    fig.text(
        0.01, 0.01,
        f"{len(events)} collapses; not shown: {unmatched} with no collapse-free "
        f"span of equal length within ±{CONTROL_RADIUS} steps, "
        f"{unrecovered} unrecovered by the end",
        fontsize=7, color=MUTED,
    )  # fmt: skip
    if cells:
        axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--window-start", type=int, default=30_000)
    parser.add_argument("--detail-seed", type=int, default=42)
    args = parser.parse_args()

    rows = [r.split(",") for r in args.row]
    labels = dict(lbl.split("=", 1) for lbl in args.label)
    summaries: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in args.summaries:
        cell, seed = cell_seed(path)
        summaries[cell][seed] = json.loads(path.read_text())
    traces: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in args.traces:
        cell, seed = cell_seed(path)
        traces[cell][seed] = path

    args.out.mkdir(parents=True, exist_ok=True)
    plot_lag_curves(summaries, rows, labels, args.out / "levels_by_lag.png")
    plot_test_loss(traces, rows, labels, args.window_start, args.out / "test_loss.png")
    plot_repairs(summaries, labels, args.out / "repairs.png")
    for cell, by_seed in traces.items():
        seed = args.detail_seed if args.detail_seed in by_seed else min(by_seed)
        plot_run_detail(
            by_seed[seed],
            f"{labels.get(cell, cell)} — seed {seed}",
            args.window_start,
            args.out / f"detail_{cell}.png",
        )


if __name__ == "__main__":
    main()

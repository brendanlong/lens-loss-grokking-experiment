"""Generate the writeup figures + formation-phase analysis.

Outputs PNGs into experiments/grok_lens/figures/ and prints the
formation-phase statistics (the part-2 seed question: does the aux loss
grow many weak components in parallel where the baseline crystallizes a
few winners?).

Writeup figures:
  1. occupancy.png         — test-acc trajectories: baseline sawtooth vs aux
                             absorbing (the hero image)
  2. knockout.png          — accuracy after frequency knockouts
  3. churn.png             — per-frequency power churn vs output accuracy
  4. drift.png             — circuit composition drift under a flat capability
  5. formation.png         — circuit concentration during formation
  6. lego_grok_fullseq.png — full-sequence grokking regime, all 6 runs
  7. probes_fullseq.png    — lens vs probe under full-sequence training
                             (needs data/analysis/fullseq-grok.json; see
                             scripts/reproduce_analyses.sh)

Legacy answer-only LEGO figures (RESULTS.md record; --legacy-answer-only):
  coalescence.png, probes.png, lego_grok.png

Usage:
    uv run python -m grok_lens.make_figures
"""

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from common.checkpoint import artifact_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
from grok_lens.analyze_drift import RUN as FFTTRACE_RUN
from grok_lens.analyze_drift import Run, power_moved, spectra
from grok_lens.analyze_fourier import fourier_power
from grok_lens.analyze_knockout import remove_frequency, test_acc
from grok_lens.config import GrokModelConfig
from grok_lens.data import train_test_split
from grok_lens.model import GrokTransformer

# Okabe-Ito subset, validated (dataviz six checks, light surface)
BLUE = "#0072B2"  # aux
VERM = "#D55E00"  # baseline
GREEN = "#009E73"  # third arm (LEGO linear)
GRAY = "#8a8a8a"
SEEDS = [42, 43, 44]
FREQ_KEYS = [f"freq_power/k_{i}" for i in range(1, 57)]


def style(ax: "Axes") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#666666", labelsize=8)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)


def load_model_and_test(
    name: str, cache: Path
) -> tuple[GrokTransformer, GrokModelConfig, torch.Tensor, torch.Tensor]:
    path = artifact_path(f"grok_lens/{name}/final.pt")
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    cfg = GrokModelConfig(
        **{
            k: v
            for k, v in ckpt["model_config"].items()
            if k in GrokModelConfig.model_fields
        }
    )
    model = GrokTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seed = int(re.search(r"-s(\d+)", name).group(1))  # type: ignore[union-attr]
    _, _, tokens, targets = train_test_split(cfg, 0.3, seed)
    return model, cfg, tokens, targets


def _run(api: wandb.Api, name: str) -> Run:
    return next(r for r in api.runs("brendanlong-com/grok-lens") if r.name == name)


def trace_history(api: wandb.Api, name: str) -> list[dict[str, float]]:
    run = next(r for r in api.runs("brendanlong-com/grok-lens") if r.name == name)
    return sorted(
        run.history(keys=[*FREQ_KEYS, "test/acc"], pandas=False, samples=600),
        key=lambda h: h["_step"],
    )


def fig_occupancy(api: wandb.Api, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)
    for seed in SEEDS:
        for kind, color, label in [
            ("lam0.0", VERM, "baseline"),
            ("lam0.3", BLUE, "aux λ=0.3"),
        ]:
            hist = trace_history(
                api, f"p113-L2-{kind}-uniform-frac0.3-s{seed}-ffttrace"
            )
            steps = [h["_step"] for h in hist]
            accs = [h["test/acc"] for h in hist]
            bold = seed == 42
            ax.plot(
                steps,
                accs,
                color=color,
                lw=1.6 if bold else 0.8,
                alpha=1.0 if bold else 0.35,
                label=label if bold else None,
            )
    style(ax)
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("test accuracy", fontsize=9)
    ax.set_ylim(-0.03, 1.06)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(
        "Baseline grokking never sticks; any aux weight makes it absorbing\n"
        "(wd = 1.0, 50k steps; bold = seed 42, faint = seeds 43/44)",
        fontsize=9.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(outdir / "occupancy.png", bbox_inches="tight")
    plt.close(fig)


def fig_knockout(cache: Path, outdir: Path) -> None:
    conditions = ["intact", "worst single\nfreq removed", "top 6\nremoved"]
    results: dict[str, list[list[float]]] = {"baseline": [], "aux": []}
    for arm, pattern in [
        ("baseline", "p113-L2-lam0.0-uniform-frac0.3-s{s}-50k"),
        ("aux", "p113-L2-lam0.3-uniform-frac0.3-s{s}"),
    ]:
        for s in SEEDS:
            model, cfg, tokens, targets = load_model_and_test(
                pattern.format(s=s), cache
            )
            with torch.no_grad():
                emb = model.embed.weight.detach()[: cfg.p].clone()
                base = test_acc(model, tokens, targets)
                power = fourier_power(emb, cfg.p)
                top = [int(k) + 1 for k in power.sort(descending=True).indices[:6]]
                worst = base
                removed_all = emb.clone()
                for f in top:
                    model.embed.weight.data[: cfg.p] = remove_frequency(emb, cfg.p, f)
                    worst = min(worst, test_acc(model, tokens, targets))
                    removed_all = remove_frequency(removed_all, cfg.p, f)
                model.embed.weight.data[: cfg.p] = removed_all
                all6 = test_acc(model, tokens, targets)
                model.embed.weight.data[: cfg.p] = emb
            results[arm].append([base, worst, all6])

    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
    width = 0.34
    for i, (arm, color) in enumerate([("baseline", VERM), ("aux", BLUE)]):
        vals = list(zip(*results[arm], strict=True))  # per-condition seed lists
        means = [sum(v) / len(v) for v in vals]
        xs = [x + (i - 0.5) * width for x in range(3)]
        ax.bar(
            xs,
            means,
            width=width * 0.94,
            color=color,
            label="aux λ=0.3" if arm == "aux" else "baseline",
        )
        for x, v in zip(xs, vals, strict=True):
            ax.scatter(
                [x] * len(v),
                v,
                s=9,
                color="#333333",
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )
    ax.axhline(1 / 113, color=GRAY, lw=0.8, ls=":")
    ax.text(2.35, 1 / 113 + 0.02, "chance", fontsize=7.5, color=GRAY)
    style(ax)
    ax.set_xticks(range(3), conditions, fontsize=8.5)
    ax.set_ylabel("test accuracy", fontsize=9)
    ax.set_title(
        "Sparse baseline circuits are one deletion from collapse;\n"
        "aux circuits survive losing their six largest frequencies",
        fontsize=9.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "knockout.png", bbox_inches="tight")
    plt.close(fig)


def fig_churn(api: wandb.Api, outdir: Path) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 4.0),
        dpi=160,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    for col, (kind, color, title) in enumerate(
        [("lam0.0", VERM, "baseline"), ("lam0.3", BLUE, "aux λ = 0.3")]
    ):
        hist = trace_history(api, f"p113-L2-{kind}-uniform-frac0.3-s42-ffttrace")
        steps = [h["_step"] for h in hist]
        axes[0][col].plot(steps, [h["test/acc"] for h in hist], color=color, lw=1.0)
        axes[0][col].set_ylim(-0.03, 1.06)
        axes[0][col].set_title(title, fontsize=9.5, loc="left")
        means = {k: sum(h[k] for h in hist) / len(hist) for k in FREQ_KEYS}
        active = sorted(means, key=lambda k: means[k], reverse=True)[:18]
        for k in active:
            axes[1][col].plot(
                steps, [h[k] for h in hist], color=GRAY, lw=0.55, alpha=0.6
            )
        axes[1][col].set_xlabel("training step", fontsize=9)
        for ax in (axes[0][col], axes[1][col]):
            style(ax)
    axes[0][0].set_ylabel("test acc", fontsize=9)
    axes[1][0].set_ylabel("per-frequency\npower share", fontsize=9)
    fig.suptitle(
        "Components churn in both regimes; only the sparse circuit's\n"
        "failures reach the output (seed 42)",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "churn.png", bbox_inches="tight")
    plt.close(fig)


def fig_drift(api: wandb.Api, outdir: Path) -> None:
    """Circuit composition drifts even where the capability is pinned.

    Answers the question the other figures don't: across the aux arm's
    post-grok tail the capability is near-constant (0.999 mean, two evals
    below 0.99 in the window shown), but the circuit underneath it is not
    the same circuit from one end of that tail to the other. Companion to
    fig_churn, which frames the same traces around the output-coupling
    result instead. The window matches analyze_drift's default so the
    figure and the quoted numbers describe the same interval.
    """
    since = 30_000
    # Component identities in the trace panel, direct-labelled (the CVD
    # check on this triple needs the secondary encoding).
    detail = [
        (32, "#009E73", "breaks and repairs", 6),
        (49, "#B07A00", "is born", -8),
        (24, "#8B4FA8", "dies", 0),
    ]
    fig = plt.figure(figsize=(7.6, 7.4), dpi=160)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[0.75, 1.7, 1.25], hspace=0.5, wspace=0.28
    )
    for col, (kind, color, title) in enumerate(
        [
            ("lam0.0", VERM, "baseline"),
            ("lam0.3", BLUE, "aux λ = 0.3  (the stable arm)"),
        ]
    ):
        steps, accs, shares = spectra(
            _run(api, FFTTRACE_RUN.format(kind=kind, seed=42)), since
        )
        ax = fig.add_subplot(gs[0, col])
        style(ax)
        ax.plot(steps, accs, color=color, lw=1.0)
        ax.set_ylim(-0.03, 1.08)
        ax.set_title(title, fontsize=9.5, loc="left", color="#333333")
        ax.set_ylabel("test acc", fontsize=8.5)

        ax2 = fig.add_subplot(gs[1, col])
        style(ax2)
        ax2.grid(False)
        active = [k for k in range(56) if max(s[k] for s in shares) >= 0.03]
        # Sort by centre of mass in time so births and deaths read as a diagonal.
        active.sort(
            key=lambda k: (
                sum(i * s[k] for i, s in enumerate(shares)) / sum(s[k] for s in shares)
            )
        )
        grid = [[s[k] for s in shares] for k in active]
        cmap = LinearSegmentedColormap.from_list(f"seq{col}", ["#ffffff", color])
        im = ax2.imshow(
            grid,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=max(max(r) for r in grid),
            extent=(steps[0], steps[-1], len(active) - 0.5, -0.5),
            interpolation="nearest",
        )
        ax2.set_yticks(range(len(active)))
        ax2.set_yticklabels([f"k={k + 1}" for k in active], fontsize=5.5)
        ax2.set_xlabel("training step", fontsize=8.5)
        if col == 0:
            ax2.set_ylabel("Fourier component of the embedding", fontsize=8.5)
        # Per-panel colour scales: the arms differ ~7x in peak share.
        cb = fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.045)
        cb.ax.tick_params(labelsize=6, colors="#666666")
        cb.ax.spines[["outline"]].set_visible(False)
        cb.set_label("power share", fontsize=6.5, color="#666666")

    ax3 = fig.add_subplot(gs[2, 0])
    style(ax3)
    ax3.grid(axis="y", color="#e6e6e6", lw=0.6)
    for kind, color in [("lam0.0", VERM), ("lam0.3", BLUE)]:
        for seed in SEEDS:
            steps, _, shares = spectra(
                _run(api, FFTTRACE_RUN.format(kind=kind, seed=seed)), since
            )
            ax3.plot(
                steps,
                [power_moved(shares[0], s) for s in shares],
                color=color,
                lw=1.4,
                alpha=0.85,
            )
    ax3.set_ylim(0, None)
    ax3.set_xlabel("training step", fontsize=8.5)
    ax3.set_ylabel("share of circuit power\nheld elsewhere than at 30k", fontsize=8.5)
    ax3.set_title(
        "drift from the step-30k circuit", fontsize=9, loc="left", color="#333333"
    )
    ax3.legend(
        handles=[
            Line2D([], [], color=VERM, lw=1.4, label="baseline"),
            Line2D([], [], color=BLUE, lw=1.4, label="aux λ = 0.3"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )

    ax4 = fig.add_subplot(gs[2, 1])
    style(ax4)
    ax4.grid(axis="y", color="#e6e6e6", lw=0.6)
    steps, accs, shares = spectra(
        _run(api, FFTTRACE_RUN.format(kind="lam0.3", seed=42)), since
    )
    for k in range(56):
        if max(s[k] for s in shares) >= 0.03 and k + 1 not in {d[0] for d in detail}:
            ax4.plot(steps, [s[k] for s in shares], color=GRAY, lw=0.5, alpha=0.35)
    for freq, color, label, dy in detail:
        ax4.plot(steps, [s[freq - 1] for s in shares], color=color, lw=1.6)
        ax4.annotate(
            f"k={freq} {label}",
            xy=(steps[-1], shares[-1][freq - 1]),
            xytext=(5, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            va="center",
        )
    ax4.set_xlabel("training step", fontsize=8.5)
    ax4.set_ylabel("power share", fontsize=8.5)
    ax4.set_xlim(steps[0], steps[-1] + 9500)
    ax4.set_xticks(list(range(steps[0], steps[-1] + 1, 5000)))
    ax4.set_title(
        "aux λ = 0.3, seed 42: individual components",
        fontsize=9,
        loc="left",
        color="#333333",
    )

    fig.suptitle(
        "Post-grok, the capability stops moving; the circuit need not",
        fontsize=10.5,
        x=0.02,
        ha="left",
        color="#222222",
    )
    fig.savefig(outdir / "drift.png", bbox_inches="tight")
    plt.close(fig)


def fig_coalescence(outdir: Path) -> None:
    """Coalescence layer l*(k) per LEGO run, computed from the checkpoints
    in lego.compare_lens_aux.RUNS, each probed on ITS OWN held-out split
    (the split seed follows the training --seed, parsed from the run name)."""
    from lego.compare_lens_aux import (
        RUNS,
        coalescence_layer,
        run_seed,
        test_examples_by_k,
    )
    from lego.training import load_model as load_lego_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = [2, 4, 6]
    probe_cache: dict[int, dict[int, list]] = {}
    data: dict[str, list[list[int]]] = {}  # arm -> per-seed [l* for each k]
    n_layers_max = 0
    for label, run_name, ckpt_file in RUNS:
        seed = run_seed(run_name)
        if seed not in probe_cache:
            probe_cache[seed] = test_examples_by_k(
                k_min=0, k_max=6, test_frac=0.2, seed=seed, n_examples=500
            )
        examples_by_k = probe_cache[seed]
        model, config = load_lego_model(
            artifact_path(f"lego/{run_name}/{ckpt_file}"), device
        )
        n_layers = config.n_layers
        n_layers_max = max(n_layers_max, n_layers)
        vals: list[int] = []
        for k in ks:
            lstar = coalescence_layer(model, examples_by_k[k], device)
            if lstar is None:
                # never reaches 95% readability — plot at n_layers, flagged
                print(
                    f"WARNING: {run_name} k={k}: lens never >=0.95; plotting at "
                    f"{n_layers} (= no layer)"
                )
                lstar = n_layers
            vals.append(lstar)
        arm = label.rsplit(" ", 1)[0]  # "baseline s42" -> "baseline"
        data.setdefault(arm, []).append(vals)
        print(f"  l*(k={ks}) {label}: {vals}")

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=160)
    for (label, seeds_vals), color in zip(
        data.items(), [VERM, BLUE, GREEN], strict=True
    ):
        # Mean over converged seeds only; non-converged (l* = n_layers,
        # lens never >=0.95) would drag the line to a fake value.
        mean = [
            sum(vs) / len(vs)
            if (vs := [v[i] for v in seeds_vals if v[i] < n_layers_max])
            else float("nan")
            for i in range(len(ks))
        ]
        ax.plot(ks, mean, color=color, lw=1.6, marker="o", ms=4, label=label)
        for j, v in enumerate(seeds_vals):
            xs = [k + (j - 1) * 0.07 for k in ks]
            conv = [(x, y) for x, y in zip(xs, v, strict=True) if y < n_layers_max]
            nonc = [(x, y) for x, y in zip(xs, v, strict=True) if y >= n_layers_max]
            if conv:
                ax.scatter(*zip(*conv, strict=True), s=8, color=color, alpha=0.45)
            if nonc:
                ax.scatter(
                    *zip(*nonc, strict=True),
                    s=14,
                    facecolors="none",
                    edgecolors=color,
                    alpha=0.7,
                )
    style(ax)
    ax.set_xticks(ks)
    ax.set_yticks(range(0, n_layers_max + 1), [*map(str, range(n_layers_max)), "n/r"])
    ax.set_xlabel("hops k", fontsize=9)
    ax.set_ylabel("first layer where the lens\nreads the answer (l*)", fontsize=9)
    ax.set_title(
        "LEGO, held-out chains: baselines answer only in the last layers;\n"
        "the aux loss front-loads, most at small k (lines = converged-seed\n"
        'means; open circles "n/r" = lens never reaches 95%)',
        fontsize=9.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "coalescence.png", bbox_inches="tight")
    plt.close(fig)


def fig_formation(api: wandb.Api, outdir: Path) -> None:
    """Circuit concentration over full training: both arms grok with a
    distributed circuit; wd then prunes the baseline sparse (aux blocks it)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)
    summary: dict[str, list[tuple[float, float]]] = {"lam0.0": [], "lam0.3": []}
    for kind, color, label in [
        ("lam0.0", VERM, "baseline"),
        ("lam0.3", BLUE, "aux \u03bb=0.3"),
    ]:
        for seed in SEEDS:
            hist = trace_history(
                api, f"p113-L2-{kind}-uniform-frac0.3-s{seed}-ffttrace"
            )
            first = next((i for i, h in enumerate(hist) if h["test/acc"] >= 0.95), None)
            if first is None:
                continue
            steps = [h["_step"] for h in hist]
            top6 = [
                sum(sorted((h[k] for k in FREQ_KEYS), reverse=True)[:6]) for h in hist
            ]
            bold = seed == 42
            ax.plot(
                steps,
                top6,
                color=color,
                lw=1.5 if bold else 0.7,
                alpha=1.0 if bold else 0.35,
                label=label if bold else None,
            )
            ax.scatter(
                [steps[first]],
                [top6[first]],
                s=26,
                color=color,
                zorder=4,
                edgecolors="white",
                linewidths=0.7,
            )
            summary[kind].append((top6[first], top6[-1]))
    style(ax)
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("top-6 frequency power share\n(circuit concentration)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(
        "Both arms grok with a distributed circuit (dots = first grokking);\n"
        "weight decay then prunes the baseline sparse - aux blocks the pruning",
        fontsize=9.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(outdir / "formation.png", bbox_inches="tight")
    plt.close(fig)

    print("\nTop-6 concentration (at first grokking -> end of run):")
    for kind, vals in summary.items():
        pretty = ", ".join(f"{a:.2f}->{b:.2f}" for a, b in vals)
        print(f"  {kind}: {pretty}")


def fig_probes(outdir: Path, probes_json: Path) -> None:
    """Probe vs lens at the <predict> position (LEGO, k = 4): intermediates
    are linearly present in lens-dark directions in both arms; only in aux
    runs does the lens track the linearly-present answer."""
    results = json.loads(probes_json.read_text())
    arms = [
        ("baseline", "S3-std-8L-splitku-base-s{s}", VERM),
        ("aux uniform λ=0.3", "S3-std-8L-splitku-lensaux0.3-uniform-s{s}", BLUE),
    ]
    k = 4
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), dpi=160, sharex=True)
    for col, (title, pattern, color) in enumerate(arms):
        for seed in SEEDS:
            r = results[pattern.format(s=seed)][f"k{k}"]
            probe = r["probe_eval_acc"]
            lens = r["lens_eval_acc"]
            layers = range(len(probe))
            bold = seed == 42
            lw = 1.6 if bold else 0.8
            alpha = 1.0 if bold else 0.35
            # row 0: best intermediate state (j = 1..k-1) per layer
            axes[0][col].plot(
                layers,
                [max(row[1:k]) for row in probe],
                color=color,
                lw=lw,
                alpha=alpha,
                label="linear probe" if bold else None,
            )
            axes[0][col].plot(
                layers,
                [max(row[1:k]) for row in lens],
                color=color,
                lw=lw,
                alpha=alpha,
                ls="--",
                label="logit lens" if bold else None,
            )
            # row 1: the answer (j = k) per layer
            axes[1][col].plot(
                layers,
                [row[k] for row in probe],
                color=color,
                lw=lw,
                alpha=alpha,
            )
            axes[1][col].plot(
                layers,
                [row[k] for row in lens],
                color=color,
                lw=lw,
                alpha=alpha,
                ls="--",
            )
        axes[0][col].set_title(title, fontsize=9.5, loc="left")
        axes[1][col].set_xlabel("layer", fontsize=9)
        for row in (0, 1):
            ax = axes[row][col]
            ax.axhline(1 / 6, color=GRAY, lw=0.8, ls=":")
            ax.set_ylim(-0.03, 1.06)
            style(ax)
    axes[0][0].set_ylabel("best intermediate state\nheld-out accuracy", fontsize=9)
    axes[1][0].set_ylabel("final answer\nheld-out accuracy", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0][0].text(6.9, 1 / 6 + 0.03, "chance", fontsize=7.5, color=GRAY)
    fig.suptitle(
        "The supervised position, probed (k = 4 chains): intermediates live in\n"
        "lens-dark directions either way; only aux models' lens tracks the answer",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outdir / "probes.png", bbox_inches="tight")
    plt.close(fig)


def fig_lego_grok_fullseq_grid(api: wandb.Api, outdir: Path) -> None:
    """All six full-sequence grokking runs (2 arms x 3 seeds): per-k
    held-out accuracy over training. Single-seed views of this regime
    mislead — seed variance dominates the arm contrast."""
    ks = [2, 3, 4, 5, 6]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        2, 3, figsize=(9.6, 4.4), dpi=160, sharey=True, sharex=True
    )
    for row, pattern in enumerate(
        [
            "S3-grok-sub10000-wd0.3-fullseqbase-s{s}-100k",
            "S3-grok-sub10000-wd0.3-fullseq-lensaux0.3-allpos-s{s}-100k",
        ]
    ):
        for col, seed in enumerate(SEEDS):
            ax = axes[row][col]
            name = pattern.format(s=seed)
            run = next(
                r for r in api.runs("brendanlong-com/grok-lens") if r.name == name
            )
            hist = sorted(
                (h for h in run.scan_history() if h.get("test_acc/k_2") is not None),
                key=lambda h: h["_step"],
            )
            steps = [h["_step"] for h in hist]
            for i, k in enumerate(ks):
                ax.plot(
                    steps,
                    [h[f"test_acc/k_{k}"] for h in hist],
                    color=cmap(0.1 + 0.8 * i / (len(ks) - 1)),
                    lw=0.9,
                    label=f"k={k}" if (row, col) == (0, 0) else None,
                )
            ax.set_ylim(-0.03, 1.06)
            style(ax)
            if row == 0:
                ax.set_title(f"seed {seed}", fontsize=9.5, loc="left")
            if row == 1:
                ax.set_xlabel("training step", fontsize=9)
    axes[0][0].set_ylabel("base objective\nheld-out accuracy", fontsize=9)
    axes[1][0].set_ylabel("+ deep supervision\nheld-out accuracy", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right", ncols=2)
    fig.suptitle(
        "Full-sequence grokking regime, all runs: the base transitions everywhere "
        "but rarely settles;\ndeep supervision delays (s42), stalls partway (s43), or "
        "matches-and-improves (s44) the transition",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(outdir / "lego_grok_fullseq.png", bbox_inches="tight")
    plt.close(fig)


def fig_probes_fullseq(outdir: Path, grid_json: Path) -> None:
    """Lens vs probe under full-sequence training (k = 4 chains), on the
    grokking-regime models that solve the k = 4 stratum. Data from
    `lego.analyze_fullseq --runs ... --json-out` on those checkpoints."""
    results = json.loads(grid_json.read_text())
    arms = [
        (
            "S3-grok-sub10000-wd0.3-fullseqbase-s43-100k",
            "full-sequence base (seed 43)",
            VERM,
        ),
        (
            "S3-grok-sub10000-wd0.3-fullseq-lensaux0.3-allpos-s44-100k",
            "full-sequence base + aux λ=0.3 (seed 44)",
            BLUE,
        ),
    ]
    k = 4
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), dpi=160, sharex=True)
    for col, (run, title, color) in enumerate(arms):
        r = results[run][f"k{k}"]
        probe = r["probe_traj_acc"]  # (P, L, k+1)
        lens = r["lens_traj_acc"]  # (L, P, k+1)
        next_acc = r["lens_next_token_acc"]  # (L, P)
        n_layers = len(lens)
        n_pos = len(probe)
        layers = range(n_layers)
        # row 0: best intermediate state (j = 1..k-1), max over positions
        probe_best = [
            max(probe[p][li][j] for p in range(n_pos) for j in range(1, k))
            for li in layers
        ]
        lens_best = [
            max(lens[li][p][j] for p in range(n_pos) for j in range(1, k))
            for li in layers
        ]
        axes[0][col].plot(layers, probe_best, color=color, lw=1.6, label="linear probe")
        axes[0][col].plot(
            layers, lens_best, color=color, lw=1.6, ls="--", label="logit lens"
        )
        # row 1: the answer at <predict> — the lens's own trained target there
        pr = 2 * k + 2
        axes[1][col].plot(
            layers,
            [probe[pr][li][k] for li in layers],
            color=color,
            lw=1.6,
        )
        axes[1][col].plot(
            layers,
            [row[pr] for row in next_acc],
            color=color,
            lw=1.6,
            ls="--",
        )
        axes[0][col].set_title(title, fontsize=9.5, loc="left")
        axes[1][col].set_xlabel("layer", fontsize=9)
        for row in (0, 1):
            ax = axes[row][col]
            ax.axhline(1 / 6, color=GRAY, lw=0.8, ls=":")
            ax.set_ylim(-0.03, 1.06)
            style(ax)
    axes[0][0].set_ylabel("best intermediate\n(any position)", fontsize=9)
    axes[1][0].set_ylabel("answer at <predict>", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0][0].text(6.9, 1 / 6 + 0.03, "chance", fontsize=7.5, color=GRAY)
    fig.suptitle(
        "Full-sequence training, models that solve k = 4: intermediates are\n"
        "linearly present but lens-invisible in both arms; the answer — the\n"
        "lens's trained target — is the one thing lens and probe agree on",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(outdir / "probes_fullseq.png", bbox_inches="tight")
    plt.close(fig)


def fig_lego_grok(
    api: wandb.Api,
    outdir: Path,
    runs: list[tuple[str, str]] | None = None,
    filename: str = "lego_grok.png",
    suptitle: str = (
        "LEGO grokking regime: the baseline grokks per-k in stages and wobbles;\n"
        "the aux loss compresses the staircase and holds it"
    ),
) -> None:
    """LEGO grokking regime (10k-chain subset, wd 0.3): per-k held-out
    accuracy over training, two arms side by side, one representative seed."""
    ks = [2, 3, 4, 5, 6]
    cmap = plt.get_cmap("viridis")
    if runs is None:
        runs = [
            ("S3-grok-sub10000-wd0.3-base-s44-100k", "baseline (seed 44)"),
            (
                "S3-grok-sub10000-wd0.3-lensaux0.3-uniform-s44-100k",
                "aux λ=0.3 (seed 44)",
            ),
        ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), dpi=160, sharey=True)
    for ax, (name, title) in zip(axes, runs, strict=True):
        run = next(r for r in api.runs("brendanlong-com/grok-lens") if r.name == name)
        hist = sorted(
            (h for h in run.scan_history() if h.get("test_acc/k_2") is not None),
            key=lambda h: h["_step"],
        )
        steps = [h["_step"] for h in hist]
        for i, k in enumerate(ks):
            ax.plot(
                steps,
                [h[f"test_acc/k_{k}"] for h in hist],
                color=cmap(0.1 + 0.8 * i / (len(ks) - 1)),
                lw=1.1,
                label=f"k={k}",
            )
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("training step", fontsize=9)
        ax.set_ylim(-0.03, 1.06)
        style(ax)
    axes[0].set_ylabel("held-out accuracy", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right", ncols=2)
    fig.suptitle(suptitle, fontsize=9.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(outdir / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes-json",
        type=Path,
        default=Path("data/analysis/probes.json"),
        help="Output of lego.analyze_probes --json-out (probes.png input)",
    )
    parser.add_argument(
        "--fullseq-grid-json",
        type=Path,
        default=Path("data/analysis/fullseq-grok.json"),
        help=(
            "Output of lego.analyze_fullseq on the grokking-regime "
            "full-sequence checkpoints (probes_fullseq.png input)"
        ),
    )
    parser.add_argument(
        "--legacy-answer-only",
        action="store_true",
        help=(
            "Also regenerate the answer-only LEGO figures (coalescence.png, "
            "probes.png, lego_grok.png) documented in RESULTS.md; these are "
            "not part of the writeup."
        ),
    )
    args = parser.parse_args()
    outdir = Path(__file__).resolve().parents[1] / "figures"
    outdir.mkdir(exist_ok=True)
    api = wandb.Api()
    fig_occupancy(api, outdir)
    print("occupancy.png done")
    fig_knockout(Path("data/hf_cache"), outdir)
    print("knockout.png done")
    fig_churn(api, outdir)
    print("churn.png done")
    fig_drift(api, outdir)
    print("drift.png done")
    fig_formation(api, outdir)
    print("formation.png done")
    fig_lego_grok_fullseq_grid(api, outdir)
    print("lego_grok_fullseq.png done")
    if args.legacy_answer_only:
        fig_coalescence(outdir)
        print("coalescence.png done")
        fig_lego_grok(api, outdir)
        print("lego_grok.png done")
        if args.probes_json.exists():
            fig_probes(outdir, args.probes_json)
            print("probes.png done")
        else:
            print(
                f"probes.png skipped: {args.probes_json} not found "
                "(run lego.analyze_probes --json-out first)"
            )
    if args.fullseq_grid_json.exists():
        fig_probes_fullseq(outdir, args.fullseq_grid_json)
        print("probes_fullseq.png done")
    else:
        print(
            f"probes_fullseq.png skipped: {args.fullseq_grid_json} not found "
            "(run the lego.analyze_fullseq --runs command in "
            "scripts/reproduce_analyses.sh first)"
        )


if __name__ == "__main__":
    main()

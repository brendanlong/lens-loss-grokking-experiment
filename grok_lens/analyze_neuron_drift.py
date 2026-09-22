"""Does the population of neurons serving a circuit role turn over?

`analyze_drift` asks whether the *embedding's* frequency content moves. This
asks the question representational drift is actually about: a role stays in
the circuit, but do the same units keep carrying it?

The role here is a Fourier frequency of the answer. In the grokked circuit
(Nanda et al. 2023) an MLP neuron's activation at the answer position is
largely a function of (a + b), concentrated on a few frequencies, so each
neuron has a *profile* over frequencies — and each frequency has a
*population* of neurons serving it. Both are measured per checkpoint:

  - ``profile[n]``: neuron n's power distribution over frequencies 1..(p-1)/2,
    taken from its activation averaged over each value of (a + b).
  - ``frac_sum[n]``: how much of n's variance is a function of (a + b) at all.
    Neurons below ``--min-frac-sum`` are not doing answer-frequency work and
    are excluded.
  - ``population(k)``: neurons devoting at least ``--member-share`` of their
    profile to frequency k. Membership is soft — a neuron can serve several
    frequencies, which is both true here (aux neurons are polysemantic, top
    frequency ~0.19 of profile) and true of real neurons.

Reported against a lag Δ, over checkpoint pairs:

  - population Jaccard per frequency — the direct analogue of "circle the
    cells encoding the maze, come back a week later".
  - profile cosine per neuron — threshold-free: did this unit change its job?
  - test accuracy at both ends, so role turnover can be read against whether
    the capability moved at all.

Requires checkpoints saved through training (``train.py --checkpoint-every``);
the published runs only kept ``final.pt``.

Usage:
    uv run python -m grok_lens.analyze_neuron_drift \\
        --checkpoints driftckpt/aux-s42 --seed 42 --label "aux λ=0.3"
"""

import argparse
import re
from pathlib import Path
from typing import cast

import torch

from grok_lens.config import GrokModelConfig
from grok_lens.data import train_test_split
from grok_lens.model import Block, GrokTransformer


def load(path: Path) -> tuple[GrokTransformer, GrokModelConfig]:
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
    return model, cfg


def all_pairs(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return torch.stack([a, b, torch.full_like(a, p)], dim=1), (a + b) % p


@torch.no_grad()
def neuron_profiles(
    model: GrokTransformer, p: int, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(profile [n, n_freqs], frac_sum [n], variance [n]) for one block's MLP.

    ``profile`` rows are L1-normalised power over frequencies 1..(p-1)/2 of
    the neuron's activation as a function of (a + b); rows whose activation
    is constant are left at zero.
    """
    tokens, sums = all_pairs(p)
    acts: dict[int, torch.Tensor] = {}
    relu = cast("torch.nn.Module", cast("Block", model.blocks[block]).mlp[1])
    handle = relu.register_forward_hook(
        lambda _m, _i, out: acts.__setitem__(0, out[:, -1].detach())
    )
    model(tokens)
    handle.remove()
    act = acts[0]  # [p^2, n_neurons]

    by_sum = torch.zeros(p, act.shape[1])
    by_sum.index_add_(0, sums, act)
    by_sum /= p  # mean activation per value of (a + b)

    variance = act.var(0, unbiased=False)
    frac_sum = (by_sum.var(0, unbiased=False) / variance.clamp_min(1e-12)).clamp(0, 1)

    centred = by_sum - by_sum.mean(0, keepdim=True)
    n_freqs = (p - 1) // 2
    spec = torch.fft.rfft(centred, dim=0)[1 : n_freqs + 1].abs().pow(2)  # [K, n]
    profile = (spec / spec.sum(0).clamp_min(1e-12)).T  # [n, K]
    return profile, frac_sum, variance


@torch.no_grad()
def test_accuracy(model: GrokTransformer, cfg: GrokModelConfig, seed: int) -> float:
    _, _, tokens, targets = train_test_split(cfg, 0.3, seed)
    logits = model(tokens)[-1]
    return (logits.argmax(-1) == targets).float().mean().item()


def populations(
    profile: torch.Tensor, active: torch.Tensor, member_share: float
) -> list[set[int]]:
    """Per frequency, the set of active neurons devoting >= member_share to it."""
    member = (profile >= member_share) & active[:, None]
    return [
        set(member[:, k].nonzero().flatten().tolist()) for k in range(member.shape[1])
    ]


def jaccard(a: set[int], b: set[int]) -> float | None:
    return len(a & b) / len(a | b) if (a or b) else None


def retention_lift(
    pops: list[list[set[int]]],
    actives: list[torch.Tensor],
    role: int,
    lag: int,
    min_pop: int = 5,
) -> float | None:
    """Chance-corrected share of a role's neurons still serving it after `lag`.

    Restricted to neurons active at both times, so a neuron that stops being
    answer-driven altogether is not scored as having left the role (that
    would bias large roles below chance). Retention is the share of P(t) still
    in P(t+lag); chance is the share of all both-times-active neurons in
    P(t+lag), i.e. the retention if later membership ignored identity. Lift
    rescales so 1 = every neuron stayed and 0 = chance, which makes roles of
    very different sizes comparable (Jaccard is not: a role holding most
    neurons overlaps heavily by default). Averaged over t; None if the role is
    never big enough to measure.
    """
    lifts = []
    for t in range(len(pops) - lag):
        both = set((actives[t] & actives[t + lag]).nonzero().flatten().tolist())
        now = pops[t][role] & both
        later = pops[t + lag][role] & both
        if len(now) < min_pop or not both:
            continue
        chance = len(later) / len(both)
        if chance >= 1:
            continue
        lifts.append((len(now & later) / len(now) - chance) / (1 - chance))
    return sum(lifts) / len(lifts) if lifts else None


def trace(
    checkpoints: Path, seed: int, since: int, block: int, min_frac_sum: float
) -> tuple[list[int], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Steps, test accuracy, neuron profiles and active masks per checkpoint."""
    paths = sorted(
        checkpoints.glob("step_*.pt"),
        key=lambda p: int(re.findall(r"step_(\d+)", p.name)[0]),
    )
    paths = [p for p in paths if int(re.findall(r"step_(\d+)", p.name)[0]) >= since]
    if len(paths) < 2:
        raise SystemExit(f"need >=2 checkpoints at/after step {since} in {checkpoints}")
    steps, accs, profiles, actives = [], [], [], []
    for path in paths:
        model, cfg = load(path)
        layer = block % cfg.n_layers
        profile, frac_sum, variance = neuron_profiles(model, cfg.p, layer)
        steps.append(int(re.findall(r"step_(\d+)", path.name)[0]))
        accs.append(test_accuracy(model, cfg, seed))
        profiles.append(profile)
        actives.append((frac_sum >= min_frac_sum) & (variance > variance.max() * 1e-3))
    return steps, accs, profiles, actives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--block", type=int, default=-1, help="MLP block (-1 = last)")
    parser.add_argument("--since", type=int, default=30_000)
    parser.add_argument("--min-frac-sum", type=float, default=0.5)
    parser.add_argument("--member-share", type=float, default=0.10)
    parser.add_argument(
        "--min-population", type=int, default=5, help="Ignore smaller populations"
    )
    args = parser.parse_args()

    steps, accs, profiles, actives = trace(
        args.checkpoints, args.seed, args.since, args.block, args.min_frac_sum
    )

    pops = [
        populations(pr, ac, args.member_share)
        for pr, ac in zip(profiles, actives, strict=True)
    ]
    n_freqs = profiles[0].shape[1]
    cadence = steps[1] - steps[0]

    print(f"\n=== {args.label}  block {args.block}  steps {steps[0]}-{steps[-1]}")
    print(f"test acc: mean {sum(accs) / len(accs):.4f}  min {min(accs):.3f}")
    print(
        f"active neurons (frac_sum >= {args.min_frac_sum}): "
        f"{int(actives[0].sum())} -> {int(actives[-1].sum())} of {len(actives[0])}"
    )
    sizes = [
        len(pops[0][k])
        for k in range(n_freqs)
        if len(pops[0][k]) >= args.min_population
    ]
    print(
        f"populations >= {args.min_population} members at window start: {len(sizes)} "
        f"(sizes {min(sizes) if sizes else 0}-{max(sizes) if sizes else 0})"
    )

    print(f"\n{'lag':>8}{'pop Jaccard':>14}{'profile cos':>14}{'n freqs':>10}")
    print("-" * 46)
    for lag in (1, 2, 5, 10, 20, 40, len(steps) - 1):
        if lag >= len(steps) or lag < 1:
            continue
        js, cos, counted = [], [], 0
        for t in range(len(steps) - lag):
            shared = actives[t] & actives[t + lag]
            if shared.any():
                a, b = profiles[t][shared], profiles[t + lag][shared]
                cos.append(
                    float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())
                )
            for k in range(n_freqs):
                if len(pops[t][k]) >= args.min_population:
                    value = jaccard(pops[t][k], pops[t + lag][k])
                    if value is not None:
                        js.append(value)
                        counted += 1
        mean_j = f"{sum(js) / len(js):>14.3f}" if js else f"{'--':>14}"
        mean_c = f"{sum(cos) / len(cos):>14.3f}" if cos else f"{'--':>14}"
        print(
            f"{lag * cadence:>8}{mean_j}{mean_c}{counted / (len(steps) - lag):>10.1f}"
        )
    print(
        "\n'pop Jaccard' is the overlap of the neuron set serving a frequency at\n"
        "t and t+lag, averaged over frequencies with enough members and over t.\n"
        "'profile cos' is the mean cosine between a neuron's own frequency\n"
        "profile at t and t+lag (threshold-free). 'n freqs' is how many\n"
        "populations met the size floor, averaged over t."
    )


if __name__ == "__main__":
    main()

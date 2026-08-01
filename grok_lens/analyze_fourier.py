"""Phase 4 circuit check: Fourier structure of the learned embeddings.

The known generalizing circuit for a + b mod p (Nanda et al. 2023)
concentrates the embedding matrix's variance on a few Fourier frequencies
of Z_p. This script measures that signature across all runs: per-frequency
power of the number-token embeddings, the share held by the top-k
frequencies, and how many frequencies are needed to reach 90% of the
(non-DC) power. Sparse power (a handful of key frequencies) = the Fourier
multiplication circuit; diffuse power = something else.

Usage:
    uv run python -m grok_lens.analyze_fourier
"""

import argparse
import math

import torch

from common.checkpoint import artifact_path
from grok_lens.config import GrokModelConfig

CELLS: list[tuple[str, list[str]]] = [
    ("L2 baseline", ["p113-L2-lam0.0-uniform-frac0.3-s{s}"]),
    ("L2 lam0.01", ["p113-L2-lam0.01-uniform-frac0.3-s{s}"]),
    ("L2 lam0.1", ["p113-L2-lam0.1-uniform-frac0.3-s{s}"]),
    ("L2 lam0.3", ["p113-L2-lam0.3-uniform-frac0.3-s{s}"]),
    ("L2 lam1.0", ["p113-L2-lam1.0-uniform-frac0.3-s{s}"]),
    ("L2 lam3.0", ["p113-L2-lam3.0-uniform-frac0.3-s{s}"]),
    ("L1 baseline", ["p113-L1-lam0.0-uniform-frac0.3-s{s}"]),
    ("L3 baseline", ["p113-L3-lam0.0-uniform-frac0.3-s{s}"]),
    ("L3 lam0.3 uniform", ["p113-L3-lam0.3-uniform-frac0.3-s{s}"]),
    ("L3 lam0.3 linear", ["p113-L3-lam0.3-linear-frac0.3-s{s}"]),
    ("L2 base 50k", ["p113-L2-lam0.0-uniform-frac0.3-s{s}-50k"]),
    ("L3 base 50k", ["p113-L3-lam0.0-uniform-frac0.3-s{s}-50k"]),
    ("L2 base muon", ["p113-L2-lam0.0-uniform-frac0.3-s{s}-muon-50k"]),
    ("L2 lam0.3 muon", ["p113-L2-lam0.3-uniform-frac0.3-s{s}-muon-50k"]),
    ("L3 base muon", ["p113-L3-lam0.0-uniform-frac0.3-s{s}-muon-50k"]),
    ("L3 lam0.3 muon", ["p113-L3-lam0.3-uniform-frac0.3-s{s}-muon-50k"]),
    # torch.optim.Muon re-runs (the canonical Muon numbers in RESULTS/WRITEUP)
    ("L2 base tmuon", ["p113-L2-lam0.0-uniform-frac0.3-s{s}-torchmuon-50k"]),
    ("L2 lam0.3 tmuon", ["p113-L2-lam0.3-uniform-frac0.3-s{s}-torchmuon-50k"]),
    ("L3 base tmuon", ["p113-L3-lam0.0-uniform-frac0.3-s{s}-torchmuon-50k"]),
    ("L3 lam0.3 tmuon", ["p113-L3-lam0.3-uniform-frac0.3-s{s}-torchmuon-50k"]),
    # no-LayerNorm architecture control (ends sparse yet trains stably)
    ("L2 base no-LN", ["p113-L2-lam0.0-uniform-frac0.3-s{s}-noln-50k"]),
]
SEEDS = [42, 43, 44]
# 500k long-horizon pair exists for seed 42 only.
CELLS_500K: list[tuple[str, list[str]]] = [
    ("L2 base 500k", ["p113-L2-lam0.0-uniform-frac0.3-s{s}-500k"]),
    ("L2 lam0.3 500k", ["p113-L2-lam0.3-uniform-frac0.3-s{s}-500k"]),
]
SEEDS_500K = [42]


def fourier_power(embed: torch.Tensor, p: int) -> torch.Tensor:
    """Power per frequency k=1..(p-1)//2 of the number-token embeddings.

    ``embed`` is [p, dim] (the "=" row excluded). Rows are mean-centered
    (drops the DC component), then projected onto the cos/sin basis of Z_p;
    power at frequency k is the squared norm of both coefficients summed
    over embedding dims, normalized to sum to 1.
    """
    x = embed - embed.mean(dim=0, keepdim=True)
    positions = torch.arange(p, dtype=torch.float32)
    n_freqs = (p - 1) // 2
    ks = torch.arange(1, n_freqs + 1, dtype=torch.float32)
    angles = 2 * math.pi * ks[:, None] * positions[None, :] / p  # [K, p]
    # Unnormalized projections are fine: cos/sin rows share the same norm
    # (sqrt(p/2)) for 0 < k < p/2, so relative power is unaffected.
    cos_coef = torch.cos(angles) @ x  # [K, dim]
    sin_coef = torch.sin(angles) @ x
    power = cos_coef.pow(2).sum(dim=1) + sin_coef.pow(2).sum(dim=1)
    return power / power.sum()


def summarize(power: torch.Tensor) -> tuple[float, int, list[int]]:
    """Return (top-6 power share, #freqs for 90% power, top-6 freq list)."""
    sorted_power, order = power.sort(descending=True)
    top6 = float(sorted_power[:6].sum())
    cum = sorted_power.cumsum(dim=0)
    n90 = int((cum < 0.90).sum()) + 1
    freqs = [int(k) + 1 for k in order[:6]]
    return top6, n90, freqs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print(f"{'cell':20s} {'seed':>4s} {'top6 power':>10s} {'n90':>4s}  top-6 freqs")
    all_cells = [(c, SEEDS) for c in CELLS] + [(c, SEEDS_500K) for c in CELLS_500K]
    for (label, patterns), seeds in all_cells:
        top6s, n90s = [], []
        for seed in seeds:
            name = patterns[0].format(s=seed)
            path = artifact_path(f"grok_lens/{name}/final.pt")
            ckpt = torch.load(path, weights_only=True, map_location="cpu")
            cfg = GrokModelConfig(
                **{
                    k: v
                    for k, v in ckpt["model_config"].items()
                    if k in GrokModelConfig.model_fields
                }
            )
            embed = ckpt["model_state_dict"]["embed.weight"][: cfg.p]
            power = fourier_power(embed, cfg.p)
            top6, n90, freqs = summarize(power)
            top6s.append(top6)
            n90s.append(n90)
            print(f"{label:20s} {seed:>4d} {top6:>10.3f} {n90:>4d}  {freqs}")
        mean_top6 = sum(top6s) / len(top6s)
        mean_n90 = sum(n90s) / len(n90s)
        print(f"{label:20s} {'mean':>4s} {mean_top6:>10.3f} {mean_n90:>4.1f}")
        print()


if __name__ == "__main__":
    main()

"""Per-frequency knockout: is the sparse Fourier circuit causally brittle?

For each checkpoint, remove one frequency at a time from the number-token
embeddings (project out that frequency's cos/sin components) and measure
the test-accuracy drop. If sparse circuits are causally brittle, knocking
out one of a baseline's ~5 key frequencies should crater accuracy, while
an aux model's ~18-frequency distributed circuit should degrade
gracefully. This discriminates "sparse circuits are brittle" from
"sparsity and slingshot instability are co-symptoms of optimizer
dynamics" (the question the Muon arm opened).

Usage:
    uv run python -m grok_lens.analyze_knockout
"""

import argparse
import math
import re
from pathlib import Path

import torch

from common.checkpoint import artifact_path
from grok_lens.analyze_fourier import CELLS, SEEDS, fourier_power
from grok_lens.config import GrokModelConfig
from grok_lens.data import train_test_split
from grok_lens.model import GrokTransformer

TOP_K = 6


def remove_frequency(embed: torch.Tensor, p: int, freq: int) -> torch.Tensor:
    """Project the frequency-``freq`` cos/sin components out of embed [p, dim]."""
    positions = torch.arange(p, dtype=torch.float32, device=embed.device)
    angle = 2 * math.pi * freq * positions / p
    out = embed.clone()
    for basis in (torch.cos(angle), torch.sin(angle)):
        basis = basis / basis.norm()
        out = out - torch.outer(basis, basis @ out)
    return out


@torch.no_grad()
def test_acc(
    model: GrokTransformer, tokens: torch.Tensor, targets: torch.Tensor
) -> float:
    lens = model(tokens)
    return float((lens[-1].argmax(dim=-1) == targets).float().mean())


@torch.no_grad()
def knockout_run(name: str, cache_dir: Path, device: torch.device) -> str:
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
    model.to(device).eval()

    seed_match = re.search(r"-s(\d+)", name)
    assert seed_match is not None
    _, _, tokens, targets = train_test_split(cfg, 0.3, int(seed_match.group(1)))
    tokens, targets = tokens.to(device), targets.to(device)

    number_embed = model.embed.weight.detach()[: cfg.p].clone()
    base_acc = test_acc(model, tokens, targets)
    power = fourier_power(number_embed.cpu(), cfg.p)
    top_freqs = [int(k) + 1 for k in power.sort(descending=True).indices[:TOP_K]]

    drops: list[float] = []
    all_removed = number_embed.clone()
    for freq in top_freqs:
        model.embed.weight.data[: cfg.p] = remove_frequency(number_embed, cfg.p, freq)
        drops.append(base_acc - test_acc(model, tokens, targets))
        all_removed = remove_frequency(all_removed, cfg.p, freq)
    model.embed.weight.data[: cfg.p] = all_removed
    all_acc = test_acc(model, tokens, targets)
    model.embed.weight.data[: cfg.p] = number_embed

    return (
        f"base {base_acc:.3f}  max1 drop {max(drops):.3f}  "
        f"mean top{TOP_K} drop {sum(drops) / len(drops):.3f}  "
        f"all-top{TOP_K} acc {all_acc:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for label, patterns in CELLS:
        for seed in SEEDS:
            name = patterns[0].format(s=seed)
            try:
                summary = knockout_run(name, Path("data/hf_cache"), device)
            except Exception as e:  # report and continue per run
                summary = f"SKIPPED ({e})"
            print(f"{label:20s} s{seed}: {summary}")


if __name__ == "__main__":
    main()

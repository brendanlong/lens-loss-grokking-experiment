"""How much the weights themselves move over a lag, from saved checkpoints.

``analyze_levels`` covers the embedding spectrum and MLP neurons. This covers
the rest of the network, on the same "1 = no more alike than unrelated" scale:

  - raw weights: for each group of matrices (embedding, attention, MLP,
    unembedding), ||W_t - W_t'||² / (||W_t||² + ||W_t'||²) summed over the
    group. 0 = identical, 1 = what two independent matrices give in
    expectation. Not invariant to re-expressing the residual stream in a
    rotated basis: a rotated residual stream with the same mechanism would
    still score high here. That is the honest answer to "are the same weights
    doing the work", but not to "is the same mechanism doing the work".
  - attention heads: each head's frequency profile — the 2-D (a, b) Fourier
    power of its output at the answer position, summed over output
    dimensions, which no change of residual basis affects. Head identity is
    fixed by the architecture, so a head can be compared with its own past:
    moved = (1 - own cosine) / (1 - cosine between different heads at the
    same time).
  - unembedding: the per-frequency power of the number tokens' unembedding
    rows, chance-corrected like the embedding in ``analyze_levels``.

Usage:
    uv run python -m grok_lens.analyze_weights CHECKPOINT_DIR --out weights.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

from grok_lens.analyze_fourier import fourier_power
from grok_lens.analyze_levels import embedding_moved, lag_pairs
from grok_lens.config import GrokModelConfig
from grok_lens.data import modular_addition_data
from grok_lens.model import Block, GrokTransformer
from grok_lens.trace import pair_frequency_power

GROUPS = {
    "embedding": re.compile(r"^embed\.weight$"),
    "attention": re.compile(r"^blocks\.\d+\.attn\..*weight$"),
    "mlp": re.compile(r"^blocks\.\d+\.mlp\.\d+\.weight$"),
    "unembedding": re.compile(r"^unembed\.weight$"),
}
WEIGHT_LEVELS = (*GROUPS, "heads", "unembed_spectrum")
MIN_CHANCE_GAP = 0.05  # below this, heads are too alike for a chance level


def load(path: Path) -> tuple[GrokTransformer, GrokModelConfig, int]:
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
    return model, cfg, int(ckpt["step"])


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[N, S, d] -> [N, H, S, d / H]."""
    return x.unflatten(-1, (n_heads, -1)).transpose(1, 2)


@torch.no_grad()
def head_outputs(model: GrokTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Each head's write to the residual at the answer position [L, H, N, d].

    Summed over heads (plus the out-projection bias) this is the attention
    sublayer's output at that position.
    """
    inputs: list[torch.Tensor] = []
    handles = [
        cast("Block", block).attn.register_forward_pre_hook(
            lambda _m, args: inputs.append(args[0].detach())
        )
        for block in model.blocks
    ]
    try:
        model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    outs = []
    for block, h in zip(model.blocks, inputs, strict=True):
        attn = cast("Block", block).attn
        n_heads, dim = attn.num_heads, attn.embed_dim
        d_head = dim // n_heads
        q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)

        scores = (
            split_heads(q, n_heads)[:, :, -1:]
            @ split_heads(k, n_heads).transpose(-1, -2)
            / d_head**0.5
        )
        z = (scores.softmax(-1) @ split_heads(v, n_heads))[
            :, :, 0
        ]  # last position attends all
        w_o = attn.out_proj.weight.unflatten(1, (n_heads, d_head))  # [d, H, dh]
        outs.append(torch.einsum("nhe,dhe->hnd", z, w_o))
    return torch.stack(outs)


def weight_moved(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], pattern: re.Pattern[str]
) -> float:
    keys = [k for k in a if pattern.match(k)]
    diff = sum(float((a[k] - b[k]).pow(2).sum()) for k in keys)
    scale = sum(float(a[k].pow(2).sum() + b[k].pow(2).sum()) for k in keys)
    return diff / scale


def heads_moved(profiles: torch.Tensor, i: int, j: int) -> float | None:
    """``profiles`` [T, heads, K]; own-head change over chance, between t=i, j."""
    own = F.cosine_similarity(profiles[i], profiles[j], dim=-1).mean()
    now = F.normalize(profiles[i], dim=-1)
    sims = now @ now.T
    n = len(now)
    chance = (sims.sum() - sims.diagonal().sum()) / (n * (n - 1))
    if 1 - chance < MIN_CHANCE_GAP:
        return None
    return float((1 - own) / (1 - chance))


def checkpoint_paths(directory: Path, since: int) -> list[Path]:
    def step(path: Path) -> int:
        found = re.findall(r"step_(\d+)", path.name)
        return int(found[0]) if found else 10**9  # final.pt is last

    return [p for p in sorted(directory.glob("*.pt"), key=step) if step(p) >= since]


def summarize(directory: Path, window_start: int, lags: list[int]) -> dict:
    states, head_profiles, unembed_power, steps = [], [], [], []
    for path in checkpoint_paths(directory, window_start):
        model, cfg, step = load(path)
        tokens, _ = modular_addition_data(cfg.p, cfg.task)
        heads = head_outputs(model, tokens)  # [L, H, N, d]
        power = pair_frequency_power(heads, cfg.p).sum(-2)  # [L, H, K]
        head_profiles.append(power.flatten(0, 1))
        unembed_power.append(fourier_power(model.unembed.weight[: cfg.p], cfg.p))
        states.append({k: v.float() for k, v in model.state_dict().items()})
        steps.append(step)
    profiles = torch.stack(head_profiles)
    unembed = torch.stack(unembed_power)
    cadence = steps[1] - steps[0]

    curve = []
    for lag_steps in lags:
        lag = lag_steps // cadence
        if lag < 1 or lag >= len(steps):
            continue
        pairs = list(zip(*lag_pairs(len(steps), lag), strict=True))
        point: dict[str, float | None] = {"lag": lag_steps, "n_pairs": len(pairs)}
        for name, pattern in GROUPS.items():
            point[name] = sum(
                weight_moved(states[i], states[j], pattern) for i, j in pairs
            ) / len(pairs)
        head_values = [
            v
            for i, j in pairs
            if (v := heads_moved(profiles, int(i), int(j))) is not None
        ]
        point["heads"] = sum(head_values) / len(head_values) if head_values else None
        point["unembed_spectrum"] = embedding_moved(
            unembed, lag_pairs(len(steps), lag)
        )[0]
        curve.append(point)
    return {"steps": [steps[0], steps[-1]], "cadence": cadence, "curve": curve}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=30_000)
    parser.add_argument(
        "--lags", type=int, nargs="+", default=[1000, 2000, 5000, 10_000, 15_000]
    )
    args = parser.parse_args()
    summary = summarize(args.checkpoints, args.window_start, args.lags)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

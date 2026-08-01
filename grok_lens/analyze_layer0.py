"""Layer-0 relocation probe, generalized across seeds/λ (was n=1).

For every 2-layer checkpoint: layer-0 lens accuracy on the held-out
split, layer-0 vs final argmax agreement, top-1/top-2 logit margin at
each layer, and cosine between the layer-0 and final residuals at the
answer position. Also checks the "one grok event" claim from wandb
histories: first eval where the layer-0 lens crosses 0.95 vs where final
test accuracy does.

Usage:
    uv run python -m grok_lens.analyze_layer0
"""

import argparse
import re
from pathlib import Path

import torch
import wandb

from common.checkpoint import artifact_path
from grok_lens.config import GrokModelConfig
from grok_lens.data import train_test_split
from grok_lens.model import GrokTransformer

L2_CELLS: list[tuple[str, str]] = [
    ("L2 baseline", "p113-L2-lam0.0-uniform-frac0.3-s{s}"),
    ("L2 lam0.01", "p113-L2-lam0.01-uniform-frac0.3-s{s}"),
    ("L2 lam0.1", "p113-L2-lam0.1-uniform-frac0.3-s{s}"),
    ("L2 lam0.3", "p113-L2-lam0.3-uniform-frac0.3-s{s}"),
    ("L2 lam1.0", "p113-L2-lam1.0-uniform-frac0.3-s{s}"),
    ("L2 lam3.0", "p113-L2-lam3.0-uniform-frac0.3-s{s}"),
    ("L2 base muon", "p113-L2-lam0.0-uniform-frac0.3-s{s}-muon-50k"),
    ("L2 lam0.3 muon", "p113-L2-lam0.3-uniform-frac0.3-s{s}-muon-50k"),
    ("L2 base tmuon", "p113-L2-lam0.0-uniform-frac0.3-s{s}-torchmuon-50k"),
    ("L2 lam0.3 tmuon", "p113-L2-lam0.3-uniform-frac0.3-s{s}-torchmuon-50k"),
    ("L2 base no-LN", "p113-L2-lam0.0-uniform-frac0.3-s{s}-noln-50k"),
]
SEEDS = [42, 43, 44]


@torch.no_grad()
def probe_run(name: str, cache_dir: Path) -> str:
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
    seed_match = re.search(r"-s(\d+)", name)
    assert seed_match is not None
    _, _, tokens, targets = train_test_split(cfg, 0.3, int(seed_match.group(1)))

    x = model.embed(tokens) + model.pos_embed
    resids: list[torch.Tensor] = []
    for block in model.blocks:
        x = block(x, model.causal_mask)
        resids.append(x[:, -1])
    lens = model.unembed(model.ln_f(torch.stack(resids)))

    def margin(logits: torch.Tensor) -> float:
        top2 = logits.topk(2, dim=-1).values
        return float((top2[:, 0] - top2[:, 1]).mean())

    acc0 = float((lens[0].argmax(-1) == targets).float().mean())
    agree = float((lens[0].argmax(-1) == lens[-1].argmax(-1)).float().mean())
    cos = float(
        torch.nn.functional.cosine_similarity(resids[0], resids[-1], dim=-1).mean()
    )
    return (
        f"L0 lens acc {acc0:.3f}  agree {agree:.3f}  "
        f"margin L0 {margin(lens[0]):.2f} -> final {margin(lens[-1]):.2f}  "
        f"cos(r0,rF) {cos:.3f}"
    )


def simultaneity(api: wandb.Api, name: str) -> str:
    runs = list(api.runs("brendanlong-com/grok-lens", filters={"display_name": name}))
    if not runs:
        return "no wandb run"
    hist = sorted(
        runs[0].history(keys=["lens_test/acc_layer_0", "test/acc"], pandas=False),
        key=lambda h: h["_step"],
    )

    def first_cross(key: str) -> int | None:
        for h in hist:
            if h.get(key) is not None and h[key] >= 0.95:
                return h["_step"]
        return None

    lens_cross = first_cross("lens_test/acc_layer_0")
    final_cross = first_cross("test/acc")
    return f"L0 cross {lens_cross!s} vs final {final_cross!s}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    api = wandb.Api()
    for label, pattern in L2_CELLS:
        for seed in SEEDS:
            name = pattern.format(s=seed)
            try:
                summary = probe_run(name, Path("data/hf_cache"))
            except Exception as e:  # report and continue per run
                summary = f"SKIPPED ({e})"
            sim = (
                ""
                if "baseline" in label or "base" in label
                else ("  |  " + simultaneity(api, name))
            )
            print(f"{label:16s} s{seed}: {summary}{sim}")


if __name__ == "__main__":
    main()

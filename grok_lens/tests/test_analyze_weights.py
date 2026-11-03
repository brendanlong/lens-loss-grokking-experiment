import re

import pytest
import torch

from grok_lens.analyze_weights import (
    checkpoint_paths,
    head_outputs,
    heads_moved,
    weight_moved,
)
from grok_lens.config import GrokModelConfig
from grok_lens.data import modular_addition_data
from grok_lens.model import Block, GrokTransformer


def test_head_outputs_sum_to_the_attention_output() -> None:
    torch.manual_seed(0)
    cfg = GrokModelConfig(p=11, dim=16, n_heads=4, n_layers=2)
    model = GrokTransformer(cfg).eval()
    tokens, _ = modular_addition_data(cfg.p, cfg.task)
    blocks = [b for b in model.blocks if isinstance(b, Block)]
    seen: list[torch.Tensor] = []
    handles = [
        block.attn.register_forward_hook(
            lambda _m, _i, out: seen.append(out[0][:, -1].detach())
        )
        for block in blocks
    ]
    heads = head_outputs(model, tokens)
    for handle in handles:
        handle.remove()
    for layer, block in enumerate(blocks):
        total = heads[layer].sum(0) + block.attn.out_proj.bias
        assert torch.allclose(total, seen[layer], atol=1e-5)


def test_weight_moved_is_zero_for_same_and_about_one_for_unrelated() -> None:
    gen = torch.Generator().manual_seed(0)
    a = {"w": torch.randn(64, 64, generator=gen)}
    b = {"w": torch.randn(64, 64, generator=gen)}
    pattern = re.compile("w")
    assert weight_moved(a, a, pattern) == 0.0
    assert weight_moved(a, b, pattern) == pytest.approx(1.0, abs=0.05)


def test_heads_moved_is_zero_for_unchanged_and_large_for_swapped_roles() -> None:
    # four heads sharing one dominant frequency, each with its own minor one
    profiles = torch.full((2, 4, 5), 0.0)
    profiles[:, :, 0] = 10.0
    for h in range(4):
        profiles[:, h, h + 1] = 1.0
    assert heads_moved(profiles, 0, 1) == pytest.approx((0.0, 0.0), abs=1e-6)
    profiles[1] = profiles[1].roll(1, 0)
    corrected, raw = heads_moved(profiles, 0, 1)
    assert raw < 0.05  # raw profiles barely change: the shared frequency dominates
    assert corrected > 1.0  # but each head now has another head's distinct role


def test_checkpoint_paths_skip_final_duplicate(tmp_path) -> None:
    for name in ("step_1000.pt", "step_2000.pt", "step_3000.pt", "final.pt"):
        (tmp_path / name).touch()
    names = [p.name for p in checkpoint_paths(tmp_path, since=2000)]
    assert names == ["step_2000.pt", "step_3000.pt"]

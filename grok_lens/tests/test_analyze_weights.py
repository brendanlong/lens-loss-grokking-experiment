import re

import pytest
import torch

from grok_lens.analyze_weights import head_outputs, heads_moved, weight_moved
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


def test_heads_moved_is_zero_for_unchanged_and_one_for_swapped_roles() -> None:
    profiles = torch.eye(4)[None].repeat(2, 1, 1)
    assert heads_moved(profiles, 0, 1) == pytest.approx(0.0)
    profiles[1] = profiles[1].roll(1, 0)
    # swapping to another head's role is beyond chance: different heads here
    # are orthogonal, so chance is cosine 0 and a swap scores 1
    assert heads_moved(profiles, 0, 1) == pytest.approx(1.0)

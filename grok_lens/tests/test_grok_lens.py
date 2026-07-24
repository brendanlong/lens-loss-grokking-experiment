"""Fast CPU tests: data correctness, split invariants, model shapes, loss."""

import torch
import torch.nn.functional as F

from grok_lens.config import GrokLensTrainingConfig, GrokModelConfig
from grok_lens.data import modular_addition_data, train_test_split
from grok_lens.model import (
    GrokTransformer,
    grok_lens_loss,
    intermediate_layer_weights,
)

P = 11  # small modulus for fast tests


def small_config(n_layers: int = 2) -> GrokModelConfig:
    return GrokModelConfig(p=P, dim=16, n_heads=2, n_layers=n_layers)


class TestData:
    def test_exhaustive_and_correct(self) -> None:
        tokens, targets = modular_addition_data(P)
        assert tokens.shape == (P * P, 3)
        assert (tokens[:, 2] == P).all()  # "=" token everywhere
        assert (targets == (tokens[:, 0] + tokens[:, 1]) % P).all()
        # every (a, b) pair appears exactly once
        pair_ids = tokens[:, 0] * P + tokens[:, 1]
        assert len(pair_ids.unique()) == P * P

    def test_split_partitions_all_pairs(self) -> None:
        cfg = small_config()
        train_tok, train_tgt, test_tok, test_tgt = train_test_split(cfg, 0.3, seed=0)
        assert len(train_tok) == round(0.3 * P * P)
        assert len(train_tok) + len(test_tok) == P * P
        train_ids = set((train_tok[:, 0] * P + train_tok[:, 1]).tolist())
        test_ids = set((test_tok[:, 0] * P + test_tok[:, 1]).tolist())
        assert train_ids.isdisjoint(test_ids)
        assert len(train_ids | test_ids) == P * P
        assert (train_tgt == (train_tok[:, 0] + train_tok[:, 1]) % P).all()
        assert (test_tgt == (test_tok[:, 0] + test_tok[:, 1]) % P).all()

    def test_split_deterministic_given_seed(self) -> None:
        cfg = small_config()
        a = train_test_split(cfg, 0.3, seed=7)
        b = train_test_split(cfg, 0.3, seed=7)
        c = train_test_split(cfg, 0.3, seed=8)
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        assert not torch.equal(a[0], c[0])


class TestModel:
    def test_forward_shapes(self) -> None:
        cfg = small_config(n_layers=3)
        model = GrokTransformer(cfg)
        tokens, _ = modular_addition_data(P)
        lens = model(tokens[:5])
        assert lens.shape == (3, 5, cfg.vocab_size)

    def test_last_lens_layer_is_output_logits(self) -> None:
        """Lens at the final layer must be exactly the model's output head."""
        cfg = small_config()
        model = GrokTransformer(cfg)
        model.eval()
        tokens, _ = modular_addition_data(P)
        with torch.no_grad():
            lens = model(tokens[:8])
            # recompute the output head by hand from the final residual
            x = model.embed(tokens[:8]) + model.pos_embed
            for block in model.blocks:
                x = block(x, model.causal_mask)
            expected = model.unembed(model.ln_f(x[:, -1]))
        assert torch.allclose(lens[-1], expected, atol=1e-6)


class TestLoss:
    def test_layer_weights(self) -> None:
        assert intermediate_layer_weights(1, "uniform").numel() == 0
        for weighting in ("uniform", "linear"):
            w = intermediate_layer_weights(4, weighting)
            assert w.shape == (3,)
            assert torch.allclose(w.sum(), torch.tensor(1.0))
        w_lin = intermediate_layer_weights(4, "linear")
        assert (w_lin[1:] > w_lin[:-1]).all()  # later layers weighted more
        assert torch.allclose(
            intermediate_layer_weights(4, "uniform"), torch.full((3,), 1 / 3)
        )

    def test_lambda_zero_is_plain_ce(self) -> None:
        torch.manual_seed(0)
        lens = torch.randn(3, 10, P + 1)
        targets = torch.randint(0, P, (10,))
        w = intermediate_layer_weights(3, "uniform")
        total, final_ce = grok_lens_loss(lens, targets, w, 0.0)
        assert torch.equal(total, final_ce)
        assert torch.allclose(final_ce, F.cross_entropy(lens[-1], targets))

    def test_aux_true_matches_manual(self) -> None:
        torch.manual_seed(0)
        lens = torch.randn(3, 10, P + 1)
        targets = torch.randint(0, P, (10,))
        w = intermediate_layer_weights(3, "linear")
        lam = 0.5
        total, final_ce = grok_lens_loss(lens, targets, w, lam)
        expected_aux = sum(
            w[layer] * F.cross_entropy(lens[layer], targets) for layer in range(2)
        )
        assert torch.allclose(total, final_ce + lam * expected_aux)

    def test_single_layer_aux_is_noop(self) -> None:
        torch.manual_seed(0)
        lens = torch.randn(1, 10, P + 1)
        targets = torch.randint(0, P, (10,))
        w = intermediate_layer_weights(1, "uniform")
        total, final_ce = grok_lens_loss(lens, targets, w, 1.0)
        assert torch.equal(total, final_ce)

    def test_aux_gradients_reach_early_layers_only_when_on(self) -> None:
        """With lambda > 0, block-0 params get gradient from the aux path even
        when the final CE is detached from them; sanity-check the wiring by
        comparing gradients with lambda on vs off."""
        cfg = small_config(n_layers=2)
        tokens, targets = modular_addition_data(P)
        w = intermediate_layer_weights(2, "uniform")

        def block0_grad_norm(lam: float) -> float:
            torch.manual_seed(0)
            model = GrokTransformer(cfg)
            lens = model(tokens[:16])
            total, _ = grok_lens_loss(lens, targets[:16], w, lam)
            total.backward()
            grads = [
                p.grad.norm()
                for p in model.blocks[0].parameters()
                if p.grad is not None
            ]
            return float(torch.stack(grads).sum())

        # identical init/seed, so any difference comes from the aux term
        assert block0_grad_norm(1.0) != block0_grad_norm(0.0)


class TestMuon:
    def test_newton_schulz_approximately_orthogonalizes(self) -> None:
        from grok_lens.muon import zeropower_via_newtonschulz5

        torch.manual_seed(0)
        for shape in [(16, 32), (32, 16), (16, 16)]:
            grad = torch.randn(*shape)
            out = zeropower_via_newtonschulz5(grad)
            assert out.shape == grad.shape
            # Quintic NS leaves singular values near 1 (roughly [0.7, 1.3])
            svals = torch.linalg.svdvals(out)
            assert (svals > 0.5).all() and (svals < 1.5).all()

    def test_param_split_partitions_model(self) -> None:
        from grok_lens.muon import split_muon_params

        model = GrokTransformer(small_config())
        muon_params, adamw_params = split_muon_params(model)
        all_ids = {id(p) for p in model.parameters()}
        split_ids = [id(p) for p in muon_params + adamw_params]
        assert len(split_ids) == len(set(split_ids)) == len(all_ids)
        assert all(p.ndim == 2 for p in muon_params)
        # embed/unembed/pos/norms must stay on the AdamW side
        adamw_ids = {id(p) for p in adamw_params}
        assert id(model.embed.weight) in adamw_ids
        assert id(model.unembed.weight) in adamw_ids
        assert id(model.pos_embed) in adamw_ids

    def test_muon_step_reduces_loss(self) -> None:
        from grok_lens.muon import Muon, split_muon_params

        torch.manual_seed(0)
        cfg = small_config()
        model = GrokTransformer(cfg)
        tokens, targets = modular_addition_data(P)
        w = intermediate_layer_weights(cfg.n_layers, "uniform")
        muon_params, adamw_params = split_muon_params(model)
        optimizers = [
            Muon(muon_params, lr=0.02, weight_decay=0.05),
            torch.optim.AdamW(adamw_params, lr=1e-3, weight_decay=1.0),
        ]

        def loss_value() -> torch.Tensor:
            lens = model(tokens)
            total, _ = grok_lens_loss(lens, targets, w, 0.0)
            return total

        start = loss_value().item()
        for _ in range(30):
            loss = loss_value()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:
                opt.step()
        assert loss_value().item() < start


class TestConfig:
    def test_train_frac_bounds(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            GrokLensTrainingConfig(train_frac=0.0)
        with pytest.raises(ValueError):
            GrokLensTrainingConfig(aux_lambda=-0.1)

    def test_model_config_validation(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            GrokModelConfig(dim=15, n_heads=2)
        with pytest.raises(ValueError):
            GrokModelConfig(n_layers=0)


def test_aux_targets_override_changes_loss() -> None:
    """Shuffled aux targets must change the aux term but not the final CE."""
    torch.manual_seed(0)
    lens = torch.randn(3, 10, P + 1)
    targets = torch.randint(0, P, (10,))
    shuffled = targets[torch.randperm(10)]
    w = intermediate_layer_weights(3, "uniform")
    total_true, ce_true = grok_lens_loss(lens, targets, w, 0.5)
    total_shuf, ce_shuf = grok_lens_loss(lens, targets, w, 0.5, aux_targets=shuffled)
    assert torch.allclose(ce_true, ce_shuf)
    assert not torch.allclose(total_true, total_shuf)


def test_subtraction_task() -> None:
    """Subtraction targets are (a-b) mod p and differ from addition's."""
    tokens_add, targets_add = modular_addition_data(P, "add")
    tokens_sub, targets_sub = modular_addition_data(P, "sub")
    assert torch.equal(tokens_add, tokens_sub)  # same input pairs
    assert (targets_sub == (tokens_sub[:, 0] - tokens_sub[:, 1]) % P).all()
    assert not torch.equal(targets_add, targets_sub)
    # split respects the task via the model config
    cfg = GrokModelConfig(p=P, dim=16, n_heads=2, n_layers=2, task="sub")
    _, train_tgt, _test_tok, _ = train_test_split(cfg, 0.3, seed=0)
    assert len(train_tgt) == round(0.3 * P * P)

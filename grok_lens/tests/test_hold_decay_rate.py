import pytest
import torch

from grok_lens.config import GrokLensTrainingConfig, GrokModelConfig
from grok_lens.data import train_test_split
from grok_lens.model import GrokTransformer
from grok_lens.train import train_grok_model


@pytest.mark.parametrize("hold", [False, True])
def test_decay_rate_follows_lr_unless_held(tmp_path, monkeypatch, hold) -> None:
    rates: list[float] = []
    step = torch.optim.AdamW.step

    def recording_step(self, *args, **kwargs):
        group = self.param_groups[0]
        rates.append(group["lr"] * group["weight_decay"])
        return step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", recording_step)
    model_config = GrokModelConfig(p=7, dim=8, n_heads=2, n_layers=1)
    config = GrokLensTrainingConfig(
        total_steps=40,
        warmup_steps=2,
        lr_schedule="wsd",
        lr_decay_start=10,
        lr_decay_end=30,
        hold_decay_rate=hold,
        compile=False,
        use_wandb=False,
        log_every_steps=0,
        eval_every_steps=0,
        checkpoint_dir=str(tmp_path),
    )
    data = train_test_split(model_config, config.train_frac, config.seed)
    train_grok_model(
        GrokTransformer(model_config), model_config, config, data,
        torch.device("cpu"), "test",
    )  # fmt: skip
    base = config.lr * config.weight_decay
    after_warmup = rates[config.warmup_steps :]
    if hold:
        assert after_warmup == pytest.approx([base] * len(after_warmup))
    else:
        assert after_warmup[-1] == pytest.approx(base * config.lr_decay_floor)

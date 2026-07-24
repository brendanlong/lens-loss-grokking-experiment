"""Train a small transformer to grok modular addition, optionally with a
logit-lens auxiliary loss supervising every intermediate layer.

Full-batch training on a fixed train split (the canonical grokking regime —
see the note in config.py on why this deviates from the repo's streaming
default). Defaults reproduce the clean grokking baseline (aux_lambda = 0).

Usage:
    # Smoke test (small p, few steps, CPU-friendly)
    uv run python -m grok_lens.train \
        --p 23 --total-steps 500 --no-wandb

    # Baseline grokking run
    uv run python -m grok_lens.train

    # Aux-loss run
    uv run python -m grok_lens.train \
        --aux-lambda 0.3 --aux-weighting uniform
"""

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import torch

from common.checkpoint import save_model_checkpoint
from common.schedule import should_log_and_eval
from common.wandb_utils import finish_wandb, init_wandb, log_metrics
from grok_lens.config import GrokLensTrainingConfig, GrokModelConfig
from grok_lens.data import train_test_split
from grok_lens.model import (
    GrokTransformer,
    grok_lens_loss,
    intermediate_layer_weights,
)
from grok_lens.muon import Muon, split_muon_params

EXPERIMENT = "grok_lens"


def build_lr_lambda(config: GrokLensTrainingConfig) -> Callable[[int], float]:
    """Linear warmup then constant or cosine decay, as a LambdaLR multiplier."""

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        if config.lr_schedule == "constant":
            return 1.0
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


@torch.no_grad()
def evaluate(
    model: GrokTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    layer_weights: torch.Tensor,
    config: GrokLensTrainingConfig,
    prefix: str,
) -> dict[str, float]:
    """Full-split loss/accuracy plus per-layer logit-lens accuracy."""
    lens_logits = model(tokens)
    total, final_ce = grok_lens_loss(
        lens_logits, targets, layer_weights, config.aux_lambda
    )
    per_layer_acc = (lens_logits.argmax(dim=-1) == targets).float().mean(dim=1)
    metrics = {
        f"{prefix}/loss": final_ce.item(),
        f"{prefix}/aux_loss": (total - final_ce).item(),  # λ-scaled aux term
        f"{prefix}/total_loss": total.item(),
        f"{prefix}/acc": per_layer_acc[-1].item(),
    }
    for layer, acc in enumerate(per_layer_acc.tolist()):
        metrics[f"lens_{prefix}/acc_layer_{layer}"] = acc
    return metrics


def train_grok_model(
    model: GrokTransformer,
    model_config: GrokModelConfig,
    config: GrokLensTrainingConfig,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    run_name: str,
) -> dict[str, float]:
    """Full-batch training loop; returns the final eval metrics."""
    train_tokens, train_targets, test_tokens, test_targets = (
        t.to(device) for t in data
    )
    layer_weights = intermediate_layer_weights(
        model_config.n_layers, config.aux_weighting
    ).to(device)
    aux_train_targets = train_targets
    if config.aux_shuffled_targets:
        perm_gen = torch.Generator().manual_seed(config.seed + 1000)
        aux_train_targets = train_targets[
            torch.randperm(len(train_targets), generator=perm_gen).to(device)
        ]

    def make_adamw(params: list[torch.Tensor]) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "muon":
        muon_params, adamw_params = split_muon_params(model)
        # Order matters below: the last optimizer's scheduler is the one
        # whose lr gets logged (the AdamW group, matching the baseline arm).
        optimizers: list[torch.optim.Optimizer] = [
            Muon(
                muon_params,
                lr=config.muon_lr,
                momentum=config.muon_momentum,
                weight_decay=config.muon_weight_decay,
            ),
            make_adamw(adamw_params),
        ]
    else:
        optimizers = [make_adamw(list(model.parameters()))]
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=build_lr_lambda(config))
        for opt in optimizers
    ]

    train_step = model
    if config.compile and device.type == "cuda":
        train_step = torch.compile(model)  # type: ignore[assignment]

    memorize_step: int | None = None
    grok_step: int | None = None
    final_metrics: dict[str, float] = {}

    for step in range(1, config.total_steps + 1):
        lens_logits = train_step(train_tokens)
        total, final_ce = grok_lens_loss(
            lens_logits,
            train_targets,
            layer_weights,
            config.aux_lambda,
            aux_targets=aux_train_targets,
        )
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total.backward()
        if math.isfinite(config.max_grad_norm):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

        do_log, do_eval = should_log_and_eval(
            step,
            log_every_steps=config.log_every_steps,
            eval_every_steps=config.eval_every_steps,
        )
        if not do_log:
            continue

        # Sync point (intentional): only reached on the log/eval cadence.
        metrics = {
            "train/loss": final_ce.item(),
            "train/aux_loss": (total - final_ce).item(),
            "train/total_loss": total.item(),
            "lr": schedulers[-1].get_last_lr()[0],
        }
        if do_eval:
            model.eval()
            if config.log_fourier:
                from grok_lens.analyze_fourier import fourier_power

                power = fourier_power(
                    model.embed.weight.detach()[: model_config.p].cpu(),
                    model_config.p,
                )
                for freq_idx, p_val in enumerate(power.tolist()):
                    metrics[f"freq_power/k_{freq_idx + 1}"] = p_val
            metrics.update(
                evaluate(
                    model, train_tokens, train_targets, layer_weights, config, "train"
                )
            )
            metrics.update(
                evaluate(
                    model, test_tokens, test_targets, layer_weights, config, "test"
                )
            )
            model.train()
            weight_sq = torch.zeros((), device=device)
            for param in model.parameters():
                weight_sq = weight_sq + param.detach().pow(2).sum()
            metrics["weight_norm"] = weight_sq.sqrt().item()

            if memorize_step is None and metrics["train/acc"] >= config.grok_threshold:
                memorize_step = step
                print(f"  step {step}: memorized (train acc >= threshold)")
            if grok_step is None and metrics["test/acc"] >= config.grok_threshold:
                grok_step = step
                print(f"  step {step}: GROKKED (test acc >= {config.grok_threshold})")
            final_metrics = metrics

        log_metrics(metrics, step, enabled=config.use_wandb)
        if do_eval:
            print(
                f"step {step:6d} | train loss {metrics['train/loss']:.4f} "
                f"acc {metrics['train/acc']:.3f} | test loss "
                f"{metrics['test/loss']:.4f} acc {metrics['test/acc']:.3f}"
            )

    if config.use_wandb:
        import wandb

        if wandb.run is not None:
            wandb.run.summary["memorize_step"] = memorize_step
            wandb.run.summary["grok_step"] = grok_step

    print(f"\nmemorize_step: {memorize_step}, grok_step: {grok_step}")

    save_model_checkpoint(
        model,
        config.total_steps,
        model_config.model_dump(),
        Path(config.checkpoint_dir),
        filename="final.pt",
    )
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grokking modular addition with a logit-lens auxiliary loss"
    )
    # Model
    parser.add_argument("--p", type=int, default=113, help="Modulus")
    parser.add_argument(
        "--task",
        default="add",
        choices=["add", "sub"],
        help="a+b or a-b (mod p); sub is the second-task generality check",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    # Data
    parser.add_argument("--train-frac", type=float, default=0.3)
    # Aux loss
    parser.add_argument(
        "--aux-lambda",
        type=float,
        default=0.0,
        help="Strength of the logit-lens auxiliary loss (0 = baseline)",
    )
    parser.add_argument(
        "--aux-weighting",
        default="uniform",
        choices=["uniform", "linear"],
        help="Layer weighting: uniform, or linear (CALM-style later-weighted)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Checkpoint path or hf:<relpath> (public HF dataset) to load "
            "initial weights from "
            "(continuation experiments). Model config comes from the "
            "checkpoint and overrides --p/--dim/--n-heads/--n-layers. "
            "Use the SAME --seed as the source run to preserve the "
            "train/test split. Optimizer state starts fresh (checkpoints "
            "don't store it) — pair every continuation with a same-source "
            "control continuation so the restart transient is matched."
        ),
    )
    # Optimization
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "muon"],
        help="muon = hybrid Muon (block weights) + AdamW (embeds/norms)",
    )
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=0.05,
        help="Decoupled; default matches AdamW per-step shrinkage (lr*wd=1e-3)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=30_000)
    parser.add_argument(
        "--lr-schedule", default="constant", choices=["cosine", "constant"]
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    # Cadence / logging
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="data/grok_lens/checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="grok-lens")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--aux-shuffled-targets",
        action="store_true",
        help=(
            "Specificity control: intermediate-layer CE against a fixed "
            "random permutation of the train labels (not answer-shaped)."
        ),
    )
    parser.add_argument(
        "--log-fourier",
        action="store_true",
        help="Log per-frequency embedding Fourier power at every eval",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resume_state: dict[str, torch.Tensor] | None = None
    if args.resume_from:
        from common.checkpoint import resolve_checkpoint

        ckpt_path = resolve_checkpoint(
            args.resume_from, Path("data/grok_lens/s3_cache")
        )
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        model_config = GrokModelConfig(
            **{
                k: v
                for k, v in ckpt["model_config"].items()
                if k in GrokModelConfig.model_fields
            }
        )
        resume_state = ckpt["model_state_dict"]
        print(f"Resuming weights from {args.resume_from}")
    else:
        model_config = GrokModelConfig(
            p=args.p,
            task=args.task,
            dim=args.dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )
    config = GrokLensTrainingConfig(
        train_frac=args.train_frac,
        aux_lambda=args.aux_lambda,
        aux_weighting=args.aux_weighting,
        aux_shuffled_targets=args.aux_shuffled_targets,
        optimizer=args.optimizer,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.total_steps,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        compile=not args.no_compile,
        log_every_steps=args.log_every_steps,
        eval_every_steps=args.eval_every_steps,
        log_fourier=args.log_fourier,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )

    if config.aux_lambda > 0.0 and model_config.n_layers < 2:
        parser.error("--aux-lambda > 0 needs --n-layers >= 2 (no intermediate layers)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(config.seed)

    data = train_test_split(model_config, config.train_frac, config.seed)
    n_train, n_test = len(data[1]), len(data[3])
    print(
        f"p={model_config.p}: {n_train} train / {n_test} test pairs "
        f"(train_frac={config.train_frac})"
    )
    print(f"Aux loss: lambda={config.aux_lambda}, weighting={config.aux_weighting}")

    model = GrokTransformer(model_config)
    if resume_state is not None:
        model.load_state_dict(resume_state)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_config.n_layers}L/{model_config.dim}d ({n_params:,} params)")

    optimizer_suffix = "" if config.optimizer == "adamw" else f"-{config.optimizer}"
    task_prefix = "" if model_config.task == "add" else model_config.task
    run_name = config.wandb_run_name or (
        f"p{model_config.p}{task_prefix}-L{model_config.n_layers}"
        f"-lam{config.aux_lambda}-{config.aux_weighting}-frac{config.train_frac}"
        f"-s{config.seed}{optimizer_suffix}"
    )
    init_wandb(
        enabled=config.use_wandb,
        project=config.wandb_project,
        run_name=run_name,
        config={
            "model": model_config.model_dump(),
            "training": config.model_dump(),
            "resume_from": args.resume_from,
        },
    )

    final_metrics = train_grok_model(
        model, model_config, config, data, device, run_name
    )
    print("\n=== Final ===")
    for key in ("train/acc", "test/acc"):
        if key in final_metrics:
            print(f"  {key}: {final_metrics[key]:.3f}")
    finish_wandb(enabled=config.use_wandb)


if __name__ == "__main__":
    main()

"""Shared training utilities for LEGO experiments.

Provides evaluate, load_model, optimizer/scheduler creation, checkpoint
saving, and a unified training loop used by train.py, sweep scripts,
and comparison scripts.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import wandb
from torch.utils.data import DataLoader

from common.checkpoint import (
    save_model_checkpoint as _save_checkpoint,
)
from common.schedule import should_log_and_eval
from lego.config import ModelConfig
from lego.data import make_eval_batch
from lego.generator import ChainExample
from lego.losses import (
    compute_answer_accuracy,
    compute_answer_only_loss,
    compute_lens_aux_loss,
    compute_lens_aux_loss_all_positions,
)
from lego.model import StandardTransformer, print_model_summary


@dataclass
class TrainingResult:
    """Result from train_lego_model."""

    converged_step: int | None
    final_eval: dict[str, float]
    checkpoint_path: Path | None
    total_steps: int


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_examples_per_k: dict[int, list[ChainExample]],
    k_max: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate per-chain-length accuracy."""
    model.eval()
    results: dict[str, float] = {}
    total_correct = torch.tensor(0.0, device=device)
    total_count = 0
    for k in sorted(test_examples_per_k):
        examples = test_examples_per_k[k]
        correct = torch.tensor(0.0, device=device)
        count = 0
        for i in range(0, len(examples), batch_size):
            chunk = examples[i : i + batch_size]
            batch = make_eval_batch(chunk, k_max)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            answer_positions = batch["answer_position"].to(
                device,
                non_blocking=True,
            )
            logits = model(input_ids)
            acc = compute_answer_accuracy(logits, input_ids, answer_positions)
            correct += acc * len(chunk)
            count += len(chunk)
        k_acc = (correct / max(1, count)).item()
        results[f"test_acc/k_{k}"] = k_acc
        total_correct += correct
        total_count += count
    results["test_acc/mean"] = (total_correct / max(1, total_count)).item()
    return results


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[StandardTransformer, ModelConfig]:
    """Load model and config from checkpoint."""
    ckpt = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=device,
    )
    model_config = ModelConfig(**ckpt["model_config"])
    model = StandardTransformer(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_config


def save_model_checkpoint(
    model: torch.nn.Module,
    step: int,
    model_config: ModelConfig,
    checkpoint_dir: Path,
) -> Path:
    """Save checkpoint, stripping torch.compile prefix from keys."""
    return _save_checkpoint(model, step, model_config.model_dump(), checkpoint_dir)


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    total_steps: int,
    weight_decay: float = 0.0,
    lr_schedule: Literal["cosine", "constant"] = "cosine",
    warmup_steps: int | None = None,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.SequentialLR]:
    """Create AdamW optimizer with warmup + cosine/constant decay."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    effective_warmup = (
        warmup_steps if warmup_steps is not None else min(200, total_steps // 2)
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / max(1, effective_warmup),
        total_iters=effective_warmup,
    )
    decay_scheduler: torch.optim.lr_scheduler.LRScheduler
    if lr_schedule == "cosine":
        decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - effective_warmup),
        )
    else:
        decay_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps,
        )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[effective_warmup],
    )
    return optimizer, scheduler


def train_lego_model(
    model: StandardTransformer,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    test_examples_per_k: dict[int, list[ChainExample]],
    model_config: ModelConfig,
    device: torch.device,
    *,
    total_steps: int,
    k_min: int = 1,
    k_max: int = 6,
    n_epochs: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    lr_schedule: Literal["cosine", "constant"] = "cosine",
    use_compile: bool = True,
    lens_aux: bool = False,
    lens_aux_weight: float = 0.3,
    lens_aux_weighting: Literal["uniform", "linear"] = "uniform",
    lens_aux_mode: Literal["answer", "all-positions"] = "answer",
    early_stop_patience: int | None = 2000,
    log_every_steps: int = 100,
    eval_every_steps: int = 500,
    eval_batch_size: int = 512,
    run_name: str = "",
    checkpoint_dir: Path | None = None,
    save_every_steps: int | None = None,
    use_wandb: bool = True,
    wandb_project: str = "lego-reasoning",
    wandb_config: dict[str, object] | None = None,
) -> TrainingResult:
    """Unified training loop for LEGO models.

    Trains the model with periodic evaluation, optional early stopping,
    wandb logging, and checkpoint saving. Convergence is detected when
    all chain lengths reach >= 99.9% accuracy.
    """
    print_model_summary(model)

    # Residual-based losses require forward_with_residuals which is
    # incompatible with torch.compile
    needs_residuals = lens_aux
    if needs_residuals and use_compile:
        print(
            "  Note: disabling torch.compile (incompatible with residual-based losses)"
        )
        use_compile = False

    if use_compile and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=lr,
        total_steps=total_steps,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
    )

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config=wandb_config or {},
            reinit=True,
        )

    global_step = 0
    running_loss = torch.tensor(0.0, device=device)
    running_lens_loss = torch.tensor(0.0, device=device)
    running_acc = torch.tensor(0.0, device=device)
    running_count = 0
    t0 = time.time()
    converged_step: int | None = None
    last_saved_step = -1
    last_checkpoint_path: Path | None = None
    stopped_early = False

    for _epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            answer_positions = batch["answer_position"].to(
                device,
                non_blocking=True,
            )

            # Forward pass: use residuals if any aux loss needs them
            if needs_residuals:
                logits, residuals = model.forward_with_residuals(input_ids)
            else:
                logits = model(input_ids)
                residuals = None

            base_loss = compute_answer_only_loss(
                logits,
                input_ids,
                answer_positions,
            )
            aux_loss = torch.tensor(0.0, device=device)

            step_lens_loss = torch.tensor(0.0, device=device)
            if lens_aux:
                assert residuals is not None
                if lens_aux_mode == "all-positions":
                    step_lens_loss = compute_lens_aux_loss_all_positions(
                        residuals,
                        input_ids,
                        model.final_norm,
                        model.tok_emb.weight,
                        weighting=lens_aux_weighting,
                    )
                else:
                    step_lens_loss = compute_lens_aux_loss(
                        residuals,
                        input_ids,
                        answer_positions,
                        model.final_norm,
                        model.tok_emb.weight,
                        weighting=lens_aux_weighting,
                    )
                aux_loss = aux_loss + lens_aux_weight * step_lens_loss

            loss = base_loss + aux_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += base_loss.detach()
            if lens_aux:
                running_lens_loss += step_lens_loss.detach()
            running_acc += compute_answer_accuracy(
                logits,
                input_ids,
                answer_positions,
            )
            running_count += 1
            global_step += 1

            # Early stopping
            if (
                early_stop_patience is not None
                and converged_step is not None
                and global_step - converged_step >= early_stop_patience
            ):
                print(
                    f"  [{run_name}] Early stop at step {global_step} "
                    f"({early_stop_patience} steps after convergence)",
                )
                stopped_early = True
                break

            # Logging also fires on eval steps so evals are never skipped
            # when eval_every_steps isn't a multiple of log_every_steps.
            do_log, do_eval = should_log_and_eval(
                global_step,
                log_every_steps=log_every_steps,
                eval_every_steps=eval_every_steps,
            )
            if do_log:
                avg_loss = (running_loss / running_count).item()
                avg_acc = (running_acc / running_count).item()
                elapsed = time.time() - t0
                steps_per_sec = running_count / elapsed

                log_dict: dict[str, float] = {
                    "train/loss": avg_loss,
                    "train/answer_acc": avg_acc,
                    "perf/steps_per_sec": steps_per_sec,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if lens_aux:
                    log_dict["train/lens_aux_loss"] = (
                        running_lens_loss / running_count
                    ).item()

                if do_eval:
                    eval_results = evaluate(
                        model,
                        test_examples_per_k,
                        k_max,
                        eval_batch_size,
                        device,
                    )
                    log_dict.update(eval_results)

                    model.train()

                    mean_acc = eval_results["test_acc/mean"]
                    k_accs = [
                        eval_results.get(f"test_acc/k_{k}", 0.0)
                        for k in range(k_min, k_max + 1)
                    ]
                    k_str = " ".join(f"{a:.0%}" for a in k_accs)

                    print(
                        f"  [{run_name}] Step {global_step:6d}/{total_steps}"
                        f" | loss={avg_loss:.4f} | train={avg_acc:.1%}"
                        f" | test={mean_acc:.1%} | k=[{k_str}]"
                        f" | {steps_per_sec:.1f} steps/s",
                        flush=True,
                    )

                    # Convergence: all k at 100%
                    if converged_step is None and all(a >= 0.999 for a in k_accs):
                        converged_step = global_step
                        print(
                            f"  [{run_name}] *** CONVERGED at step {global_step} ***",
                        )
                        if (
                            checkpoint_dir is not None
                            and last_saved_step != global_step
                        ):
                            last_checkpoint_path = save_model_checkpoint(
                                model,
                                global_step,
                                model_config,
                                checkpoint_dir,
                            )
                            last_saved_step = global_step
                else:
                    print(
                        f"  [{run_name}] Step {global_step:6d}/{total_steps}"
                        f" | loss={avg_loss:.4f} | train={avg_acc:.1%}"
                        f" | {steps_per_sec:.1f} steps/s",
                        flush=True,
                    )

                if use_wandb:
                    wandb.log(log_dict, step=global_step)

                running_loss = torch.tensor(0.0, device=device)
                running_acc = torch.tensor(0.0, device=device)
                running_count = 0
                t0 = time.time()

            # Periodic checkpoint
            if (
                save_every_steps is not None
                and checkpoint_dir is not None
                and global_step % save_every_steps == 0
                and last_saved_step != global_step
            ):
                last_checkpoint_path = save_model_checkpoint(
                    model,
                    global_step,
                    model_config,
                    checkpoint_dir,
                )
                last_saved_step = global_step

        if stopped_early:
            break

    # Final checkpoint
    if checkpoint_dir is not None and last_saved_step != global_step:
        last_checkpoint_path = save_model_checkpoint(
            model,
            global_step,
            model_config,
            checkpoint_dir,
        )
        if converged_step is None:
            print(f"  [{run_name}] WARNING: Did not converge!")

    # Final eval
    final_eval = evaluate(
        model,
        test_examples_per_k,
        k_max,
        eval_batch_size,
        device,
    )

    if use_wandb:
        wandb.log(final_eval, step=global_step)
        wandb.finish()

    return TrainingResult(
        converged_step=converged_step,
        final_eval=final_eval,
        checkpoint_path=last_checkpoint_path,
        total_steps=global_step,
    )

"""Training loop for LEGO group composition task (S3).

Trains decoder-only transformers on S3 group composition chains.
Measures per-chain-length accuracy to track how many sequential
composition steps the model has learned.

The dataset is the FULL enumeration of chains with k in [k_min, k_max]
(335,922 chains for k in [0, 6]), split into disjoint train/test sets with
a seeded shuffle — evaluation is always on held-out chains the model never
trained on. The default --n-epochs 40 over the ~269k-example train split
(test_frac=0.2) gives ~10.7M examples seen, matching the old streaming
default of 10M generated examples.

Usage:
    # Smoke test (tiny model)
    uv run python -m lego.train \
        --k-max 3 --n-layers 4 --dim 64 --n-heads 2 \
        --n-epochs 1 --batch-size 64 --no-wandb

    # Baseline (6 hops)
    uv run python -m lego.train --k-max 6

    # With the lens auxiliary loss (deep supervision)
    uv run python -m lego.train --k-max 6 --lens-aux --lens-aux-weight 0.3

    # Full-sequence lens aux (next-token CE at every position)
    uv run python -m lego.train --k-max 6 --lens-aux --lens-aux-mode all-positions
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lego.config import LegoTrainingConfig, lego_model_config
from lego.data import ChainDataset, collate_s3, make_k_uniform_sampler
from lego.generator import S3, enumerate_split, group_by_k
from lego.model import create_model
from lego.tokenizer import Tokenizer
from lego.training import train_lego_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LEGO group composition models",
    )

    # Data
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help=(
            "Fraction of the full chain enumeration held out for test "
            "(stratified per chain length k)"
        ),
    )

    # Model
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)

    # Training
    parser.add_argument(
        "--lens-aux",
        action="store_true",
        help=(
            "grok_lens-style deep supervision: per-layer logit-lens CE "
            "against the FINAL answer at the <predict> position."
        ),
    )
    parser.add_argument(
        "--lens-aux-weight",
        type=float,
        default=0.3,
        help="Weight (lambda) for the lens auxiliary loss.",
    )
    parser.add_argument(
        "--lens-aux-weighting",
        default="uniform",
        choices=["uniform", "linear"],
        help="Layer weighting: uniform, or linear (CALM-style later-weighted).",
    )
    parser.add_argument(
        "--lens-aux-mode",
        default="answer",
        choices=["answer", "all-positions"],
        help=(
            "answer: lens CE at the <predict> position only (default); "
            "all-positions: next-token lens CE at every non-pad position."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=40,
        help=(
            "Epochs over the train split (~269k examples at defaults; "
            "40 epochs ≈ 10.7M examples seen)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        default="cosine",
        choices=["cosine", "constant"],
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # Eval / logging / checkpoint (step-based)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=1000,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=100,
        help="Steps between train metric logging",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=5000,
        help="Steps between checkpoint saves",
    )
    parser.add_argument("--checkpoint-dir", default="data/lego/checkpoints")

    # Wandb
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lego-reasoning")
    parser.add_argument("--wandb-run-name", default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    # Group and tokenizer (S3 only)
    group = S3
    tokenizer = Tokenizer(group)
    print(f"Group: {group.name} ({group.order} elements, vocab={tokenizer.vocab_size})")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)

    # Config
    config = LegoTrainingConfig(
        k_min=args.k_min,
        k_max=args.k_max,
        test_frac=args.test_frac,
        lens_aux=args.lens_aux,
        lens_aux_weight=args.lens_aux_weight,
        lens_aux_weighting=args.lens_aux_weighting,
        lens_aux_mode=args.lens_aux_mode,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        lr_schedule=args.lr_schedule,
        eval_every_steps=args.eval_every_steps,
        log_every_steps=args.log_every_steps,
        save_every_steps=args.save_every_steps,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )

    model_config = lego_model_config(
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vocab_size=tokenizer.vocab_size,
    )

    run_name = args.wandb_run_name or (
        f"{group.name}_std"
        f"_{args.dim}d_{args.n_heads}h_{args.n_layers}L"
        f"_k{args.k_min}-{args.k_max}"
    )

    # Data: enumerate ALL chains and split into disjoint train/test sets.
    # The test split is stratified per chain length k, so per-k eval always
    # has held-out examples for every k.
    train_examples, test_examples = enumerate_split(
        args.k_min,
        args.k_max,
        test_frac=args.test_frac,
        seed=args.seed,
        group=group,
    )
    test_examples_per_k = group_by_k(test_examples)
    n_total = len(train_examples) + len(test_examples)
    print(
        f"Enumerated {n_total} chains, k in [{args.k_min}, {args.k_max}]: "
        f"{len(train_examples)} train / {len(test_examples)} test "
        f"(test_frac={args.test_frac}, seed={args.seed})"
    )
    per_k_str = " ".join(
        f"k{k}:{len(v)}" for k, v in sorted(test_examples_per_k.items())
    )
    print(f"Held-out test examples per k: {per_k_str}")

    train_dataset = ChainDataset(train_examples, args.k_max, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=make_k_uniform_sampler(train_dataset, seed=args.seed),
        collate_fn=collate_s3,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = len(train_examples) // config.batch_size
    total_steps = steps_per_epoch * config.n_epochs

    # Model
    model = create_model(model_config)
    model = model.to(device)

    if config.lens_aux:
        print(
            f"Lens aux loss: enabled (mode={config.lens_aux_mode}, "
            f"weight={config.lens_aux_weight}, "
            f"weighting={config.lens_aux_weighting})"
        )
    print(f"Training for {config.n_epochs} epoch(s) ({total_steps} steps)")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(
        f"Eval every {config.eval_every_steps} steps, "
        f"log every {config.log_every_steps} steps"
    )

    # Train
    result = train_lego_model(
        model,
        train_loader,
        test_examples_per_k,
        model_config,
        device,
        total_steps=total_steps,
        k_min=args.k_min,
        k_max=args.k_max,
        n_epochs=config.n_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_schedule=config.lr_schedule,
        use_compile=not args.no_compile,
        lens_aux=config.lens_aux,
        lens_aux_weight=config.lens_aux_weight,
        lens_aux_weighting=config.lens_aux_weighting,
        lens_aux_mode=config.lens_aux_mode,
        early_stop_patience=None,
        log_every_steps=config.log_every_steps,
        eval_every_steps=config.eval_every_steps,
        eval_batch_size=config.batch_size,
        run_name=run_name,
        checkpoint_dir=Path(config.checkpoint_dir),
        save_every_steps=config.save_every_steps,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_config={
            "model": model_config.model_dump(),
            "training": config.model_dump(),
            "group": group.name,
        },
        tokenizer=tokenizer,
    )

    # Final summary
    print("\n=== Final Evaluation (held-out test split) ===")
    for k in range(args.k_min, args.k_max + 1):
        key = f"test_acc/k_{k}"
        print(f"  k={k:2d}: {result.final_eval.get(key, 0.0):.1%}")
    print(f"  Mean: {result.final_eval['test_acc/mean']:.1%}")
    if result.converged_step is not None:
        print(f"  Converged at step {result.converged_step}")


if __name__ == "__main__":
    main()

"""Training loop for LEGO group composition task.

Trains decoder-only transformers on group composition chains.
Measures per-chain-length accuracy to track how many sequential
composition steps the model has learned.

Usage:
    # Smoke test (tiny model, S3)
    uv run python -m lego.train \
        --k-max 3 --n-layers 4 --dim 64 --n-heads 2 \
        --generate-n 100000 --batch-size 64 --no-wandb

    # Primary experiment (streaming, 6 hops, S3)
    uv run python -m lego.train \
        --k-max 6 --generate-n 5000000 --batch-size 512

    # S5 experiment (larger model needed)
    uv run python -m lego.train \
        --group S5 --dim 256 --n-heads 8 \
        --generate-n 10000000 --batch-size 256
"""

import argparse
from pathlib import Path
from typing import get_args

import torch
from torch.utils.data import DataLoader, Dataset

from lego.config import LegoTrainingConfig, lego_model_config
from lego.data import (
    S3FixedDataset,
    S3StreamingDataset,
    collate_s3,
)
from lego.generator import (
    GroupName,
    generate_fixed_dataset,
    generate_mixed_dataset,
    get_group,
)
from lego.model import (
    create_model,
)
from lego.tokenizer import Tokenizer
from lego.training import train_lego_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LEGO group composition models",
    )

    # Group
    parser.add_argument(
        "--group",
        type=str,
        default="S3",
        choices=list(get_args(GroupName)),
        help="Group to use: S3 (6 elts), S4 (24), A5 (60), S5 (120)",
    )

    # Data
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Training set size (fixed dataset mode)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Test examples per chain length k",
    )
    parser.add_argument(
        "--generate-n",
        type=int,
        default=None,
        help="Streaming mode: generate N examples, 1 epoch",
    )
    parser.add_argument(
        "--k-power",
        type=float,
        default=0.0,
        help="Power for k-weighting: weight(k)=k^power. "
        "0=uniform (default), 2=quadratic (k=6 gets 36x k=1)",
    )

    # Model
    parser.add_argument("--weight-shared", action="store_true")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument(
        "--pos-encoding",
        default="learned",
        choices=["rope", "pope", "learned"],
    )
    parser.add_argument("--init-std", type=float, default=None)

    # Training
    parser.add_argument(
        "--loss-mode",
        default="answer-only",
        choices=["answer-only", "full-sequence"],
        help=(
            "answer-only: loss at answer position only. "
            "full-sequence: autoregressive loss on all non-pad tokens."
        ),
    )
    parser.add_argument(
        "--staircase-loss",
        action="store_true",
        help=(
            "Add auxiliary loss at <op> positions at specific layers, "
            "targeting intermediate trajectory values. Forces sequential "
            "algorithm where layer j computes trajectory[j]."
        ),
    )
    parser.add_argument(
        "--staircase-start-layer",
        type=int,
        default=0,
        help=(
            "First layer for staircase loss targets. 0 for 6L models, 1 for 7L models."
        ),
    )
    parser.add_argument(
        "--staircase-weight",
        type=float,
        default=1.0,
        help="Weight for the staircase auxiliary loss.",
    )
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
        "--align-loss",
        action="store_true",
        help=(
            "Add soft alignment auxiliary loss encouraging residual stream "
            "to stay near token embedding manifold at all layers."
        ),
    )
    parser.add_argument(
        "--align-weight",
        type=float,
        default=0.1,
        help="Weight for the alignment auxiliary loss.",
    )
    parser.add_argument(
        "--align-temp",
        type=float,
        default=1.0,
        help="Temperature for alignment loss cosine similarity.",
    )
    parser.add_argument(
        "--repel-loss",
        action="store_true",
        help=(
            "Add embedding repulsion loss pushing element token "
            "embeddings apart (cosine similarity)."
        ),
    )
    parser.add_argument(
        "--repel-weight",
        type=float,
        default=0.1,
        help="Weight for the embedding repulsion loss.",
    )
    parser.add_argument(
        "--repel-margin",
        type=float,
        default=0.0,
        help="Cosine similarity margin for repulsion loss.",
    )
    parser.add_argument(
        "--repel-all-tokens",
        action="store_true",
        help=(
            "Apply repulsion to all tokens (including PAD and special), "
            "not just element tokens. Prevents PAD from becoming an "
            "alignment attractor."
        ),
    )
    parser.add_argument("--n-epochs", type=int, default=200)
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

    # Group and tokenizer
    group_name: GroupName = args.group
    group = get_group(group_name)
    tokenizer = Tokenizer(group)
    print(f"Group: {group_name} ({group.order} elements, vocab={tokenizer.vocab_size})")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)

    # Config
    n_train = args.n_train or 100_000
    config = LegoTrainingConfig(
        k_min=args.k_min,
        k_max=args.k_max,
        n_train=n_train,
        n_test=args.n_test,
        generate_n=args.generate_n,
        loss_mode=args.loss_mode,
        staircase_loss=args.staircase_loss,
        staircase_start_layer=args.staircase_start_layer,
        staircase_weight=args.staircase_weight,
        lens_aux=args.lens_aux,
        lens_aux_weight=args.lens_aux_weight,
        lens_aux_weighting=args.lens_aux_weighting,
        align_loss=args.align_loss,
        align_weight=args.align_weight,
        align_temp=args.align_temp,
        repel_loss=args.repel_loss,
        repel_weight=args.repel_weight,
        repel_margin=args.repel_margin,
        repel_all_tokens=args.repel_all_tokens,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=1 if args.generate_n else args.n_epochs,
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
        weight_shared=args.weight_shared,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        pos_encoding=args.pos_encoding,
        init_std=args.init_std,
        vocab_size=tokenizer.vocab_size,
    )

    # Run name — resolve and preflight the checkpoint namespace *before* the
    # expensive data generation / model build, so a reused name fails in seconds.
    run_name = args.wandb_run_name or (
        f"{group_name}_{'ws' if args.weight_shared else 'std'}"
        f"_{args.dim}d_{args.n_heads}h_{args.n_layers}L"
        f"_k{args.k_min}-{args.k_max}"
    )

    # Test set — generate per-k for clean evaluation
    print(
        f"Generating test set: {args.n_test} examples per k, "
        f"k in [{args.k_min}, {args.k_max}]"
    )
    test_examples_per_k = {
        k: generate_fixed_dataset(k, args.n_test, seed=args.seed + 1 + k, group=group)
        for k in range(args.k_min, args.k_max + 1)
    }

    # Training data
    if args.generate_n:
        print(
            f"Streaming mode: k in [{args.k_min}, {args.k_max}], "
            f"generate_n={args.generate_n}"
        )
        train_dataset: Dataset[dict[str, torch.Tensor]] = S3StreamingDataset(
            args.k_min,
            args.k_max,
            args.generate_n,
            seed=args.seed + 100,
            k_power=args.k_power,
            group=group,
            tokenizer=tokenizer,
        )
        steps_per_epoch = args.generate_n // config.batch_size
    else:
        print(f"Fixed dataset: k in [{args.k_min}, {args.k_max}], n_train={n_train}")
        train_examples = generate_mixed_dataset(
            args.k_min,
            args.k_max,
            n_train,
            seed=args.seed,
            group=group,
        )
        train_dataset = S3FixedDataset(train_examples, args.k_max, tokenizer)
        steps_per_epoch = n_train // config.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=not args.generate_n,
        collate_fn=collate_s3,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    total_steps = steps_per_epoch * config.n_epochs

    # Model
    model = create_model(model_config)
    model = model.to(device)

    # Validation
    if config.staircase_loss and config.align_loss:
        parser.error(
            "--staircase-loss and --align-loss are mutually exclusive. "
            "Staircase is supervised (targets specific trajectory values); "
            "alignment is unsupervised (pushes toward embedding manifold)."
        )

    print(f"\nLoss mode: {config.loss_mode}")
    if config.staircase_loss:
        print(
            f"Staircase loss: enabled (start_layer={config.staircase_start_layer}, "
            f"weight={config.staircase_weight})"
        )
    if config.lens_aux:
        print(
            f"Lens aux loss: enabled (weight={config.lens_aux_weight}, "
            f"weighting={config.lens_aux_weighting})"
        )
    if config.align_loss:
        print(
            f"Alignment loss: enabled (weight={config.align_weight}, "
            f"temp={config.align_temp})"
        )
    if config.repel_loss:
        scope = "all tokens" if config.repel_all_tokens else "elements only"
        print(
            f"Repulsion loss: enabled (weight={config.repel_weight}, "
            f"margin={config.repel_margin}, {scope})"
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
        full_sequence_loss=(config.loss_mode == "full-sequence"),
        staircase_loss=config.staircase_loss,
        staircase_start_layer=config.staircase_start_layer,
        staircase_weight=config.staircase_weight,
        lens_aux=config.lens_aux,
        lens_aux_weight=config.lens_aux_weight,
        lens_aux_weighting=config.lens_aux_weighting,
        align_loss=config.align_loss,
        align_weight=config.align_weight,
        align_temp=config.align_temp,
        repel_loss=config.repel_loss,
        repel_weight=config.repel_weight,
        repel_margin=config.repel_margin,
        repel_all_tokens=config.repel_all_tokens,
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
            "group": group_name,
        },
        tokenizer=tokenizer,
    )

    # Final summary
    print("\n=== Final Evaluation ===")
    for k in range(args.k_min, args.k_max + 1):
        key = f"test_acc/k_{k}"
        print(f"  k={k:2d}: {result.final_eval.get(key, 0.0):.1%}")
    print(f"  Mean: {result.final_eval['test_acc/mean']:.1%}")
    if result.converged_step is not None:
        print(f"  Converged at step {result.converged_step}")


if __name__ == "__main__":
    main()

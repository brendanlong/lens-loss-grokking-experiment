"""Configuration for model architecture and LEGO training."""

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

from lego.tokenizer import VOCAB_SIZE


class ModelConfig(BaseModel):
    """Model architecture configuration.

    No defaults — always construct via lego_model_config() or deserialize
    from a checkpoint dict. The architecture itself is fixed: pre-norm
    LayerNorm blocks, learned positional embeddings, GELU MLP, tied
    embeddings, PyTorch-default init. Checkpoints from older code may
    contain extra architecture-selection keys; pydantic ignores them.
    """

    dim: int
    n_heads: int
    n_layers: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    norm_eps: float = 1e-5
    dropout: float

    @model_validator(mode="after")
    def _validate_architecture(self) -> "ModelConfig":
        if self.dim % self.n_heads != 0:
            msg = f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def lego_model_config(
    dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 8,
    dropout: float = 0.0,
) -> ModelConfig:
    """Create ModelConfig for LEGO experiments (S3 vocabulary, 10 tokens)."""
    return ModelConfig(
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        intermediate_dim=dim * 4,
        vocab_size=VOCAB_SIZE,
        max_seq_len=128,  # plenty of room for any k_max
        dropout=dropout,
    )


class LegoTrainingConfig(BaseModel):
    """Training hyperparameters for LEGO S3 experiments."""

    # Data -- chain length range. The dataset is the FULL enumeration of
    # chains with k in [k_min, k_max] (335,922 for S3, k in [0, 6]),
    # split into disjoint train/test sets.
    k_min: int = 0
    k_max: int = 6
    test_frac: float = 0.2  # held-out fraction per chain length k
    # Grokking regime: train on a small memorizable subset of the train
    # split (per-k waterfill subsample; None = full split), for a fixed
    # number of optimizer steps (None = n_epochs governs).
    train_subset: int | None = None
    total_steps: int | None = None

    # Training: answer-only cross-entropy at the <predict> position,
    # optionally with grok_lens-style deep supervision (Phase 5).
    # lens_aux_mode "answer" supervises intermediate layers at the
    # <predict> position only; "all-positions" applies next-token
    # logit-lens CE at every non-pad position.
    lens_aux: bool = False
    lens_aux_weight: float = 0.3
    lens_aux_weighting: Literal["uniform", "linear"] = "uniform"
    lens_aux_mode: Literal["answer", "all-positions"] = "answer"
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.0
    # 40 epochs over the ~269k-example train split (test_frac=0.2 of
    # 335,922) ≈ 10.7M examples seen, matching the old streaming default
    # of --generate-n 10000000.
    n_epochs: int = 40
    lr_schedule: Literal["cosine", "constant"] = "cosine"

    # Step-based logging and evaluation
    eval_every_steps: int = 1000
    log_every_steps: int = 100
    save_every_steps: int = 5000

    # Checkpointing
    checkpoint_dir: str = "data/lego/checkpoints"

    # Wandb
    wandb_project: str = "lego-reasoning"
    wandb_run_name: str | None = None
    use_wandb: bool = True

    # Reproducibility
    seed: int = 42

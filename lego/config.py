"""Configuration for model architecture and LEGO training."""

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator


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
    vocab_size: int | None = None,
) -> ModelConfig:
    """Create ModelConfig for LEGO experiments.

    Args:
        vocab_size: Override vocabulary size. If None, uses S3 default (10).
    """
    if vocab_size is None:
        from lego.tokenizer import VOCAB_SIZE

        vocab_size = VOCAB_SIZE

    return ModelConfig(
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        intermediate_dim=dim * 4,
        vocab_size=vocab_size,
        max_seq_len=128,  # plenty of room for any k_max
        dropout=dropout,
    )


class LegoTrainingConfig(BaseModel):
    """Training hyperparameters for LEGO S3 experiments."""

    # Data -- chain length range
    k_min: int = 0
    k_max: int = 6
    n_test: int = 1_000  # test examples per chain length k

    # Streaming mode
    generate_n: int | None = None

    # Fixed dataset mode (ignored when generate_n is set)
    n_train: int = 100_000

    # Training: answer-only cross-entropy at the <predict> position,
    # optionally with grok_lens-style deep supervision (Phase 5)
    lens_aux: bool = False
    lens_aux_weight: float = 0.3
    lens_aux_weighting: Literal["uniform", "linear"] = "uniform"
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.0
    n_epochs: int = 200
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

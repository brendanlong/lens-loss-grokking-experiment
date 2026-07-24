"""Configuration for model architecture and LEGO training."""

from typing import Literal, TypedDict

from pydantic import BaseModel, computed_field, model_validator


class HeadlineModelSpec(TypedDict):
    """Specification for a headline LEGO model (Phase 9 results)."""

    label: str
    short: str
    weight_shared: bool
    dim: int
    n_heads: int
    n_layers: int


HEADLINE_MODELS: list[HeadlineModelSpec] = [
    {
        "label": "Large Standard (128d/4h/8L)",
        "short": "std_128d",
        "weight_shared": False,
        "dim": 128,
        "n_heads": 4,
        "n_layers": 8,
    },
    {
        "label": "Small Standard (48d/3h/8L)",
        "short": "std_48d",
        "weight_shared": False,
        "dim": 48,
        "n_heads": 3,
        "n_layers": 8,
    },
    {
        "label": "Large WS (256d/8h/8L)",
        "short": "ws_256d",
        "weight_shared": True,
        "dim": 256,
        "n_heads": 8,
        "n_layers": 8,
    },
    {
        "label": "Small WS (96d/6h/8L)",
        "short": "ws_96d",
        "weight_shared": True,
        "dim": 96,
        "n_heads": 6,
        "n_layers": 8,
    },
]


class ModelConfig(BaseModel):
    """Model architecture configuration.

    No defaults — always construct via lego_model_config() or deserialize
    from a checkpoint dict.
    """

    dim: int
    n_heads: int
    n_layers: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 100000.0
    norm_eps: float = 1e-5
    weight_shared: bool
    use_iteration_embed: bool
    use_input_injection: bool
    dropout: float
    pos_encoding: Literal["rope", "pope", "learned"]
    norm_type: Literal["rmsnorm", "layernorm"]
    activation: Literal["swiglu", "gelu"]
    # None = PyTorch default (Kaiming). 0.02 is standard for LLMs but too
    # small for bAbI-scale models (attention scores ~0.05 std, too flat).
    init_std: float | None

    @model_validator(mode="after")
    def _validate_architecture(self) -> "ModelConfig":
        if self.dim % self.n_heads != 0:
            msg = f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        if (
            self.pos_encoding in ("rope", "pope")
            and (self.dim // self.n_heads) % 2 != 0
        ):
            msg = (
                f"head_dim ({self.dim // self.n_heads}) must be even "
                f"for {self.pos_encoding}"
            )
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def lego_model_config(
    weight_shared: bool = False,
    dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 8,
    pos_encoding: Literal["rope", "pope", "learned"] = "learned",
    norm_type: Literal["rmsnorm", "layernorm"] = "layernorm",
    activation: Literal["swiglu", "gelu"] = "gelu",
    init_std: float | None = None,
    dropout: float = 0.0,
    use_iteration_embed: bool = True,
    use_input_injection: bool = False,
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
        weight_shared=weight_shared,
        use_iteration_embed=use_iteration_embed and weight_shared,
        use_input_injection=use_input_injection and weight_shared,
        dropout=dropout,
        pos_encoding=pos_encoding,
        norm_type=norm_type,
        activation=activation,
        init_std=init_std,
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

    # Training
    loss_mode: Literal["answer-only", "full-sequence"] = "answer-only"
    staircase_loss: bool = False
    staircase_start_layer: int = 0
    staircase_weight: float = 1.0
    # grok_lens-style deep supervision at the answer position (Phase 5)
    lens_aux: bool = False
    lens_aux_weight: float = 0.3
    lens_aux_weighting: Literal["uniform", "linear"] = "uniform"
    # Alignment auxiliary losses
    align_loss: bool = False
    align_weight: float = 0.1
    align_temp: float = 1.0
    repel_loss: bool = False
    repel_weight: float = 0.1
    repel_margin: float = 0.0
    repel_all_tokens: bool = False
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

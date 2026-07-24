"""Configuration for grok_lens: grokking modular addition under a
logit-lens auxiliary loss.

Defaults reproduce the canonical Nanda et al. (2023) grokking setup
(p = 113, train_frac = 0.3, full-batch AdamW with weight_decay = 1.0,
constant LR 1e-3), with the auxiliary loss OFF (aux_lambda = 0.0) so the
default run is the clean grokking baseline.
"""

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

from common.config import BaseTrainingConfig

# Sequence is always [a, b, =] with the answer read at the "=" position.
SEQ_LEN = 3


class GrokModelConfig(BaseModel):
    """Architecture for the small grokking transformer.

    Standard pre-LayerNorm decoder-only transformer with learned positional
    embeddings, ReLU MLP, and an untied unembedding — the Nanda et al.
    modular-addition architecture, except we keep LayerNorm (they removed it
    for circuit analysis; we need `ln_f` for the logit lens to be the model's
    own readout).
    """

    p: int = 113  # modulus; vocab is the p residues plus one "=" token
    # Task: a+b or a-b (mod p). Subtraction is the second-task generality
    # check — non-commutative, groks in the same regime (Power et al.), and
    # keeps the circuit basis on Z_p so all Fourier analyses port unchanged.
    # Stored on the model config so checkpoints are self-describing and the
    # analyses can reconstruct the right split.
    task: Literal["add", "sub"] = "add"
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4

    @model_validator(mode="after")
    def _validate(self) -> "GrokModelConfig":
        if self.dim % self.n_heads != 0:
            msg = f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        if self.n_layers < 1:
            msg = f"n_layers must be >= 1, got {self.n_layers}"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eq_token(self) -> int:
        return self.p


class GrokLensTrainingConfig(BaseTrainingConfig):
    """Training hyperparameters.

    Deliberate deviation from the repo's streaming-data default: grokking
    requires a small FIXED train set revisited every step — the
    memorize-then-generalize transition is the object of study, so streaming
    unique examples would remove the phenomenon. Training is full-batch (the
    whole train split every optimizer step, the canonical grokking regime),
    so the inherited ``batch_size`` field is unused.
    """

    # Data
    train_frac: float = 0.3  # fraction of the p^2 pairs used for training

    # Logit-lens auxiliary loss (the experimental knob; 0.0 = baseline)
    aux_lambda: float = 0.0
    # Weighting over the L-1 non-final layers: uniform = 1/(L-1) each,
    # linear = CALM-style later-weighted (w_l ∝ l+1). Aux targets are the
    # true answer token (per-layer cross-entropy) unless
    # aux_shuffled_targets is set.
    aux_weighting: Literal["uniform", "linear"] = "uniform"
    # Specificity control: supervise intermediate layers against a FIXED
    # random permutation of the train labels instead of the true answers —
    # matched functional form/magnitude, but not answer-shaped.
    aux_shuffled_targets: bool = False

    # Optimization (canonical grokking settings)
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 1.0
    # Optimizer-robustness arm: muon = hybrid Muon (2-D block weights) +
    # AdamW (embed/unembed/norms/biases, using lr/weight_decay above).
    optimizer: Literal["adamw", "muon"] = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    # Decoupled wd chosen so per-step shrinkage muon_lr * muon_weight_decay
    # matches the AdamW baseline's lr * weight_decay = 1e-3 — otherwise the
    # regularization pressure (load-bearing for grokking) wouldn't be
    # comparable across optimizer arms.
    muon_weight_decay: float = 0.05
    lr_schedule: Literal["cosine", "constant"] = "constant"
    warmup_steps: int = 10
    total_steps: int = 30_000
    # No gradient clipping (inf disables it): the canonical grokking setup
    # doesn't clip, and a fixed clip norm would bite harder as aux_lambda
    # grows — re-introducing the optimization-speed confound the normalized
    # layer weights are meant to remove.
    max_grad_norm: float = float("inf")
    # Full-batch training is precision-sensitive (slingshot effects), so we
    # stay in fp32 — the model is tiny, autocast buys nothing here.
    compile: bool = True

    # A run counts as memorized/grokked when train/test accuracy first
    # crosses this threshold.
    grok_threshold: float = 0.95

    # Log per-frequency embedding Fourier power at every eval (56 metrics)
    # — for tracking component-level churn through training.
    log_fourier: bool = False

    # Cadence: grokking transitions are sharp, so eval densely.
    log_every_steps: int = 100
    eval_every_steps: int = 100

    # Checkpointing / wandb
    checkpoint_dir: str = "data/grok_lens/checkpoints"
    wandb_project: str = "grok-lens"

    @model_validator(mode="after")
    def _validate(self) -> "GrokLensTrainingConfig":
        if not 0.0 < self.train_frac < 1.0:
            msg = f"train_frac must be in (0, 1), got {self.train_frac}"
            raise ValueError(msg)
        if self.aux_lambda < 0.0:
            msg = f"aux_lambda must be >= 0, got {self.aux_lambda}"
            raise ValueError(msg)
        return self

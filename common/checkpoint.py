"""Checkpoint saving and artifact resolution.

Training saves checkpoints locally. All checkpoints referenced by the
analyses are hosted publicly on the Hugging Face Hub
(https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment)
and are downloaded on demand into the local HF cache.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

HF_DATASET = "brendanlong/lens-loss-grokking-experiment"


def save_model_checkpoint(
    model: torch.nn.Module,
    step: int,
    model_config: dict[str, object],
    checkpoint_dir: Path,
    filename: str | None = None,
) -> Path:
    """Save a checkpoint, stripping any torch.compile ``_orig_mod.`` prefix.

    The payload is ``{"step", "model_state_dict", "model_config"}`` so
    loaders can reconstruct the model from its config dict. Loadable with
    ``torch.load(..., weights_only=True)``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
    }
    ckpt_path = checkpoint_dir / (filename or f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "model_config": model_config,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")
    return ckpt_path


def artifact_path(relpath: str) -> Path:
    """Resolve an artifact, preferring a local mirror over the HF dataset.

    ``relpath`` is e.g. ``grok_lens/<run_name>/final.pt`` or
    ``lego/<run_name>/step_39000.pt``. If ``data/artifacts/<relpath>``
    exists (the escape hatch for analyzing fresh local runs before they
    are uploaded), it is used directly; otherwise the file is downloaded
    from the public HF dataset and cached by huggingface_hub.
    """
    local = Path("data/artifacts") / relpath
    if local.exists():
        return local
    return Path(
        hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=relpath)
    )


def resolve_checkpoint(source: str, cache_dir: Path | None = None) -> Path:
    """Resolve a checkpoint source to a local path.

    - A local path is returned as-is.
    - ``hf:<relpath>`` downloads from the public HF dataset.

    ``cache_dir`` is accepted for signature compatibility; HF manages its
    own cache.
    """
    if source.startswith("hf:"):
        return artifact_path(source.removeprefix("hf:"))
    return Path(source)

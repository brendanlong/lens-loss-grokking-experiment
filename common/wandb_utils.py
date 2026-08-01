"""wandb run-lifecycle helpers.

Thin wrappers so training loops aren't littered with ``if use_wandb:``
branches and every experiment initializes runs the same way.
"""

from collections.abc import Mapping

import wandb


def init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    config: Mapping[str, object] | None = None,
) -> None:
    """Start a wandb run (no-op when disabled)."""
    if not enabled:
        return
    wandb.init(
        project=project,
        name=run_name,
        config=dict(config) if config is not None else None,
        reinit=True,
    )


def log_metrics(
    metrics: Mapping[str, float], step: int, *, enabled: bool = True
) -> None:
    """Log metrics to the active wandb run (no-op when disabled)."""
    if not enabled:
        return
    wandb.log(dict(metrics), step=step)


def finish_wandb(*, enabled: bool = True) -> None:
    """Finish the active wandb run (no-op when disabled)."""
    if not enabled:
        return
    wandb.finish()

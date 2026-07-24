"""Muon optimizer (Jordan et al. 2024) with decoupled weight decay.

Used in the optimizer-robustness arm: is the sparse-circuit ↔ slingshot-
instability correlation an AdamW-regime artifact? Muon orthogonalizes the
momentum matrix via Newton-Schulz iteration before each update. Standard
hybrid usage (as in "Muon is Scalable", Kimi K2): Muon for the 2-D
hidden-layer weight matrices only; AdamW for embeddings, unembedding,
positional embeddings, and all 1-D params (norms, biases).

Weight decay is decoupled (p *= 1 - lr*wd), following the Moonshot variant —
the original Muon has none, but weight decay is load-bearing for grokking.
"""

from collections.abc import Callable, Iterable

import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalize ``grad`` (≈ UV^T of its SVD).

    Quintic Newton-Schulz iteration from Jordan et al.; coefficients tuned
    for fast convergence, leaving singular values in roughly [0.7, 1.3]
    rather than exactly 1. Run in fp32 — models here are tiny.
    """
    if grad.ndim != 2:
        msg = f"Muon update requires 2-D gradients, got ndim={grad.ndim}"
        raise ValueError(msg)
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * gram @ gram
        x = a * x + poly @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters, with nesterov momentum and decoupled wd."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override] — closure is unused but kept for API parity
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            weight_decay: float = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                update = p.grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, group["ns_steps"])
                # Scale so the update RMS is comparable across matrix shapes
                # (Jordan et al.): sqrt(max(1, rows/cols)).
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-lr * scale)
        return loss


def split_muon_params(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split params into (muon_params, adamw_params).

    Muon gets the 2-D weight matrices inside transformer blocks; everything
    else (embed, pos_embed, unembed, norms, biases) goes to AdamW.
    """
    muon_params: list[torch.Tensor] = []
    adamw_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if name.startswith("blocks.") and param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params

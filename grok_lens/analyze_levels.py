"""How much of each level moves over a lag: capability, embedding, neurons.

Reads one ``grok_lens.trace`` file and measures how much each level moves
between evals t and t + lag (0 = identical):

  - capability: share of test answers whose correctness differs. A literal
    share, not chance-corrected: it measures the function, and a model right
    on 99.9% of examples at both ends has changed at most 0.2% of its
    answers however much its internals moved. That bound is the claim.
  - embedding: total-variation distance between the embedding's
    per-frequency power shares, divided by its expectation if frequency
    identity were shuffled (``embedding_raw`` keeps the undivided share of
    power sitting on different frequencies).
  - neurons: 1 - chance-corrected retention of each frequency's neuron
    population (see ``analyze_neuron_drift.retention_lift``), averaged over
    frequencies. Neurons of every block are pooled, so a role handed from one
    block's neurons to another's counts as turnover. Per-block curves are
    reported too.

Embedding and neurons therefore share a scale — 1 = no more alike than with
identities shuffled — and can be compared directly; capability is on its own
literal scale.

Two uses of the same measurement:

  - lag curves over a post-grok window (from ``window_start`` or first
    grok, whichever is later; none for runs that never grok).
  - repair events: for each collapse (test acc falls below ``DIP`` after
    first grok), the state just before it against the state at recovery,
    alongside the same measurement over equally long collapse-free spans
    starting within ``CONTROL_RADIUS`` steps of it (so an early collapse is
    not compared with late, quieter training). "Repairs itself onto the same
    circuit" predicts event ≈ control.

Usage:
    uv run python -m grok_lens.analyze_levels TRACE --out summary.json
"""

import argparse
import json
from pathlib import Path

import torch

GROK = 0.95  # first-grok / "working" threshold (matches train.py)
DIP = 0.90  # below this after first grok counts as a collapse
LEVELS = ("capability", "embedding", "neurons")
CONTROL_RADIUS = 5_000
N_SHUFFLES = 64
Pairs = tuple[torch.Tensor, torch.Tensor]


def load_trace(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=False, map_location="cpu")


def neuron_roles(
    trace: dict[str, torch.Tensor],
    member_share: float,
    min_frac_answer: float,
    blocks: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(member [T, N, K], active [T, N]) with the chosen blocks' neurons pooled.

    Active: at least ``min_frac_answer`` of the neuron's variance is a
    function of the answer, and it is not dead (variance above 1e-3 of its
    block's largest at that eval).
    """
    profile = trace["profile"].float()  # [T, L, n, K]
    frac = trace["frac_answer"].float()
    var = trace["variance"]
    if blocks is not None:
        profile, frac, var = profile[:, blocks], frac[:, blocks], var[:, blocks]
    alive = var > var.amax(-1, keepdim=True) * 1e-3
    active = (frac >= min_frac_answer) & alive
    member = (profile >= member_share) & active[..., None]
    return member.flatten(1, 2), active.flatten(1, 2)


def capability_moved(correct: torch.Tensor, pairs: Pairs) -> float:
    i, j = pairs
    return (correct[i] != correct[j]).float().mean().item()


def embedding_moved(power: torch.Tensor, pairs: Pairs) -> tuple[float, float]:
    """(TV distance / its frequency-shuffled expectation, raw TV distance)."""
    shares = power / power.sum(-1, keepdim=True)
    i, j = pairs
    tv = (0.5 * (shares[i] - shares[j]).abs().sum(-1)).mean()
    gen = torch.Generator().manual_seed(0)
    perms = torch.stack(
        [torch.randperm(shares.shape[-1], generator=gen) for _ in range(N_SHUFFLES)]
    )
    shuffled = shares[j][:, perms]  # [P, S, K]
    chance = (0.5 * (shares[i][:, None] - shuffled).abs().sum(-1)).mean()
    return (tv / chance).item(), tv.item()


def neuron_retention(
    member: torch.Tensor, active: torch.Tensor, pairs: Pairs, min_pop: int
) -> float | None:
    """Mean over frequencies of the mean over pairs of chance-corrected retention.

    Vectorised ``analyze_neuron_drift.retention_lift`` generalised from fixed
    lags to arbitrary (t, t') pairs.
    """
    i, j = pairs
    both = active[i] & active[j]  # [P, N]
    now = member[i] & both[..., None]  # [P, N, K]
    later = member[j] & both[..., None]
    n_now = now.sum(1).float()
    n_later = later.sum(1).float()
    n_both = both.sum(1, keepdim=True).float()
    chance = n_later / n_both.clamp_min(1)
    valid = (n_now >= min_pop) & (chance < 1) & (n_both > 0)
    kept = (now & later).sum(1).float() / n_now.clamp_min(1)
    lift = torch.where(valid, (kept - chance) / (1 - chance).clamp_min(1e-9), 0.0)
    counts = valid.sum(0)
    per_role = lift.sum(0)[counts > 0] / counts[counts > 0]
    return per_role.mean().item() if len(per_role) else None


def lag_pairs(n: int, lag: int, start: int = 0) -> Pairs:
    i = torch.arange(start, n - lag)
    return i, i + lag


def first_grok(acc: torch.Tensor) -> int | None:
    hits = (acc >= GROK).nonzero().flatten()
    return int(hits[0]) if len(hits) else None


def collapse_events(acc: torch.Tensor) -> list[tuple[int, int, int | None]]:
    """(last working eval before, lowest eval, recovery eval or None) per collapse."""
    start = first_grok(acc)
    if start is None:
        return []
    events: list[tuple[int, int, int | None]] = []
    t = start
    while t < len(acc):
        drops = (acc[t:] < DIP).nonzero().flatten()
        if not len(drops):
            break
        drop = t + int(drops[0])
        pre = drop - 1 - int((acc[:drop].flip(0) >= GROK).int().argmax())
        rec_hits = (acc[drop:] >= GROK).nonzero().flatten()
        rec = drop + int(rec_hits[0]) if len(rec_hits) else None
        end = rec if rec is not None else len(acc)
        trough = drop + int(acc[drop:end].argmin())
        events.append((pre, trough, rec))
        if rec is None:
            break
        t = rec
    return events


def collapse_free_pairs(
    acc: torch.Tensor, lag: int, start: int, stop: int | None = None
) -> Pairs:
    """(t, t + lag) spans, start <= t < stop, that never drop below DIP."""
    ok = acc >= DIP
    i, j = lag_pairs(len(acc), lag, start)
    if stop is not None:
        i, j = i[i < stop], j[i < stop]
    if not len(i):
        return i, j
    clean_prefix = torch.cat([torch.zeros(1), (~ok).float().cumsum(0)])
    clean = (clean_prefix[j + 1] - clean_prefix[i]) == 0
    clean &= acc[i] >= GROK
    clean &= acc[j] >= GROK
    return i[clean], j[clean]


def levels_moved(
    trace: dict[str, torch.Tensor],
    roles: tuple[torch.Tensor, torch.Tensor],
    pairs: Pairs,
    min_pop: int,
) -> dict[str, float | None]:
    if not len(pairs[0]):
        return {**dict.fromkeys(LEVELS), "n_pairs": 0}
    retention = neuron_retention(*roles, pairs, min_pop)
    embedding, embedding_raw = embedding_moved(trace["embed_power"], pairs)
    return {
        "capability": capability_moved(trace["test_correct"], pairs),
        "embedding": embedding,
        "embedding_raw": embedding_raw,
        "neurons": None if retention is None else 1 - retention,
        "n_pairs": len(pairs[0]),
    }


def summarize(
    trace: dict[str, torch.Tensor],
    window_start: int,
    lags: list[int],
    member_share: float,
    min_frac_answer: float,
    min_pop: int,
) -> dict[str, object]:
    steps = trace["steps"]
    acc = trace["test_correct"].float().mean(1)
    loss = trace["test_loss"].float().mean(1)
    cadence = int(steps[1] - steps[0])
    n_blocks = trace["profile"].shape[1]
    pooled = neuron_roles(trace, member_share, min_frac_answer)
    per_block = [
        neuron_roles(trace, member_share, min_frac_answer, [b]) for b in range(n_blocks)
    ]

    grok = first_grok(acc)
    win = max(int((steps < window_start).sum()), grok if grok is not None else 0)
    post = acc[grok:] if grok is not None else acc[:0]

    curves: dict[str, list[dict[str, float | None]]] = {"pooled": []}
    for b in range(n_blocks):
        curves[f"block{b}"] = []
    for lag_steps in lags:
        lag = lag_steps // cadence
        if grok is None or lag < 1 or win + lag >= len(steps):
            continue
        pairs = lag_pairs(len(steps), lag, win)
        curves["pooled"].append(
            {"lag": lag_steps, **levels_moved(trace, pooled, pairs, min_pop)}
        )
        for b, roles in enumerate(per_block):
            retention = neuron_retention(*roles, pairs, min_pop)
            curves[f"block{b}"].append(
                {
                    "lag": lag_steps,
                    "neurons": None if retention is None else 1 - retention,
                }
            )

    events = []
    for pre, trough, rec in collapse_events(acc):
        event: dict[str, object] = {
            "pre_step": int(steps[pre]),
            "trough_step": int(steps[trough]),
            "recovery_step": None if rec is None else int(steps[rec]),
            "min_acc": acc[trough].item(),
        }
        if rec is not None and grok is not None:
            span = (torch.tensor([pre]), torch.tensor([rec]))
            event["moved"] = levels_moved(trace, pooled, span, min_pop)
            radius = CONTROL_RADIUS // cadence
            control = collapse_free_pairs(
                acc, rec - pre, max(grok, pre - radius), pre + radius + 1
            )
            event["collapse_free_control"] = levels_moved(
                trace, pooled, control, min_pop
            )
        events.append(event)

    return {
        "run_name": trace["meta"]["run_name"],  # type: ignore[index]
        "last_step": int(steps[-1]),
        "window_start": int(steps[min(win, len(steps) - 1)]),
        "first_grok": None if grok is None else int(steps[grok]),
        "collapses": len(events),
        "evals_below_dip": int((post < DIP).sum()),
        "occupancy": (post >= GROK).float().mean().item() if len(post) else None,
        "window_acc_min": acc[win:].min().item(),
        "window_loss_mean": loss[win:].mean().item(),
        "active_neurons_per_block_at_end": [int(r[1][-1].sum()) for r in per_block],
        "curves": curves,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=30_000)
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000, 2000, 5000, 10_000, 15_000, 19_900],
    )
    parser.add_argument("--member-share", type=float, default=0.10)
    parser.add_argument("--min-frac-answer", type=float, default=0.5)
    parser.add_argument("--min-population", type=int, default=5)
    args = parser.parse_args()

    summary = summarize(
        load_trace(args.trace),
        args.window_start,
        args.lags,
        args.member_share,
        args.min_frac_answer,
        args.min_population,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

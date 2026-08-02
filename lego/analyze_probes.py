"""Linear probes at the supervised position: the direction half of the
dark-space prediction.

The lens-aux LEGO models provably compute intermediate trajectory states
(they solve k >= 3), yet below the coalescence layer the logit lens at the
supervised <predict> position reads near-uniform. Two places the
intermediates could live: other token positions (established — they are
lens-decodable at the <op> positions), or lens-invisible *directions* of
the <predict>-position residual itself. This script tests the second: for
every layer l and trajectory index j, fit a plain linear probe (no bias,
no norm — multinomial logistic regression on the raw residual) for
trajectory[j] on the <predict>-position residual at layer l.

Probes are fit on each run's OWN train-split chains and evaluated on that
run's reconstructed held-out split (the split seed follows the training
seed, parsed from the run name). Probe fitting is deterministic: zero
init + full-batch Adam on a convex objective.

Interpretation:
- probe >> lens below l*  => intermediates live in lens-invisible
  directions of the supervised position (dark space in the direction
  sense);
- probe ~ lens (both near chance) => the supervised position's residual
  is answer-subspace-only and the intermediates live entirely at the
  unsupervised positions.

Usage:
    uv run python -m lego.analyze_probes
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from common.checkpoint import artifact_path
from lego.analyze_logit_lens import make_logit_lens_fn, probe_predict_position
from lego.compare_lens_aux import RUNS, run_seed
from lego.generator import S3, ChainExample, enumerate_split, group_by_k
from lego.model import StandardTransformer
from lego.tokenizer import answer_position, encode
from lego.training import load_model

N_ELEMENTS = S3.order
PROBE_KS = (2, 4, 6)


@torch.no_grad()
def predict_position_residuals(
    model: StandardTransformer,
    examples: list[ChainExample],
    device: torch.device,
    batch_size: int = 512,
) -> Tensor:
    """<predict>-position residuals for every layer.

    Returns (n_layers, n_examples, dim), fp32.
    """
    k = len(examples[0].ops)
    pos = answer_position(k) - 1
    chunks: list[Tensor] = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        input_ids = torch.tensor(
            [encode(ex) for ex in chunk],
            dtype=torch.long,
            device=device,
        )
        _logits, residuals = model.forward_with_residuals(input_ids)
        chunks.append(torch.stack([r[:, pos, :] for r in residuals]))
    return torch.cat(chunks, dim=1).float()


@torch.no_grad()
def op_position_residuals(
    model: StandardTransformer,
    examples: list[ChainExample],
    j: int,
    device: torch.device,
    batch_size: int = 512,
) -> Tensor:
    """Residuals at the <op> position after operand j+1, every layer.

    Position 2*(j+2) is the first position that has causally seen the
    start element and operands 1..j+1 (same grid as probe_op_positions;
    for j = k-1 this is the <predict> position). Returns
    (n_layers, n_examples, dim), fp32.
    """
    pos = 2 * (j + 2)
    chunks: list[Tensor] = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        input_ids = torch.tensor(
            [encode(ex) for ex in chunk],
            dtype=torch.long,
            device=device,
        )
        _logits, residuals = model.forward_with_residuals(input_ids)
        chunks.append(torch.stack([r[:, pos, :] for r in residuals]))
    return torch.cat(chunks, dim=1).float()


def trajectory_targets(examples: list[ChainExample], device: torch.device) -> Tensor:
    """Element-class targets, shape (k+1, n_examples), values 0..5."""
    n_states = len(examples[0].ops) + 1
    return torch.tensor(
        [[ex.trajectory[j] for ex in examples] for j in range(n_states)],
        dtype=torch.long,
        device=device,
    )


def fit_probes(
    train_feats: Tensor,
    train_targets: Tensor,
    steps: int = 2000,
    lr: float = 1e-2,
) -> Tensor:
    """Fit plain linear probes (no bias) for every (layer, trajectory index).

    train_feats: (L, N, D); train_targets: (J, N) with classes 0..5.
    One probe per (layer, j) pair, all fit jointly as a batched tensor —
    the objectives are independent, this is just vectorization. Zero init
    on a convex objective makes the fit deterministic.

    Returns probe weights of shape (L, J, n_classes, D).
    """
    n_layers, _n, dim = train_feats.shape
    n_targets = train_targets.shape[0]
    weights = torch.zeros(
        n_layers,
        n_targets,
        N_ELEMENTS,
        dim,
        device=train_feats.device,
        requires_grad=True,
    )
    opt = torch.optim.Adam([weights], lr=lr)
    for _ in range(steps):
        logits = torch.einsum("lnd,ljcd->ljnc", train_feats, weights)
        loss = F.cross_entropy(
            logits.reshape(-1, N_ELEMENTS),
            train_targets.expand(n_layers, -1, -1).reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return weights.detach()


@torch.no_grad()
def probe_accuracy(feats: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    """Top-1 accuracy of each (layer, j) probe: (L, J)."""
    logits = torch.einsum("lnd,ljcd->ljnc", feats, weights)
    preds = logits.argmax(dim=-1)
    return (preds == targets.unsqueeze(0)).float().mean(dim=-1)


def format_matrix(mat: Tensor, row_prefix: str = "L") -> list[str]:
    return [
        f"      {row_prefix}{i}: " + " ".join(f"{v:.2f}" for v in row)
        for i, row in enumerate(mat.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-fit",
        type=int,
        default=4096,
        help="Max train-split chains per k used to fit probes",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=1024,
        help="Max held-out chains per k used to evaluate probes",
    )
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the full accuracy tensors as JSON",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Splits are reconstructed per training seed (parsed from the run name):
    # each run trained on its own seeded split, so probes must fit on that
    # run's train chains and evaluate on that run's held-out chains.
    split_cache: dict[
        int,
        tuple[dict[int, list[ChainExample]], dict[int, list[ChainExample]]],
    ] = {}
    results: dict[str, dict[str, object]] = {}

    for label, run_name, ckpt_file in RUNS:
        seed = run_seed(run_name)
        if seed not in split_cache:
            train, test = enumerate_split(
                args.k_min, args.k_max, test_frac=args.test_frac, seed=seed
            )
            split_cache[seed] = (group_by_k(train), group_by_k(test))
        train_by_k, test_by_k = split_cache[seed]

        path = artifact_path(f"lego/{run_name}/{ckpt_file}")
        model, _config = load_model(path, device)
        print(f"\n=== {label} ({run_name}) ===")
        run_result: dict[str, object] = {}

        for k in PROBE_KS:
            fit_examples = train_by_k[k][: args.n_fit]
            eval_examples = test_by_k[k][: args.n_eval]

            fit_feats = predict_position_residuals(model, fit_examples, device)
            eval_feats = predict_position_residuals(model, eval_examples, device)
            fit_targets = trajectory_targets(fit_examples, device)
            eval_targets = trajectory_targets(eval_examples, device)

            torch.manual_seed(0)  # Adam has no stochasticity here; belt & braces
            weights = fit_probes(fit_feats, fit_targets, steps=args.probe_steps)
            fit_acc = probe_accuracy(fit_feats, fit_targets, weights)
            eval_acc = probe_accuracy(eval_feats, eval_targets, weights)

            # Lens accuracy on the same held-out examples, same (L, j) grid.
            eval_ids = torch.tensor(
                [encode(ex) for ex in eval_examples],
                dtype=torch.long,
                device=device,
            )
            _logits, residuals = model.forward_with_residuals(eval_ids)
            lens_acc = probe_predict_position(
                residuals, eval_examples, make_logit_lens_fn(model), k, device
            )

            print(
                f"\n  k={k}  ({len(fit_examples)} fit / {len(eval_examples)} "
                f"held-out chains; cols = trajectory index j, chance = "
                f"{1 / N_ELEMENTS:.2f})"
            )
            print("    probe held-out accuracy:")
            print("\n".join(format_matrix(eval_acc)))
            print("    lens held-out accuracy (same examples):")
            print("\n".join(format_matrix(lens_acc)))
            print("    probe fit (train) accuracy:")
            print("\n".join(format_matrix(fit_acc)))

            run_result[f"k{k}"] = {
                "probe_eval_acc": eval_acc.tolist(),
                "probe_fit_acc": fit_acc.tolist(),
                "lens_eval_acc": lens_acc.tolist(),
                "n_fit": len(fit_examples),
                "n_eval": len(eval_examples),
            }
        results[run_name] = run_result

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()

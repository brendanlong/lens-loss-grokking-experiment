"""Compare the logit-lens staircase with vs without the lens-aux loss.

grok_lens Phase 5 analysis (see experiments/grok_lens/EXPERIMENT_PLAN.md):
for the baseline and lens-aux LEGO runs, probe at the <predict> position
per layer and per chain length k:

- answer readability: fraction of examples where the lens top-1 equals the
  FINAL answer (trajectory[k]) — its first >= 0.95 layer is the
  coalescence layer l*(k);
- element-restricted entropy of the lens distribution (nats; uniform over
  the 6 S3 elements = ln 6 ~ 1.79) — the "dark space" prediction is
  ~uniform before l*(k);
- the <op>-position trajectory staircase (does the baseline's staircase
  survive when the answer position is supervised?).

Usage:
    uv run python -m lego.compare_lens_aux
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from common.checkpoint import artifact_path
from lego.analyze_logit_lens import (
    analyze_op_positions,
    analyze_predict_position,
)
from lego.generator import ChainExample, enumerate_split, group_by_k
from lego.model import AnyModel
from lego.tokenizer import answer_position, encode
from lego.training import load_model

RUNS: list[tuple[str, str, str]] = [
    ("baseline s42", "S3-std-8L-lensaux-base-s42", "step_39000.pt"),
    ("baseline s43", "S3-std-8L-lensaux-base-s43-stream", "step_19531.pt"),
    ("baseline s44", "S3-std-8L-lensaux-base-s44-stream", "step_19531.pt"),
    ("aux uniform s42", "S3-std-8L-lensaux0.3-uniform-s42", "step_39000.pt"),
    ("aux uniform s43", "S3-std-8L-lensaux0.3-uniform-s43-stream", "step_19531.pt"),
    ("aux uniform s44", "S3-std-8L-lensaux0.3-uniform-s44-stream", "step_19531.pt"),
    ("aux linear s42", "S3-std-8L-lensaux0.3-linear-s42", "step_39000.pt"),
    ("aux linear s43", "S3-std-8L-lensaux0.3-linear-s43-stream", "step_19531.pt"),
    ("aux linear s44", "S3-std-8L-lensaux0.3-linear-s44-stream", "step_19531.pt"),
]
N_ELEMENTS = 6


@torch.no_grad()
def element_entropy_at_predict(
    model: AnyModel,
    examples: list[ChainExample],
    k: int,
    device: torch.device,
) -> Tensor:
    """Per-layer entropy (nats) of the lens distribution over element
    tokens at the <predict> position. Uniform over 6 elements = ln 6."""
    input_ids = torch.tensor([encode(ex) for ex in examples], device=device)
    _, residuals = model.forward_with_residuals(input_ids)
    pos = answer_position(k) - 1  # <predict> position
    # element tokens are ids 1..6
    entropies = []
    for r in residuals:
        logits = F.linear(model.final_norm(r[:, pos]), model.tok_emb.weight)
        probs = F.softmax(logits[:, 1 : 1 + N_ELEMENTS], dim=-1)
        entropies.append(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean())
    return torch.stack(entropies)


def test_examples_by_k(
    k_min: int,
    k_max: int,
    test_frac: float,
    seed: int,
    n_examples: int,
) -> dict[int, list[ChainExample]]:
    """Reconstruct the held-out test split used in training, grouped by k.

    Uses the same (k_min, k_max, test_frac, seed) as lego.train, so the
    probe examples are exactly the chains the model never trained on.
    Caps each k at n_examples (small k have few held-out chains).
    """
    _train, test = enumerate_split(k_min, k_max, test_frac=test_frac, seed=seed)
    return {k: exs[:n_examples] for k, exs in group_by_k(test).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-examples", type=int, default=500)
    # Must match the training run's data settings so the reconstructed
    # test split is identical (and disjoint from the training data).
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples_by_k = test_examples_by_k(
        args.k_min, args.k_max, args.test_frac, args.seed, args.n_examples
    )

    for label, run_name, ckpt_file in RUNS:
        path = artifact_path(f"lego/{run_name}/{ckpt_file}")
        model, _config = load_model(path, device)
        print(f"\n=== {label} ({run_name}) ===")
        for k in (2, 4, 6):
            examples = examples_by_k[k]
            heat = analyze_predict_position(model, examples, device)
            answer_col = heat[:, k]  # per-layer P(lens top-1 == final answer)
            entropy = element_entropy_at_predict(model, examples, k, device)
            coalesce = next(
                (i for i, a in enumerate(answer_col.tolist()) if a >= 0.95), None
            )
            acc_str = " ".join(f"{a:.2f}" for a in answer_col.tolist())
            ent_str = " ".join(f"{e:.2f}" for e in entropy.tolist())
            print(f"k={k}  l*={coalesce!s:>4s}  answer acc/layer: {acc_str}")
            print(f"      (ln6={math.log(6):.2f})  elem entropy/layer: {ent_str}")
        # op-position trajectory staircase at k=6 (diag = layer j holds traj[j+1])
        examples = examples_by_k[6]
        op_heat = analyze_op_positions(model, examples, device)
        print("      op-position staircase (rows=layers, cols=traj steps):")
        for layer_idx in range(op_heat.shape[0]):
            row = " ".join(f"{v:.2f}" for v in op_heat[layer_idx].tolist())
            print(f"        L{layer_idx}: {row}")


if __name__ == "__main__":
    main()

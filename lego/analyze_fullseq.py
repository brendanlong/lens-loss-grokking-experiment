"""Position-by-position lens and probe analysis for the full-sequence arms.

Three readouts per (layer, position), on held-out chains (fit sets come
from each run's own train split; split seed = training seed):

1. **Own-output progression** — lens top-1 vs the position's actual next
   token, plus full-vocab lens entropy: does the lens show a clean
   distribution → token progression toward what that position is trained
   to emit? (At positions whose next token is a uniform-random operand,
   the CE-optimal "clean" answer is calibrated near-uniform.)
2. **Intermediates in the lens** — lens top-1 vs trajectory[j] for every
   j at every position: are intermediate values visible to the lens
   anywhere?
3. **Intermediates via probes** — plain linear probes (no bias) for
   trajectory[j] on the residual at every position: are the
   intermediates linearly present even where the lens is blind?

The condensed printout shows the op-position grid (the <op> token after
operand m is the first position that has causally seen operands 1..m,
i.e. everything needed for trajectory[m]; for m = k it is the <predict>
position) and the <predict> column; the full (position x layer x target)
tensors go to --json-out.

Usage:
    uv run python -m lego.analyze_fullseq
    uv run python -m lego.analyze_fullseq \
        --runs "grok fullseq base s42=\
            S3-grok-sub10000-wd0.3-fullseqbase-s42-100k=step_100000.pt"
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from common.checkpoint import artifact_path
from lego.analyze_probes import fit_probes, probe_accuracy, trajectory_targets
from lego.compare_lens_aux import run_seed
from lego.generator import ChainExample, enumerate_split, group_by_k
from lego.model import StandardTransformer
from lego.tokenizer import element_token, encode
from lego.training import load_model

ANALYZE_KS = (2, 4, 6)

# The Phase 9 arms plus the answer-only pair for side-by-side comparison.
DEFAULT_RUNS: list[tuple[str, str, str]] = [
    ("fullseq base s42", "S3-std-8L-splitku-fullseqbase-s42", "step_20960.pt"),
    (
        "fullseq aux s42",
        "S3-std-8L-splitku-fullseq-lensaux0.3-allpos-s42",
        "step_20960.pt",
    ),
    ("answer-only base s42", "S3-std-8L-splitku-base-s42", "step_20960.pt"),
    (
        "answer-only aux s42",
        "S3-std-8L-splitku-lensaux0.3-uniform-s42",
        "step_20960.pt",
    ),
]


@torch.no_grad()
def all_position_residuals(
    model: StandardTransformer,
    examples: list[ChainExample],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Residuals at every position, every layer, plus the input ids.

    Returns (residuals (L, N, T, D) fp32, input_ids (N, T)); all examples
    share one chain length so T = 2k + 4 with no padding.
    """
    res_chunks: list[Tensor] = []
    id_chunks: list[Tensor] = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        input_ids = torch.tensor(
            [encode(ex) for ex in chunk],
            dtype=torch.long,
            device=device,
        )
        _logits, residuals = model.forward_with_residuals(input_ids)
        res_chunks.append(torch.stack(residuals))
        id_chunks.append(input_ids)
    return torch.cat(res_chunks, dim=1).float(), torch.cat(id_chunks, dim=0)


@torch.no_grad()
def lens_readouts(
    model: StandardTransformer,
    residuals: Tensor,
    input_ids: Tensor,
    k: int,
    examples: list[ChainExample],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lens accuracy tensors over (layer, position).

    Returns:
        next_acc (L, P): lens top-1 == actual next token, positions 0..2k+2.
        entropy (L, P): full-vocab lens entropy (nats), same positions.
        traj_acc (L, P, k+1): lens top-1 == element_token(trajectory[j]).
    """
    n_pos = 2 * k + 3  # positions with a next token
    lens_logits = F.linear(
        model.final_norm(residuals[:, :, :n_pos, :]),
        model.tok_emb.weight,
    )  # (L, N, P, V)
    top1 = lens_logits.argmax(dim=-1)
    next_tokens = input_ids[:, 1 : n_pos + 1]  # (N, P)
    next_acc = (top1 == next_tokens.unsqueeze(0)).float().mean(dim=1)
    probs = F.softmax(lens_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean(dim=1)  # (L, P)
    targets = torch.tensor(
        [[element_token(ex.trajectory[j]) for j in range(k + 1)] for ex in examples],
        dtype=torch.long,
        device=device,
    )  # (N, k+1)
    traj_acc = (
        (top1.unsqueeze(-1) == targets.unsqueeze(0).unsqueeze(2)).float().mean(dim=1)
    )  # (L, P, k+1)
    return next_acc, entropy, traj_acc


def position_labels(k: int) -> list[str]:
    """Human-readable labels for positions 0..2k+2."""
    labels: list[str] = []
    for p in range(2 * k + 3):
        if p == 0:
            labels.append("<s>")
        elif p == 1:
            labels.append("e0")
        elif p == 2 * k + 2:
            labels.append("<pr>")
        elif p % 2 == 0:
            labels.append("<op>")
        else:
            labels.append(f"g{(p - 1) // 2}")
    return labels


def fmt_row(vals: list[float]) -> str:
    return " ".join(f"{v:.2f}" for v in vals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fit", type=int, default=4096)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument(
        "--runs",
        action="append",
        default=None,
        help=(
            "label=run_name=ckpt_file (repeatable); default = the Phase 9 "
            "data-rich pair plus the answer-only pair"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = [tuple(r.split("=", 2)) for r in args.runs] if args.runs else DEFAULT_RUNS

    split_cache: dict[
        int,
        tuple[dict[int, list[ChainExample]], dict[int, list[ChainExample]]],
    ] = {}
    results: dict[str, dict[str, object]] = {}

    for label, run_name, ckpt_file in runs:
        seed = run_seed(run_name)
        if seed not in split_cache:
            train, test = enumerate_split(
                args.k_min, args.k_max, test_frac=args.test_frac, seed=seed
            )
            split_cache[seed] = (group_by_k(train), group_by_k(test))
        train_by_k, test_by_k = split_cache[seed]

        model, _config = load_model(
            artifact_path(f"lego/{run_name}/{ckpt_file}"), device
        )
        print(f"\n=== {label} ({run_name}) ===")
        run_result: dict[str, object] = {}

        for k in ANALYZE_KS:
            fit_examples = train_by_k[k][: args.n_fit]
            eval_examples = test_by_k[k][: args.n_eval]
            labels = position_labels(k)

            fit_res, _fit_ids = all_position_residuals(model, fit_examples, device)
            eval_res, eval_ids = all_position_residuals(model, eval_examples, device)
            fit_targets = trajectory_targets(fit_examples, device)
            eval_targets = trajectory_targets(eval_examples, device)

            next_acc, entropy, traj_lens = lens_readouts(
                model, eval_res, eval_ids, k, eval_examples, device
            )

            # Probes at every position with a next token (0..2k+2).
            n_pos = 2 * k + 3
            probe_acc_all = torch.zeros(n_pos, eval_res.shape[0], k + 1)
            for p in range(n_pos):
                weights = fit_probes(
                    fit_res[:, :, p, :],
                    fit_targets,
                    steps=args.probe_steps,
                )
                probe_acc_all[p] = probe_accuracy(
                    eval_res[:, :, p, :], eval_targets, weights
                )

            n_layers = eval_res.shape[0]
            print(
                f"\n  k={k}  ({len(eval_examples)} held-out; positions: "
                f"{' '.join(labels)}; chance over elements = 0.17)"
            )
            print("    [1] lens next-token accuracy (rows = layers):")
            for li in range(n_layers):
                print(f"      L{li}: {fmt_row(next_acc[li].tolist())}")
            print(
                f"    [1] lens entropy at <predict> by layer (ln6={math.log(6):.2f}):"
                f" {fmt_row(entropy[:, 2 * k + 2].tolist())}"
            )
            # Op-position grid: trajectory[m] read at the <op> after operand m
            # (= position 2m+2; m = k is the <predict> position).
            print(
                "    [2] lens on trajectory[m] at its op position "
                "(rows = layers, cols m=1..k):"
            )
            for li in range(n_layers):
                row = [traj_lens[li, 2 * m + 2, m].item() for m in range(1, k + 1)]
                print(f"      L{li}: {fmt_row(row)}")
            print("    [3] probe on trajectory[m] at its op position:")
            for li in range(n_layers):
                row = [probe_acc_all[2 * m + 2, li, m].item() for m in range(1, k + 1)]
                print(f"      L{li}: {fmt_row(row)}")
            # Best lens/probe cell anywhere for each intermediate state.
            for m in range(1, k):
                lens_best = traj_lens[:, :, m]
                probe_best = probe_acc_all[:, :, m]
                lb = lens_best.max().item()
                pb = probe_best.max().item()
                lb_where = divmod(int(lens_best.argmax()), lens_best.shape[1])
                pb_pos = int(probe_best.argmax() // probe_best.shape[1])
                pb_layer = int(probe_best.argmax() % probe_best.shape[1])
                print(
                    f"    traj[{m}] anywhere: lens max {lb:.2f} "
                    f"(L{lb_where[0]},{labels[lb_where[1]]}) | probe max {pb:.2f} "
                    f"(L{pb_layer},{labels[pb_pos]})"
                )

            run_result[f"k{k}"] = {
                "position_labels": labels,
                "lens_next_token_acc": next_acc.tolist(),
                "lens_entropy": entropy.tolist(),
                "lens_traj_acc": traj_lens.tolist(),
                "probe_traj_acc": probe_acc_all.tolist(),
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

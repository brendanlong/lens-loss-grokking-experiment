#!/bin/bash
# Retrain everything from scratch. ~160 runs; each 30k-step run takes
# ~12 min on an RTX 3060 Ti-class GPU (50k ~19 min, 100k ~38 min) — the
# full sweep is roughly a day of consumer-GPU time. Runs log to wandb by
# default (pass --no-wandb to disable). Checkpoints land in
# data/grok_lens/checkpoints/<run-name>/final.pt.
#
# Exact per-phase commands and their original results: see RESULTS.md.
# This script reproduces the core arms; uncomment sections for the rest.
set -euo pipefail
cd "$(dirname "$0")/.."
train() { local run="$1"; shift; uv run python -m grok_lens.train --wandb-run-name "$run" --checkpoint-dir "data/grok_lens/checkpoints/$run" "$@"; }

# --- Phase 1: baselines (grokking is unstable; use 50k to see it) ---
for s in 42 43 44; do train "p113-L2-lam0.0-uniform-frac0.3-s${s}-50k" --seed $s --total-steps 50000 --log-fourier; done

# --- Core aux arm ---
for s in 42 43 44; do train "p113-L2-lam0.3-uniform-frac0.3-s${s}" --aux-lambda 0.3 --seed $s --log-fourier; done

# --- Lambda sweep (flat dose-response) ---
# for lam in 0.01 0.1 1.0; do for s in 42 43 44; do train "p113-L2-lam${lam}-uniform-frac0.3-s${s}" --aux-lambda $lam --seed $s; done; done
# for s in 42 43 44; do train "p113-L2-lam3.0-uniform-frac0.3-s${s}" --aux-lambda 3.0 --seed $s --total-steps 50000; done

# --- Depth / weighting / optimizer arms ---
# for s in 42 43 44; do train "p113-L1-lam0.0-uniform-frac0.3-s${s}" --n-layers 1 --seed $s; done
# for s in 42 43 44; do train "p113-L3-lam0.0-uniform-frac0.3-s${s}" --n-layers 3 --seed $s; done
# for s in 42 43 44; do train "p113-L3-lam0.3-uniform-frac0.3-s${s}" --n-layers 3 --aux-lambda 0.3 --seed $s; done
# for s in 42 43 44; do train "p113-L3-lam0.3-linear-frac0.3-s${s}" --n-layers 3 --aux-lambda 0.3 --aux-weighting linear --seed $s; done
# for s in 42 43 44; do train "p113-L2-lam0.0-uniform-frac0.3-s${s}-muon-50k" --optimizer muon --seed $s --total-steps 50000; done
# for s in 42 43 44; do train "p113-L2-lam0.3-uniform-frac0.3-s${s}-muon-50k" --optimizer muon --aux-lambda 0.3 --seed $s --total-steps 50000; done
# for s in 42 43 44; do train "p113-L3-lam0.0-uniform-frac0.3-s${s}-muon-50k" --optimizer muon --n-layers 3 --seed $s --total-steps 50000; done
# for s in 42 43 44; do train "p113-L3-lam0.3-uniform-frac0.3-s${s}-muon-50k" --optimizer muon --n-layers 3 --aux-lambda 0.3 --seed $s --total-steps 50000; done

# --- Controls: wd sweep, shuffled targets, wd=0 ---
# for wd in 0.25 0.5; do for s in 42 43 44; do train "p113-L2-lam0.0-uniform-frac0.3-s${s}-wd${wd}-50k" --weight-decay $wd --seed $s --total-steps 50000 --log-fourier; done; done
# for s in 42 43 44; do train "p113-L2-lam0.3-shuf-frac0.3-s${s}-50k" --aux-lambda 0.3 --aux-shuffled-targets --seed $s --total-steps 50000 --log-fourier; done
# for s in 42 43 44; do train "p113-L2-lam0.0-uniform-frac0.3-s${s}-wd0-50k" --weight-decay 0 --seed $s --total-steps 50000; done

# --- Second task: modular subtraction ---
# for s in 42 43 44; do train "p113sub-L2-lam0.0-uniform-frac0.3-s${s}-50k" --task sub --seed $s --total-steps 50000 --log-fourier; done
# for s in 42 43 44; do train "p113sub-L2-lam0.3-uniform-frac0.3-s${s}-100k" --task sub --aux-lambda 0.3 --seed $s --total-steps 100000 --log-fourier; done

# --- Objective-switching continuations (from your own trained checkpoints, or hf: paths) ---
# train "p113-L2-cont-auxoff-from-lam0.3-s42" --resume-from "hf:grok_lens/p113-L2-lam0.3-uniform-frac0.3-s42/final.pt" --aux-lambda 0 --seed 42
# train "p113-L2-cont-rescue-from-base-s42" --resume-from "hf:grok_lens/p113-L2-lam0.0-uniform-frac0.3-s42-50k/final.pt" --aux-lambda 0.3 --seed 42
# controls (same source, objective unchanged):
# train "p113-L2-cont-auxon-from-lam0.3-s42" --resume-from "hf:grok_lens/p113-L2-lam0.3-uniform-frac0.3-s42/final.pt" --aux-lambda 0.3 --seed 42
# train "p113-L2-cont-base-from-base-s42" --resume-from "hf:grok_lens/p113-L2-lam0.0-uniform-frac0.3-s42-50k/final.pt" --aux-lambda 0 --seed 42

# --- Architecture control: no LayerNorm (instability requires LN) ---
# for s in 42 43 44; do train "p113-L2-lam0.0-uniform-frac0.3-s${s}-noln-50k" --no-layernorm --seed $s --total-steps 50000 --log-fourier; done

# --- torch.optim.Muon re-runs (canonical Muon arm) ---
# for s in 42 43 44; do train "p113-L2-lam0.0-uniform-frac0.3-s${s}-torchmuon-50k" --optimizer muon --seed $s --total-steps 50000 --log-fourier; done
# for s in 42 43 44; do train "p113-L2-lam0.3-uniform-frac0.3-s${s}-torchmuon-50k" --optimizer muon --aux-lambda 0.3 --seed $s --total-steps 50000 --log-fourier; done
# for s in 42 43 44; do train "p113-L3-lam0.0-uniform-frac0.3-s${s}-torchmuon-50k" --optimizer muon --n-layers 3 --seed $s --total-steps 50000 --log-fourier; done
# for s in 42 43 44; do train "p113-L3-lam0.3-uniform-frac0.3-s${s}-torchmuon-50k" --optimizer muon --n-layers 3 --aux-lambda 0.3 --seed $s --total-steps 50000 --log-fourier; done

# --- 500k long-horizon pair (~3.5 h each) ---
# train "p113-L2-lam0.0-uniform-frac0.3-s42-500k" --seed 42 --total-steps 500000 --log-fourier
# train "p113-L2-lam0.3-uniform-frac0.3-s42-500k" --aux-lambda 0.3 --seed 42 --total-steps 500000 --log-fourier

# --- LEGO multi-hop (front-loading; enumerated data, disjoint per-k split,
# --- k-uniform sampling; ~10 min/run). Note: --seed also seeds each run's
# --- train/test split; the analyses reconstruct the split per run name. ---
# for s in 42 43 44; do
#   uv run python -m lego.train --seed $s --wandb-run-name "S3-std-8L-splitku-base-s${s}" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-base-s${s}"
#   uv run python -m lego.train --seed $s --lens-aux --lens-aux-weight 0.3 --wandb-run-name "S3-std-8L-splitku-lensaux0.3-uniform-s${s}" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-lensaux0.3-uniform-s${s}"
#   uv run python -m lego.train --seed $s --lens-aux --lens-aux-weight 0.3 --lens-aux-weighting linear --wandb-run-name "S3-std-8L-splitku-lensaux0.3-linear-s${s}" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-lensaux0.3-linear-s${s}"
# done
# for s in 42 43; do uv run python -m lego.train --seed $s --lens-aux --lens-aux-weight 0.3 --lens-aux-mode all-positions --wandb-run-name "S3-std-8L-splitku-lensaux0.3-allpos-s${s}" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-lensaux0.3-allpos-s${s}"; done

# --- LEGO grokking regime search (memorizable subset × weight decay;
# --- constant LR, 50k steps, ~25 min/run) ---
# for n in 2000 5000 10000; do for wd in 0.1 0.3 1.0; do
#   run="S3-grok-sub${n}-wd${wd}-s42-50k"
#   uv run python -m lego.train --train-subset $n --weight-decay $wd --lr-schedule constant --total-steps 50000 --eval-every-steps 500 --seed 42 --wandb-run-name "$run" --checkpoint-dir "data/lego/checkpoints/$run"
# done; done

# --- LEGO grokking regime, baseline vs aux (sub10000-wd0.3, 100k steps,
# --- ~50 min/run; per-k stability via lego.analyze_grok_stability) ---
# for s in 42 43 44; do
#   run="S3-grok-sub10000-wd0.3-base-s${s}-100k"
#   uv run python -m lego.train --train-subset 10000 --weight-decay 0.3 --lr-schedule constant --total-steps 100000 --eval-every-steps 500 --seed $s --wandb-run-name "$run" --checkpoint-dir "data/lego/checkpoints/$run"
#   run="S3-grok-sub10000-wd0.3-lensaux0.3-uniform-s${s}-100k"
#   uv run python -m lego.train --train-subset 10000 --weight-decay 0.3 --lr-schedule constant --total-steps 100000 --eval-every-steps 500 --lens-aux --lens-aux-weight 0.3 --seed $s --wandb-run-name "$run" --checkpoint-dir "data/lego/checkpoints/$run"
# done

# --- Full-sequence (realistic) objective: data-rich pair + attribution arm,
# --- and the grokking-regime pair (delay-and-destabilize) ---
# uv run python -m lego.train --base-loss all-positions --seed 42 --wandb-run-name "S3-std-8L-splitku-fullseqbase-s42" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-fullseqbase-s42"
# uv run python -m lego.train --base-loss all-positions --lens-aux --lens-aux-weight 0.3 --lens-aux-mode all-positions --seed 42 --wandb-run-name "S3-std-8L-splitku-fullseq-lensaux0.3-allpos-s42" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-fullseq-lensaux0.3-allpos-s42"
# uv run python -m lego.train --base-loss all-positions --lens-aux --lens-aux-weight 0.3 --seed 42 --wandb-run-name "S3-std-8L-splitku-fullseqbase-lensaux0.3-answer-s42" --checkpoint-dir "data/lego/checkpoints/S3-std-8L-splitku-fullseqbase-lensaux0.3-answer-s42"
# for s in 42 43 44; do
#   run="S3-grok-sub10000-wd0.3-fullseqbase-s${s}-100k"
#   uv run python -m lego.train --base-loss all-positions --train-subset 10000 --weight-decay 0.3 --lr-schedule constant --total-steps 100000 --eval-every-steps 500 --seed $s --wandb-run-name "$run" --checkpoint-dir "data/lego/checkpoints/$run"
#   run="S3-grok-sub10000-wd0.3-fullseq-lensaux0.3-allpos-s${s}-100k"
#   uv run python -m lego.train --base-loss all-positions --lens-aux --lens-aux-weight 0.3 --lens-aux-mode all-positions --train-subset 10000 --weight-decay 0.3 --lr-schedule constant --total-steps 100000 --eval-every-steps 500 --seed $s --wandb-run-name "$run" --checkpoint-dir "data/lego/checkpoints/$run"
# done

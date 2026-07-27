#!/bin/bash
# Retrain everything from scratch. ~130 runs; each 30k-step run takes
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

# --- LEGO multi-hop (front-loading; ~15 min/run) ---
# for s in 42 43 44; do
#   uv run python -m lego.train --generate-n 10000000 --seed $s --wandb-run-name "S3-std-8L-lensaux-base-s${s}"
#   uv run python -m lego.train --generate-n 10000000 --seed $s --lens-aux --lens-aux-weight 0.3 --wandb-run-name "S3-std-8L-lensaux0.3-uniform-s${s}"
#   uv run python -m lego.train --generate-n 10000000 --seed $s --lens-aux --lens-aux-weight 0.3 --lens-aux-weighting linear --wandb-run-name "S3-std-8L-lensaux0.3-linear-s${s}"
# done

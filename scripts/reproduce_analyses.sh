#!/bin/bash
# Reproduce every table in RESULTS.md / WRITEUP.md from the hosted
# checkpoints (https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment).
# No GPU or training needed; checkpoints download on first use (~300 MB).
#
# The wandb-history-based analyses (analyze_stability, make_figures) read the
# public wandb project https://wandb.ai/brendanlong-com/grok-lens and need a
# (free) `wandb login` first.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Fourier spectra (sparse baseline vs distributed aux circuits) ==="
uv run python -m grok_lens.analyze_fourier

echo "=== Per-frequency knockouts (causal brittleness) ==="
uv run python -m grok_lens.analyze_knockout

echo "=== Layer-0 relocation probe ==="
uv run python -m grok_lens.analyze_layer0

echo "=== LEGO lens staircase / coalescence ==="
uv run python -m lego.compare_lens_aux

echo "=== LEGO direction probes at the supervised position ==="
uv run python -m lego.analyze_probes --json-out data/analysis/probes.json

echo "=== LEGO full-sequence arms: position-by-position lens/probe grid ==="
uv run python -m lego.analyze_fullseq --json-out data/analysis/fullseq.json
uv run python -m lego.analyze_fullseq \
  --runs "grok fullseq base s43=S3-grok-sub10000-wd0.3-fullseqbase-s43-100k=step_100000.pt" \
  --runs "grok fullseq aux s44=S3-grok-sub10000-wd0.3-fullseq-lensaux0.3-allpos-s44-100k=step_100000.pt" \
  --json-out data/analysis/fullseq-grok.json

echo "=== Stability metrics (occupancy, dips; needs wandb login) ==="
uv run python -m grok_lens.analyze_stability || echo "skipped (wandb login required)"

echo "=== Figures (needs wandb login) ==="
uv run python -m grok_lens.make_figures || echo "skipped (wandb login required)"

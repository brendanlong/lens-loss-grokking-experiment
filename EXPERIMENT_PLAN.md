# Grokking Under a Logit-Lens Auxiliary Loss

## Hypothesis

A logit-lens auxiliary loss (project every layer's residual stream through
the final LayerNorm + unembedding and score it against the target) forces
every layer to be "answer-shaped", biasing the model toward monotone
iterative refinement and penalizing non-answer-shaped intermediate
computation. For a task whose generalizing solution is a specific
late-forming circuit — modular addition's Fourier-multiplication circuit
(Nanda et al. 2023) — a strong early-layer auxiliary weight should **delay or
suppress grokking**. The less likely alternative is that the denser gradient
signal *accelerates* generalization.

Supporting tension: McGrath et al. (2023) found late-layer MLPs often
*reduce* the top token's probability — the natural computation is not
monotonic, and a strict per-layer loss fights this.

**Predictions (written before running):**

1. Baseline (λ = 0) reproduces canonical grokking: train accuracy hits ~100%
   early, test accuracy stays near chance for thousands of full-batch steps,
   then jumps to ~100%.
2. Small λ with later-weighted layers (CALM-style) leaves grokking timing
   roughly intact.
3. Large λ with uniform weighting (heavy early-layer pressure) shifts
   `grok_step` later, and at some λ prevents grokking within the step budget
   (test accuracy stays low while train accuracy is high).
4. The layer-weighting schedule w_ℓ is the pivotal knob: at matched λ,
   uniform weighting hurts more than later-weighted.

What would refute the hypothesis: `grok_step` unchanged (or earlier) across
a λ sweep spanning 0.01–3.0 with uniform weighting.

## Novelty

The auxiliary loss itself is deep supervision (Lee et al.) with a shared
unembedding head — LayerSkip, CALM, DistillLens are existing instances, all
motivated by **inference efficiency** (early exit / self-speculative
decoding). Nobody appears to have studied its effect on **grokking**, where
the interesting question is what per-layer answer-shaping does to a known
delayed-generalization phase transition and a known circuit.

## Approach: reproduce the pieces, then compose

**Phase 1 — clean grokking baseline.** λ = 0 with the canonical setup
(p = 113, train_frac 0.3, 2-layer transformer, full-batch AdamW, wd = 1.0,
constant LR 1e-3, ~30k steps). Success = the standard delayed-generalization
curve, with `memorize_step` ≪ `grok_step` visible in wandb. Run a few seeds.

**Phase 2 — aux-loss mechanism check.** Moderate λ (e.g. 0.3–1.0), verify
the aux loss does what it claims: per-layer logit-lens accuracy at
intermediate layers rises well above the λ = 0 baseline (where layer-0 lens
accuracy typically stays near chance). This validates the mechanism
independently of any grokking claim.

**Phase 3 — the sweep.** λ ∈ {0, 0.01, 0.1, 0.3, 1.0, 3.0} ×
weighting ∈ {uniform, linear}, ≥3 seeds each.
Primary outcome: `grok_step` (first eval step with test acc ≥ 0.95);
secondary: `memorize_step`, whether grokking happens at all, per-layer lens
accuracy trajectories.

**Phase 4 (stretch) — circuit check.** For representative runs, test whether
the Fourier-multiplication circuit still forms: periodicity in the embedding
matrix (FFT of embedding columns, concentration on a few key frequencies).
Analysis script added if Phase 3 shows an interesting effect.

## Architecture

| Component | Details |
|---|---|
| Task | a + b (mod p), p = 113; tokens `[a, b, =]`, answer read at `=` |
| Data | all p² = 12,769 pairs; seeded disjoint split, train_frac 0.3 (3,831 train) |
| Model | 2-layer pre-LN decoder-only transformer, d = 128, 4 heads, ReLU MLP (4×), learned positions, untied unembed (~530k params) |
| Loss | CE(final) + λ · Σ_ℓ w_ℓ · CE_ℓ (true-token targets) over the L−1 intermediate layers, lens = ln_f + unembed |
| Optimizer | full-batch AdamW, lr 1e-3 (constant, 10-step warmup), β = (0.9, 0.98), wd = 1.0, no grad clipping |
| Budget | 30k full-batch steps (= epochs); eval every 100 steps |

Note: unlike Nanda et al. we keep LayerNorm — the logit lens needs `ln_f` to
be the model's real readout. Grokking is known to be robust to this choice.

## Metrics

- `memorize_step` / `grok_step`: first eval step with train/test acc ≥ 0.95
  (wandb summary keys).
- `lens_train/acc_layer_ℓ`, `lens_test/acc_layer_ℓ`: per-layer logit-lens
  accuracy over training — the mechanism readout.
- `weight_norm`: global parameter L2, the standard grokking dynamics probe
  (generalization coincides with weight-norm decline under high wd).
- Phase 4: FFT of embeddings → key-frequency concentration.

## Deviations from repo defaults (deliberate)

- **Fixed dataset, not streaming**: memorization of a small fixed train set
  is the phenomenon under study; streaming unique data would remove it.
- **Full-batch, fp32, no autocast**: canonical grokking regime; tiny model,
  precision-sensitive dynamics (slingshot effects).

## Risks

| Risk | Mitigation |
|---|---|
| Baseline doesn't grok in 30k steps | Known-good hyperparameters (Nanda et al.); if needed raise steps to 50–100k — runs are minutes on the local GPU |
| 2 layers → only one intermediate lens point, weighting knob degenerate | Uniform vs linear identical at L = 2; run the weighting contrast at L = 3–4 where they differ |
| Grokking-step variance across seeds swamps the λ effect | ≥3 seeds per cell; report per-seed grok steps, not just means |
| Aux loss changes effective LR scale (more loss terms → bigger grads) | Normalized layer weights (Σw = 1); report train-loss trajectories so an optimization-speed confound is visible |

## Success criteria

Informative regardless of outcome:

- **λ delays/suppresses grokking (dose-dependent)**: per-layer
  answer-shaping is in tension with forming the generalizing circuit —
  a clean, legible negative interaction between deep supervision and
  delayed generalization.
- **λ accelerates grokking**: denser gradient signal speeds circuit
  formation — practical and surprising.
- **No effect**: grokking is robust to intermediate-layer supervision;
  the aux loss only reshapes *where* the answer appears, not *when* the
  circuit forms.

## Phase 5 (extension): lens aux loss on LEGO multi-hop composition

Added 2026-07-19, after Phases 1–4 concluded (see RESULTS.md). Modular
addition has no *necessary* intermediate quantities, so answer-shaping
every layer "only" forced a shallower circuit. k-hop S3 composition
(`experiments/lego/`) is the opposite regime: intermediate layers must
carry non-answer-shaped partial products, so a final-target lens loss
conflicts with the computation itself. The question shifts from *when*
generalization happens to *whether the task is still learnable and where
the computation goes*.

**Design.** Reuse the lego harness (S3, k ∈ [0, 6], standard
128d/4h/8L, streaming `--generate-n 5000000`, batch 512 — known-good
defaults). Add a `--lens-aux` loss: via the existing
`forward_with_residuals`, at the *answer position only*, per-layer logit
lens (final norm + tied unembedding) CE against the **final answer**, at
every layer, weights uniform (or `linear`), scaled by
`--lens-aux-weight` (λ). This is deliberately narrower than grok_lens
(supervise one position, not the sequence): the `<op>` positions stay
unsupervised, so "dark space" = both the unread directions of the
answer-position residual *and* the other token positions.

**Conditions (quick first pass, 1 seed, extend if interesting):**
baseline (matched rerun), λ=0.3 uniform, λ=0.3 linear. Existing staircase
runs in lego's RESULTS.md give the third arm of the conceptual contrast
(per-layer supervision toward *trajectory* values) without new runs.

**Measurements:**
1. Capability: per-k test accuracy + convergence step vs baseline (the
   "does it still get the right answer at all" question).
2. Answer-position lens trajectory: per-layer lens accuracy AND
   per-layer lens distribution entropy (restricted to the 6 element
   tokens) at the answer position, per k. Defines a coalescence layer
   ℓ*(k) = first layer whose lens acc ≥ 0.95.
3. Dark space probe (post-hoc, reusing lego's tuned-lens/probe tooling):
   decompose the answer-position residual into the unembedding row-space
   vs its orthogonal complement; linear-probe trajectory[j] from each
   component. "Computation in dark space" = trajectory decodable from
   the orthogonal component while the lens shows uniform.

**Pre-registered predictions:**
- P1: capability largely intact (the computation can route through the
  unsupervised `<op>` positions), possibly slower convergence at high k.
  Failure mode worth watching: accuracy caps at low k.
- P2 (Brendan): at intermediate layers the answer-position lens shows
  ≈ uniform probabilities over the 6 group elements — which is exactly
  the CE-minimizing "answer-shaped but ignorant" output, so the aux loss
  actively sculpts it — while the real composition happens in dark space
  (other positions / lens-orthogonal directions).
- P3 (Brendan): at the final position the distribution coalesces cleanly:
  lens acc vs layer transitions sharply at ℓ*(k), monotone in layer
  (unlike baseline lens trajectories), with ℓ*(k) increasing in k — a
  staircase in k rather than in supervised targets.

**Budget:** 3 runs × (lego 5M-example run, a few hours each on the local
GPU) — queue via `./train.sh local lego`. Implementation ~40 lines in
`experiments/lego/data.py`/`training.py` + CLI flags, plus a small
analysis script for (2)/(3).

## Phase 6 (extension): optimizer robustness — Muon

Added 2026-07-19, after the 50k un-censoring runs. Slingshot instability
is an adaptive-optimizer phenomenon (Thilak et al.), and Muon is now used
at frontier scale (Kimi K2), so: is the sparse-circuit ↔ instability
correlation — and the aux loss's circuit-distributing effect — specific to
the AdamW regime? Prior work (arXiv:2504.16041) shows Muon accelerates
grokking *onset*; nobody has looked at post-grok stability or circuit
sparsity under Muon.

**Design.** `--optimizer muon`: hybrid Muon (2-D block weights; decoupled
wd = 0.05 so per-step shrinkage lr·wd = 1e-3 matches the AdamW arm) +
AdamW (embeds/unembed/norms/biases, canonical settings). 12 runs at 50k
steps: {L2, L3} × {λ=0, λ=0.3 uniform} × 3 seeds, named
`<cell>-s<seed>-muon-50k`.

**Pre-registered predictions:**
- P1: Muon baselines reach first-crossing faster than AdamW baselines
  (replicating 2504.16041 directionally).
- P2 (key): the stability ↔ circuit-sparsity correlation holds *within*
  Muon runs. If Muon baselines learn sparse circuits and don't slingshot,
  the brittleness story is AdamW-specific; if Muon's orthogonalized
  updates yield distributed circuits AND stability, the circuit-level
  story survives with the optimizer as another route to the same circuit
  property.
- P3: the aux loss still produces distributed circuits and ~100%
  occupancy under Muon (its circuit-selection effect is not
  optimizer-mediated).

## References

- Power et al. 2022 — Grokking (arXiv:2201.02177)
- Nanda et al. 2023 — Progress measures for grokking (arXiv:2301.05217)
- Lee et al. 2015 — Deeply-Supervised Nets
- LayerSkip (arXiv:2404.16710); CALM / Schuster et al. 2022; DistillLens
- Nostalgebraist 2020 — logit lens; Belrose et al. 2023 — tuned lens
- McGrath et al. 2023 — late-layer MLPs suppress the max-likelihood token

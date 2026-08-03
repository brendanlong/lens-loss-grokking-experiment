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

## Phase 7 (extension): the direction half of the dark-space prediction

Added 2026-08-01, after the Phase 5 held-out re-runs. Phase 5's dark-space
prediction named two places supervision could push necessary intermediate
computation: **other positions** and **unread directions of the supervised
position's residual**. The position half is answered (intermediates are
lens-decodable at the unsupervised `<op>` positions; the `<predict>` lens
below ℓ* reads near-uniform). The direction half is not: the aux models
provably compute intermediates (they solve k ≥ 3), but nobody has checked
whether those intermediates are *linearly recoverable* from the
`<predict>`-position residual at layers where the lens is blind.

**Design.** No training; the hosted `S3-std-8L-splitku-*` checkpoints.
For each layer ℓ and trajectory index j, fit a plain linear probe (no
bias, no norm — probe capacity controlled so the contrast is about the
representation) for trajectory[j] on the `<predict>`-position residual at
layer ℓ. Fit on each run's own train-split chains, evaluate on that run's
reconstructed held-out split (split seed = training seed). Compare probe
vs lens accuracy per layer, aux vs baseline, k ∈ {2, 4, 6}.

**Pre-registered interpretation:**
- probe ≫ lens below ℓ* ⇒ intermediates live in lens-invisible directions
  of the supervised position (dark space confirmed in the direction sense);
- probe ≈ lens (both near chance) ⇒ the intermediates genuinely live
  elsewhere (the `<op>` positions) and the supervised position's residual
  is answer-subspace-only.

## Phase 8 (extension): LEGO in a grokking regime

Added 2026-08-01. The stability findings are established only on
depth-1-sufficient tasks, where per-layer answer supervision is never in
tension with computation the model *needs*. LEGO is the task where that
tension exists, but its arms were trained only in a promptly-generalizing
regime (wd = 0, cosine LR, 80% of all chains — no memorization plateau,
no transition). Does the headline phenotype (delayed first grokking, then
near-absorbing stability; baseline sawtooth) survive when depth is
required?

**Design.**
1. **Regime search**: subsample the enumerated train split to a
   memorizable set (2k / 5k / 10k chains, per-k waterfill — the
   short-chain curriculum finding still applies) × weight decay
   (0.1 / 0.3 / 1.0), AdamW, constant LR, 50k-step budget, per-k test
   tracking (multi-hop may grok per-k in stages, a result on its own).
2. **If a grokking regime exists**: baseline vs aux λ = 0.3 uniform,
   3 seeds each; per-k first-crossing, dips, occupancy.

**Pre-registered predictions:**
- The LEGO model keeps LayerNorm, so the two-ingredient account (LN+wd
  churn × brittle circuit) predicts baseline instability *if* it grokks.
- Open question worth stating in advance: can the aux loss still
  stabilize when it cannot collapse the computation into one block? On
  the arithmetic tasks its stable solution was the shallow one — that
  exit is closed here. Aux failing to stabilize on LEGO would bound the
  mechanism's scope; succeeding would show redundancy maintenance works
  for genuinely deep circuits.

Small-strata caveat: at small train-set sizes the k ≤ 1 strata are tiny;
report per-k over k ≥ 2 and keep the split-seed = training-seed
convention so analyses reconstruct each run's split.

## Phase 9 (extension): full-sequence supervision — the realistic, adversarial form

Added 2026-08-02, correcting a misunderstanding in the Phase 5 design.
Phase 5 supervised the answer position only, which aligns the supervision
with the graded quantity — a setting where deep supervision turned out to
be benign-to-helpful. The original question (Brendan) was the opposite
one: in a *realistic* LM setup every position is trained next-token, so
apply the deep supervision to **all** positions and ask whether it hurts.

**Pre-registered predictions (Brendan):**
- P1: full-sequence deep supervision makes the task harder and may delay
  or break grokking in the Phase 8 regime (in contrast to answer-only
  aux, which accelerates it there).
- P2: under full-sequence supervision the logit lens shows the output
  token's progression extremely clearly — a broad distribution
  sharpening to the right token across layers — but shows the
  intermediate trajectory values *not at all*, even where linear probes
  recover them. (The op-position staircase visible in answer-only
  models' lenses should be erased: those positions' lens directions are
  now spent on next-token targets, which are uniform-random operands.)

**Design.**
- New `--base-loss all-positions`: final-layer next-token CE at every
  non-pad position (the answer remains readable as next-token at
  `<predict>`; per-k answer accuracy stays the capability metric).
- Arms, both with the full-sequence base loss: no aux vs
  `--lens-aux --lens-aux-mode all-positions --lens-aux-weight 0.3`.
- Grokking regime (Phase 8 cell: sub10000, wd 0.3, constant LR, 100k,
  3 seeds each) for P1; data-rich regime (canonical 40-epoch settings,
  seed 42 first) for P2's lens/probe analysis on models that certainly
  learn.
- Dose-response follow-up (added after the λ = 0.3 results): full-seq
  aux at λ ∈ {0.01, 0.1}, data-rich, seed 42. Question: is the harm
  presence-not-strength, like the answer-shaped benefits were on the
  arithmetic tasks, or dose-dependent? Open prediction either way —
  presence-not-strength would mean any per-layer noise-target pressure
  suffices to block the task; dose-dependence would locate a usable
  low-λ regime.
- Analysis, per (position, layer), on held-out chains (Brendan's
  three-readout spec):
  1. **Own-output progression**: lens top-1 vs that position's actual
     next token, plus lens entropy — expect a clean distribution →
     sharpened-token progression at every position (at positions whose
     next token is a uniform-random operand, "clean" means calibrated
     near-uniform over elements, the CE-optimal prediction).
  2. **Intermediates in the lens**: lens top-1 vs trajectory[j] for
     every intermediate j, at every position — expect maybe-visible in
     the no-aux baseline (the answer-only models' op-position staircase
     is the precedent), invisible under full-sequence aux.
  3. **Intermediates via probes**: plain linear probes for
     trajectory[j] at every position — expect recoverable in both
     arms, and at earlier layers under the aux loss.

## References

- Power et al. 2022 — Grokking (arXiv:2201.02177)
- Nanda et al. 2023 — Progress measures for grokking (arXiv:2301.05217)
- Lee et al. 2015 — Deeply-Supervised Nets
- LayerSkip (arXiv:2404.16710); CALM / Schuster et al. 2022; DistillLens
- Nostalgebraist 2020 — logit lens; Belrose et al. 2023 — tuned lens
- McGrath et al. 2023 — late-layer MLPs suppress the max-likelihood token

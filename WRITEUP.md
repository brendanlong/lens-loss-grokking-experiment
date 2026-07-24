# Grokking doesn't stick: deep supervision reveals — and counteracts — the instability of grokked circuits

*Technical writeup (2026-07-24). ~130 runs, all load-bearing results multi-seed, incorporating two adversarial substance reviews and the resulting controls (specificity, a second grokking task, power-matched knockouts, early-exit verification). Full per-seed tables, exact commands, and the complete correction log in [RESULTS.md](RESULTS.md); pre-registered predictions in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md); training curves in the [public wandb project](https://wandb.ai/brendanlong-com/grok-lens); checkpoints on the [HF dataset](https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment).*

## TL;DR

*Grokking*: on small algorithmic tasks, a network memorizes the training set almost immediately, sits at chance on held-out data for thousands of steps, then abruptly generalizes (Power et al. 2022). The canonical setup — modular addition with high weight decay — is a workhorse of mechanistic interpretability because the generalizing circuit is fully mapped (Nanda et al. 2023).

We added a **deep-supervision auxiliary loss** to this setup: every layer's residual stream is trained, through the model's own unembedding, to predict the known answer — the logit lens turned into a (supervised) training objective, as in LayerSkip/CALM. We predicted it would break grokking while making the logit lens artificially legible. What actually happened:

1. **A metric the field relies on misled us first.** Our initial result — "the aux loss delays grokking ~2×" — died under adversarial review: "grok step = first threshold crossing" credits a state the model doesn't hold. Measured properly, **baseline grokking is unstable**: models generalize, fall out of it, and regain it, dozens of times, indefinitely.
2. **The aux loss delays first grokking (~2× on average) but makes it stick.** This is a real cost in one regime — if you monitor and stop training at first grokking — and a large benefit in every other.
3. **The mechanism is redundancy, not calm.** Baselines end with the known sparse ~5-frequency circuit, which is one component-failure away from collapse; aux models keep the computation spread over ~18 frequencies. Component-level tracking shows substantial breakage in both — the aux models just have enough backups that in three seeds' worth of tracked training, *not one* of 159 component failures reached the output.
4. **The sparse circuit — and its fragility — is a *post-grok product of weight decay*, not what grokking finds.** Both arms reach first grokking with the same moderately distributed circuit (top-6 power share ≈ 0.4); weight decay's post-grok "cleanup" then prunes the baseline to the brittle sparse form (→ 0.75–0.91) while the aux loss holds concentration fixed (0.41–0.50). The aux loss doesn't build the redundancy — it *prevents the pruning* of what grokking naturally built. Consistent with this, switching it off lets wd prune the redundancy away (instability returns, 3/3 seeds), and switching it on late rebuilds redundancy only slowly — with stability tracking exactly how much gets rebuilt.

5. **The effect is specific to answer-shaped supervision, and general across tasks.** Reduced weight decay cannot reproduce it (it slows generalization and pruning *together*; the aux loss decouples them), and matched-form supervision against shuffled labels is actively destructive. The full story — instability, post-grok pruning, causal brittleness, aux stabilization with arrested pruning — replicates on a second task (modular subtraction, non-commutative, 3 seeds per arm), with a larger first-arrival cost on the harder task (3.2–4.7× vs ~2×).

Effect strength is not the knob: λ from 0.01 to 3.0 (a 1%-weight loss to a 300% one) produces the same delay, the same distributed circuit, and the same stability. Presence, not strength.

## Setup

**Task and training.** a + b (mod 113): all 12,769 ordered pairs, 30% train split, sequences `[a, b, =]`, answer read at `=`. Full-batch AdamW (lr 1e-3, β = (0.9, 0.98), weight decay 1.0, no clipping), 1–3-layer pre-LN transformers (d = 128, 4 heads, ReLU MLP), matching Nanda et al. (2023) except depth — their canonical model is 1-layer; intermediate-layer supervision needs ≥ 2, and depth turns out to matter (below).

**Aux loss.** At each non-final layer ℓ, project the answer-position residual through the final LayerNorm and the shared unembedding — exactly the logit lens — and add cross-entropy against the true answer: total = CE(final) + λ · Σ_ℓ w_ℓ · CE_ℓ, with w_ℓ either uniform (1/(L−1)) or later-weighted ("linear": w_ℓ ∝ ℓ+1, normalized; CALM's schedule).

**Metrics.** *First crossing*: first eval (100-step cadence) with test accuracy ≥ 0.95. *Dips*: post-first-crossing evals with accuracy < 0.90. *Occupancy*: fraction of post-first-crossing evals ≥ 0.95. (Two thresholds are used so occupancy isn't inflated by borderline evals; conclusions are unchanged at a single threshold — sensitivity note in RESULTS.) *Stable crossing*: first crossing after which accuracy never drops below 0.95 again — right-censored by the budget, which is exactly how it fooled us.

**Arms.** λ ∈ {0, 0.01, 0.1, 0.3, 1, 3} at 2 layers; a 1-layer control; a 3-layer uniform-vs-linear contrast (10 seeds); 50k-step baseline extensions; a Muon arm (hybrid: Newton–Schulz orthogonalized updates for 2-D block weights with decoupled wd = 0.05 matching AdamW's per-step shrinkage, AdamW for embeddings/norms); wd = 0 controls; per-frequency knockouts on every checkpoint; per-frequency power traces through training; and objective-switching continuations from trained checkpoints. 3 seeds per cell unless noted.

## Results

![Test accuracy over 50k steps: baseline runs collapse repeatedly; aux runs stabilize permanently](figures/occupancy.png)

*Figure 1 — the phenomenon. Baseline runs (vermillion) touch full test accuracy early and then collapse below 90% dozens of times, indefinitely; aux runs (blue) arrive later and never leave. Bold = seed 42, faint = seeds 43/44; all wd = 1.0, 50k steps.*

### Baseline grokking is unstable; first-crossing metrics hide it

Extended to a 50k-step budget, every AdamW baseline at 2–3 layers shows chronic post-grok collapse: 33–39 (L2) and 41–55 (L3) separate drops below 90% test accuracy, occupancy 0.86–0.90, with the apparent "stable point" always pinned to the end of whatever budget is used (pure censoring — at 30k it appears at ~29k, at 50k at ~49k). The same holds for Muon at 2 layers (41–50 collapses); the **Muon 3-layer baseline is the exception** — only 6–14 dips, occupancy 0.96–0.98 — so the precise scope of the instability claim is *AdamW at 2–3 layers and Muon at 2 layers*. These unstable models are still grokking models: they generalize, lose it, and regain it, repeatedly.

Why did nobody notice? Partly metric (first-crossing under-reports; see the correction log — an adversarial review of our own v1 caught us making exactly this error), and partly a fortunate canonical choice: Nanda et al.'s 1-layer model sits in the stable corner of depth-space. Our 1-layer control groks very late (13.2k / 19.5k / 27.1k across seeds, the last requiring a 50k budget) and then holds (0–1 dips; the 50k stability observation is single-seed).

Weight decay appears to be the driver on both sides of the ledger. At wd = 0, nothing groks at all — 0/6 runs in 50k steps, with or without the aux loss (which lifts the test plateau from ~0.20 to ~0.29, 3/3 seeds, but produces no transition): wd is the generalization driver, and the aux loss cannot substitute for it. And the continuation experiments below show wd actively eroding circuit redundancy when nothing opposes it. We could not measure post-grok stability at wd = 0 (nothing groks there), and slingshot oscillations are known to involve adaptive-optimizer dynamics generally (Thilak et al. 2022), so we scope the attribution as wd-plus-adaptive-optimizer.

### The aux loss: later first grokking, then (nearly) absorbing

| L2 AdamW cell | first crossing (per seed) | dips <0.90 | occupancy |
|---|---|---|---|
| baseline (50k budget) | 7.6k / 7.8k / 7.9k | 33–39 | 0.89–0.90 |
| λ = 0.01 | 14.7k / 16.9k / 20.4k | 0–3 | 0.98–0.99 |
| λ = 0.1 | 11.2k / 13.4k / 14.4k | 0–3 | 0.98–1.00 |
| λ = 0.3 | 11.5k / 13.7k / 19.4k | 0–3 | 0.98–0.99 |
| λ = 1.0 | 15.3k / 16.2k / 19.3k | 0–3 | 0.98–1.00 |
| λ = 3.0 (50k budget) | 16.2k / 20.3k / 27.2k | 0–3 | 0.98–0.99 |

(Dip counts are the verified across-cells range; per-cell tabulation in the analysis-script output.) Memorization is never affected (train accuracy ~100% by step 100–200 in every run in the study); full generalization is never prevented; the first-arrival delay is flat in λ across 2.5 orders of magnitude.

At 3 layers with 10 seeds, the baseline first-crosses at 3.1–3.7k with a mean of 22.4 collapses per 30k run (occupancy 0.90), while **uniform** aux delays first crossing to 5.1–16.4k with 0–2 dips (mean 0.7) and **linear** to 3.8–20.9k with **0–10** dips (mean 2.8). The linear arm's three high-dip seeds (10/6/7) are not noise — they are exactly the three aux runs whose circuits stayed sparse, and they carry the mechanism section below. The uniform-vs-linear first-arrival contrast itself is directional but marginal (uniform later on 8/10 seeds, sign test p = 0.055).

Under Muon, first grokking is ~2× faster for everyone at 2 layers (baseline 3.3–4.2k; replicating Tveit et al. 2025), the aux first-arrival delay reproduces (8.9–10.0k), and the L2 stabilization reproduces sharply (0/5/0 dips vs 41–50). At Muon-L3, aux gives no stability benefit (11–14 dips vs the baseline's 6–14) — the one standing exception, discussed below.

### Mechanism: redundant circuits under constant fire

**Sparse vs distributed (FFT).** AdamW baselines concentrate 92–97% of embedding Fourier power in ~5 frequencies — the known circuit. Every uniform-weighted aux model spreads 90% of power over 16–20 of 56 frequencies, identically from λ = 0.01 to 3.0.

![Knockout: baselines collapse when frequencies are deleted; aux models barely notice](figures/knockout.png)

*Figure 2 — causal brittleness. Bars are 3-seed means (dots = seeds): deleting the single worst frequency halves baseline accuracy on average; deleting the top six sends baselines to chance while aux models retain ~98%.*

**Brittleness is causal (knockout).** Projecting a *single* top frequency out of a baseline's embedding costs up to 90 points of test accuracy; removing its top six sends every AdamW baseline to chance. Removing the top six from an aux model leaves 83–100% accuracy — under both optimizers. This is not an artifact of "six" meaning different things per arm: at a *power-matched* ablation (removing top frequencies until 60% of embedding power is gone from each), aux models retain ~0.55 accuracy vs the baselines' ~0.12; at 90% removed both arms die — the distributed circuit is *wider and gracefully degrading*, not unkillable, which is the precise sense in which we use "redundant" (what is strictly shown is *distributed and knockout-robust*; whether it contains literal backup subcircuits or one widened algorithm is open). *Functional redundancy* (accuracy after top-6 removal) turns out to be the right circuit metric: raw spectral spread misleads (Muon baselines look diffuse in power but are functionally top-6 circuits — and at L2 they slingshot accordingly). Within AdamW, functional redundancy predicts post-grok stability across all 20 aux runs with no exceptions — including the three linear-arm seeds that kept sparse circuits despite the aux loss and were exactly the three unstable aux runs. And critically, the metric was then tested *out of sample*: in the rescue continuations (below), it predicted post-switch stability in a context it was not selected on. The correlation is clean *within AdamW only*: the Muon-L3 pair (functionally concentrated baseline that is fairly stable; fully redundant aux arm that is mildly unstable) shows Muon contributes optimizer-specific dynamics on both arms — the continuation experiments, which hold the optimizer fixed and flip only the objective, are what isolate the circuit's causal role from optimizer dynamics.

![Per-frequency churn under both regimes; only baseline failures reach the output](figures/churn.png)

*Figure 3 — churn without consequence. Per-frequency power shares (bottom; note the different y-scales — the aux model has no dominant components, max share ~0.1 vs ~0.7) churn in both regimes, but only the baseline's component failures appear in test accuracy (top).*

**Components break constantly either way** *(instrumented, 3 seeds)*. Logging per-frequency power at every eval: component collapse events (an active frequency losing >50% of its power between evals) occur throughout post-grok training in both regimes — baselines 55–76 events per run, aux runs 13–92 (rates vary widely by seed; our initial single-seed "same rate" observation was coincidence). The robust difference is the coupling to the output: **5–22 collapses per baseline run coincide with an accuracy collapse; 0 of 159 pooled aux collapses do, in every seed.** Consistent with this, aux models' logit margins are quiescent (median test loss ~10⁻⁴ in accuracy-intact evals vs ~10⁻² for baselines *between* their visible collapses). Stability is failure isolation — enough backups that they never all break at once — not the absence of damage.

**Where the sparse circuit comes from (formation traces, 3 seeds).** Tracking circuit concentration through *full* training answers what we had earlier flagged as the main open question — and refutes our own guess. We had speculated the dense per-layer gradient steers formation toward accumulating many weak components. It doesn't: **both arms reach first grokking with statistically indistinguishable concentration** (top-6 share at first crossing: baseline 0.39/0.48/0.43 vs aux 0.39/0.33/0.35). The divergence is entirely post-grok: baselines continue to concentrate under weight decay's pruning (→ 0.89/0.75/0.91 by 50k) while aux runs plateau (0.44/0.41/0.50). So grokking naturally discovers a distributed circuit; the celebrated sparse Fourier circuit of the interpretability literature is the product of the post-grok "cleanup" phase (Nanda et al.'s own term) — and that same cleanup is what creates the brittleness and the chronic instability. The aux loss is simply an anti-cleanup brake.

![Circuit concentration through training: identical until grokking, divergent after](figures/formation.png)

*Figure 4 — the resolution of the mechanism question. Top-6 frequency power share over training; dots mark first grokking. Both arms grok at concentration ≈ 0.4; weight decay then prunes baselines to ≈ 0.9 while the aux loss holds the line.*

**Maintenance and slow rebuilding (objective switching, 3 seeds).** From a stabilized aux checkpoint, continuing with λ = 0 lets weight decay prune the redundancy back to a functionally sparse circuit in every seed (all three aux-off finals collapse to chance on top-6 removal) and the instability returns (27–49 dips vs the same-source aux-on controls' 0–2), with dips *accelerating* as the ensemble erodes. So the aux gradient continuously replaces failing components against wd's pruning. In the reverse direction — λ = 0.3 switched on from a slingshotting baseline — the layer-0 readout aligns within 100 steps, but redundancy rebuilds only slowly and partially, at seed-dependent rates (accuracy-after-top-6-removal reaches just 0.01 / 0.15 / 0.44 in 30k steps, vs 0.83–1.0 for aux-from-scratch) — and **post-switch stability tracks the rebuilt redundancy monotonically** (24 / 4 / 1 dips) — an *out-of-sample prediction* by the functional-redundancy metric in a context it was not selected on, which is what defends it against a metric-fishing reading. One honesty note: the baseline-continuation controls are bursty (2–76 dips per 30k window — baseline instability is heavy-tailed), so short-window behavioral comparisons are noisy and the circuit metric is the reliable readout.

### Where the computation goes

The pre-registered "dark space" prediction — that per-layer answer supervision would push unrelated computation into lens-invisible directions — was **refuted**. In all 18 two-layer aux runs (every λ, both optimizers), the layer-0 lens alone scores 1.000 on held-out pairs and agrees with the final output; later blocks approach answer-preserving identities as λ grows (cos(r₀, r_final) 0.845 → 0.98). Since the layer-0 lens *is* the truncated model, every aux network — including the 3-layer ones, whose layer-0 lens is also 1.000 (a different, 30-run subset) — can be cut to one block at zero accuracy cost.

### Specificity: it's the answer-shaping, not generic regularization

Two controls separate "answer-shaped deep supervision" from "any pressure that opposes weight-decay sparsification":

- **Reduced weight decay is not a substitute.** wd = 0.5 baselines still prune to sparse (top-6 → 0.89–0.94) and still slingshot (14–20 dips); wd = 0.25 is stable-so-far only at 3.5× the first-grok cost, with the circuit *still sparsifying* at 50k (0.44 → 0.85 and rising) and a short, censored post-grok tail. Lowering wd slows generalization and pruning **together**; the aux loss uniquely **decouples** them (near-normal grokking speed, permanently arrested pruning).
- **Matched-form non-answer supervision is destructive, not stabilizing.** Supervising intermediate layers against a fixed random permutation of the labels (same functional form and magnitude) blocks generalization outright in 2/3 seeds and produces the least stable run in the entire study in the third (48 dips, occupancy 0.47) — while pinning the layer-0 lens to memorized noise (test lens accuracy 0.01). Despite low spectral concentration (~0.2), there is no stability — one more independent confirmation that *functional answer-circuit* redundancy, not diffuseness, is the operative property.

Answer-shapedness matters twice: generalization survives the supervision only when the targets are true, and the maintained redundancy is redundancy *of the answer circuit*.

### A second task: modular subtraction

Everything above could have been a fact about one circuit. On a − b (mod 113) — non-commutative (which also removes addition's commutative-pair leakage; its pre-grok test accuracy sits at true chance), known to grok more slowly — the complete baseline story replicates 3/3: first grok at 10–12k, chronic instability (33–72 dips, occupancy 0.78–0.90, *worse* than addition), post-grok sparsification (top-6 ≈ 0.5 at grok → 0.92–0.97), and causal knockout brittleness (top-6 removal → chance). The aux phenotype also replicates 3/3, at a higher price on the harder task: first grok at 33k / 52.5k / 58.2k (3.2–4.7× that task's baseline vs addition's ~2×), then near-absorbing (0–3 dips, occupancy 0.99–1.00) with pruning largely arrested (0.40 at grok → 0.56–0.70, vs baselines' 0.92–0.97).

![LEGO coalescence layer vs hop count for the three arms](figures/coalescence.png)

*Figure 5 — front-loading on the multi-hop task. First layer at which the lens reads the final answer, vs hop count (lines = 3-seed means, dots = seeds): baselines answer only in the last layers regardless of difficulty; aux models answer as early as each seed's circuit allows.*

**A second task, briefly** *(3 seeds per arm; the two newer seeds use streaming unique data, removing an earlier repeated-epoch deviation)*: on k-hop non-abelian group composition (S₃ "LEGO" chains, 8-layer models), the aux loss leaves capability untouched (100% at all k ≤ 6 in all 9 runs, equal convergence) and *front-loads* the computation rather than hiding it. Baselines only produce the answer in the final layers regardless of difficulty (coalescence layer ℓ* = 5–7 for k = 2–6, all seeds); aux models produce it 1–5 layers earlier at every seed and every k (k = 2 readable by layer 0–2; k = 6 by layer 3–6 depending on seed, vs always layer 7 for baselines), with intermediate states readable earlier at unsupervised positions, and below ℓ* the lens shows a calibrated anytime estimate where the baseline lens is confidently wrong. The one residue of the dark-space intuition: mid-chain prefix states are only ~half-decodable, hinting at hierarchical rather than fully sequential composition. This task corroborates the relocation/front-loading story only; it is not a grokking task and says nothing about stability.

## Discussion

**For grokking research**: report occupancy or stability-aware crossings, not first-crossing — first-crossing time stays decision-relevant only in the monitored-early-stopping regime, where the aux loss's ~2× delay is a genuine cost. Treat 1-layer post-grok behavior as the special case. "Unstable" does not mean "not grokking."

**Speculation, clearly labeled**: if grokking-like phase transitions occur during large-scale training and share this instability, models at scale could be acquiring and losing specific capabilities continuously under ongoing regularization pressure, with aggregate metrics too coarse to show it. We have no evidence beyond this toy setting; we flag it because the failure mode — capability churn masked by averaged evals — is structurally the same mistake our first-crossing metric made.

**For interpretability**: training a model to be lens-legible changed the object under study — the circuit's location, spectrum, and robustness class all moved. "Train for transparency" proposals should expect the object, not just the view, to move.

**For training practice**: a 1%-weight deep-supervision term acted as a strong implicit regularizer toward redundant, failure-isolating solutions — at the cost of ~2× later first generalization — and yields free early-exit/truncation structure (LayerSkip's use-case, rediscovered from the science side). Whether LayerSkip-style training quietly buys the same robustness at scale is an open empirical question.

## Open questions

1. **~~Why does answer-shaping induce redundancy at formation?~~ Resolved by the formation traces**: it doesn't — grokking finds the distributed circuit on its own, and the aux loss merely blocks the post-grok pruning that would concentrate it (our earlier formation-steering guess is refuted). The question this *opens* is sharper: **why does per-layer answer supervision stop weight decay's cleanup?** Mechanically, the aux gradient must penalize the pruning direction — plausibly because removing a frequency's embedding component transiently degrades the layer-0 readout before the remaining components compensate — but we have not verified this gradient-level account. Also unexplained: the wd = 0 plateau lift (+0.1 accuracy, 3/3). We deliberately did not reverse-engineer the distributed circuit's algorithm — the knockout/churn evidence carries the stability claims without it.
2. The Muon-L3 pair (stability decoupled from circuit redundancy in both directions).
3. Whether the sparse trap is escapable on longer horizons.

## Limitations

The stability story now spans two modular-arithmetic tasks but remains within one task *family* (the LEGO task corroborates only relocation/front-loading). The redundancy↔stability link is exception-free only within AdamW; Muon-L3 breaks it in both directions (the fixed-optimizer continuations are the rebuttal, but Muon's own dynamics remain unexplained). The weighting-schedule contrast is marginal (8/10 seeds, p = 0.055). The front-loading *magnitude* on LEGO is seed-varying (ℓ*(6) from 3 to 6), and subtraction-aux pruning is only *largely* arrested (concentration drifts to 0.56–0.70). "Redundant" is established at the embedding-frequency level and means *distributed and knockout-robust* — the algorithm on those frequencies is not reverse-engineered, so backup-subcircuits vs one-widened-algorithm is open. A residual puzzle we flag rather than resolve: if the aux gradient is a continuous brake on pruning, its effect should scale with λ, but concentration plateaus at the same level from λ = 0.01 to 3.0 — the maintenance account is descriptively right (the continuations prove it) while its gradient-level dynamics are not understood. Tiny models, grokking regime; transfer to realistic training is unknown.

## References

- Power, Burda, Edwards, Babuschkin, Misra (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177.
- Nanda, Chan, Lieberum, Smith, Steinhardt (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* arXiv:2301.05217.
- Liu, Michaud, Tegmark (2022). *Omnigrok: Grokking Beyond Algorithmic Data.* arXiv:2210.01117. (Weight-norm account of grokking; our wd-as-driver and wd-as-eroder findings sit in this territory.)
- Thilak, Littwin, Zhai, Saremi, Paiss, Susskind (2022). *The Slingshot Mechanism.* arXiv:2206.04817.
- Merrill, Tsilivis, Shukla (2023). *A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks.* arXiv:2303.11873. (Direct precedent for sparse-vs-distributed circuit competition.)
- Frankle, Carbin (2018). *The Lottery Ticket Hypothesis.* arXiv:1803.03635. (Sparse-winning-subnetwork framing; our results are a case where the sparse winner is the fragile one.)
- Srivastava et al. (2014). *Dropout.* JMLR. (Redundancy-as-robustness lineage.)
- Lee, Xie, Gallagher, Zhang, Tu (2015). *Deeply-Supervised Nets.* AISTATS; Szegedy et al. (2015). *Going Deeper with Convolutions.* CVPR. (Deep-supervision lineage.)
- Elhoushi et al. (2024). *LayerSkip.* arXiv:2404.16710; Schuster et al. (2022). *CALM.* NeurIPS. (The shared-unembedding early-exit loss and later-weighted schedule.)
- Tveit, Remseth, Skogvold (2025). *Muon Optimizer Accelerates Grokking.* arXiv:2504.16041.
- nostalgebraist (2020). *Interpreting GPT: the Logit Lens.* LessWrong.
- McGrath et al. (2023). *The Hydra Effect / late-layer suppression observations.* (Motivated the original hypothesis.)

## Reproducibility

All runs are logged to the [public wandb project](https://wandb.ai/brendanlong-com/grok-lens) with exact launch commands in [RESULTS.md](RESULTS.md); final checkpoints are on the [HF dataset](https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment); the analysis scripts in this repo (`grok_lens/analyze_*.py`, `grok_lens/make_figures.py`, `lego/compare_lens_aux.py`) reproduce every table and figure from those artifacts — see the [README](README.md) for the two-tier reproduction guide.

# grok_lens results — full experimental log

> **Note for the public release.** This log is kept verbatim as it was
> written during the experiment (it is the provenance record, including
> the correction lineage). The commands were run in a private research
> monorepo; to map them onto this repo:
> `experiments.grok_lens.X` → `grok_lens.X`, `experiments.lego.X` →
> `lego.X`, and the `./train.sh local grok_lens -- ARGS` SkyPilot wrapper
> → `uv run python -m grok_lens.train ARGS`. `s3://…` checkpoint URIs
> refer to the original private store; public copies of every referenced
> checkpoint are on the
> [HF dataset](https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment)
> under the same run names, and wandb run IDs link into the
> [public project](https://wandb.ai/brendanlong-com/grok-lens).
> `--save-checkpoint` in historical commands performed the private S3
> upload and has no equivalent here (checkpoints always save locally).
> Entries dated 2026-08-01 onward were run from this repo directly, via
> `skypilot/local.yaml` on the maintainers' local cluster — those
> commands are copy-pasteable as recorded (the S3 sync they reference is
> the same private store; public checkpoint copies are on the HF
> dataset under the same run names).

Does a logit-lens auxiliary loss — every layer's residual stream projected
through the shared unembedding and scored against the target — delay,
suppress, or accelerate grokking on modular addition? See
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for hypotheses and predictions.

## Runs

Every entry must include: the exact copy-pasted command, model config, batch
size (full-batch here — record the train-set size), LR, schedule, GPU, wandb
run ID, `memorize_step` / `grok_step`, and a result summary.

### 2026-07-18 — Phase 1 baseline (λ = 0), 3 seeds ✅ grokking reproduced

Ran directly on the local GPU (local SkyPilot's Kind cluster is down —
`kind-skypilot` context gone post-OS-upgrade, recovery needs brendanlong
privileges), sequentially in tmux:

```bash
for seed in 42 43 44; do
  run=p113-L2-lam0.0-uniform-frac0.3-s${seed}
  uv run python -m experiments.grok_lens.train \
    --seed ${seed} --save-checkpoint \
    --checkpoint-dir data/grok_lens/checkpoints/${run}
done
```

- **Config**: all defaults at commit `695f23f` — p=113, train_frac=0.3
  (3,831 train / 8,938 test pairs, full-batch), 2L/128d/4h ReLU transformer
  (531k params), AdamW lr=1e-3 constant (10-step warmup), β=(0.9, 0.98),
  weight_decay=1.0, no grad clipping, 30,000 full-batch steps, fp32,
  torch.compile, eval every 100 steps. aux_lambda=0 (clean baseline).
- **GPU**: local RTX 3060 Ti, ~11 min/run.
- **Runs** (wandb project `grok-lens`; checkpoints at
  `s3://brendanlong-experiments/grok_lens/checkpoints/<run_name>/final.pt`):

| seed | wandb ID | memorize_step | grok_step | final train/test acc |
|---|---|---|---|---|
| 42 | `u6kf97dt` | 100 | 8600 | 1.000 / 0.998 |
| 43 | `holfno24` | 100 | 8100 | 1.000 / 1.000 |
| 44 | `1t5nrb4p` | 100 | 7800 | 0.964 / 0.932 |

**Outcome: prediction 1 confirmed.** Textbook delayed generalization on all
three seeds: train accuracy hits 100% by step 100, test accuracy stays low
for thousands of steps, then jumps past 95% at step 7800–8600 (tight
across seeds — good news for resolving λ effects against seed variance).
Seed 44's final-step accuracies are lower because the last eval landed on a
post-grok slingshot fluctuation (common under wd=1.0); its curve sits near
1.0 after grokking like the others. Per-layer lens accuracy and weight-norm
trajectories are in wandb (`lens_test/acc_layer_*`, `weight_norm`).

Next: Phase 2 (mechanism check — moderate λ raises intermediate-layer lens
accuracy), then the Phase 3 λ × weighting sweep.

### 2026-07-18 — Phase 2 mechanism check (λ ∈ {0.3, 1.0}, 2 seeds) ✅ + grokking delayed in all 4 runs

Local SkyPilot recovered (Kind cluster rebuilt), so these went through the
normal path (jobs 2–5 on the `local-gpu` cluster):

```bash
for lam in 0.3 1.0; do for seed in 42 43; do
  run="p113-L2-lam${lam}-uniform-frac0.3-s${seed}"
  RUN_NAME="${run}" ./train.sh local grok_lens -- \
    --aux-lambda ${lam} --seed ${seed} --save-checkpoint
done; done
```

- **Config**: identical to Phase 1 (commit `889a5ff`, defaults) except
  `--aux-lambda`; uniform weighting (degenerate at L=2 — one intermediate
  layer), true-token targets. Full-batch 3,831 examples, 30k steps,
  RTX 3060 Ti, ~11.5 min/run.
- **Runs** (wandb `grok-lens`; checkpoints in S3 under the run name):

| λ | seed | wandb ID | grok_step (baseline same seed) | layer-0 lens test acc (max / final) | final test acc |
|---|---|---|---|---|---|
| 0.3 | 42 | `pauojiax` | **19400** (8600) | 1.000 / 1.000 | 1.000 |
| 0.3 | 43 | `1msd47ww` | **11500** (8100) | 1.000 / 1.000 | 1.000 |
| 1.0 | 42 | `pfj0wq6j` | **16200** (8600) | 1.000 / 1.000 | 1.000 |
| 1.0 | 43 | `4l3eq2o3` | **15300** (8100) | 1.000 / 1.000 | 1.000 |

Baseline (λ=0) layer-0 lens test accuracy for comparison: final 0.04–0.08
(near chance) across all three seeds, with a *transient* mid-training bump
that varies wildly by seed (max 0.345 / 0.688 / 0.966 around steps 11–14k,
decaying back toward chance by 30k).

**Outcomes:**

1. **Mechanism verified.** The aux loss does exactly what it claims: with
   λ ≥ 0.3 the intermediate layer becomes fully answer-shaped — layer-0
   logit-lens accuracy reaches 1.000 *on held-out test pairs* and stays
   there, vs near-chance final values at λ=0. Layer 0 alone effectively
   solves the task post-grok.
2. **Grokking delayed in 4/4 runs** (prediction 3, preliminary): grok_step
   1.4–2.3× the matched-seed baseline. Memorization unaffected (step 100
   everywhere), final test acc still 1.000 — the aux loss delays but does
   not suppress generalization at these λ.
3. **Dose-response is not monotone across these two λ values** (19400 vs
   16200 at s42, but 11500 vs 15300 at s43) — per-seed grok-step variance
   under aux loss looks larger than the λ=0.3→1.0 difference. The Phase 3
   sweep needs ≥3 seeds/cell and more λ values before any dose-response
   claim.

Interpretation guardrail: with only one intermediate layer at L=2, the aux
loss forces layer 0 to be the whole answer — the "iterative refinement vs
late-forming circuit" story really needs the L=3–4 runs where uniform vs
later-weighted schedules differ.

### 2026-07-18 — Layer-0 probe: where does the circuit live under aux loss?

Ad-hoc analysis (wandb histories + S3 checkpoints, λ=0.3 s42 vs baseline
s42): (1) in aux runs, layer-0 lens and final test accuracy cross 95%
*simultaneously* (within one 100-step eval interval) — one grok event, with
the circuit forming in layer 0; (2) in the aux checkpoint, layer-0 lens
gets 99.96% on test and agrees with the final argmax 99.96% of the time —
the answer computation is complete after one block — while block 1 still
writes a large residual update (‖Δ‖/‖r0‖ ≈ 0.98, cos(r0, r1) = 0.93) that
only sharpens confidence (top-1/top-2 margin 6.9 → 8.4); (3) in the
baseline checkpoint, layer-0 lens is at chance (3.8%) and cos(r0, r1) =
0.07 — there the answer direction is constructed by block 1. Motivated the
1-layer control below.

### 2026-07-18 — Phase 3 sweep: dose-response, 1L control, L=3 weighting arm (23 runs)

SkyPilot local jobs 6–28, RTX 3060 Ti, ~11.5 min per 30k-step run
(λ=3.0 runs used `--total-steps 50000`, ~19 min). Config identical to
Phase 1/2 (commit `204b9e4` defaults) except the flags shown. Launch
commands (RUN_NAME always set to the auto run name):

```bash
launch() { local run="$1"; shift; RUN_NAME="$run" ./train.sh local grok_lens -- "$@"; }
for s in 42 43 44; do launch "p113-L1-lam0.0-uniform-frac0.3-s$s" --n-layers 1 --seed $s --save-checkpoint; done
launch "p113-L2-lam0.3-uniform-frac0.3-s44" --aux-lambda 0.3 --seed 44 --save-checkpoint
launch "p113-L2-lam1.0-uniform-frac0.3-s44" --aux-lambda 1.0 --seed 44 --save-checkpoint
for lam in 0.01 0.1; do for s in 42 43 44; do launch "p113-L2-lam${lam}-uniform-frac0.3-s$s" --aux-lambda $lam --seed $s --save-checkpoint; done; done
for s in 42 43 44; do launch "p113-L2-lam3.0-uniform-frac0.3-s$s" --aux-lambda 3.0 --seed $s --total-steps 50000 --save-checkpoint; done
for s in 42 43 44; do launch "p113-L3-lam0.0-uniform-frac0.3-s$s" --n-layers 3 --seed $s --save-checkpoint; done
for s in 42 43 44; do launch "p113-L3-lam0.3-uniform-frac0.3-s$s" --n-layers 3 --aux-lambda 0.3 --seed $s --save-checkpoint; done
for s in 42 43 44; do launch "p113-L3-lam0.3-linear-frac0.3-s$s" --n-layers 3 --aux-lambda 0.3 --aux-weighting linear --seed $s --save-checkpoint; done
```

All runs reached final test acc 1.000 unless noted. `memorize_step` was 100
for every L2/L3 run and 200 for every L1 run — the aux loss never affects
memorization. wandb IDs per run are in the project (`grok-lens`), names
match `RUN_NAME`; checkpoints in S3 under the run name.

**grok_step by cell (seeds 42 / 43 / 44, mean):**

| cell | grok_step | mean | ×baseline |
|---|---|---|---|
| L2 λ=0 (baseline) | 8600 / 8100 / 7800 | 8167 | 1.0 |
| L2 λ=0.01 | 14700 / 16900 / 20400 | 17333 | 2.1 |
| L2 λ=0.1 | 14400 / 13400 / 11200 | 13000 | 1.6 |
| L2 λ=0.3 | 19400 / 11500 / 13700 | 14867 | 1.8 |
| L2 λ=1.0 | 16200 / 15300 / 19300 | 16933 | 2.1 |
| L2 λ=3.0 (50k budget) | 27200 / 16200 / 20300 | 21233 | 2.6 |
| L1 λ=0 (control) | >30000† / 19500 / 13200 | ≥20900 | ≥2.6 |
| L3 λ=0 (baseline) | 3200 / 3100 / 3700 | 3333 | 1.0 |
| L3 λ=0.3 uniform | 5100 / 16400 / 8300 | 9933 | 3.0 |
| L3 λ=0.3 linear | 3800 / 9200 / 4500 | 5833 | 1.75 |

† did not cross 0.95 in 30k steps; test acc 0.738 and climbing (slow grok,
not failure). wandb `xb299aba`.

**Findings:**

1. **The delay is a step function of the constraint being on, not its
   strength.** λ spanning 2.5 orders of magnitude (0.01 → 3.0) produces a
   statistically flat 1.6–2.6× delay (cell means overlap within seed
   spread). A lens loss at 1% of the main loss delays grokking as much as
   one at 300%. No λ suppresses grokking — every run reaches test acc 1.0
   (prediction 3: "delay" confirmed, "suppress" not observed up to λ=3).
2. **The 1L control largely explains the L=2 delay.** A bare 1-layer model
   groks slowly and erratically (13.2k / 19.5k / >30k) — the 1-block
   solution exists (cf. Nanda et al.'s 1-layer model) but is hard to find.
   Aux-L2 (which forces the answer computation into block 0; see layer-0
   probe above) groks at worst similarly and usually faster than bare 1L at
   every matched seed. So the aux delay ≈ the cost of the forced shallow
   circuit, partially offset by the dense supervision signal.
3. **Prediction 4 (weighting schedule is the pivotal knob) confirmed,
   paired across all 3 seeds at L=3**: uniform (early-layer pressure)
   delays 1.6× / 5.3× / 2.2×; CALM-style later-weighted delays only
   1.2× / 3.0× / 1.2× (uniform > linear at every seed). Early-layer
   answer-shaping is the harmful component, matching the McGrath-motivated
   hypothesis.
4. **Depth speeds up baseline grokking** (L3 baseline 3.3k vs L2 8.2k vs
   L1 ≥20.9k, decreasing variance with depth) — the aux loss pushes a model
   *down* this ladder toward its shallow-circuit behavior.
5. **Aux supervision inflates grok-time variance**: baseline seed ranges
   are tight (L2: 0.8k; L3: 0.6k) while aux cells span 5–11k. Deep
   supervision makes *when* the circuit forms much less predictable.

**Status of the central hypothesis:** supported in refined form. Per-layer
answer-shaping doesn't fight generalization per se — it relocates the
circuit into earlier layers (layer-0 probe), and the delay is the search
cost of that shallower circuit, driven by the *presence* of early-layer
pressure (findings 1, 3), not the loss magnitude.

Remaining: Phase 4 circuit check (FFT of embeddings across cells — same
Fourier circuit, or a different one?), then writeup.

### 2026-07-19 — Phase 4 circuit check: the aux loss learns a *less sparse* Fourier representation

```bash
uv run python -m experiments.grok_lens.analyze_fourier
```

Fourier power of the number-token embeddings for all 30 checkpoints:
share of (non-DC) power in the top-6 frequencies, and number of
frequencies needed for 90% of power (`n90`; 56 frequencies total, so a
uniform spectrum would give n90 ≈ 50). Cell means (per-seed values in the
script output):

| cell | top-6 power | n90 |
|---|---|---|
| L2 baseline | 0.922 | 5.3 |
| L2 λ=0.01 … 3.0 (5 cells) | 0.35–0.37 | 17.7–20.0 |
| L1 baseline | 0.324 | 28.3 |
| L3 baseline | 0.963 | 3.7 |
| L3 λ=0.3 uniform | 0.395 | 16.3 |
| L3 λ=0.3 linear | 0.787 | 8.3 |

**Findings:**

1. **Baselines show the textbook circuit** (Nanda et al.): 92–97% of
   embedding power in ~5 key frequencies.
2. **Every uniform-weighted aux cell learns a much less sparse
   representation** (top-6 ≈ 0.35, n90 ≈ 16–20) — and identically so at
   λ=0.01 and λ=3.0, mirroring the flat dose-response. This *revises the
   Phase 3 framing*: the aux loss doesn't relocate an unchanged circuit,
   it shifts the model to a more frequency-distributed solution. (Caveat:
   n90 ≈ 18 of 56 is still far from uniform — plausibly the same
   trig-multiplication algorithm spread over more frequencies rather than
   a different algorithm; a per-frequency logit attribution would settle
   this. Optional follow-up.)
3. **Bare 1L is similarly diffuse** (n90 20–45): aux-L2 matches the 1L
   control in *both* grok timing and circuit signature. There's a clean
   depth ladder — sparsity (n90 3.7 < 5.3 < 28.3) and grok speed (3.3k <
   8.2k < ≥21k) improve together with depth, and the aux loss pushes a
   model down it.
4. **Within L3-linear, circuit sparsity tracks grok timing exactly**: the
   two seeds that grokked at baseline speed (3.8k, 4.5k) kept the sparse
   baseline circuit (n90 = 6, top-6 ≈ 0.92); the one delayed seed (9.2k)
   is diffuse (n90 = 13, top-6 = 0.52). Later-weighting is gentler
   *because it usually leaves the circuit intact* — and when it doesn't,
   the delay reappears.

**Refined story for the writeup:** logit-lens deep supervision, at any
strength, knocks modular-addition grokking off the sparse-Fourier solution
onto a more distributed variant that is ~2× slower to generalize (but
still generalizes fully). The harmful ingredient is answer-pressure on
early layers; weighting supervision toward late layers mostly preserves
both the circuit and the timing.

> **Superseded in part** — see the metric-robustness correction below:
> the "~2× slower" timing claim is first-crossing-specific and does not
> survive a stability-aware grok metric.

### 2026-07-19 — Correction after adversarial validity review: grok timing is metric-dependent; the robust effect is slingshot damping

An independent validity review caught that `grok_step` (FIRST eval with
test acc ≥ 0.95) is unstable under wd = 1.0 slingshot dynamics, and that
this contaminates the headline delay claim. Verified independently with
full wandb histories (`analyze_stability.py`); numbers below reproduce.

```bash
uv run python -m experiments.grok_lens.analyze_stability
```

**The problem:** baselines keep collapsing after first crossing — L2
baseline seeds have 22 / 3 / 27 post-crossing evals below 0.90 (seed 44
*never* holds 0.95 to the end of its 30k budget), L3 baselines 15–40 —
while aux runs have 0–3 across every cell. First-crossing therefore
credits baselines with a "grok" they don't hold. Under a stable-crossing
metric (first eval after which acc never drops below 0.95 again), aux
runs stabilize at 12.7k–28.9k while L2/L3 baselines stabilize only at
28.0k–29.8k or never — but baseline stable values are **right-censored**
(they sit within a few evals of the 30k budget, so nothing remained to
disconfirm them; true values may be later). The censoring means we can
say aux runs reach durable generalization *no later* than baselines, but
not cleanly that they're faster.

**Claims revised accordingly:**

1. *"Aux delays grokking 1.6–2.6×, flat in λ"* → **downgraded**: the aux
   loss delays the *first arrival* of ≥95% test accuracy ~2× (flat in λ,
   still true and still interesting), but does **not** delay — and may
   accelerate — *durable* generalization. The robust, unambiguous effect
   is that the aux loss **damps post-grok slingshot instability** by an
   order of magnitude (dips <0.90 after first crossing: baselines 3–40,
   aux 0–3; final-5k-step fraction of evals ≥0.95: baselines 0.80–0.98,
   aux 0.96–1.00).
2. *"Never suppresses / all reach 1.0"* → holds, but restated on tail
   behavior (above) rather than the final-step sample, which is itself a
   slingshot draw (that's why Phase 1 s44 "final acc 0.932" looked odd).
3. *"Uniform > linear, paired 3/3"* → **downgraded to suggestive**: a
   3/3 sign agreement has p = 0.125 one-sided under the null, and the
   ordering does not persist under stability-aware metrics. Needs ~8–10
   seeds to test properly.
4. *"1L control explains the delay"* → survives only for first-arrival
   timing; inherits the metric caveat. The 1L cell is also n=3 with one
   censored run.
5. *"Circuit relocates into layer 0"* (Phase 2 probe) → mechanism solid
   (per-layer lens accuracies reproduce across all runs), but the
   block-1-only-sharpens / simultaneous-grok probe used one checkpoint
   pair (λ=0.3 s42) — generalizing it needs the probe run across seeds.
6. *FFT diffuseness* → the strongest evidence survives: within
   L3-linear, same 30k budget, the two fast seeds kept sparse circuits
   (n90 = 6) and the slow seed is diffuse (n90 = 13) — budget can't
   explain that. But the cross-cell baseline-vs-aux comparison is
   confounded by *time since stable grok* (aux runs had less post-grok
   cleanup time), and L1-vs-aux-L2 "same signature" confounds depth with
   the intervention. To de-confound: compare embeddings at a fixed
   number of steps past each run's own stable-grok point.

**Cheap decisive follow-ups (no conclusions should firm up without
them):** (a) extend the six L2/L3 baselines to 50k steps to un-censor
their stable-crossing (~2 h local queue); (b) rerun the layer-0 probe
across seeds/λ; (c) more seeds for the weighting contrast; (d)
fixed-steps-past-stable-grok FFT comparison.

### 2026-07-19 — 50k baseline extensions: baselines NEVER stabilize; stability tracks circuit sparsity

Follow-up (a) above, plus occupancy metrics ("what % of the time is the
model grokking", suggested by Brendan — censoring-robust because it
integrates the whole tail). Seven runs, SkyPilot local jobs 29–35, fresh
seeds-matched 50k-budget baselines named `<cell>-s<seed>-50k`:

```bash
launch() { local run="$1"; shift; RUN_NAME="$run" ./train.sh local grok_lens -- --wandb-run-name "$run" --total-steps 50000 --save-checkpoint "$@"; }
for s in 42 43 44; do launch "p113-L2-lam0.0-uniform-frac0.3-s${s}-50k" --seed $s; done
for s in 42 43 44; do launch "p113-L3-lam0.0-uniform-frac0.3-s${s}-50k" --n-layers 3 --seed $s; done
launch "p113-L1-lam0.0-uniform-frac0.3-s42-50k" --n-layers 1 --seed 42
```

Results (`analyze_stability.py`, now with occupancy + tail-10k loss):

| cell (50k budget) | first | dips <0.90 | occupancy | tail10k loss |
|---|---|---|---|---|
| L2 baseline s42/s43/s44 | 7.9k / 7.8k / 7.6k | 33 / 39 / 38 | 0.90 / 0.89 / 0.89 | 0.13 / 0.13 / 0.04 |
| L3 baseline s42/s43/s44 | 3.0k / 3.0k / 3.6k | 55 / 43 / 41 | 0.86 / 0.90 / 0.90 | 0.34 / 0.31 / 0.38 |
| L1 baseline s42 | 27.1k | 0 | 0.99 | 0.002 |

(For comparison, every aux cell: occupancy 0.98–1.00, dips 0–3, tail
losses ~0.001–0.02.)

**Findings:**

1. **The open question from the correction entry is resolved: L2/L3
   baselines never durably grok within 50k.** Their "stable at 28–29k" in
   the 30k runs was pure end-of-budget censoring — at 50k their
   stable-crossings simply move to 49.1–49.7k (the censoring artifact
   reproduced at the new budget) while they rack up 33–55 collapses.
   Baseline grokking at these hyperparameters is a chronically unstable
   state occupied ~86–90% of the time; aux grokking is an absorbing state
   entered at 13–29k. So "the aux loss delays stable grokking" is not
   just unproven — it's false on any budget we've measured: **there is no
   baseline stable grokking to delay.**
2. **Stability tracks the circuit, not the aux loss per se.** Bare L1
   (no aux, diffuse embeddings) groks late (27.1k) and then *holds* —
   0 dips, occupancy 0.99, exactly like the aux runs. Combined with the
   within-L3-linear split (its two sparse-circuit seeds have 10 and 6
   dips *despite the aux loss being on*; its one diffuse seed has 1),
   the pattern across every cell is: **sparse Fourier circuit ↔ slingshot
   instability; distributed circuit ↔ absorbing grokking** — regardless
   of whether the diffuseness came from the aux loss or from shallowness.
   The aux loss's causal role is (only) to reliably induce the
   distributed circuit; the stability is a property of the circuit.
3. Unified story, now well-supported: deep supervision at any strength
   biases learning toward a frequency-distributed circuit. Distributed
   circuits arrive later (~2× first-touch delay; harder search) but are
   an order of magnitude stabler under wd-driven slingshots; sparse
   circuits arrive early and never settle. The λ dose is irrelevant
   because the mediator is *which circuit forms*, which is binary-ish.
   Direct next test of the mechanism: per-frequency knockout — delete
   individual key frequencies from each checkpoint's embedding and
   measure the accuracy drop (brittleness should track sparsity).

### 2026-07-19 — Phase 6, Muon arm: aux effects replicate, but the circuit-mediation story breaks

12 runs (SkyPilot jobs 36–47, 50k steps): {L2, L3} × {λ=0, λ=0.3
uniform} × 3 seeds under hybrid Muon (`muon.py`: Newton-Schulz
orthogonalized updates for 2-D block weights, decoupled wd=0.05 matching
the AdamW arm's per-step shrinkage; AdamW for embeds/unembed/norms).
Pre-registered predictions in EXPERIMENT_PLAN.md Phase 6.

```bash
launch() { local run="$1"; shift; RUN_NAME="$run" ./train.sh local grok_lens -- --wandb-run-name "$run" --optimizer muon --total-steps 50000 --save-checkpoint "$@"; }
for s in 42 43 44; do launch "p113-L2-lam0.0-uniform-frac0.3-s${s}-muon-50k" --seed $s; done
for s in 42 43 44; do launch "p113-L2-lam0.3-uniform-frac0.3-s${s}-muon-50k" --aux-lambda 0.3 --seed $s; done
for s in 42 43 44; do launch "p113-L3-lam0.0-uniform-frac0.3-s${s}-muon-50k" --n-layers 3 --seed $s; done
for s in 42 43 44; do launch "p113-L3-lam0.3-uniform-frac0.3-s${s}-muon-50k" --n-layers 3 --aux-lambda 0.3 --seed $s; done
```

| cell (50k) | first-cross | dips <0.90 | occupancy | FFT n90 mean |
|---|---|---|---|---|
| L2 base muon | 3.3–4.2k | 41–50 | 0.88–0.90 | 36.3 |
| L2 λ0.3 muon | 8.9–10.0k | 0 / 5 / 0 | 0.98–1.00 | 42.3 |
| L3 base muon | 3.3–3.5k | 6–14 | 0.96–0.98 | 25.0 |
| L3 λ0.3 muon | 8.3–9.0k | 11–14 | 0.95–0.96 | 32.7 |
| (AdamW 50k L2 base) | 7.6–7.9k | 33–39 | 0.89–0.90 | 6.0 |
| (AdamW 50k L3 base) | 3.0–3.6k | 41–55 | 0.86–0.90 | 4.7 |

**Prediction verdicts:**

- **P1 (Muon accelerates first arrival): confirmed at L2** (3.8k vs 7.8k
  mean, ~2×, replicating arXiv:2504.16041 directionally), **no effect at
  L3** (3.4k vs 3.2k — already fast).
- **P2 (stability tracks circuit sparsity within Muon): REFUTED as a
  clean mechanism.** Muon L2 baselines have *diffuse* embedding spectra
  (n90 26–44 — Muon's orthogonalized updates spread energy, as
  anticipated) yet slingshot as violently as AdamW baselines (41–50
  dips). Diffuseness is NOT sufficient for stability. The unrefuted
  remainder is one-directional: every *sparse* run in the study (all
  under AdamW) is unstable, but no sparse Muon run exists to test the
  converse, and diffuse runs can be destabilized by other dynamics
  (Muon has its own instability modes — cf. MuonClip's raison d'être).
- **P3 (aux effects replicate): split.** The ~2.3–2.5× first-arrival
  delay replicates at both depths, and aux ⇒ even-more-diffuse spectra
  replicates. The *stabilization* replicates dramatically at L2 (0–5
  dips vs 41–50; two seeds stable at first crossing) but **not at L3**
  (11–14 dips vs the baseline's 6–14 — no benefit).

**Honest state of the mechanism story:** under AdamW the
sparse↔unstable correlation is exception-free, but the Muon arm shows
embedding-spectrum diffuseness alone doesn't confer stability, so
"the circuit property mediates everything" was too strong. What's
robust across both optimizers: (1) the aux loss delays first arrival
~2× regardless of λ/optimizer; (2) it never prevents full
generalization; (3) it reshapes the learned spectrum toward
distributed; (4) it strongly stabilizes post-grok behavior at L2. The
per-frequency knockout test is now the critical discriminator between
"sparse circuits are causally brittle" and "sparsity and instability
are co-symptoms of something else (e.g. optimizer dynamics)."

### 2026-07-20 — Per-frequency knockout: causal brittleness confirmed; functional redundancy supersedes the FFT-power metric

```bash
uv run python -m experiments.grok_lens.analyze_knockout
```

For every checkpoint: remove one top-power frequency at a time from the
number-token embeddings (project out its cos/sin components), measure
test-accuracy drop; also remove all top-6 at once. Cell summaries (full
per-seed output in the script):

| cell | max single-freq drop | acc after removing all top-6 |
|---|---|---|
| L2/L3 AdamW baselines (30k + 50k) | 0.16–0.90 (one seed 0.03) | **0.008–0.124 (chance)** |
| every aux cell, both optimizers | **0.000–0.001** | **0.83–1.00** |
| L2 Muon baselines | 0.24–0.41 | 0.013–0.043 (chance) |
| L3 Muon baselines | 0.04–0.09 | 0.034–0.069 (chance) |
| L3-linear sparse seeds (s42/s44) | 0.50 / 0.33 | 0.030 / 0.019 |
| L3-linear diffuse seed (s43) | 0.000 | 0.668 |
| L1 baselines (stable seeds) | 0.000 | 0.92–0.97 |

**Findings:**

1. **Sparse circuits are causally brittle; aux circuits are massively
   redundant.** Deleting ONE frequency costs AdamW baselines up to 90
   points of test accuracy; deleting the SIX highest-power frequencies
   costs aux models almost nothing (83–100% remaining, both
   optimizers). This is the direct causal version of the brittleness
   story the FFT could only correlate.
2. **The Muon-L2-baseline anomaly from Phase 6 resolves.** Their FFT
   *power* looked diffuse (n90 26–44), but *functionally* they are
   still top-6 circuits — removing those six sends them to chance. With
   "functional redundancy" (all-top6 accuracy) replacing FFT-n90 as the
   circuit metric, the instability correlation is restored: every
   functionally-concentrated run in the study slingshots; every
   functionally-redundant run is stable — with ONE residual exception,
   L3-aux-Muon (redundant yet 11–14 dips; Muon-specific dynamics).
3. The within-cell splits keep lining up: L3-linear's two sparse seeds
   are functionally concentrated (and were the unstable ones); its
   diffuse seed and the stable L1 seeds are redundant.

### 2026-07-20 — Layer-0 relocation probe generalized: 18/18, was n=1

```bash
uv run python -m experiments.grok_lens.analyze_layer0
```

Across ALL 18 L2 aux runs (λ ∈ {0.01…3.0} × 3 seeds + Muon λ=0.3):
layer-0 lens test accuracy = 1.000 and layer-0/final argmax agreement =
1.000 in every run; block 1 only sharpens margins (e.g. 6.4→11.3 at
λ=0.01, 19.0→19.4 at Muon λ=0.3). Baselines (both optimizers): layer-0
lens at 0.04–0.33, cos(r0, r_final) 0.07–0.52. The "one grok event"
claim also generalizes: layer-0 lens crosses 0.95 within 0–300 steps of
the final head in 17/18 runs (1,100 steps at λ=0.01 s42). Dose detail:
cos(r0, r_final) rises monotonically with λ (0.845 → 0.98), i.e. block 1
approaches an answer-preserving near-identity as aux pressure grows.
The Phase 2 n=1 caveat is closed.

### 2026-07-20 — Weighting contrast at 10 seeds: marginal; stabilization and circuit correlation now strong

Seeds 45–51 added to all three L3 cells (21 runs, SkyPilot jobs 48–68,
30k steps, same launch pattern as Phase 3 with `--seed $s`). First-cross
(`grok_step`), post-crossing dips <0.90, and occupancy from wandb
histories; knockout extended to the 14 new aux checkpoints.

| seed | base first | uniform first (dips) | linear first (dips) | unif > lin? |
|---|---|---|---|---|
| 42 | 3200 | 5100 (1) | 3800 (10) | yes |
| 43 | 3100 | 16400 (0) | 9200 (1) | yes |
| 44 | 3700 | 8300 (2) | 4500 (6) | yes |
| 45 | 3600 | 15400 (0) | 20900 (0) | no |
| 46 | 3100 | 5700 (2) | 5100 (1) | yes |
| 47 | 3600 | 8900 (1) | 10300 (0) | no |
| 48 | 3200 | 11500 (0) | 10400 (1) | yes |
| 49 | 3100 | 8900 (0) | 8800 (1) | yes |
| 50 | 3200 | 12300 (1) | 4800 (7) | yes |
| 51 | 3200 | 8800 (0) | 8200 (1) | yes |

Cell means: baseline first 3.3k (dips 22.4, occupancy 0.90); uniform
10.1k (dips 0.7, occ 0.99); linear 8.6k (dips 2.8, occ 0.99).

**Findings:**

1. **Prediction 4 lands as directional-but-modest, not pivotal.**
   Uniform delays first arrival more than linear on 8/10 seeds
   (one-sided sign test p = 0.0547 — the marginal edge of significance),
   but the mean gap (10.1k vs 8.6k) is small next to the shared
   aux-vs-baseline effect (both ~2.6–3.1× the 3.3k baseline). The
   presence of early-layer supervision matters far more than how it's
   weighted — consistent with the flat-λ result.
2. **The L3/AdamW stabilization result is now unambiguous at 10
   seeds**: baseline 22.4 mean collapses (occupancy 0.90) vs 0.7 / 2.8
   for the two aux arms (occupancy 0.99 both).
3. **The circuit-redundancy ↔ stability correlation extends to n = 20
   AdamW aux runs with zero exceptions.** Knockout on the 14 new
   checkpoints: exactly three aux runs in the study are functionally
   concentrated (all-top6 accuracy ≤ 0.042 — linear s42/s44/s50), and
   they are exactly the three unstable aux runs (6–10 dips); all 17
   functionally redundant runs have ≤ 2 dips.

### Conclusions (prediction scorecard)

49 + 21 = **70 runs** total. Against the pre-registered predictions:

1. Baseline grokking reproduced — **yes**, but the canonical picture is
   incomplete: first arrival is early and *chronically unstable* at 2–3
   layers (never durably grokking in 50k under either optimizer).
2. Small λ / later-weighted leaves timing roughly intact — **partly**:
   nothing about λ matters (flat over [0.01, 3]); later-weighting is
   only marginally gentler (p ≈ 0.055).
3. Strong early-layer weight delays or suppresses — **delays first
   arrival ~2–3×, never suppresses**; and under stability-aware metrics
   the aux loss is the only regime in which grokking *sticks*.
4. Weighting schedule pivotal — **downgraded**: directional but modest.
5. (self-distillation variant — dropped early as out of scope.)

The finding that outgrew the predictions: deep supervision reliably
induces a **functionally redundant** circuit (survives deletion of its
six largest frequencies) where baselines learn a brittle ~5-frequency
one (one deletion → collapse), and across all 20 AdamW aux runs circuit
redundancy predicts post-grok stability without exception. Muon
replicates everything at L2 but adds an unexplained mild instability at
L3 despite redundancy — the one open loose end, alongside the LEGO
"dark space" extension (Phase 5, planned).

### 2026-07-22 — Phase 5, LEGO multi-hop: computation doesn't hide, it front-loads

Implemented `--lens-aux` in the lego harness (per-layer logit-lens CE
against the FINAL answer at the `<predict>` position; commit `fc46525`
with unit tests) and ran three matched runs on the local cluster
(SkyPilot jobs 69–71, ~15 min each, wandb project `lego-reasoning`,
checkpoints `s3://…/lego/checkpoints/<run>/step_39000.pt`):

```bash
launch() { local run="$1"; shift; RUN_NAME="$run" ./train.sh local lego -- --wandb-run-name "$run" --save-checkpoint "$@"; }
launch "S3-std-8L-lensaux-base-s42"
launch "S3-std-8L-lensaux0.3-uniform-s42" --lens-aux --lens-aux-weight 0.3
launch "S3-std-8L-lensaux0.3-linear-s42" --lens-aux --lens-aux-weight 0.3 --lens-aux-weighting linear
```

Config: lego defaults at `fc46525` — S3, k ∈ [0, 6], std 128d/4h/8L,
batch 512, lr 3e-4 cosine. **Deviation note:** defaults meant
*fixed-dataset mode* (n_train=100k × 200 epochs = 39k steps), not
streaming — matched across arms and evaluated on held-out per-k test
sets, so the comparison is valid, but follow-up seeds should pass
`--generate-n`. Single seed (42) so far. Analysis:
`uv run python -m experiments.lego.compare_lens_aux`.

**P1 (capability survives): confirmed, emphatically.** All three runs
reach 100% test accuracy at every k ≤ 6; the aux runs converge as fast
as baseline (8–9k steps).

**P2 ("dark space"): refuted — the opposite happens.** The computation
doesn't retreat into lens-invisible directions; the *entire pipeline
moves earlier and stays lens-visible*. At `<op>` positions, trajectory
states that the baseline only exposes in layers 4–7 are readable from
layers 1–2 under aux (e.g. traj[1]: 0.90 at L2 vs baseline's 0.06).
One caveat matching the spirit of the prediction: mid-chain prefix
states plateau at ~0.5 decodability under uniform aux (vs ~0.87
baseline at L7) — the aux model appears not to compute *every*
sequential prefix, consistent with a parallel/hierarchical composition
rather than hidden computation.

**P3 (clean coalescence with k-dependent depth): confirmed, and
sharper than predicted.** Coalescence layer ℓ*(k) (first layer whose
`<predict>` lens reads the final answer at ≥95%), k = 1…6:

| | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| baseline | 5 | 5 | 6 | 6 | 7 | 7 |
| aux uniform | 0 | 0 | 1 | 2 | 2 | 3 |
| aux linear (k=2/4/6) | | 1 | | 2 | | 4 |

Both curves have the same ~k/2 slope — the network composes about two
operations per layer either way — but the aux loss removes the
baseline's ~5-layer preamble entirely: **the answer for a 6-hop problem
is fully formed by layer 3** (baseline: layer 7, the last one). Below
ℓ*, the aux model's lens shows exactly the predicted anytime estimate:
elevated element-entropy (up to 1.38 nats vs ln 6 ≈ 1.79; baseline
sits at 0.0–0.3, confidently wrong) collapsing monotonically to 0 at
ℓ*, with above-chance partial accuracy on the way (0.33 at L0 for
k=6). Later-weighted aux is intermediate: slightly later coalescence
(ℓ*(6)=4) and better-preserved prefix states at `<op>` (~0.77–0.99).

**Interpretation:** on a task with genuine intermediate quantities,
answer-position deep supervision doesn't fight the computation or push
it into dark space — it *front-loads* it, converting a
last-layers-only answer into an anytime, monotone estimate while
leaving capability untouched and making intermediate states *more*
visible earlier. Together with Phases 1–4: deep supervision's
consistent signature is relocation of computation toward the input end
plus circuit reorganization, never capability loss. Single seed —
seed expansion (with `--generate-n`) before any strong claims.

### 2026-07-22 — Continuation quartet: the aux loss *maintains* redundancy against weight decay; the sparse basin is a trap

Motivated by Brendan's landscape hypothesis (aux closes off unstable
solutions so the grokking basin can stabilize). Added `--resume-from`
(commit `2b3a2fd`; weights + model config from checkpoint, fresh
optimizer — hence paired same-source controls). Four 30k-step
continuations (jobs 72–75, seed 42):

```bash
S3P="s3://brendanlong-experiments/grok_lens/checkpoints"
launch() { local run="$1"; shift; RUN_NAME="$run" ./train.sh local grok_lens -- --wandb-run-name "$run" --seed 42 --total-steps 30000 --save-checkpoint "$@"; }
launch "p113-L2-cont-auxoff-from-lam0.3-s42" --resume-from "$S3P/p113-L2-lam0.3-uniform-frac0.3-s42/final.pt" --aux-lambda 0
launch "p113-L2-cont-auxon-from-lam0.3-s42"  --resume-from "$S3P/p113-L2-lam0.3-uniform-frac0.3-s42/final.pt" --aux-lambda 0.3
launch "p113-L2-cont-rescue-from-base-s42"   --resume-from "$S3P/p113-L2-lam0.0-uniform-frac0.3-s42-50k/final.pt" --aux-lambda 0.3
launch "p113-L2-cont-base-from-base-s42"     --resume-from "$S3P/p113-L2-lam0.0-uniform-frac0.3-s42-50k/final.pt" --aux-lambda 0
```

| arm | source | λ | dips <0.90 | final circuit: FFT top6 / n90 / all-top6-knockout acc |
|---|---|---|---|---|
| aux-off | stable aux ckpt | 0 | **49** (min acc 0.014) | **0.912 / 5 / 0.009 — re-sparsified** |
| aux-on control | stable aux ckpt | 0.3 | 1 | 0.416 / 16 / 0.958 — redundancy maintained |
| rescue | slingshotting base ckpt | 0.3 | 24 (occ 0.92) | **0.959 / 5 / 0.009 — still sparse** |
| base control | slingshotting base ckpt | 0 | 76 (occ 0.70) | 0.938 / 3 / 0.008 — sparse |

**Findings:**

1. **The redundant solution is NOT a stable attractor of the baseline
   objective — the aux loss actively maintains it.** Remove the loss
   and weight decay prunes the redundancy back to the canonical
   brittle ~5-frequency circuit within ~30k steps, with slingshots
   resuming on exactly the erosion timeline (first dip only at 4.3k;
   1 dip in the first 5k steps, 8 in the last 5k). The same-source
   aux-on control (1 dip, circuit unchanged) rules out the
   fresh-optimizer transient.
2. **Reframes baseline slingshotting**: under wd = 1.0 the flow
   perpetually seeks sparser circuits, overshoots into brittleness,
   collapses, rebuilds — the slingshot cycle *is* the sparsification
   dynamic. The aux term parks the model short of the brittle edge.
3. **The sparse basin is a trap (hysteresis).** Turning aux ON from a
   sparse baseline checkpoint aligns the layer-0 readout within 100
   steps (lens ≥0.95) but does NOT rebuild redundancy in 30k steps
   (final circuit fully sparse and knockout-brittle) and does not
   reach absorbing stability. Redundancy is built during circuit
   formation, not retrofitted: **alignment is cheap, redundancy is
   path-dependent.**
4. **There is a moderate active-damping component after all**: on the
   *same sparse circuit*, aux-on gives 24 dips vs the no-aux control's
   76 (occupancy 0.92 vs 0.70). So the full mechanism has three
   parts: (a) during grokking, aux pressure steers formation to the
   redundant circuit; (b) the redundant circuit is passively robust
   (knockout-flat basin) — the dominant term (1 dip vs 24); (c) the
   aux gradient also mildly damps oscillations on any circuit — real
   but insufficient alone (24 vs 0–1).

Single seed; the quartet should be replicated across seeds before the
writeup's mechanism section is updated wholesale.

### 2026-07-22 — wd=0 control: weight decay is the generalization driver; the aux loss cannot substitute

Six 50k-step runs (jobs 76–81): {λ=0, λ=0.3 uniform} × seeds {42, 43,
44} with `--weight-decay 0`, all else default.

```bash
launch() { local run="$1"; shift; RUN_NAME="$run" ./train.sh local grok_lens -- --wandb-run-name "$run" --weight-decay 0 --total-steps 50000 --save-checkpoint "$@"; }
for s in 42 43 44; do launch "p113-L2-lam0.0-uniform-frac0.3-s${s}-wd0-50k" --seed $s; done
for s in 42 43 44; do launch "p113-L2-lam0.3-uniform-frac0.3-s${s}-wd0-50k" --aux-lambda 0.3 --seed $s; done
```

Pre-registered predictions and outcomes:

1. *wd=0 baseline groks much later or not at all* — **confirmed**: 0/3
   seeds grok in 50k; memorization at step 100 as always, then a weak
   plateau (final test acc 0.192 / 0.215 / 0.196).
2. *wd=0 + aux* (the open cell) — **aux cannot substitute for wd**:
   0/3 grok; final test acc 0.297 / 0.291 / 0.297. The dense
   per-layer gradient consistently lifts the plateau (~0.29 vs ~0.20,
   3/3 seeds, layer-0 lens tracking test acc as usual) but does not
   create the phase transition.

**Takeaways:** (a) in this setup, weight decay is not merely an
accelerant — on a 50k budget it is effectively *the* generalization
driver (consistent with Power et al.'s finding that wd dominates
data-efficiency, and not inconsistent with grokking-without-wd being
possible on much longer horizons / other implicit mechanisms). (b) The
aux loss's role is therefore precisely scoped: it cannot drive
generalization, it *shapes and stabilizes* the generalization that wd
drives — capping wd's sparsification short of the brittle edge
(continuation quartet) while leaving its useful pressure intact. The
full picture: **wd giveth (generalization) and taketh away (post-grok
instability via sparsification overshoot); the aux term keeps the
first and blocks the second.** (c) The (unpredicted) plateau lift at
wd=0 (+0.1 acc, 3/3) hints the dense signal has some weak
generalization pressure of its own — noted, not pursued.

### 2026-07-22 — FFT-trace pair: stability is failure isolation, not calm (Brendan's masked-churn hypothesis CONFIRMED)

Hypothesis (Brendan): maybe redundancy doesn't remove the instability —
individual features keep breaking under the aux model too, and it's
just rare that enough break at once. Added `--log-fourier` (per-freq
embedding power at every eval, commit `a760990`) and ran an
instrumented pair (jobs 82–83, wd=1, 50k, seed 42; run names
`…-s42-ffttrace`). Loss-level pre-check on existing runs: aux models'
accuracy-intact evals sit at median test loss ~1e-4 with 0–2 mild
excursions vs baselines' ~1.5e-2 with 6–20 events >0.2 — no *margin*
strain, but output-level metrics can't exclude perfectly-compensated
churn. The traces can:

| post-grok window | baseline | aux λ=0.3 |
|---|---|---|
| active-freq collapse events (>50% power drop between evals) | 55 / 407 evals | 54 / 358 evals — **same rate** |
| …coinciding with test acc < 0.90 | **22** | **0** |
| consecutive top-10 Jaccard | 0.894 | 0.917 |
| top-6 component power CV | 0.45–1.88 | 0.16–0.27 |

**Component churn continues at an undiminished rate under the aux
model; none of it reaches the output.** *(Superseded in part by the
3-seed replication below: the "same rate" observation was seed-42
coincidence — churn rates vary widely — but the zero output-coupling
replicates perfectly.)* In the sparse baseline, 40% of
component collapses coincide with an accuracy collapse (with ~5
load-bearing components, one failure often suffices); in the redundant
circuit, 0/54 do. The aux ensemble has a steadier core (top-6 CV 5×
lower) with churn concentrated in peripheral components — but the
defining property is the decoupling of component failure from output
failure. **Stability = failure isolation, not quiescence.** This also
sharpens the aux-off continuation: removing the loss doesn't introduce
breakage (always ongoing), it stops the replacement of broken
components, so churn + wd pruning shrink the ensemble until single
failures are load-bearing again — matching the observed *accelerating*
dip rate. Single instrumented pair (seed 42); replicate before leaning
hard on the exact rates.

### 2026-07-23 — 18-run replication batch: everything holds, two claims sharpened

Jobs 84–101 (SkyPilot local): continuation quartet × seeds 43/44, churn
traces × seeds 43/44, LEGO trio × seeds 43/44 with streaming data
(`--generate-n 10000000`, removing the repeated-epoch deviation).
Commands identical to the original entries with `--seed`/names updated.

**Continuation quartet (now 3 seeds, dips <0.90 per seed 42/43/44):**

| arm | dips | circuit at end |
|---|---|---|
| aux-off | 49 / 27 / 40 | functionally sparse 3/3 (all-top6 → 0.009 each) |
| aux-on control | 1 / 0 / 2 | redundant, unchanged |
| rescue (aux-on from base) | 24 / 4 / 1 | all-top6 acc 0.009 / 0.151 / 0.439 |
| base-from-base control | 76 / 14 / 2 | sparse |

1. **Maintenance replicates 3/3**: removing the aux loss re-sparsifies
   the circuit functionally (every aux-off final collapses to chance on
   top-6 removal) and instability returns (27–49 dips vs controls' 0–2).
2. **The "sparse basin is a trap" claim softens**: rescues *can*
   partially rebuild redundancy at seed-dependent rates (all-top6
   0.009 → 0.151 → 0.439 across seeds), still far short of
   from-scratch aux (0.83–1.0) — and **stability tracks the rebuilt
   redundancy monotonically** (24 / 4 / 1 dips), extending the
   circuit↔stability coupling to a third independent context. Note the
   base-from-base controls are bursty (2–76 dips per 30k window):
   baseline instability is heavy-tailed, so short-window behavioral
   comparisons are noisy — the circuit metric is the reliable readout.

**Churn traces (now 3 seeds):** collapse events / post-grok evals —
baseline 55/407, 76/424, 65/419 with **5–22** coinciding with accuracy
collapse; aux 54/358, 13/383, 92/357 with **0** coinciding, every seed
(0/159 pooled). The seed-42 "same rate" observation was coincidence —
churn *rates* vary widely (13–92) — but the failure-isolation claim
(zero output-coupling under redundancy) replicates perfectly.

**LEGO (now 3 seeds; seeds 43/44 with streaming data):** 100% test
accuracy at every k ≤ 6 in all 6 new runs — the capability result is
now clean of the repeated-epoch confound. Coalescence layers ℓ*(k=2/4/6):
baselines 6/6/7 (both new seeds; s42 was 5/6/7) vs aux-uniform 1/3/5
(s43), 1/3/6 (s44) and aux-linear 2/4/5, 1/3/5. Front-loading
replicates at every seed and every k (aux always ≥1–5 layers earlier;
k=2 readable by layer 0–2 everywhere), but the *magnitude* is
seed-varying: the original "6-hop answer complete by layer 3" was the
strongest seed; across seeds ℓ*(6) ∈ {3, 5, 5, 6} vs baseline {7, 7, 7}.

### 2026-07-23 — Formation traces: the sparse circuit is a POST-GROK product of weight decay; figures added

```bash
uv run python -m experiments.grok_lens.make_figures
```

Circuit concentration (top-6 embedding power share) tracked through full
training in the six `-ffttrace` runs. **Both arms reach first grokking
with the same moderately distributed circuit** — top-6 share at first
crossing: baseline 0.39 / 0.48 / 0.43 vs aux 0.39 / 0.33 / 0.35. The
divergence is entirely post-grok: baselines concentrate to 0.89 / 0.75 /
0.91 by 50k under wd pruning while aux runs plateau at 0.44 / 0.41 /
0.50.

**This resolves the "why does the aux loss induce redundancy" open
question, refuting our own formation-steering hypothesis**: the aux loss
induces nothing at formation — grokking naturally discovers a
distributed circuit, weight decay's post-grok "cleanup" phase (Nanda et
al.'s term) prunes it into the canonical sparse form, and that pruning
is what creates the brittleness and chronic instability. The aux loss is
an anti-cleanup brake. Newly opened question: the gradient-level account
of *why* per-layer answer supervision blocks the pruning direction.

Also added `make_figures.py` (palette validated per the dataviz checks)
producing `figures/{occupancy,knockout,churn,coalescence,formation}.png`,
now embedded in WRITEUP.md (draft v4).

### 2026-07-23 — Reviewer-requested analyses: LEGO circularity-breaker + power-matched knockout

**LEGO early-exit agreement (breaks the ℓ* circularity).** The substance
review noted ℓ*(k) is lens-based and the lens is what the loss trains,
so front-loading needed the non-circular test mod-add got. On held-out
k=6 inputs, the lens at each aux run's ℓ* agrees with the final model's
prediction on **97.0 / 98.7 / 99.8%** of inputs (ℓ* = 3/5/6), with
post-ℓ* residual cosines 0.79–0.98 — an early exit at ℓ* recovers the
model's behavior on novel inputs. Front-loading is functional, not
definitional.

**Power-matched knockout (removes the "top-6 is definitional"
asymmetry).** Removing top frequencies until a matched *fraction of
embedding power* is gone from each arm:

| removed | baseline acc (2–4 freqs / 4–7 freqs) | aux acc (10–12 / 16–19 freqs) |
|---|---|---|
| ≥60% of power | 0.046 / 0.084 / 0.242 | 0.509 / 0.607 / 0.526 |
| ≥90% of power | 0.009 / 0.011 / 0.030 | 0.012 / 0.016 / 0.020 |

At matched 60% power removed, aux models retain ~0.55 accuracy vs the
baselines' ~0.12 — the robustness gap survives power-matching. At 90%
both arms die: the distributed circuit is wider, not unkillable, which
properly bounds the "redundancy" language (more components carrying the
same total function; graceful degradation at intermediate ablation
fractions, not immunity).

### 2026-07-23 — Specificity controls: the effect belongs to answer-shaped supervision specifically

Reviewer-requested controls (jobs 102–110, 50k steps, `--log-fourier`,
3 seeds each): does "just less weight decay" or generic (non-answer)
intermediate supervision reproduce the aux phenotype?

**wd sweep (λ=0):**

| wd | first grok | dips <0.90 | occupancy | top6 at grok → 50k |
|---|---|---|---|---|
| 1.0 (ref) | 7.6–7.9k | 33–39 | 0.89–0.90 | ~0.4 → 0.89–0.91 |
| 0.5 | 14.5–16.4k | 14–20 | 0.92–0.95 | 0.41–0.53 → 0.89–0.94 |
| 0.25 | 27.8–30.0k | 3 | 0.98–0.99 | 0.44–0.51 → **0.81–0.87 and rising** |

Lowering wd slows generalization and pruning *together*: wd=0.5 still
sparsifies fully and still slingshots; wd=0.25 is stable-so-far but at
3.5× the first-grok cost, with the circuit still sparsifying at 50k and
only ~20k of censored post-grok tail. **You cannot buy the aux
combination (near-normal grokking speed + permanently arrested pruning)
from wd alone** — the aux loss decouples what wd couples.

**Shuffled-target control (λ=0.3, intermediate CE vs a fixed random
label permutation — matched functional form/magnitude, not
answer-shaped):** s42 no grok in 50k (test 0.28); s43 groks at 26.2k
then is the least stable run in the study (48 dips, occupancy 0.47);
s44 no grok (0.74). Layer-0 lens on test = 0.01 in all three (the
intermediate layers memorize the noise mapping, which cannot
generalize), and despite low spectral concentration (top-6 0.16–0.24)
there is **no stability** — a further independent confirmation that
functional answer-circuit redundancy, not spectral diffuseness, is the
operative property.

**Specificity verdict: established.** Neither reduced weight decay nor
matched-form non-answer supervision reproduces the aux phenotype;
answer-shaped deep supervision specifically is what preserves the
distributed generalizing circuit. The "training for lens-legibility
changes the circuit" thesis stands on its own mechanism rather than as
a proxy for generic regularization.

### 2026-07-23 — Second task (a − b mod p): core phenomenon replicates 3/3; aux arm needs a longer budget

Jobs 111–116 (`--task sub`, 50k, `--log-fourier`, 3 seeds/arm; run names
`p113sub-*`). Subtraction is non-commutative (also removing the
commutative-twin caveat — its pre-grok test accuracy sits at true
chance, unlike addition's leakage-inflated ~0.28) and groks slower, as
in Power et al.

**Baselines — the full story generalizes, 3/3:** first grok
10.3–12.3k; chronically unstable (33–72 dips, occupancy 0.78–0.90 —
*worse* than addition); post-grok sparsification (top-6 ≈ 0.46–0.56 at
grok → 0.92–0.97 at 50k); knockout-brittle (max single-freq drop
0.48–0.86; all-top-6 → 0.009). Instability, cleanup-pruning, and causal
brittleness are not addition-specific.

**Aux λ=0.3 — bigger first-arrival delay on the harder task:** only
s43 grokked within 50k (at 33k) and shows the exact aux phenotype —
0 dips, occupancy 0.99, concentration arrested (0.39 → 0.56),
knockout-robust (max single drop 0.004, all-top-6 0.33). s42/s44 were
still mid-transition at 50k (test 0.63 / 0.17, climbing). 100k-budget
reruns queued (jobs 117–118) to characterize the cell; until then the
subtraction aux claim is scoped to "where it groks, the stable
arrested-pruning phenotype reproduces; first-arrival delay is larger
than on addition."

### 2026-07-24 — Subtraction aux cell complete at 100k: phenotype 3/3

Jobs 117–118 (100k budgets for the two mid-transition seeds). Combined
subtraction aux cell: first grok 33.0k / 58.2k / 52.5k (3.2–4.7× the
subtraction baselines — a larger first-arrival multiple than addition's
~2×, on the harder task), dips 0 / 1 / 3, occupancy 0.99–1.00,
concentration 0.39–0.41 at grok → 0.56–0.70 at end (partially arrested:
above addition-aux's 0.44–0.50 plateau but far below the subtraction
baselines' 0.92–0.97; stability holds regardless). **The second task now
replicates the complete story**: baseline instability + post-grok
sparsification + causal brittleness (3/3) and the aux
stabilization-with-arrested-pruning phenotype (3/3, at higher
first-arrival cost).

### 2026-07-27 — Review re-runs I: no-LN control, torch.optim.Muon swap, 500k long-horizon

Post-release review batch, run from this repo on the local SkyPilot
cluster via a thin queue wrapper around `skypilot/reproduce.yaml`'s
setup (one job per run; each job trains, logs to wandb, and syncs
`final.pt` + `train.log` to the private S3 mirror; curated checkpoints
land on the HF dataset). Commands per cell (seeds 42/43/44):

```
uv run python -m grok_lens.train --wandb-run-name <run> \
  --checkpoint-dir data/grok_lens/checkpoints/<run> --log-fourier \
  [--no-layernorm | --optimizer muon [--n-layers 3]] \
  [--aux-lambda 0.3] --seed <s> [--total-steps 50000 | --total-steps 500000]
```

**No-LN control (`*-noln-50k`, reviewer question: is the instability
LN-dependent?): YES — the headline instability requires LayerNorm.**

| cell (50k) | first grok | dips <0.90 | occupancy | top-6 power | knockout (worst single / top-6) |
|---|---|---|---|---|---|
| LN baseline (ref) | 7.6–7.9k | 33–55 | 0.86–0.90 | ~0.9 | −90pts / chance |
| **no-LN baseline** | 7.7–14.1k | **1–4** | **0.99** | 0.95–0.97 | 0.10–0.23 acc / 0.008–0.011 |

No-LN baselines end *sparser* and just as ablation-brittle, yet train
stably — so static brittleness ≠ training instability; the churn that
topples the sparse circuit is LN-mediated. Consistent with the
scale-invariance literature (van Laarhoven 2017; Zhang et al. 2019;
Li & Arora 2020; Lobacheva et al. 2021 periodic destabilization): wd
shrinks norms of LN-invariant weights, effective LR grows, and
post-grok (near-zero gradients) nothing opposes the decay. Nanda et
al.'s architecture is LN-free — this, rather than budget censoring, is
the clean explanation for why the literature didn't report the
sawtooth. Writeup rescoped accordingly: the instability claim is about
LN networks under the canonical recipe; the aux loss stabilizes from
the circuit side.

**torch.optim.Muon re-runs (`*-torchmuon-50k`, 12 runs): the hand-rolled
Muon arm replicates cell-for-cell on the stock optimizer** (torch 2.13's
`torch.optim.Muon`; `muon.py` is now only the parameter split). These
runs supersede the `*-muon-50k` arm as canonical.

| cell (50k) | first grok | dips <0.90 | occupancy | (hand-rolled ref) |
|---|---|---|---|---|
| L2 base | 3.3–3.5k | 39–48 | 0.89–0.91 | 41–50 / 0.88–0.90 |
| L2 λ0.3 | 8.2–9.9k | 3–5 | 0.98–0.99 | 0–5 / 0.98–1.00 |
| L3 base | 3.3–3.6k | 7–17 | 0.95–0.98 | 6–14 / 0.96–0.98 |
| L3 λ0.3 | 7.8–9.2k | 11–15 | 0.96 | 11–14 / 0.95–0.96 |

The L3-aux-Muon no-benefit anomaly reproduces exactly.

**500k long-horizon (`*-s42-500k`, both arms): the asymmetry grows with
budget.** At full eval resolution (5,000 evals via `scan_history` —
first-pass numbers used wandb's default 500-sample downsampling, an
adversarial-review catch; corrected same day): baseline 270 dips,
occupancy 0.930, **never stabilizes** (no suffix of the run stays
≥0.95; still dipping in the final 5k). Aux λ0.3: 13 dips over 500k —
20× fewer — occupancy 0.996, ≥0.95 without exception only from ~459k
(right-censored). Per-50k dip rate: baseline ~27 (consistent with the
50k cells' 33–39), aux ~1.3 (consistent with 0–3). So the aux arm is
*not* strictly absorbing at this horizon — it dips rarely but
indefinitely — while the baseline never approaches stability; the
50k-budget contrast extrapolates cleanly.

Provenance notes: job 119 (noln s42) shows FAILED in the queue — an
S3-credentials issue in the sync trap after training completed; its
checkpoint was recovered manually and verified. A tmux-launched
duplicate of the same run (killed mid-flight at 43k when the queue
moved to SkyPilot) is renamed `*-aborted-tmux` in wandb.

### 2026-07-27 — Review re-runs II: LEGO on a corrected split; honest numbers shrink but don't kill the story

Reviewer-caught validity bug: the original LEGO runs sampled i.i.d.
from only 335,922 possible chains, so test sets overlapped the training
stream's coverage — "test accuracy" partly measured seen chains. The
pipeline was rebuilt (this repo, commits f0b7ab8→d09300f): full
enumeration, disjoint per-k-stratified split (test_frac 0.2, seed 42),
map-style dataset, streaming machinery deleted. Two failed intermediate
states are preserved for provenance: the first 11 re-runs
(`*-split-*-kprop-failed` in wandb) sampled chains proportionally to
the enumeration (83% k=6) and **never lifted off chance** (train acc
1/6, loss = ln 6, 21k steps) — the short-chain curriculum is required
for learnability, so `make_k_uniform_sampler` now restores the old
per-k-uniform training distribution over the train split only.

Canonical runs (`S3-std-8L-splitku-*`, jobs 147–157, 20,960 steps =
40 epochs × 524, batch 512, lr 3e-4 cosine, seeds 42/43/44; note
`--seed` seeds the train/test split too, so **each run has its own
split** — analyses must reconstruct the split per run):

```
uv run python -m lego.train --wandb-run-name <run> --checkpoint-dir <dir> \
  [--lens-aux --lens-aux-weight 0.3 [--lens-aux-weighting linear] \
   [--lens-aux-mode all-positions]] --seed <s>
```

**Held-out accuracy** (per-k test strata: 1/7/43/259/1.5k/9.3k/56k examples):

| arm | train | k0 | k1 | k2 | k3–k6 |
|---|---|---|---|---|---|
| baseline | 1.000 | 1/0/1 | 0.71–1.0 | 1.00 | 1.00 |
| aux uniform | 1.000 | **0/0/0** | 0.43–0.86 | 0.86–0.98 | ~1.00 |
| aux linear | 0.90–1.00 | 0/1/0 | 0.43 | 0.81–0.98 | s43 1.0; s42/s44 partial |
| aux all-positions | 0.66–0.72 | 0 | 0.14 | 0.09–0.37 | 0.12–0.65 |

Baselines were genuinely generalizing all along (100% at k≥2). The aux
loss has a real short-chain cost the leaky eval hid (the held-out k=0
identity chain fails 3/3 uniform seeds; k1 degraded). The
reviewer-requested **all-positions variant** (next-token lens CE at
every position) substantially harms the task at matched budget —
answer-shaped supervision is the benign form, consistent with the
specificity controls.

**Held-out coalescence ℓ*(k)** (compare_lens_aux, each run probed on
ITS OWN reconstructed split — a first pass probed all runs on the
seed-42 split, putting ~80% trained-on chains in the s43/s44 probe
sets; adversarial-review catch, corrected same day. None = no layer,
incl. the last, reaches 95% on that run's held-out chains):

| arm | ℓ*(2) | ℓ*(4) | ℓ*(6) |
|---|---|---|---|
| baseline | 6/5/6 | 7/7/6 | 7/7/7 |
| aux uniform | 5/None/2 | 5/4/3 | 7/6/5 |
| aux linear | None/None/3 | 5/4/4 | None/5/None |

Front-loading survives held-out evaluation but is again smaller than
first reported: it is **robust at k=4** (aux 3–5 vs baseline 6–7, all
seeds), present at k=6 where the run converges (5–6 vs 7, uniform
s43/s44), and **largely undefined at k=2** — most aux runs' held-out
k=2 accuracy itself is below 0.95 (the short-chain cost), which caps
lens readability at every layer, so their ℓ*(2) is None rather than
early. The leaky evals' "k=2 readable at layer 0–1" was memorization
read through the lens. The anytime-estimate contrast is confirmed on
held-out chains: pre-ℓ* aux lens entropy ≈ ln 6 (calibrated
uncertainty), baseline entropy 0.1–0.5 at accuracy 0.0 (confidently
wrong). coalescence.png regenerated (per-run splits; means over
converged seeds; non-converged marked "n/r"); WRITEUP LEGO section
restated accordingly.

### 2026-08-01 — Phase 7, direction probes at the supervised position: dark space confirmed in the direction sense; the aux loss aligns the *answer* with the lens

The direction half of the Phase 5 dark-space prediction (EXPERIMENT_PLAN
Phase 7), on the 9 hosted `S3-std-8L-splitku-*` checkpoints. Per layer ℓ
and trajectory index j: plain linear probe (no bias, no norm) for
trajectory[j] on the `<predict>`-position residual at ℓ, fit on each
run's own train-split chains (≤4096 per k), evaluated on that run's own
reconstructed held-out split (≤1024 per k; split seed = training seed).
Zero-init full-batch Adam on the convex objective — deterministic. Run
via SkyPilot on the local cluster (job 158, RTX 3060 Ti, ~8 min):

```
sky exec local-gpu skypilot/local.yaml -d \
  --env RUN_NAME=S3-probes-predictpos \
  --env RUN_CMD='uv run python -m lego.analyze_probes --json-out "${RESULTS_DIR}/probes.json"' \
  --secret WANDB_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY
```

**Intermediate states (j ∈ 1..k−1) at layers below that run's ℓ\*(k)**,
held-out accuracy (per-seed s42/s43/s44; chance 0.17):

| arm | k | probe max | lens max | probe mean | lens mean |
|---|---|---|---|---|---|
| baseline | 2 | 0.42/0.49/0.58 | 0.09/0.12/0.14 | 0.33/0.30/0.43 | 0.02/0.03/0.02 |
| baseline | 4 | 0.94/0.93/0.75 | 0.17/0.18/0.18 | 0.37/0.50/0.40 | 0.04/0.05/0.04 |
| baseline | 6 | 0.55/0.68/0.61 | 0.22/0.24/0.23 | 0.26/0.31/0.29 | 0.05/0.08/0.08 |
| aux uniform | 2 | 0.56/0.65/0.47 | 0.16/0.42/0.30 | 0.44/0.50/0.45 | 0.13/0.22/0.17 |
| aux uniform | 4 | 0.95/0.66/0.52 | 0.24/0.22/0.34 | 0.45/0.35/0.25 | 0.17/0.18/0.18 |
| aux uniform | 6 | 0.72/0.74/0.72 | 0.21/0.23/0.28 | 0.29/0.32/0.27 | 0.17/0.17/0.17 |
| aux linear | 2 | 0.70/0.53/0.56 | 0.19/0.33/0.09 | 0.53/0.43/0.41 | 0.13/0.21/0.08 |
| aux linear | 4 | 0.81/0.62/0.67 | 0.24/0.25/0.29 | 0.37/0.33/0.39 | 0.18/0.18/0.18 |
| aux linear | 6 | 0.73/0.72/0.88 | 0.21/0.20/0.26 | 0.30/0.29/0.31 | 0.17/0.17/0.16 |

(For runs whose held-out lens never reaches 0.95 at any layer — the
non-converged aux cells — "below ℓ\*" means all 8 layers.)

**The answer itself (j = k) at layers below ℓ\***, probe vs lens max:

| arm | k=2 | k=4 | k=6 |
|---|---|---|---|
| baseline | 1.00/0.98/1.00 vs 0.81/0.70/0.93 | 1.00/0.97/0.94 vs 0.91/0.84/0.62 | 0.74/0.71/0.83 vs 0.59/0.68/0.81 |
| aux uniform | 0.93/0.91/0.16 vs 0.91/0.88/0.19 | 0.95/0.73/0.43 vs 0.94/0.73/0.42 | 0.37/0.87/0.70 vs 0.37/0.87/0.73 |
| aux linear | 0.84/0.95/0.58 vs 0.86/0.91/0.63 | 0.84/0.65/0.72 vs 0.83/0.64/0.72 | 0.51/0.74/0.92 vs 0.54/0.76/0.93 |

**Findings.**
1. **Probe ≫ lens below ℓ\* in every run and every k — dark space
   confirmed in the direction sense, in both arms.** Intermediate
   trajectory states are substantially linearly recoverable from the
   supervised position's residual at layers where the lens reads
   near-chance (best cells 0.55–0.95 vs lens ≤ 0.34 at k ∈ {4, 6};
   the tiny k=2 stratum reaches lens 0.42). The supervised
   position is not answer-subspace-only: it carries intermediates in
   lens-invisible directions. This is a property of the architecture/task,
   not of the aux loss — baselines show it at least as strongly.
2. **The aux-specific effect is on the answer direction, not the
   intermediates.** In baselines the final answer is linearly present
   well below ℓ\* (probe up to 1.00 where the lens reads 0.62–0.93 at
   best — the lens *under-reports* the baseline's answer); in aux runs
   probe ≈ lens for the answer at every layer (|gap| ≤ 0.05 in all 18
   aux seed-cells vs gaps up to 0.32 in baselines). Per-layer answer
   supervision aligns the answer
   information with the unembedding as soon as it exists — it makes the
   lens a *faithful* readout of the answer while leaving the
   intermediates in dark directions.
3. Decodability of intermediates is partial (probe max 0.4–0.95, not
   1.0) and peaks in the staircase region (late-middle layers, later j
   at higher layers), consistent with the `<op>`-position staircase.

**Caveats.** k=2 probes fit on only 173 train chains (768 probe params):
fit accuracy 1.00 vs held-out ~0.4–0.7 — overfit, so k=2 probe values
are noisy lower bounds on linear decodability; k=4/k=6 probes (4096 fit
chains) have small fit–eval gaps (e.g. 0.99 vs 0.94). The lens argmaxes
over the full 10-token vocab while probes are 6-way; this asymmetry is
inherent to the pre-registered contrast and can only *help* the lens on
the intermediate rows (its near-chance values are not an artifact).

### 2026-08-01 — Phase 8 regime search: LEGO grokks, per-k, in stages — and the baseline sawtooth appears on a depth-requiring task

Grokking-regime search (EXPERIMENT_PLAN Phase 8 step 1): train on a
small per-k-waterfill subsample of the train split (`--train-subset`),
AdamW constant LR 3e-4, batch 512, 50k steps, eval on the full held-out
split every 500 steps. 3 subset sizes × 3 weight decays, seed 42.
SkyPilot jobs 159–167 on the local cluster (~24 min each, RTX 3060 Ti):

```
for n in 2000 5000 10000; do for wd in 0.1 0.3 1.0; do
  run="S3-grok-sub${n}-wd${wd}-s42-50k"
  sky exec local-gpu skypilot/local.yaml -d \
    --env RUN_NAME="$run" \
    --env RUN_CMD="uv run python -m lego.train --train-subset ${n} --weight-decay ${wd} --lr-schedule constant --total-steps 50000 --eval-every-steps 500 --seed 42 --wandb-project grok-lens --wandb-run-name ${run} --checkpoint-dir \"\${CHECKPOINT_DIR}\"" \
    --secret WANDB_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY
done; done
```

Per-k metrics from `lego.analyze_grok_stability` (first crossing ≥ 0.95
/ dips < 0.90 after it / occupancy; wandb IDs in the project under the
run names):

| cell | memorize | k2 | k3 | k4 | k5 | k6 |
|---|---|---|---|---|---|---|
| sub2000-wd0.1 | 2000 | — (0.35) | — (0.24) | — (0.32) | — (0.33) | — (0.32) |
| sub2000-wd0.3 | 2300 | — (0.33) | — (0.23) | — (0.31) | — (0.32) | — (0.30) |
| sub2000-wd1.0 | 5400 | — (0.54) | — (0.29) | — (0.31) | — (0.33) | — (0.33) |
| sub5000-wd0.1 | 4000 | — (0.63) | — (0.46) | — (0.37) | — (0.32) | — (0.32) |
| sub5000-wd0.3 | 5300 | — (0.63) | — (0.44) | — (0.35) | — (0.31) | — (0.31) |
| sub5000-wd1.0 | 18800 | 21.5k, 19 dips, occ 0.19 | — (0.77) | — (0.60) | — (0.40) | — (0.38) |
| sub10000-wd0.1 | 6400 | 13.5k, 0, occ 0.89 | 13.5k, 0, occ 1.00 | 21.5k, 0, occ 0.98 | 33.5k, 0, occ 0.97 | 45k, 0, occ 0.55 |
| sub10000-wd0.3 | 9400 | 16k, 10 dips, occ 0.49 | 32.5k, 0, occ 0.83 | — (0.90) | — (0.72) | — (0.34) |
| sub10000-wd1.0 | 19000 | 12.5k, 34 dips, occ 0.53 | 12.5k, 35 dips, occ 0.53 | 15k, 34 dips, occ 0.45 | — (0.49) | — (0.41) |

("—" = never crosses 0.95 in 50k; parenthesis = final accuracy.)

**Findings.**
1. **A grokking regime exists**: at 10k chains the model memorizes
   (train 100% by 6–19k steps) then generalizes *per-k in stages* —
   k2/k3 first, k6 last (sub10000-wd0.1: 13.5k → 45k) — the staged
   transition the plan flagged as a result on its own. At 2k chains
   nothing crosses (test drifts to ~0.3); 5k chains is marginal.
2. **The baseline sawtooth appears on the depth-requiring task**, and
   weight decay modulates it exactly as the two-ingredient account
   predicts: wd 0.1 is near-stable post-crossing (0 dips), wd 0.3 gives
   k2 10 dips / occupancy 0.49, wd 1.0 gives chronic instability (34–35
   dips per crossed k, occupancy ~0.5, final accuracy *degrading* to
   0.61–0.86). The LEGO model keeps LayerNorm — baseline instability
   *if it grokks* was the pre-registered prediction.
3. Under wd 1.0 memorization itself is late and entangled with the
   transitions (memorize step 19k > k2 first crossing 12.5k) — the
   plateau structure is cleanest at wd 0.3.

**Phase 2 cell**: sub10000-wd0.3 (delayed staged generalization +
visible instability + budget headroom), extended to 100k steps.

### 2026-08-01 — Phase 8 main result: when depth is required, the aux loss *accelerates* staged grokking — and still stabilizes it

Baseline vs aux λ = 0.3 uniform in the chosen grokking cell
(sub10000-wd0.3, constant LR 3e-4, 100k steps, eval/500), 3 seeds each.
SkyPilot jobs 168–173 (~47 min baseline / ~55 min aux per run):

```
for s in 42 43 44; do for arm in base lensaux; do
  # base:    run="S3-grok-sub10000-wd0.3-base-s${s}-100k";                extra=""
  # lensaux: run="S3-grok-sub10000-wd0.3-lensaux0.3-uniform-s${s}-100k"; extra="--lens-aux --lens-aux-weight 0.3"
  sky exec local-gpu skypilot/local.yaml -d \
    --env RUN_NAME="$run" \
    --env RUN_CMD="uv run python -m lego.train --train-subset 10000 --weight-decay 0.3 --lr-schedule constant --total-steps 100000 --eval-every-steps 500 ${extra} --seed ${s} --wandb-project grok-lens --wandb-run-name ${run} --checkpoint-dir \"\${CHECKPOINT_DIR}\"" \
    --secret WANDB_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY
done; done
```

Per-k first crossing (steps, per seed s42/s43/s44; `analyze_grok_stability`):

| arm | k2 | k3 | k4 | k5 | k6 |
|---|---|---|---|---|---|
| baseline | 20k/6.5k/23.5k | 37.5k/6.5k/24k | 62k/7.5k/48.5k | 87k/9.5k/74.5k | **—**/16.5k/90.5k |
| aux uniform | 9k/8.5k/10k | 9k/8.5k/8k | 9.5k/11.5k/8.5k | 12k/15.5k/10.5k | 15.5k/18.5k/13.5k |

Post-crossing dips < 0.90 (summed over k) and occupancy range; final
test mean:

| arm | seed | total dips | occupancy (min–max over k) | final mean |
|---|---|---|---|---|
| baseline | 42 | 1 | 0.83–0.93 (k6 never crosses; 0.920 at end) | 0.932 |
| baseline | 43 | 15 | 0.83–0.98 (k6: 11 dips) | 0.987 |
| baseline | 44 | 14 | 0.89–0.95 | 0.969 |
| aux | 42 | 1 | 0.98–1.00 | 1.000 |
| aux | 43 | 4 | 0.95–0.99 | 0.997 |
| aux | 44 | 1 | 0.87–1.00 (k2 occ 0.87 = borderline evals on the 43-example stratum, 0 dips) | 1.000 |

wandb IDs: b7780wrm / kq5trf6k / dfhj5xcv (base),
fqarlnit / 4aahoi1b / nspjct9b (aux). Memorize steps 7.2–10.7k (base),
8.4–8.9k (aux) — both arms have a genuine plateau before the k ≥ 3
transitions.

**Findings.**
1. **The first-arrival delay reverses sign when depth is required.** On
   the arithmetic tasks the aux loss delayed first grokking ~2–4.7×; on
   LEGO-in-a-grokking-regime it *accelerates* every k ≥ 4 stage in the
   two slow-baseline seeds and on the median (k6 first crossing: aux
   15.5k vs baseline 90.5k, with one baseline seed never crossing in
   100k), compressing the whole staircase into ~8–18.5k steps. The
   fast-baseline outlier s43 crosses each stage slightly before its aux
   counterpart — acceleration is a median/2-of-3-seed effect; what is
   3/3 is that aux timing never blows up while baseline timing can.
2. **Baseline first-crossing is heavy-tailed across seeds** (k6 range
   16.5k → never); aux is tight (13.5–18.5k). The aux loss regularizes
   the transition's *timing*, not just its stability.
3. **Stabilization survives where the shallow-collapse exit is closed.**
   Baselines show the sawtooth in 2/3 seeds (15 and 14 dips; the
   s42 exception has only 1 dip but chronic sub-0.95 wobble, occupancy
   0.83–0.93); aux runs are near-absorbing in 3/3 (≤ 1 dip per k,
   occupancy 0.95–1.00 modulo the k2 small-stratum note).
4. **No short-chain capability cost in this regime** (final mean
   0.997–1.000 incl. k0/k1), unlike the promptly-generalizing regime's
   k ≤ 1 losses.

**Caveats (Phase 8 head-to-head).** Run-to-run variance in this regime is substantial even at
fixed seed-relevant config: the 50k search run at the same cell/seed
(663lqx6q) crossed k2 at 16k with 10 dips where the 100k rerun
(b7780wrm, identical settings for its first 50k) crossed at 20k with 1 —
GPU nondeterminism moves threshold crossings and dip counts; the
qualitative phenotype (staged transitions; baseline instability; aux
acceleration + stability) is consistent everywhere, and all claims
above rest on the 3-seed contrasts, not single runs. Baseline
instability at wd 0.3 is milder than the arithmetic-task sawtooth; the
search showed it scales with wd (34–35 dips per k at wd 1.0), so the
stability contrast here is conservative.

### 2026-08-02 — Phase 9, data-rich pair: the realistic full-sequence objective slows the task; full-sequence deep supervision on top of it kills it

Phase 9 (EXPERIMENT_PLAN) corrects the Phase 5 framing: supervise **all**
positions, as a real LM would be trained, and ask whether deep
supervision hurts. New `--base-loss all-positions` = final-layer
next-token CE at every non-pad position. Data-rich regime (canonical
40-epoch settings), seed 42, SkyPilot jobs 174–175 (~20 min each):

```
uv run python -m lego.train --base-loss all-positions --seed 42 \
  --wandb-run-name S3-std-8L-splitku-fullseqbase-s42 --checkpoint-dir <dir>
uv run python -m lego.train --base-loss all-positions --lens-aux \
  --lens-aux-weight 0.3 --lens-aux-mode all-positions --seed 42 \
  --wandb-run-name S3-std-8L-splitku-fullseq-lensaux0.3-allpos-s42 --checkpoint-dir <dir>
```

(Run via `sky exec local-gpu skypilot/local.yaml` as in Phase 8; launch
env identical.) Held-out accuracy at the matched 20,960-step budget,
wandb 4qx4msct / y8blx6eq:

| arm | k2 | k3 | k4 | k5 | k6 | mean |
|---|---|---|---|---|---|---|
| answer-only base (canonical, for reference) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ~1.00 |
| full-seq base, no aux | 0.51 | 0.76 | 0.79 | 0.62 | 0.46 | 0.49 |
| full-seq base + full-seq aux λ=0.3 | 0.09 | 0.08 | 0.14 | 0.16 | 0.16 | 0.16 (chance) |

The realistic base objective alone already slows task acquisition badly
at matched budget (train answer accuracy never reaches 0.95); adding
full-sequence deep supervision reduces the model to chance on every
stratum. Position-by-position lens/probe readouts (`analyze_fullseq`,
CPU sanity pass; the recorded GPU pass is job 182): the full-seq aux
model's lens is *perfectly legible everywhere* — structural next tokens
(the `<op>`/`<predict>` markers) read at 1.00 from layer 0, and the
`<predict>` distribution sits at exactly ln 6 entropy at every layer for
k = 4 — while probes find no intermediates because the model never
computes any. Deep supervision made every layer answer-shaped and the
task died: legibility without competence. P2's lens-vs-probe contrast is
therefore evaluable only on arms that learn (the full-seq base, and the
grokking-regime arms below).

### 2026-08-02 — Phase 9, grokking regime: under the realistic objective the aux loss delays or breaks grokking and stabilizes nothing

Same cell and settings as the Phase 8 head-to-head (sub10000, wd 0.3,
constant LR, 100k, eval/500), with the full-sequence base loss; aux arm
adds `--lens-aux --lens-aux-mode all-positions --lens-aux-weight 0.3`.
SkyPilot jobs 176–181 (~42 min base / ~55 min aux per run); commands as
in the data-rich entry plus the Phase 8 grokking flags. wandb:
rjqa6atc / 4ledieft / ul1z17eh (base), peyc9taf / pg5o9enz / 88c2es51
(aux).

Per-k first crossing (s42/s43/s44):

| arm | memorize | k2 | k3 | k4 | k5 | k6 |
|---|---|---|---|---|---|---|
| full-seq base | 17.1k/15.5k/19.9k | 12.5k/14.5k/17k | 20k/13.5k/19.5k | —/14.5k/— | —/17k/— | —/21.5k/— |
| full-seq + aux | 40k/52.4k/22.3k | 30.5k/—/30.5k | 56k/—/19k | —/—/21k | —/—/37k | —/—/— |

Post-crossing dips < 0.90 / occupancy at the crossed strata:

| run | worst strata |
|---|---|
| base s42 | k2: 69 dips, occ 0.25; k3: occ 0.11 |
| base s43 | k6: 122 dips, occ 0.04 (k2–k5 occ 0.87–1.00) |
| base s44 | k2: 154 dips, occ 0.01; k3: 66 dips, occ 0.04 |
| aux s42 | k3: 30 dips, occ 0.01; k2: 7 dips, occ 0.69 |
| aux s43 | nothing ever crosses (final k2/k3 0.67/0.81) |
| aux s44 | k5: 27 dips, occ 0.02 (k2–k4: 3–5 dips, occ 0.91–0.94) |

Final test mean: base 0.38/0.88/0.37; aux 0.45/0.18/0.74.

**Findings (P1 scorecard).**
1. **The realistic base objective alone degrades both speed and
   stability**: vs the answer-only baseline in the same cell, staged
   grokking still happens but only 1/3 seeds completes the staircase in
   100k, and the sawtooth is far more violent (69–154 dips at the worst
   stratum vs ≤ 15 total for answer-only baselines).
2. **Full-sequence deep supervision delays the transition — sometimes
   past the budget — with heterogeneous outcomes where it lands**:
   memorization is delayed 1.1–3.4× in every seed, most measurable
   first crossings are delayed ~1.8–2.8× (s44 k3 is unchanged within
   noise), one seed never crosses any stratum in 100k (k ≤ 4 stall at 0.58–0.81; k ≥ 5 at chance), and
   k6 never crosses under the aux loss. But "destructive" would
   overstate it: in the two seeds that do transition, the aux run ends
   with *higher* final accuracy on every stratum than its base
   counterpart (final means 0.45/0.74 vs 0.38/0.37) and is markedly
   more stable at its crossed strata in s44 (k2–k4: 3–5 dips,
   occupancy 0.91–0.94, vs the base's 154 dips at k2) — though not in
   s42 (k3: 30 dips, occupancy 0.01). Later-but-better where the
   transition comes; stuck at chance where it doesn't. Seed variance
   dominates every cross-arm comparison in this regime.
3. Together with the data-rich pair: whether deep supervision helps or
   harms is entirely a property of *what is supervised*. Supervising a
   true, task-relevant quantity (the answer) at one position is benign
   to strongly beneficial; supervising every position's next token —
   most of which are irreducible noise on this task — is harmful at the
   base level and disastrous as per-layer supervision, consistent with
   the shuffled-target specificity control on the arithmetic tasks.

### 2026-08-02 — Phase 9 attribution arm: on the realistic base, even answer-shaped aux is harmful

One-seed control (data-rich, canonical budget, SkyPilot job 183, wandb
yjtu14kn): full-sequence base loss + the *answer-mode* aux
(`--base-loss all-positions --lens-aux --lens-aux-weight 0.3`, uniform).
Result: chance on every stratum (final test mean 0.173; k2–k6
0.10–0.23). So the harm is not only "noise targets at every layer": on
the full-sequence base objective, even the aux form that was benign and
beneficial on the answer-only base destroys learning at matched budget.
The benignness of answer-shaped deep supervision is contingent on the
base objective, not just on the aux targets. (Single seed, one regime —
scoped accordingly.)

### 2026-08-02 — Phase 9 lens/probe verdict (P2): full-sequence supervision hides the intermediates from the lens everywhere; probes still see them

Recorded position-by-position analysis (`lego.analyze_fullseq`; SkyPilot
jobs 182 and 185; JSONs in the run results). Held-out chains; "lens max /
probe max anywhere" = best cell over all (layer, position); chance 0.17.

Data-rich four-way, k=4 intermediates (traj[1]/traj[2]/traj[3]):

| arm | lens max anywhere | probe max anywhere |
|---|---|---|
| answer-only base | 0.97 / 1.00 / 1.00 (op staircase) | 1.00 / 1.00 / 1.00 |
| answer-only aux | ≤ 0.34 below ℓ\* (Phase 7) | 0.95 at best cells (Phase 7) |
| full-seq base | 0.26 / 0.18 / 0.18 | 1.00 / 1.00 / 0.29 |
| full-seq base + full-seq aux | 0.30 / 0.18 / 0.20 | 1.00 / 0.41 / 0.18 |

Grokking-regime spot checks on the arms that learn (fullseq base s43,
which solves k2–k6; fullseq aux s44, which solves k2–k5): identical
pattern — lens ≤ 0.30 on every intermediate at every position and layer,
probes recover traj[1] at 1.00 and traj[2] at 0.69–0.80.

### 2026-08-02 — Phase 9 dose-response: the full-sequence harm is presence, not strength

Data-rich, seed 42, λ ∈ {0.01, 0.1} full-sequence aux on the
full-sequence base (SkyPilot jobs 186–187; wandb 10u7e220 / qzbki6a2;
commands as the λ = 0.3 run with `--lens-aux-weight` swapped):

| λ | held-out mean at 20,960 steps |
|---|---|
| 0.01 | 0.164 (chance) |
| 0.1 | 0.165 (chance) |
| 0.3 | 0.162 (chance) |

A 1%-weight full-sequence deep-supervision term blocks the task as
completely as a 30% one — the exact mirror of the arithmetic-task
finding that the answer-shaped *benefits* are flat from λ = 0.01 to
3.0. In both directions, what matters is whether per-layer supervision
is present and what it points at, not how hard it pushes. (Single seed
per λ, one regime.)

### 2026-08-02 — Phase 9 learnability check: the full-sequence base objective converges given budget; the full-sequence aux failure is not a budget or difficulty artifact

Before interpreting full-sequence grokking dynamics, establish that the
objective can be trained at all (SkyPilot jobs 188–193, seed 42;
`--n-epochs 120` for k6ext, `--k-max 4/3 --total-steps 20000` for the
easier tasks; wandb qajoz9rz / odxejh03 / z8rcvn3a / xxopj1kw /
dusvswcb / and the k3-aux run):

| config | base (no aux) | + full-seq aux λ = 0.3 |
|---|---|---|
| k_max=6, 21k steps (matched budget) | 0.49 partial | 0.16 chance |
| k_max=6, 63k steps (3× budget) | **0.983** (k3–k6 cross 13k/16k/23k/35k; k2 stratum 0.67, never crosses) | **0.164 chance** |
| k_max=4, 20k steps | 0.945 (k3 crosses 9.5k) | 0.117 chance |
| k_max=3, 20k steps | 0.40 (1,243 train chains — data-starved, not comparable) | chance |

**Findings.**
1. **The full-sequence base objective is trainable to near-full task
   competence** — the matched-budget deficit was a budget effect, not a
   learnability failure. Grokking-dynamics claims about the full-seq
   baseline therefore rest on a demonstrably learnable objective.
   (Anomaly worth flagging: the 43-example k=2 stratum lags badly under
   full-sequence training in every run that otherwise converges —
   0.67–0.84 — where answer-only training gets it to 1.00.)
2. **The full-sequence aux failure survives every margin we gave it**:
   3× budget at k_max=6, an easier task (k_max=4) that the base solves
   within the same 20k budget, and λ down to 0.01 (dose-response entry).
   Its inability to learn in the data-rich regime is a property of the
   objective combination, not of budget, task size, or weight. (The
   grokking-regime aux runs' partial learning — k2–k4 in 2/3 seeds —
   remains the only setting where this arm learns anything; contrast
   scoped accordingly.)
3. k_max=3 leaves only 1,554 total chains, too few for the data-rich
   framing; k_max=4 is the right "easier task" control.

### 2026-08-03 — Phase 9 long-horizon aux check: the marginal-solution basin is escapable at k_max=4 — ~10× delayed, with a permanent short-chain deficit

Does the full-seq aux arm *eventually* learn (pre-registered
long-horizon check)? k_max=4, λ = 0.3 all-positions on the full-seq
base, 200k steps (SkyPilot job 194, wandb 4ricj7xh):

| metric | value |
|---|---|
| memorize | 30.5k (vs base 6.3k at k_max=4 — ~5× later) |
| k4 first crossing | 52k, then 0 dips, occupancy 1.00, final 0.992 |
| k3 | never crosses; plateaus ~0.78 |
| k2 | never crosses; plateaus ~0.61 |
| final test mean | 0.950 |

The layer-0-satisfiable marginal solution is **not a terminal basin at
k_max=4**: the run is flat at chance until ~30k, then transitions —
flat-then-sudden, grokking-shaped — and the hardest stratum converges
and holds perfectly. But the short strata plateau *below what the base
objective reached at one-tenth the budget* (base at 20k: k3 0.985,
k2 0.837), so the difficulty ordering inverts: under full-sequence deep
supervision the easy strata are the casualties, consistent with the
answer-dilution account (short chains contribute the fewest informative
targets, so their answer gradient is weakest against the per-layer
noise-target pressure). Whether the k_max=6 aux arm is likewise
escapable on some longer horizon (it is flat through 63k) remains open.
(Single seed.)

**P2 confirmed, strengthened.** The pre-registered prediction was that
full-sequence deep supervision would make intermediates lens-invisible
while probes still find them; what the data show is that the
*full-sequence objective itself* already does this — the answer-only
baseline's op-position staircase (lens up to 1.00 on intermediates) is
erased the moment those positions acquire next-token targets, in the
base arm as much as the aux arm, including in models that demonstrably
compute the intermediates (they solve those strata). The lens's
own-output readout behaves exactly as predicted throughout: structural
next tokens read at 1.00 from layer 0 under the aux loss, unpredictable
positions sit at calibrated near-uniform, and the `<predict>`
distribution sharpens to the answer where the answer is learned.
Layer-0-legible structural tokens vs layer-7 answers also means the
progression depth tracks target difficulty. The general statement that
survives all four arms: **the logit lens shows exactly what the
training objective put into the unembedding basis — and nothing else;
linear probes see whatever the model actually computes.**

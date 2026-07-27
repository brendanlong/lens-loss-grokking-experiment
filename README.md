# Grokking doesn't stick: deep supervision reveals — and counteracts — the instability of grokked circuits

What happens to grokking if you train **every** layer of a transformer to
predict the answer through the model's own unembedding — the logit lens
turned into a training objective (the LayerSkip/CALM loss, borrowed from
the inference-efficiency literature)?

We expected it to break grokking. Instead we found, across ~160 runs on
modular arithmetic (plus a multi-hop composition task):

1. **Canonical grokking never actually sticks.** Run past the grok point
   and 2–3-layer baselines fall out of generalization dozens of times,
   indefinitely, under AdamW *and* Muon. The field's "grok step =
   first threshold crossing" metric hides this (it initially fooled us
   too — an adversarial review of our own first writeup caught it).
2. **The aux loss delays first grokking ~2× but makes it permanent** —
   at *any* strength from 1% to 300% of the main loss (presence, not
   strength).
3. **Mechanism:** grokking naturally finds a *distributed* Fourier
   circuit; weight decay's post-grok "cleanup" prunes it into the
   celebrated sparse ~5-frequency circuit — which is causally brittle
   (delete one frequency → up to −90 points; churn tracking shows
   component failures constantly reaching the output). The aux loss
   blocks that pruning: components still break, but 0/159 tracked
   failures ever reached the output. **Stability is failure isolation,
   not calm.**
4. Controls: the effect is specific to *answer-shaped* supervision (a
   weight-decay sweep and a shuffled-target control both fail to
   reproduce it), and the story replicates on a second task (modular
   subtraction).

![Baseline grokking collapses repeatedly; aux runs stabilize permanently](figures/occupancy.png)

**Read the full story in [WRITEUP.md](WRITEUP.md).** The complete
experimental log — per-seed tables, exact commands, and the correction
lineage — is in [RESULTS.md](RESULTS.md), with the pre-registered
predictions in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). Training curves
for every run: [public wandb project](https://wandb.ai/brendanlong-com/grok-lens).
All 120 final checkpoints:
[HF dataset](https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment).

## Repo layout

```
grok_lens/          # modular-arithmetic grokking: model, aux loss, training, analyses
  train.py          #   uv run python -m grok_lens.train --help
  analyze_*.py      #   stability / FFT / knockout / layer-0 analyses
  make_figures.py   #   regenerates figures/ from wandb + checkpoints
  muon.py           #   hybrid Muon optimizer (optimizer-robustness arm)
lego/               # S3 multi-hop composition task (front-loading results)
  train.py          #   uv run python -m lego.train --help
  compare_lens_aux.py  # lens staircase / coalescence analysis
common/             # shared config / schedule / streaming / checkpoint utilities
scripts/            # reproduction entry points (see below)
figures/            # pre-generated figures used in the writeup
```

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest   # 79 CPU tests, ~5 s
```

A GPU is optional for the analyses (checkpoints are downloaded) and
strongly recommended for training (any ~8 GB card is plenty; everything
here was trained on a single RTX 3060 Ti).

## Reproduce the analyses (no training, ~minutes)

Every table in the writeup regenerates from the hosted checkpoints:

```bash
./scripts/reproduce_analyses.sh
```

The stability/occupancy metrics and the figures additionally read training
histories from the public wandb project, which requires a free
`wandb login`. Checkpoint-only analyses (FFT, knockouts, layer-0 probes,
LEGO staircase) need no accounts at all.

## Reproduce the training (~a day of consumer GPU)

```bash
./scripts/reproduce_training.sh   # core arms enabled; sweeps/controls commented
```

Or run any single configuration directly — e.g. a baseline that shows the
instability, and an aux run that doesn't:

```bash
uv run python -m grok_lens.train --total-steps 50000 --seed 42 --no-wandb
uv run python -m grok_lens.train --total-steps 50000 --seed 42 --aux-lambda 0.3 --no-wandb
```

Each 30k-step run is ~12 minutes on an RTX 3060 Ti (50k ≈ 19 min,
100k ≈ 38 min). See RESULTS.md for the exact command behind every number
in the writeup (a header note there maps the original monorepo commands
onto this repo's layout).

### On a cloud GPU (SkyPilot)

No local GPU? [`skypilot/reproduce.yaml`](skypilot/reproduce.yaml) runs
the same reproduction on any cloud
[SkyPilot](https://docs.skypilot.co) supports:

```bash
sky launch skypilot/reproduce.yaml --infra <your-cloud> --down -y
# or a single run:
sky launch skypilot/reproduce.yaml --infra <your-cloud> --down -y \
  --env RUN_CMD="uv run python -m grok_lens.train --total-steps 50000 --seed 42 --no-wandb"
```

The full core reproduction is roughly a GPU-day on an 8 GB card —
typically a few dollars on spot instances. Pass `--secret WANDB_API_KEY`
to log to your own wandb.

## Provenance

This repo is extracted from a private research monorepo where the runs
were executed (via SkyPilot on a local GPU). wandb run IDs in RESULTS.md
link into the public project; S3 URIs in historical commands refer to the
original private checkpoint store — the public copies live on the
[HF dataset](https://huggingface.co/datasets/brendanlong/lens-loss-grokking-experiment).

**All of the code in this repository was written and run by Claude
(Anthropic's Claude Code), based on Brendan Long's prompts and direction,
with most intermediate results reviewed by Claude subagents; Brendan
reviewed the full PR before merging and the experimental log records
which analyses and corrections came from which side of that
collaboration.**

## License

[MIT](LICENSE)

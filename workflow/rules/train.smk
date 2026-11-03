rule train:
    input:
        code=TRAIN_CODE,
    output:
        trace=RESULTS / "runs/{cell}/s{seed}/trace.pt",
        checkpoints=directory(RESULTS / "runs/{cell}/s{seed}/checkpoints"),
    log:
        RESULTS / "logs/train/{cell}/s{seed}.log",
    resources:
        gpu=1,
        max_runtime_min=120,
    params:
        args=lambda wc: config["cells"][wc.cell]["args"],
        steps=config["total_steps"],
        checkpoint_every=config["checkpoint_every"],
        run_name="p113-{cell}-s{seed}-grid",
        wandb="" if config["wandb"] else "--no-wandb",
    shell:
        "REQUIRE_CUDA=1 PYTHONUNBUFFERED=1 uv run --no-sync python -m grok_lens.train"
        " --seed {wildcards.seed} --total-steps {params.steps} {params.args}"
        " --checkpoint-every {params.checkpoint_every}"
        " --checkpoint-dir {output.checkpoints} --trace-out {output.trace}"
        " --wandb-run-name {params.run_name} {params.wandb} > {log} 2>&1"

rule analyze:
    input:
        trace=rules.train.output.trace,
        code=ANALYZE_CODE,
    output:
        RESULTS / "summaries/{cell}/s{seed}.json",
    log:
        RESULTS / "logs/analyze/{cell}/s{seed}.log",
    params:
        window_start=lambda wc: window_start(wc.cell),
        lags=" ".join(map(str, config["lags"])),
    shell:
        "uv run --no-sync python -m grok_lens.analyze_levels {input.trace}"
        " --out {output} --window-start {params.window_start}"
        " --lags {params.lags} > {log} 2>&1"


rule figures:
    input:
        summaries=expand(rules.analyze.output[0], cell=CELLS, seed=SEEDS),
        traces=expand(rules.train.output.trace, cell=CELLS, seed=SEEDS),
        code=PLOT_CODE,
    output:
        expand(RESULTS / "figures/{name}.png", name=FIGURES),
    log:
        RESULTS / "logs/figures.log",
    params:
        out=RESULTS / "figures",
        rows=" ".join(f"--row {','.join(r)}" for r in config["figure_rows"]),
        labels=" ".join(
            "--label " + shlex.quote(f"{c}={v['label']}")
            for c, v in config["cells"].items()
        ),
        windows=" ".join(f"--window {c}={window_start(c)}" for c in CELLS),
    shell:
        "uv run --no-sync python -m grok_lens.plot_levels --out {params.out}"
        " --summaries {input.summaries} --traces {input.traces}"
        " {params.rows} {params.labels} {params.windows}"
        " > {log} 2>&1"

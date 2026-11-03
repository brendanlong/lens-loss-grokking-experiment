"""Build the stability-grid report from the result files in report/data.

    uv run python report/build_report.py   # writes report/out/index.html

data/ holds copies of the workflow's summaries/ and weights/ (from
s3://brendanlong-experiments/grok_lens/stability-grid/) and a few test
accuracy traces extracted from the training traces.
"""

import json
import random
import statistics as st
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from svg import STYLE, Figure, Scale, legend

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
RENDER = Path.home() / (
    ".claude/plugins/cache/personal-agent-skills/personal-agent-skills/1.11.1/"
    "skills/html-doc/scripts/render.py"
)
SURGE_DOMAIN = "grokking-stability-3f9c1a7e.surge.sh"

LABELS = {
    "L1-noln": "1 layer, no LayerNorm (Nanda)",
    "L2-noln": "2 layers, no LayerNorm",
    "L2-noln-aux": "2 layers, no LayerNorm, aux loss",
    "L1-ln": "1 layer, LayerNorm",
    "L2-ln": "2 layers, LayerNorm",
    "L3-ln": "3 layers, LayerNorm",
    "L2-ln-aux": "2 layers, LayerNorm, aux loss",
    "L2-ln-dropout": "2 layers, LayerNorm, dropout 0.1",
    "L2-ln-lrdrop": "2 layers, LayerNorm, LR ×0.1 at 20k",
    "L2-ln-anneal": "2 layers, LayerNorm, LR annealed",
    "L2-ln-anneal-heldwd": "2 layers, LayerNorm, annealed, decay rate held",
}
SHORT = {
    "L1-noln": "1 layer, no LN (Nanda)", "L2-noln": "2 layers, no LN",
    "L2-noln-aux": "2 layers, no LN, aux", "L1-ln": "1 layer, LN", "L2-ln": "2 layers, LN",
    "L3-ln": "3 layers, LN", "L2-ln-aux": "2 layers, LN, aux",
    "L2-ln-dropout": "2 layers, LN, dropout", "L2-ln-lrdrop": "2L LN, instant LR drop",
    "L2-ln-anneal": "2L LN, annealed", "L2-ln-anneal-heldwd": "2L LN, annealed, decay held",
}  # fmt: skip
CONSTANT_CELLS = [
    "L1-noln",
    "L2-noln",
    "L2-noln-aux",
    "L1-ln",
    "L2-ln",
    "L3-ln",
    "L2-ln-aux",
]
LR_CELLS = ["L2-ln", "L2-ln-lrdrop", "L2-ln-anneal", "L2-ln-anneal-heldwd"]
TAIL_CELLS = [
    "L2-ln",
    "L2-ln-dropout",
    "L2-ln-lrdrop",
    "L2-ln-anneal",
    "L2-ln-anneal-heldwd",
    "L2-ln-aux",
]
LEVELS = [("capability", "s1", "test answers changed"),
          ("embedding", "s2", "embedding frequencies"),
          ("neurons", "s3", "neurons per frequency")]  # fmt: skip


def load(kind: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for f in sorted((DATA / kind).glob("*/s*.json")):
        out.setdefault(f.parent.name, {})[int(f.stem[1:])] = json.loads(f.read_text())
    return out


S = load("summaries")
W = load("weights")
TRACES = json.loads((DATA / "accuracy_traces.json").read_text())


def at_lag(summary: dict, lag: int, key: str, curve: str = "pooled") -> float | None:
    for p in summary["curves"][curve]:
        if p["lag"] == lag:
            return p[key]
    return None


def mean(xs: Iterable[float | None]) -> float:
    """Mean of the non-missing values; a value missing everywhere is a bug."""
    present = [x for x in xs if x is not None]
    if not present:
        raise ValueError("no values")
    return sum(present) / len(present)


def fisher_two_sided(a: int, b: int, n: int) -> float:
    """a of n vs b of n with an event; two-sided Fisher exact p (doubling)."""
    k = a + b
    hi, lo = max(a, b), min(a, b)
    one = sum(comb(n, x) * comb(n, k - x) for x in range(hi, min(n, k) + 1)) / comb(
        2 * n, k
    )
    return min(1.0, 2 * one) if hi != lo else 1.0


def jitter(i: int, n: int, spread: float = 0.28) -> float:
    rnd = random.Random(i * 7919 + n)
    return (rnd.random() - 0.5) * 2 * spread


# ---------------------------------------------------------------- numbers
def collapses(cell: str) -> list[int]:
    return [s["collapses"] for _, s in sorted(S[cell].items())]


def tail_collapsing(cell: str) -> int:
    return sum(s["tail"]["collapses"] > 0 for s in S[cell].values())


N = {c: len(S[c]) for c in S}
tail = {c: tail_collapsing(c) for c in TAIL_CELLS}
p_anneal = fisher_two_sided(tail["L2-ln"], tail["L2-ln-anneal"], 10)
p_held = fisher_two_sided(tail["L2-ln"], tail["L2-ln-anneal-heldwd"], 10)
p_held_vs_anneal = fisher_two_sided(
    tail["L2-ln-anneal-heldwd"], tail["L2-ln-anneal"], 10
)
end_broken = {
    c: sum(s["tail"]["final_acc"] < 0.95 for s in S[c].values()) for c in TAIL_CELLS
}


def level_at(cell: str, level: str, lag: int = 10_000) -> float:
    return mean(at_lag(s, lag, level) for s in S[cell].values())


def events(cell: str) -> list[dict]:
    return [
        e for s in S[cell].values() for e in s["events"]
        if "moved" in e and e["collapse_free_control"]["n_pairs"]
    ]  # fmt: skip


all_events = [e for c in S for e in events(c)]


def event_ratio(cells: list[str], level: str) -> float:
    ev = [e for c in cells for e in events(c)
          if e["moved"][level] is not None and e["collapse_free_control"][level] is not None]  # fmt: skip
    return st.median(e["moved"][level] for e in ev) / st.median(
        e["collapse_free_control"][level] for e in ev
    )


def weight_at(cell: str, key: str, lag: int = 10_000) -> float | None:
    vals = [
        p[key]
        for w in W[cell].values()
        for p in w["curve"]
        if p["lag"] == lag and p[key] is not None
    ]
    return mean(vals) if vals else None


def norm_at(cell: str, idx: int) -> float:
    return mean(
        sum(v[idx] ** 2 for v in w["weight_norms"].values()) ** 0.5
        for w in W[cell].values()
    )


# ---------------------------------------------------------------- figures
def fig_accuracy() -> str:
    runs = [("L1-noln_s42", "L1-noln"), ("L1-ln_s42", "L1-ln"),
            ("L2-ln_s42", "L2-ln"), ("L2-ln-anneal_s42", "L2-ln-anneal")]  # fmt: skip
    ph, gap, top, left, w = 70, 34, 22, 44, 640
    fig = Figure(
        left + w + 16,
        top + len(runs) * (ph + gap) + 6,
        "Test accuracy over training for four runs",
    )
    for i, (key, cell) in enumerate(runs):
        tr = TRACES[key]
        p = fig.panel(
            left,
            top + i * (ph + gap),
            w,
            ph,
            Scale(0, 50_000, 0, w),
            Scale(0, 1, ph, 0),
        )
        p.title(LABELS[cell] + ", seed 42")
        p.ygrid([0, 0.5, 1], ["0", "0.5", "1"])
        p.xgrid([0, 10_000, 20_000, 30_000, 40_000, 50_000],
                ["0", "10k", "20k", "30k", "40k", "50k"], show_labels=i == len(runs) - 1)  # fmt: skip
        p.polyline(tr["steps"], tr["test_acc"], "ink", width=1.2)
        p.frame()
    fig.text(
        12,
        top + 2 * (ph + gap) - gap / 2,
        "test accuracy",
        cls="ink2",
        anchor="middle",
        rotate=True,
    )
    fig.text(left + w / 2, fig.height - 2, "training step", cls="ink2", anchor="middle")
    return fig.svg()


def strip(rows: list[str], values: Mapping[str, Sequence[float]], xmax: float, xticks: list[float],
          xlabel: str, label: str, xfmt=lambda t: f"{t:g}", mean_tick: bool = True) -> str:  # fmt: skip
    left, top, rh, w = 300, 8, 26, 380
    fig = Figure(left + w + 20, top + rh * len(rows) + 36, label)
    p = fig.panel(
        left,
        top,
        w,
        rh * len(rows),
        Scale(0, xmax, 0, w),
        Scale(0, len(rows), 0, rh * len(rows)),
    )
    p.xgrid(xticks, [xfmt(t) for t in xticks])
    for r, cell in enumerate(rows):
        y = r + 0.5
        fig.text(
            left - 8,
            p.Y(y) + 4,
            f"{LABELS[cell]} ({len(values[cell])})",
            cls="ink",
            anchor="end",
        )
        for i, v in enumerate(values[cell]):
            p.dot(
                v,
                y + jitter(i, r) * 0.9,
                "ink2",
                r=3.2,
                opacity=0.65,
                tip=f"{LABELS[cell]}: {v:g}",
            )
        if mean_tick and values[cell]:
            p.tick(mean(values[cell]), y, 9, "s2")
    p.frame()
    fig.text(left + w / 2, fig.height - 4, xlabel, cls="ink2", anchor="middle")
    return fig.svg()


def fig_collapses() -> str:
    rows = ["L1-noln", "L2-noln", "L2-noln-aux", "L1-ln", "L2-ln", "L3-ln", "L2-ln-aux"]
    return strip(rows, {c: collapses(c) for c in rows}, 55, [0, 10, 20, 30, 40, 50],
                 "collapses after first grokking (test accuracy falls below 0.90, then recovers to 0.95)",
                 "Collapses per run by architecture")  # fmt: skip


def fig_levels(cells: list[str], label: str) -> str:
    cols = 4 if len(cells) == 4 else 3
    nrows = -(-len(cells) // cols)
    pw, ph, gx, gy, left, top = (150 if cols == 4 else 200), 120, 26, 56, 52, 24
    fig = Figure(left + cols * (pw + gx), top + nrows * (ph + gy) + 4, label)
    xs = Scale(100, 20_000, 0, pw, log=True)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        p = fig.panel(
            left + c * (pw + gx), top + r * (ph + gy), pw, ph, xs, Scale(0, 1, ph, 0)
        )
        p.title(SHORT[cell])
        p.ygrid([0, 0.5, 1], ["0", "0.5", "1"], show_labels=c == 0)
        p.xgrid([100, 1000, 10_000], ["100", "1k", "10k"])
        for level, series, _ in LEVELS:
            seeds = list(S[cell].values())
            for s in seeds:
                pts = [
                    (q["lag"], q[level])
                    for q in s["curves"]["pooled"]
                    if q[level] is not None
                ]
                if pts:
                    p.polyline(
                        [a for a, _ in pts],
                        [b for _, b in pts],
                        series,
                        width=0.8,
                        opacity=0.25,
                    )
            lags = sorted({q["lag"] for s in seeds for q in s["curves"]["pooled"]})
            m = [
                (lag, mean(vals))
                for lag in lags
                if (
                    vals := [
                        v for s in seeds if (v := at_lag(s, lag, level)) is not None
                    ]
                )
            ]
            if m:
                p.polyline([a for a, _ in m], [b for _, b in m], series, width=2.2)
        p.frame()
        fig.text(
            p.x0 + pw,
            p.y0 + ph + 26,
            f"lag (steps) · {N[cell]} seeds",
            cls="ink2",
            anchor="end",
        )
    fig.text(
        14,
        top + nrows * (ph + gy) / 2,
        "moved (0 = unchanged)",
        cls="ink2",
        anchor="middle",
        rotate=True,
    )
    return fig.svg()


def fig_repairs() -> str:
    pw, ph, gx, left, top = 260, 220, 60, 46, 24
    fig = Figure(
        left + 2 * (pw + gx),
        top + ph + 44,
        "Change across each collapse vs equal-length collapse-free spans",
    )
    for i, (level, name, hi) in enumerate([("embedding", "embedding frequencies", 0.6),
                                           ("neurons", "neurons per frequency", 1.0)]):  # fmt: skip
        p = fig.panel(
            left + i * (pw + gx), top, pw, ph, Scale(0, hi, 0, pw), Scale(0, hi, ph, 0)
        )
        p.title(name)
        ticks = [0, 0.2, 0.4, 0.6] if hi < 1 else [0, 0.25, 0.5, 0.75, 1]
        p.xgrid(ticks)
        p.ygrid(ticks)
        fig.line(p.X(0), p.Y(0), p.X(hi), p.Y(hi), "ink2", dash="3 3")
        for cell in S:
            for e in events(cell):
                x, y = e["collapse_free_control"][level], e["moved"][level]
                if x is None or y is None:
                    continue
                one_layer = cell == "L1-ln"
                p.dot(min(x, hi), min(y, hi), "s2" if one_layer else "ink2", r=3.6 if one_layer else 2.6,
                      opacity=1 if one_layer else 0.45, hollow=not one_layer,
                      tip=f"{LABELS[cell]}: collapse at {e['pre_step']}")  # fmt: skip
        p.frame()
        fig.text(
            p.x0 + pw / 2,
            top + ph + 30,
            "collapse-free span of the same length",
            cls="ink2",
            anchor="middle",
        )
        if i == 0:
            fig.text(
                12,
                top + ph / 2,
                "just before collapse → just after recovery",
                cls="ink2",
                anchor="middle",
                rotate=True,
            )
    return fig.svg()


def fig_weights() -> str:
    groups = [("embedding", "embedding weights"), ("attention", "attention weights"),
              ("mlp", "MLP weights"), ("unembedding", "unembedding weights"),
              ("heads", "attention head roles"), ("unembed_spectrum", "unembedding frequencies")]  # fmt: skip
    rows = [*CONSTANT_CELLS, "L2-ln-anneal", "L2-ln-anneal-heldwd"]
    left, top, rh, pw, gx = 300, 40, 22, 84, 12
    fig = Figure(
        left + len(groups) * (pw + gx),
        top + rh * len(rows) + 40,
        "Weight movement over 10k steps by part of the network",
    )
    for g, (key, name) in enumerate(groups):
        p = fig.panel(
            left + g * (pw + gx),
            top,
            pw,
            rh * len(rows),
            Scale(0, 1, 0, pw),
            Scale(0, len(rows), 0, rh * len(rows)),
        )
        words = name.split(" ")
        fig.text(p.x0, top - 22, words[0], cls="ink", weight="600")
        fig.text(p.x0, top - 8, " ".join(words[1:]), cls="ink", weight="600")
        p.xgrid([0, 0.5, 1], ["0", ".5", "1"])
        for r, cell in enumerate(rows):
            y = r + 0.5
            vals = [
                pt[key]
                for w in W[cell].values()
                for pt in w["curve"]
                if pt["lag"] == 10_000 and pt[key] is not None
            ]
            for i, v in enumerate(vals):
                p.dot(
                    min(v, 1),
                    y + jitter(i, r + g) * 0.8,
                    "ink2",
                    r=2.6,
                    opacity=0.55,
                    tip=f"{LABELS[cell]}: {v:.2f}",
                )
            if vals:
                p.tick(min(mean(vals), 1), y, 8, "s2")
        p.frame()
    for r, cell in enumerate(rows):
        fig.text(
            left - 8, top + (r + 0.5) * rh + 4, LABELS[cell], cls="ink", anchor="end"
        )
    fig.text(
        left + len(groups) * (pw + gx) / 2,
        fig.height - 4,
        "moved over 10k steps (0 = unchanged, 1 = unrelated)",
        cls="ink2",
        anchor="middle",
    )
    return fig.svg()


def fig_tail() -> str:
    rows = TAIL_CELLS
    vals = {c: [s["tail"]["occupancy"] for _, s in sorted(S[c].items())] for c in rows}
    return strip(rows, vals, 1.0, [0, 0.25, 0.5, 0.75, 1],
                 "share of evals in the last 10k steps at ≥ 95% test accuracy",
                 "Late-training stability per seed")  # fmt: skip


# ---------------------------------------------------------------- tables
def tail_table() -> str:
    head = ("<tr><th>arm</th><th>seeds</th><th>seeds with a collapse in the last 10k steps</th>"
            "<th>runs ending below 95%</th><th>worst accuracy in last 10k</th></tr>")  # fmt: skip
    rows = []
    for c in TAIL_CELLS:
        rows.append(
            f"<tr><td>{LABELS[c]}</td><td>{N[c]}</td><td>{tail[c]}</td><td>{end_broken[c]}</td>"
            f"<td>{min(s['tail']['min_acc'] for s in S[c].values()):.2f}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def collapse_table() -> str:
    rows = "".join(
        f"<tr><td>{LABELS[c]}</td><td>{N[c]}</td><td>{min(collapses(c))}–{max(collapses(c))}</td>"
        f"<td>{st.median(collapses(c)):g}</td></tr>"
        for c in [
            "L1-noln",
            "L2-noln",
            "L2-noln-aux",
            "L1-ln",
            "L2-ln",
            "L3-ln",
            "L2-ln-aux",
        ]
    )
    return ("<table><tr><th>cell</th><th>seeds</th><th>collapses (range)</th><th>median</th></tr>"
            f"{rows}</table>")  # fmt: skip


def level_table(cells: list[str]) -> str:
    rows = "".join(
        f"<tr><td>{LABELS[c]}</td>"
        + "".join(
            f"<td>{level_at(c, lvl, lag):.3f}</td>"
            for lag in (1000, 10_000)
            for lvl, _, _ in LEVELS
        )
        + "</tr>"
        for c in cells
    )
    head = "".join(
        f"<th>{n} @{lag // 1000}k</th>" for lag in (1000, 10_000) for _, _, n in LEVELS
    )
    return f"<table><tr><th>cell</th>{head}</tr>{rows}</table>"


def weights_table() -> str:
    keys = [
        "embedding",
        "attention",
        "mlp",
        "unembedding",
        "heads",
        "heads_raw",
        "unembed_spectrum",
    ]
    rows = "".join(
        f"<tr><td>{LABELS[c]}</td>" + "".join(
            f"<td>{v:.2f}</td>" if (v := weight_at(c, k)) is not None else "<td>–</td>" for k in keys
        ) + "</tr>"
        for c in [*CONSTANT_CELLS, "L2-ln-dropout", "L2-ln-lrdrop", "L2-ln-anneal", "L2-ln-anneal-heldwd"]
    )  # fmt: skip
    return f"<table><tr><th>cell</th>{''.join(f'<th>{k}</th>' for k in keys)}</tr>{rows}</table>"


# ---------------------------------------------------------------- page
def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=HERE).stdout.strip()  # fmt: skip


def body() -> str:
    lv = lambda c, k: f"{level_at(c, k):.2f}"  # noqa: E731
    wt_f = lambda c, k: weight_at(c, k)  # noqa: E731
    wt = lambda c, k: f"{weight_at(c, k):.2f}"  # noqa: E731
    r_emb = event_ratio(["L2-ln", "L3-ln", "L2-ln-dropout"], "embedding")
    r_neu = event_ratio(["L2-ln", "L3-ln", "L2-ln-dropout"], "neurons")
    r_l1 = event_ratio(["L1-ln"], "embedding")
    n_l1_events = len(events("L1-ln"))
    c = collapses
    return f"""{STYLE}
<p><em>Modular addition (a + b mod 113), small transformers trained full-batch with AdamW
(learning rate 10⁻³, weight decay 1.0) for 50k steps. {sum(N.values())} runs across
{len(S)} configurations. Results as of 2026-11-03.</em></p>

<h2>Summary</h2>
<p>We asked how stable a grokked modular-addition circuit is after it forms, at three levels:
the capability (which test answers it gets right), the embedding's Fourier frequencies, and
the MLP neurons serving each frequency, then extended this to every weight matrix.</p>
<ul>
<li><strong>Collapses need LayerNorm and at least two layers.</strong> 2-layer LayerNorm models
lose and regain test accuracy {min(c("L2-ln"))}–{max(c("L2-ln"))} times per run
({N["L2-ln"]} seeds), 3-layer {min(c("L3-ln"))}–{max(c("L3-ln"))} ({N["L3-ln"]} seeds).
1-layer LayerNorm models collapse at most {max(c("L1-ln"))} time ({N["L1-ln"]} seeds), and
no-LayerNorm models at most {max(c("L2-noln") + c("L1-noln"))}.</li>
<li><strong>With a constant learning rate, LayerNorm models keep drifting after grokking even
when they never collapse.</strong> Over 10k steps a 1-layer LayerNorm model's embedding spectrum
moves {lv("L1-ln", "embedding")} of the way to chance, against {lv("L1-noln", "embedding")} for Nanda
et al.'s LayerNorm-free model; its embedding, attention and MLP weights move {wt("L1-ln", "embedding")},
{wt("L1-ln", "attention")} and {wt("L1-ln", "mlp")} of the way to unrelated, against at most
{max(mean([wt_f("L1-noln", k)]) for k in ("embedding", "attention", "mlp")):.2f}. Neuron populations are the exception: they turn over
without LayerNorm too ({lv("L1-noln", "neurons")} vs {lv("L1-ln", "neurons")} for the 1-layer pair). In LayerNorm
models the unembedding moves least.</li>
<li><strong>At a 10k-step lag, the share of test answers that change is below the embedding's
movement, which is below the neurons', in every configuration.</strong> The capability measure is a
literal share and the other two are relative to chance, so this compares different scales. In the
2–3 layer LayerNorm models the answers still change, by {lv("L2-ln", "capability")}–{lv("L3-ln", "capability")}, because of the collapses.</li>
<li><strong>Repairs land back near the same circuit.</strong> In 2–3 layer LayerNorm models the change
from just before a collapse to just after recovery is close to the change over a collapse-free span
of the same length (median ratio {r_emb:.1f}× for the embedding, {r_neu:.1f}× for neurons,
{len(all_events)} collapses in all). The {n_l1_events} collapses in 1-layer LayerNorm models
reorganise the embedding more ({r_l1:.1f}×).</li>
<li><strong>Annealing the learning rate ends on a working model.</strong> In the last 10k steps,
{tail["L2-ln"]} of 10 constant-rate 2-layer LayerNorm seeds collapse and {end_broken["L2-ln"]} end
below 95% accuracy; with a gradual anneal {tail["L2-ln-anneal"]} of 10 collapse and none end broken
(Fisher p ≈ {p_anneal:.3f}). Keeping the per-step weight-decay shrinkage fixed during the anneal
changes little ({tail["L2-ln-anneal-heldwd"]} of 10, p ≈ {p_held:.3f} vs constant), so the anneal's effect
comes mostly from the lower learning rate, not from weaker weight decay. Under annealing, drift stops.</li>
<li><strong>The aux loss stabilises the capability but not the implementation</strong>:
{tail["L2-ln-aux"]} of {N["L2-ln-aux"]} seeds collapse late, while embedding and neurons drift as much
as in the unstable baseline.</li>
<li><strong>Two stabilisers failed.</strong> Dropout 0.1 left collapses within the baseline's range
({min(c("L2-ln-dropout"))}–{max(c("L2-ln-dropout"))} per run, 3 seeds); dropping the learning rate 10× instantly at
20k froze each seed in whatever state it was in, one of them broken.</li>
</ul>
<p><strong>Caveats.</strong> Several configurations have 3 seeds. The decay-rate-held arm's weights were
still shrinking at the end of training, so a longer run could differ. The neuron measure includes
real fast turnover at short lags, so read neuron curves for their trend, not their intercept. The
head-role measure starts high even at short lags in the aux models, whose heads are nearly identical, so don't read it there. Raw weight change
counts function-preserving changes (rotating the residual stream) as movement. Test loss oscillates in
every configuration, including without LayerNorm; only the accuracy collapses depend on LayerNorm.</p>

<h2>Setup</h2>
<p>Each run trains a pre-LayerNorm transformer (width 128, 4 heads, ReLU MLP) on 30% of all
113² pairs and evaluates on the rest every 100 steps. "No LayerNorm" removes every LayerNorm,
as in Nanda et al. The <em>aux loss</em> also trains the first layer's output, read through the
model's own unembedding, to predict the answer (λ = 0.3). Learning-rate arms change only the
schedule after step 20k: an instant 10× drop, or a linear anneal to 10% between 20k and 40k
(with, in one arm, weight decay raised by the same factor so that learning rate × weight decay,
AdamW's per-step shrinkage, stays fixed).</p>
<p>A <em>collapse</em> is test accuracy falling below 0.90 after the run has first reached 0.95,
counted once until it recovers to 0.95. Stability is measured over steps 30k–50k (40k–50k for the
annealed arms, so the learning rate is flat), as how much changes between two evals some lag apart:</p>
<ul>
<li><strong>test answers</strong>: share of test examples whose right/wrong status differs.</li>
<li><strong>embedding frequencies</strong>: share of the embedding's Fourier power that sits on different
frequencies, divided by the same quantity with frequency identities shuffled (1 = no more alike than chance).</li>
<li><strong>neurons per frequency</strong>: for each frequency, the set of MLP neurons devoting ≥ 10% of their
2-D (a, b) Fourier power to it; 1 − chance-corrected share of that set still serving it after the lag
(1 = no more overlap than if neuron identities were shuffled).</li>
<li><strong>weights</strong>, from checkpoints every 1k steps: per group of matrices,
‖ΔW‖² / (‖W‖² + ‖W′‖²) (1 = what two unrelated matrices give); each attention head's frequency role,
relative to the other heads; and the unembedding's frequency spectrum, chance-corrected like the embedding.</li>
</ul>

<h2>A collapse is a sudden loss of test accuracy that repairs itself; only multi-layer LayerNorm models have them</h2>
<figure>{fig_accuracy()}<figcaption>Test accuracy over training for one seed of four configurations.</figcaption></figure>
<figure>{fig_collapses()}<figcaption>Collapses after first grokking, one dot per seed; orange tick = mean.
Number of seeds in brackets.</figcaption></figure>
<details><summary>Table</summary>{collapse_table()}</details>

<h2>With a constant learning rate, LayerNorm models keep drifting after grokking: test answers least, neurons most</h2>
<p>Each panel shows, for one configuration, how much each level changes between two evals as a function of the gap between them.
Compare the 1-layer models in the first column: same depth, with and without LayerNorm.</p>
{legend([(s, n, "line") for _, s, n in LEVELS])}
<figure>{fig_levels(CONSTANT_CELLS, "Change versus lag, constant learning rate")}
<figcaption>Bold = mean over seeds, thin = individual seeds. Window: steps 30k–50k.</figcaption></figure>
<details><summary>Table</summary>{level_table(CONSTANT_CELLS)}</details>

<h2>In LayerNorm models the drift covers the whole network except the unembedding</h2>
<figure>{fig_weights()}<figcaption>Change over 10k steps, one dot per seed; orange tick = mean. Solid parameters on the left four panels;
the right two measure roles that don't depend on the residual stream's basis.</figcaption></figure>
<details><summary>Table (lag 10k; heads_raw is the head measure without chance correction)</summary>{weights_table()}</details>

<h2>After a collapse, the model returns to roughly the circuit it had</h2>
<figure>{fig_repairs()}{legend([("s2", "1-layer LayerNorm", "dot"), ("ink2", "all other configurations", "hollow")])}
<figcaption>Each point is one collapse that recovered. Vertical: change from the last eval at ≥ 95% before the collapse to the first eval back at ≥ 95%.
Horizontal: mean change over collapse-free spans of the same length within 5k steps of it. Dashed = equal.</figcaption></figure>

<h2>Annealing the learning rate ends on a working model, and the drift stops</h2>
<figure>{fig_tail()}<figcaption>Share of the last 10k steps each seed spends at ≥ 95% test accuracy; orange tick = mean.</figcaption></figure>
{tail_table()}
<p>Fisher exact tests on seeds with a late collapse: anneal vs constant p ≈ {p_anneal:.3f}; decay-rate-held vs constant
p ≈ {p_held:.3f}; decay-rate-held vs anneal p ≈ {p_held_vs_anneal:.2f}.</p>
{legend([(s, n, "line") for _, s, n in LEVELS])}
<figure>{fig_levels(LR_CELLS, "Change versus lag, learning-rate arms")}
<figcaption>Constant rate (30k–50k window) against the three learning-rate arms (instant drop: 30k–50k; annealed: 40k–50k, so lags stop at 10k).</figcaption></figure>
<details><summary>Table</summary>{level_table(LR_CELLS)}</details>
<p>The decay-rate-held arm ends with smaller weights (total norm {norm_at("L2-ln-anneal-heldwd", -1):.0f} vs
{norm_at("L2-ln-anneal", -1):.0f} for the plain anneal at 50k) and was still shrinking, which is why its raw
weights keep moving in the figure above while its roles and capability don't.</p>

<h2>What we didn't find or didn't test</h2>
<ul>
<li><strong>No stabiliser other than annealing and the aux loss worked.</strong> Dropout 0.1 didn't reduce collapses;
an instant learning-rate drop froze whatever state each seed was in.</li>
<li><strong>We didn't show why LayerNorm causes collapses.</strong> The configurations separate LayerNorm and depth,
but nothing here tests a mechanism.</li>
<li><strong>We didn't reverse-engineer the drifting circuits.</strong> Every measure is in Fourier-frequency terms
(roles, spectra, populations), so "the same circuit" means the same frequencies served by some neurons and heads,
not a verified identical algorithm.</li>
<li><strong>The decay-rate-held arm didn't reach equilibrium</strong> in 50k steps, so whether held weight decay
eventually destabilises the annealed model is open.</li>
<li><strong>One task, one model size, one optimiser setting</strong> (AdamW, learning rate 10⁻³, weight decay 1.0,
30% training data).</li>
<li><strong>3 seeds</strong> for the no-LayerNorm, 3-layer, dropout and instant-drop configurations.</li>
</ul>

<h2>Reproduce</h2>
<p>Code: <a href="https://github.com/brendanlong/lens-loss-grokking-experiment">brendanlong/lens-loss-grokking-experiment</a>,
PRs #8 and #9; this report built at commit <code>{commit()}</code>. The grid is a Snakemake workflow
(<code>config/config.yaml</code>) that submits training through gpuc:</p>
<pre>uv run --with "gpu-coordinator @ git+https://github.com/brendanlong/gpu-coordinator" \\
  snakemake --workflow-profile workflow/profiles/gpuc --config results_dir=/abs/path</pre>
<p>Full outputs (traces, checkpoints every 1k steps, summaries, figures; 14.5 GB) are in
<code>s3://brendanlong-experiments/grok_lens/stability-grid/</code>; training curves are in the
<a href="https://wandb.ai/brendanlong-com/grok-lens">grok-lens wandb project</a> as
<code>p113-&lt;cell&gt;-s&lt;seed&gt;-grid</code>. This page is built by <code>report/build_report.py</code>
from the copies in <code>report/data/</code>.</p>
"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frag = OUT / "body.html"
    frag.write_text(body())
    subprocess.run(
        [sys.executable, str(RENDER), str(frag), "--title",
         "Grokked modular addition: stable answers, drifting circuits", "-o", str(OUT / "index.html")],
        check=True,
    )  # fmt: skip
    frag.unlink()


if __name__ == "__main__":
    main()

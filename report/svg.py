"""Minimal inline-SVG charting for the report.

Marks carry literal light-theme colours (so a saved, stylesheet-free copy
still renders) plus a class the page stylesheet maps to the theme variables.
"""

import math
from collections.abc import Callable, Sequence
from html import escape

PALETTE = {
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "s3": "#1baf7a",
    "s4": "#eda100",
}
STROKE_CLASS = {
    "ink": "lk",
    "ink2": "lk2",
    "grid": "grid",
    "axis": "axis",
    "s1": "s1",
    "s2": "s2",
    "s3": "s3",
    "s4": "s4",
}
STYLE = """<style>
svg.chart text { font: 11px system-ui, sans-serif; }
svg.chart .ink { fill: var(--ink); } svg.chart .ink2 { fill: var(--ink-2); }
svg.chart .grid { stroke: var(--grid); } svg.chart .axis { stroke: var(--ink-2); }
svg.chart .s1 { stroke: var(--series-1); } svg.chart .s2 { stroke: var(--series-2); }
svg.chart .s3 { stroke: var(--series-3); } svg.chart .s4 { stroke: var(--series-4); }
svg.chart .f1 { fill: var(--series-1); } svg.chart .f2 { fill: var(--series-2); }
svg.chart .f3 { fill: var(--series-3); } svg.chart .f4 { fill: var(--series-4); }
svg.chart .fi { fill: var(--ink); } svg.chart .fi2 { fill: var(--ink-2); }
svg.chart .lk { stroke: var(--ink); } svg.chart .lk2 { stroke: var(--ink-2); }
figure { margin: 1.5rem 0; } figcaption { color: var(--ink-2); font-size: 0.9rem; }
.legend { display: flex; flex-wrap: wrap; gap: 0.3rem 1.2rem; font-size: 0.85rem; color: var(--ink-2); margin: 0.3rem 0; }
.legend span { white-space: nowrap; }
</style>"""


def fmt(v: float) -> str:
    return f"{v:.1f}"


class Scale:
    def __init__(self, lo: float, hi: float, a: float, b: float, log: bool = False):
        self.lo, self.hi, self.a, self.b, self.log = lo, hi, a, b, log

    def __call__(self, v: float) -> float:
        f: Callable[[float], float] = math.log10 if self.log else (lambda x: x)
        t = (f(v) - f(self.lo)) / (f(self.hi) - f(self.lo))
        return self.a + t * (self.b - self.a)


class Panel:
    """A plot area at (x0, y0) of size (w, h) inside a Figure."""

    def __init__(self, fig: "Figure", x0: float, y0: float, w: float, h: float,
                 xs: Scale, ys: Scale):  # fmt: skip
        self.fig, self.x0, self.y0, self.w, self.h = fig, x0, y0, w, h
        self.xs, self.ys = xs, ys

    def X(self, v: float) -> float:
        return self.x0 + self.xs(v)

    def Y(self, v: float) -> float:
        return self.y0 + self.ys(v)

    def title(self, text: str) -> None:
        self.fig.text(self.x0, self.y0 - 8, text, cls="ink", weight="600")

    def xgrid(self, ticks: Sequence[float], labels: Sequence[str] | None = None,
              show_labels: bool = True) -> None:  # fmt: skip
        for i, t in enumerate(ticks):
            x = self.X(t)
            self.fig.line(x, self.y0, x, self.y0 + self.h, "grid")
            if show_labels:
                lab = labels[i] if labels else f"{t:g}"
                self.fig.text(
                    x, self.y0 + self.h + 13, lab, cls="ink2", anchor="middle"
                )

    def ygrid(self, ticks: Sequence[float], labels: Sequence[str] | None = None,
              show_labels: bool = True) -> None:  # fmt: skip
        for i, t in enumerate(ticks):
            y = self.Y(t)
            self.fig.line(self.x0, y, self.x0 + self.w, y, "grid")
            if show_labels:
                lab = labels[i] if labels else f"{t:g}"
                self.fig.text(self.x0 - 5, y + 4, lab, cls="ink2", anchor="end")

    def frame(self) -> None:
        self.fig.line(
            self.x0, self.y0 + self.h, self.x0 + self.w, self.y0 + self.h, "axis"
        )
        self.fig.line(self.x0, self.y0, self.x0, self.y0 + self.h, "axis")

    def polyline(self, xs: Sequence[float], ys: Sequence[float], series: str,
                 width: float = 2, opacity: float = 1, dash: str = "") -> None:  # fmt: skip
        pts = " ".join(
            f"{fmt(self.X(x))},{fmt(self.Y(y))}" for x, y in zip(xs, ys, strict=True)
        )
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.fig.parts.append(
            f'<polyline class="{STROKE_CLASS[series]}" points="{pts}" fill="none" '
            f'stroke="{PALETTE[series]}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}" stroke-linejoin="round"{extra}/>'
        )

    def dot(self, x: float, y: float, fill: str, r: float = 3.5, opacity: float = 1,
            hollow: bool = False, tip: str = "") -> None:  # fmt: skip
        cls = {
            "s1": "f1",
            "s2": "f2",
            "s3": "f3",
            "s4": "f4",
            "ink": "fi",
            "ink2": "fi2",
        }[fill]
        title = f"<title>{escape(tip)}</title>" if tip else ""
        if hollow:
            self.fig.parts.append(
                f'<circle class="{STROKE_CLASS[fill]}" cx="{fmt(self.X(x))}" cy="{fmt(self.Y(y))}" r="{r}" '
                f'fill="none" stroke="{PALETTE[fill]}" stroke-width="1.3" '
                f'stroke-opacity="{opacity}">{title}</circle>'
            )
        else:
            self.fig.parts.append(
                f'<circle class="{cls}" cx="{fmt(self.X(x))}" cy="{fmt(self.Y(y))}" r="{r}" '
                f'fill="{PALETTE[fill]}" fill-opacity="{opacity}">{title}</circle>'
            )

    def tick(
        self, x: float, y: float, half: float, series: str, width: float = 2.5
    ) -> None:
        self.fig.line(
            self.X(x), self.Y(y) - half, self.X(x), self.Y(y) + half, series, width
        )


class Figure:
    def __init__(self, width: float, height: float, label: str):
        self.width, self.height, self.label = width, height, label
        self.parts: list[str] = []

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str,
             width: float = 1, dash: str = "") -> None:  # fmt: skip
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line class="{STROKE_CLASS[cls]}" x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
            f'stroke="{PALETTE[cls if cls in PALETTE else "ink2"] if cls != "axis" else PALETTE["ink2"]}" '
            f'stroke-width="{width}"{extra}/>'
        )

    def text(self, x: float, y: float, s: str, cls: str = "ink", anchor: str = "start",
             weight: str = "400", size: float | None = None, rotate: bool = False) -> None:  # fmt: skip
        tr = f' transform="rotate(-90 {fmt(x)} {fmt(y)})"' if rotate else ""
        sz = f' font-size="{size}"' if size else ""
        self.parts.append(
            f'<text class="{cls}" x="{fmt(x)}" y="{fmt(y)}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{PALETTE[cls]}"{sz}{tr}>{escape(s)}</text>'
        )

    def panel(
        self, x0: float, y0: float, w: float, h: float, xs: Scale, ys: Scale
    ) -> Panel:
        return Panel(self, x0, y0, w, h, xs, ys)

    def svg(self) -> str:
        return (
            f'<svg class="chart" viewBox="0 0 {self.width} {self.height}" role="img" '
            f'aria-label="{escape(self.label)}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(self.parts)
            + "</svg>"
        )


def legend(items: Sequence[tuple[str, str, str]]) -> str:
    """HTML legend: (series, label, kind) with kind 'line' | 'dot' | 'hollow'."""
    out = []
    for series, label, kind in items:
        c = PALETTE[series]
        if kind == "line":
            sw = f'<svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" stroke="{c}" class="{series}" stroke-width="2.5"/></svg>'
        elif kind == "hollow":
            sw = f'<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="{c}" class="{series}" stroke-width="1.5"/></svg>'
        else:
            cls = {
                "s1": "f1",
                "s2": "f2",
                "s3": "f3",
                "s4": "f4",
                "ink": "fi",
                "ink2": "fi2",
            }[series]
            sw = f'<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="{c}" class="{cls}"/></svg>'
        out.append(f"<span>{sw} {escape(label)}</span>")
    return '<div class="legend">' + "".join(out) + "</div>"

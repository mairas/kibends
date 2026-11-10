"""Draw a smoothing run for review; needs the `plot` extra (matplotlib)."""

# matplotlib's stubs leave **kwargs untyped on its drawing methods, which
# pyright strict reports at every call; this module only draws.
# pyright: reportUnknownMemberType=false

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kibends.chain import Point2
from kibends.energy import DistanceField, FloatArray

MARGIN = 2.0
"""mm of board drawn around the chain."""
GRID_POINTS = 300
"""Distance samples along the longer side of each drawn area."""
FIGURE_WIDTH = 9.0
TITLE_AND_LEGEND_HEIGHT = 2.0
DPI = 120


@dataclass(frozen=True)
class Panel:
    """One chain over the field its centreline must stay out of."""

    field: DistanceField
    original: FloatArray
    optimised: FloatArray
    simplified: FloatArray
    title: str
    vias: Sequence[tuple[Point2, Point2]] = ()
    """The start and end of each moving via the chain ends at."""


def render(path: Path, panels: Sequence[Panel]) -> None:
    """Write a figure with one subplot per chain, each over its own field.

    Grey is where the centreline would violate clearance (other copper
    inflated by clearance plus half the track width). The y axis points down,
    as in KiCad. The file format follows the extension of `path`.
    """
    extents = [_extent(p) for p in panels]
    heights = [
        FIGURE_WIDTH * float(high[1] - low[1]) / float(high[0] - low[0])
        + TITLE_AND_LEGEND_HEIGHT
        for low, high in extents
    ]
    figure = Figure(figsize=(FIGURE_WIDTH, sum(heights)))
    axes = figure.subplots(
        len(panels), 1, squeeze=False, gridspec_kw={"height_ratios": heights}
    )
    for ax, panel, (low, high) in zip(axes[:, 0], panels, extents, strict=True):
        _draw(ax, panel, low, high)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")


def _extent(panel: Panel) -> tuple[FloatArray, FloatArray]:
    points = np.concatenate([panel.original, panel.optimised])
    return points.min(axis=0) - MARGIN, points.max(axis=0) + MARGIN


def _draw(axes: Axes, panel: Panel, low: FloatArray, high: FloatArray) -> None:
    step = float((high - low).max()) / GRID_POINTS
    xs = np.arange(low[0], high[0] + step, step)
    ys = np.arange(low[1], high[1] + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = panel.field.signed_distance(
        np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    )[0].reshape(grid_x.shape)

    if distance.min() < 0.0:
        axes.contourf(
            grid_x, grid_y, distance, levels=[distance.min(), 0.0], colors=["#d9d9d9"]
        )
        axes.contour(
            grid_x, grid_y, distance, levels=[0.0], colors=["#555555"], linewidths=1
        )
    axes.plot(*panel.original.T, "--", color="#999999", label="original")
    axes.plot(
        *panel.optimised.T, "-", color="#1f77b4", linewidth=1.5, label="optimised"
    )
    axes.plot(
        *panel.simplified.T,
        "o",
        color="#d62728",
        markersize=3,
        label="result vertices",
    )
    for k, (start, end) in enumerate(panel.vias):
        axes.plot(
            *start,
            "o",
            markerfacecolor="none",
            color="#2ca02c",
            markersize=10,
            label=None if k else "via start",
        )
        axes.plot(
            *end, "o", color="#2ca02c", markersize=10, label=None if k else "via end"
        )
    axes.set_aspect("equal")
    axes.set_xlim(low[0], high[0])
    axes.set_ylim(high[1], low[1])
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4)
    axes.set_title(panel.title, fontsize=9)

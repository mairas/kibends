"""Draw a smoothing run for review; needs the `plot` extra (matplotlib)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from kibends.energy import DistanceField, FloatArray
from kibends.optimize import FrozenVertex

MARGIN = 2.0
"""mm of board drawn around the chain."""
GRID_POINTS = 300
"""Distance samples along the longer side of the drawn area."""
FIGURE_WIDTH = 9.0
TITLE_AND_LEGEND_HEIGHT = 2.0
DPI = 120


def render(
    path: Path,
    field: DistanceField,
    *,
    original: FloatArray,
    optimised: FloatArray,
    simplified: FloatArray,
    frozen: Sequence[FrozenVertex],
    title: str,
) -> None:
    """Write a figure of the chain over the region its centreline must avoid.

    Grey is where the centreline would violate clearance (other copper
    inflated by clearance plus half the track width). The y axis points down,
    as in KiCad. The file format follows the extension of `path`.
    """
    low = np.minimum(original.min(axis=0), optimised.min(axis=0)) - MARGIN
    high = np.maximum(original.max(axis=0), optimised.max(axis=0)) + MARGIN
    step = float((high - low).max()) / GRID_POINTS
    xs = np.arange(low[0], high[0] + step, step)
    ys = np.arange(low[1], high[1] + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = field.signed_distance(
        np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    )[0].reshape(grid_x.shape)

    width, height = high - low
    figure = Figure(
        figsize=(FIGURE_WIDTH, FIGURE_WIDTH * height / width + TITLE_AND_LEGEND_HEIGHT)
    )
    axes = figure.subplots()
    if distance.min() < 0.0:
        axes.contourf(
            grid_x, grid_y, distance, levels=[distance.min(), 0.0], colors=["#d9d9d9"]
        )
        axes.contour(
            grid_x, grid_y, distance, levels=[0.0], colors=["#555555"], linewidths=1
        )
    axes.plot(*original.T, "--", color="#999999", label="original")
    axes.plot(*optimised.T, "-", color="#1f77b4", linewidth=1.5, label="optimised")
    axes.plot(
        *simplified.T, "o", color="#d62728", markersize=3, label="result vertices"
    )
    if frozen:
        axes.plot(
            *np.array([v.position for v in frozen]).T,
            "x",
            color="#000000",
            markersize=8,
            label="frozen vertices",
        )
    axes.set_aspect("equal")
    axes.set_xlim(low[0], high[0])
    axes.set_ylim(high[1], low[1])
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4)
    axes.set_title(title, fontsize=9)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")

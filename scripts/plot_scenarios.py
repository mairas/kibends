"""Render the optimizer's synthetic scenarios to PNGs for visual review.

Usage: uv run --extra plot python scripts/plot_scenarios.py OUTPUT_DIR
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import scenarios

from kibends.optimize import optimize
from kibends.simplify import simplify

GRID_STEP = 0.02
MARGIN = 2.5
FIGURE_WIDTH = 9.0
TITLE_AND_LEGEND_HEIGHT = 2.0

SCENARIOS = [
    scenarios.wrap(stiffness=0.1),
    scenarios.wrap(stiffness=1.5),
    scenarios.channel(repulsion=0.0),
    scenarios.channel(repulsion=1.0),
    scenarios.topology(),
]


def render(scenario: scenarios.Scenario, output: Path) -> Path:
    result = optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params
    )
    simplified = simplify(result.polyline, result.fixed, scenario.field)

    low = scenario.polyline.min(axis=0) - MARGIN
    high = scenario.polyline.max(axis=0) + MARGIN
    xs = np.arange(low[0], high[0], GRID_STEP)
    ys = np.arange(low[1], high[1], GRID_STEP)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance, _ = scenario.field.signed_distance(
        np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    )
    distance = distance.reshape(grid_x.shape)

    width, height = high - low
    figure, axes = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_WIDTH * height / width + TITLE_AND_LEGEND_HEIGHT)
    )
    axes.contourf(
        grid_x, grid_y, distance, levels=[distance.min(), 0.0], colors=["#d9d9d9"]
    )
    axes.contour(
        grid_x, grid_y, distance, levels=[0.0], colors=["#555555"], linewidths=1
    )
    axes.plot(*scenario.polyline.T, "--", color="#999999", label="initial")
    axes.plot(
        *result.polyline.T, "-", color="#1f77b4", linewidth=1.5, label="optimised"
    )
    axes.plot(
        *simplified.T, "o", color="#d62728", markersize=3, label="simplified vertices"
    )
    axes.set_aspect("equal")
    axes.set_xlim(low[0], high[0])
    axes.set_ylim(low[1], high[1])
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3)
    after = result.after
    axes.set_title(
        f"{scenario.name}: {result.termination.value}, {result.iterations} iterations\n"
        f"length {result.before.length:.3f} → {after.length:.3f} mm, "
        f"tightest bend {after.tightest_bend_radius:.3f} mm, "
        f"min clearance {after.min_clearance:.4f} mm, "
        f"{len(simplified) - 1} segments simplified",
        fontsize=9,
    )
    path = output / f"{scenario.name}.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        print(render(scenario, output))


if __name__ == "__main__":
    main()

"""Render the optimizer's synthetic scenarios to PNGs for visual review.

Usage: uv run --extra plot python scripts/plot_scenarios.py OUTPUT_DIR
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import scenarios

from kibends.optimize import optimize
from kibends.plot import Panel, render
from kibends.simplify import simplify

SCENARIOS = [
    scenarios.wrap(),
    scenarios.channel(repulsion=0.0),
    scenarios.channel(repulsion=1.0),
    scenarios.topology(),
]


def plot_scenario(scenario: scenarios.Scenario, output: Path) -> Path:
    result = optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params
    )
    simplified = simplify(result.polyline, result.fixed, scenario.field)
    after = result.after
    path = output / f"{scenario.name}.png"
    panel = Panel(
        scenario.field,
        original=scenario.polyline,
        optimised=result.polyline,
        simplified=simplified,
        frozen=result.frozen,
        title=f"{scenario.name}: {result.termination.value}, "
        f"{result.iterations} iterations\n"
        f"length {result.before.length:.3f} → {after.length:.3f} mm, "
        f"tightest bend {after.tightest_bend_radius:.3f} mm, "
        f"min clearance {after.min_clearance:.4f} mm, "
        f"{len(simplified) - 1} segments simplified",
    )
    render(path, [panel])
    return path


def main() -> None:
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        print(plot_scenario(scenario, output))


if __name__ == "__main__":
    main()

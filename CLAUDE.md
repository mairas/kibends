# kibends

## Test speed

A slow suite slows every edit-test cycle for the person running it. Treat test run time as a requirement, not a detail.

- `./run test` (the fast tier) finishes in under 10 s wall-clock.
- No test in the fast tier takes more than 1 s on its own, setup included. Measure with `.venv/bin/python -m pytest -n0 -m "not slow" --durations=10`; the parallel run hides slow tests.
- Make a slow test faster before marking it `slow`: coarser optimizer settings, a smaller scenario, a fake instead of the real kicad-cli. Mark it `slow` only when it needs the real thing, such as the kicad-cli tests.
- Every change that adds or modifies tests checks the durations above and reports the fast-tier time.

## Commands

- `./run sync` installs dependencies, including the `plot` extra that `TestPlot` needs.
- `./run test` runs the fast tier; `./run test-all` runs everything, including `slow` and real kicad-cli tests.

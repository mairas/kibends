"""Draw the KiCad plugin's icons: a smooth track between two pads.

Writes the toolbar icons of the plugin's action (light and dark, 24 and 48 px)
into ``pcm/plugins/`` and the 64 px package icon that KiCad's Plugin and
Content Manager lists into ``pcm/resources/icon.png``. Needs the ``plot``
extra. Run it after changing the glyph and commit the PNGs:

    .venv/bin/python scripts/make_icons.py
"""

# pyright strict reports at every matplotlib call; this script only draws.
# pyright: reportUnknownMemberType=false

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

PCM = Path(__file__).resolve().parent.parent / "pcm"

# Dark on KiCad's light toolbars, light on its dark ones.
LIGHT_THEME = "#2b2b2b"
DARK_THEME = "#e0e0e0"
# Copper reads on both the light and the dark PCM dialog.
COPPER = "#c87533"

TOOLBAR_SIZES = (24, 48)
PACKAGE_SIZE = 64

# The glyph in a unit square, y up: pads at lower left and upper right, and
# a track that leaves each pad level and turns diagonally through two round
# bends.
PAD_RADIUS = 0.12
TRACK_WIDTH = 0.09
LEFT_PAD = (0.14, 0.25)
RIGHT_PAD = (0.86, 0.75)
# Where the track starts and stops turning.
BEND_START = 0.3
BEND_END = 0.7
BEND_SAMPLES = 64


def _track() -> tuple[np.ndarray, np.ndarray]:
    """The centreline: level out of each pad, a smoothstep in between."""
    x = np.linspace(LEFT_PAD[0], RIGHT_PAD[0], BEND_SAMPLES)
    t = np.clip((x - BEND_START) / (BEND_END - BEND_START), 0, 1)
    smooth = t * t * (3 - 2 * t)
    return x, LEFT_PAD[1] + (RIGHT_PAD[1] - LEFT_PAD[1]) * smooth


def draw(path: Path, size: int, colour: str) -> None:
    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    points_per_unit = size * 72 / dpi
    x, y = _track()
    ax.plot(
        x,
        y,
        color=colour,
        linewidth=TRACK_WIDTH * points_per_unit,
        solid_capstyle="round",
    )
    for centre in (LEFT_PAD, RIGHT_PAD):
        ax.add_patch(Circle(centre, PAD_RADIUS, color=colour))
    fig.savefig(path, dpi=dpi, transparent=True, metadata={"Software": None})


def main() -> None:
    for theme, colour in (("light", LIGHT_THEME), ("dark", DARK_THEME)):
        for size in TOOLBAR_SIZES:
            draw(PCM / "plugins" / f"icon-{theme}-{size}.png", size, colour)
    draw(PCM / "resources" / "icon.png", PACKAGE_SIZE, COPPER)


if __name__ == "__main__":
    main()

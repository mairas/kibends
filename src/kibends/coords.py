"""Coordinates as kibends prints them: relative to the board's grid origin,
which `snapshot.read_user_frame` reads, in mm with y pointing down."""

from dataclasses import dataclass

DECIMALS = 3


@dataclass(frozen=True)
class UserFrame:
    """Maps KiCad's board frame (mm, page top-left origin, y down) to the
    coordinates kibends prints."""

    origin: tuple[float, float] = (0.0, 0.0)
    """The grid origin in the board frame, mm."""

    def to_user(self, p: tuple[float, float]) -> tuple[float, float]:
        return p[0] - self.origin[0], p[1] - self.origin[1]

    def point(self, p: tuple[float, float]) -> str:
        """`p`, given in the board frame, as "(x, y)" in user coordinates."""
        # Adding 0.0 turns a -0.0 from rounding into 0.0.
        x, y = (round(v, DECIMALS) + 0.0 for v in self.to_user(p))
        return f"({x:.{DECIMALS}f}, {y:.{DECIMALS}f})"


BOARD_FRAME = UserFrame()
"""The board frame itself, as a grid origin at the page origin gives it."""

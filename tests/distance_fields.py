"""Analytic DistanceField and ObstacleField implementations for tests, Rust
fields of `kibends._core.testing` (`kibends_core::testing`).

Each shape is one obstacle, but `Walls`, two, so their smooth distance
(`DistanceField.smooth_distance`) is their exact one, and that of `Walls`
and `Union` the smooth minimum over theirs. `ReachIgnored` answers as
another field with no reach, and `NanField` NaN everywhere."""

from kibends._core.testing import (
    FAR,
    Box,
    Disc,
    FreeSpace,
    MisleadingWalls,
    NanField,
    ReachIgnored,
    Union,
    Walls,
)
from kibends.energy import BoolArray, FloatArray

__all__ = [
    "FAR",
    "BoolArray",
    "Box",
    "Disc",
    "FloatArray",
    "FreeSpace",
    "MisleadingWalls",
    "NanField",
    "ReachIgnored",
    "Union",
    "Walls",
]

"""The panel's between-run reads of the board (`Prober`): which board KiCad
shows, what is selected and, for a board not read before, its totals and
ground nets."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import kipy
from kipy.errors import ConnectionError as KiCadConnectionError

from kibends.run import KICAD_TIMEOUT_MS, open_board
from kibends.snapshot import BoardSnapshot, GroundNets, SelectionError, read_selection


@dataclass(frozen=True)
class BoardId:
    """Which board KiCad shows."""

    name: str
    """The board's file name."""
    project: str
    """The project's directory; empty when KiCad gives none."""


class Counts(NamedTuple):
    tracks: int
    """Tracks and arcs, those of selected groups included."""
    vias: int


@dataclass(frozen=True)
class Totals:
    """What a board holds, as a whole-board run reads it."""

    tracks: int
    vias: int
    ground: GroundNets
    """The ground nets a run on the board detects."""


@dataclass(frozen=True)
class Probe:
    board: BoardId
    selected: Counts | None
    """What a run would read from the selection; None when nothing is
    selected, so a run reads the whole board."""
    totals: Totals | None
    """None when the board is the one the caller knows."""


def connect() -> kipy.KiCad:
    return kipy.KiCad(timeout_ms=KICAD_TIMEOUT_MS)


class Prober:
    """Reads the board on one KiCad connection of its own, made on first
    use and made again after it is lost. Not for two threads at once."""

    def __init__(self, make: Callable[[], kipy.KiCad] = connect) -> None:
        self._make = make
        self._kicad: kipy.KiCad | None = None

    def __call__(self, known: BoardId | None) -> Probe:
        """The board KiCad shows and its selection, with its totals unless
        it is `known`."""
        try:
            if self._kicad is None:
                self._kicad = self._make()
            board = open_board(self._kicad)
        except KiCadConnectionError:
            self._kicad = None
            raise
        document = board.document
        found = BoardId(document.board_filename, document.project.path)
        totals = None
        if found != known:
            # Reads more than detection needs; whether a lighter read pays
            # is to be measured on a large board.
            ground = BoardSnapshot.read(board, ()).ground_nets()
            totals = Totals(len(board.get_tracks()), len(board.get_vias()), ground)
        selected = None
        if board.get_selection():
            try:
                selection = read_selection(board)
                selected = Counts(
                    len({s.id for s in selection.segments}), len(selection.vias)
                )
            except SelectionError:
                selected = Counts(0, 0)
        return Probe(found, selected, totals)

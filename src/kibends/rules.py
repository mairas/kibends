"""Edge and hole clearance from the project's Board Setup constraints.

kicad-python cannot read a board's design rules, so a run reads the two
it models from the project file, the JSON `.kicad_pro` in the project's
directory, at `board.design_settings.rules`: `min_copper_edge_clearance`
and `min_hole_clearance`, in mm. A value the caller gives wins over the
file. With no project, no project file or no such rule, the defaults of
`obstacles.EDGE_CLEARANCE` and `HOLE_CLEARANCE` apply; a file that is no
project file, or a rule that is no clearance, also warns. A rule below its
default is kept, since the DRC checks the same rule, but warns: project
files often hold 0 for rules their author never set. The file is what
KiCad last saved: Board Setup changes not yet saved are not seen. The run
logs where each value came from.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple, cast

import kipy.board
from loguru import logger

from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE

PROJECT_SUFFIX = ".kicad_pro"
EDGE_RULE = "min_copper_edge_clearance"
HOLE_RULE = "min_hole_clearance"
# The command line options that override each rule.
FLAGS = {EDGE_RULE: "--edge-clearance", HOLE_RULE: "--hole-clearance"}
RULES_PATH = ("board", "design_settings", "rules")


class Clearances(NamedTuple):
    edge: float
    """To the board edge, mm."""
    hole: float
    """Between drilled holes and other nets' copper, mm."""


def read_clearances(
    board: kipy.board.Board, *, edge: float | None, hole: float | None
) -> Clearances:
    """`edge` and `hole` where given, else the project file's rules, else
    the defaults, with one log line saying where each came from. The
    project file is read only when a value is not given."""
    rules: dict[str, float] = {}
    name = ""
    if edge is None or hole is None:
        path = _project_file(board)
        if path is None:
            logger.debug("The board has no project: no project file to read")
        else:
            name = path.name
            rules = _read_rules(path)
    (edge_value, edge_source), (hole_value, hole_source) = (
        _resolved(given, rule, rules.get(rule), default, name)
        for given, rule, default in (
            (edge, EDGE_RULE, EDGE_CLEARANCE),
            (hole, HOLE_RULE, HOLE_CLEARANCE),
        )
    )
    logger.info(
        f"Edge clearance {edge_value:g} mm {edge_source}, "
        f"hole clearance {hole_value:g} mm {hole_source}"
    )
    return Clearances(edge_value, hole_value)


def _resolved(
    given: float | None, rule: str, ruled: float | None, default: float, name: str
) -> tuple[float, str]:
    """The value and where it came from, warning when the project file's
    `rule` is below `default`."""
    if given is not None:
        return given, "as given"
    if ruled is not None:
        if ruled < default:
            logger.warning(
                f"{name} sets {rule} to {ruled:g} mm, below the default "
                f"{default:g} mm; give {FLAGS[rule]} to override it"
            )
        return ruled, f"from {name}"
    return default, "by default"


def _project_file(board: kipy.board.Board) -> Path | None:
    """The project file, or None when the board has no project. KiCad gives
    the project's directory and its name without the suffix."""
    project = board.get_project()
    if not project.path:
        return None
    path = Path(project.path)
    if path.suffix == PROJECT_SUFFIX:
        return path
    if not project.name:
        return None
    return path / f"{project.name}{PROJECT_SUFFIX}"


def _read_rules(path: Path) -> dict[str, float]:
    """The edge and hole clearance rules `path` holds, by name; none when it
    is missing, with a warning when it is no project file."""
    try:
        settings: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug(f"No project file {path.name}")
        return {}
    except (OSError, ValueError) as e:
        logger.warning(
            f"Cannot read {path.name} ({e}); using the default edge and hole clearances"
        )
        return {}
    rules = settings
    for key in RULES_PATH:
        rules = _member(rules, key)
    if not isinstance(rules, dict):
        logger.warning(
            f"{path.name} holds no Board Setup rules; using the default edge "
            "and hole clearances"
        )
        return {}
    found: dict[str, float] = {}
    for rule in (EDGE_RULE, HOLE_RULE):
        value = cast(dict[str, object], rules).get(rule)
        if value is None and rule not in rules:
            continue
        if _is_clearance(value):
            found[rule] = float(cast(float, value))
        else:
            logger.warning(
                f"{path.name} gives {rule} as {value!r}, which is no clearance; "
                "using the default"
            )
    return found


def _member(value: object, key: str) -> object:
    """`value[key]` when `value` is a JSON object, else None."""
    if isinstance(value, dict):
        return cast(dict[str, object], value).get(key)
    return None


def _is_clearance(value: object) -> bool:
    """Whether `value` is a finite, non-negative JSON number."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )

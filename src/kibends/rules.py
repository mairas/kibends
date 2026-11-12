"""Edge, hole and hole-to-hole clearance from the project's Board Setup
constraints.

kicad-python cannot read a board's design rules, so a run reads the three
it models from the project file, the JSON `.kicad_pro` in the project's
directory, at `board.design_settings.rules` (`RULES`): `min_copper_edge_clearance`,
`min_hole_clearance` and `min_hole_to_hole`, in mm. A value the caller
gives wins over the file. With no project, no project file or no such
rule, the defaults of `obstacles.EDGE_CLEARANCE`, `HOLE_CLEARANCE` and
`HOLE_TO_HOLE` apply; a file that is no
project file, or a rule that is no clearance, also warns. A rule below its
default is kept, since the DRC checks the same rule, but warns: project
files often hold 0 for rules their author never set. The file is what
KiCad last saved: Board Setup changes not yet saved are not seen. The run
logs where each value came from.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, cast

import kipy.board
from loguru import logger

from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE, HOLE_TO_HOLE

PROJECT_SUFFIX = ".kicad_pro"
RULES_PATH = ("board", "design_settings", "rules")


class Rule(NamedTuple):
    key: str
    """Its name among the project file's rules."""
    default: float
    """mm, where the project file gives none."""
    label: str
    """What the log calls it."""


RULES = {
    "edge": Rule("min_copper_edge_clearance", EDGE_CLEARANCE, "edge clearance"),
    "hole": Rule("min_hole_clearance", HOLE_CLEARANCE, "hole clearance"),
    "hole_to_hole": Rule("min_hole_to_hole", HOLE_TO_HOLE, "hole to hole"),
}
"""The rules a run reads, by the `Clearances` field each gives."""


class Clearances(NamedTuple):
    edge: float
    """To the board edge, mm."""
    hole: float
    """Between drilled holes and other nets' copper, mm."""
    hole_to_hole: float
    """Between the edges of drilled holes, mm."""


def read_clearances(
    board: kipy.board.Board,
    *,
    edge: float | None,
    hole: float | None,
    hole_to_hole: float | None,
    override_names: Mapping[str, str],
) -> Clearances:
    """`edge`, `hole` and `hole_to_hole` where given, else the project
    file's rules, else the defaults, with one log line saying where each
    came from. The project file is read only when a value is not given.

    `override_names` names, by `Clearances` field, what the caller's user
    gives to override each rule, for the warning about a rule below its
    default; a rule it does not name is warned about without it."""
    given = {"edge": edge, "hole": hole, "hole_to_hole": hole_to_hole}
    rules: dict[str, float] = {}
    name = ""
    if None in given.values():
        path = _project_file(board)
        if path is None:
            logger.debug("The board has no project: no project file to read")
        else:
            name = path.name
            rules = _read_rules(path)
    resolved = {
        field: _resolved(
            given[field], rule, rules.get(rule.key), name, override_names.get(field)
        )
        for field, rule in RULES.items()
    }
    line = ", ".join(
        f"{RULES[field].label} {value:g} mm {source}"
        for field, (value, source) in resolved.items()
    )
    logger.info(line[0].upper() + line[1:])
    return Clearances(**{field: value for field, (value, _) in resolved.items()})


def _resolved(
    given: float | None,
    rule: Rule,
    ruled: float | None,
    name: str,
    override: str | None,
) -> tuple[float, str]:
    """The value and where it came from, warning when the project file's
    `rule` is below its default, naming `override` if given."""
    if given is not None:
        return given, "as given"
    if ruled is not None:
        if ruled < rule.default:
            hint = "" if override is None else f"; give {override} to override it"
            logger.warning(
                f"{name} sets {rule.key} to {ruled:g} mm, below the default "
                f"{rule.default:g} mm{hint}"
            )
        return ruled, f"from {name}"
    return rule.default, "by default"


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
    """The clearance rules of `RULES` that `path` holds, by key; none when it
    is missing, with a warning when it is no project file."""
    try:
        settings: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug(f"No project file {path.name}")
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Cannot read {path.name} ({e}); using the default clearances")
        return {}
    rules = settings
    for key in RULES_PATH:
        rules = _member(rules, key)
    if not isinstance(rules, dict):
        logger.warning(
            f"{path.name} holds no Board Setup rules; using the default clearances"
        )
        return {}
    found: dict[str, float] = {}
    for rule in (r.key for r in RULES.values()):
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

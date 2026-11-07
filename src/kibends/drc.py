"""Refuse results that add DRC violations, using kicad-cli on the board text.

kicad-cli checks the board's own rules, which KiCad's API does not expose:
custom rules, board minimums, edge and hole clearances. It runs twice on
copies in a scratch directory: once on the board as it is, once with the old
chain removed and the new tracks added, and the reports are compared.

`locate_kicad_cli`, `kicad_cli_version`, `project_files`, the DRC run,
`parse_report`, `violation_nets`, `rising_connectivity_nets` and `inject`
are copied and adapted from kistitch's drc.py (MIT, Copyright (c) 2025
Hat Labs Oy).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from loguru import logger

from kibends.chain import Point2

KICAD_CLI = "kicad-cli"
KNOWN_KICAD_CLI_PATHS: tuple[str, ...] = (
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
    "/usr/bin/kicad-cli",
    "/usr/local/bin/kicad-cli",
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
)

# Violation keys compare positions at 1 µm.
POSITION_DECIMALS = 3
# KiCad's 1 nm resolution in mm.
COORD_DECIMALS = 6

# Zone refill moves the anchors these report, so they are counted per net
# instead of compared by position.
CONNECTIVITY_TYPES = frozenset({"unconnected_items", "isolated_copper"})
# DRC item descriptions name the item's net in brackets: "Pad 1 [GND] of J1".
NET_TAG = re.compile(r"\[(.*)\]")
# Connectivity violations naming no net are counted together under this.
UNNAMED_NET = "(unnamed)"
# Stands in for a chain track in a violation key: a track's description
# includes its length and its position is its start, so neither survives
# the chain being replaced.
CHAIN_ITEM_KEY = ("(chain track)", 0.0, 0.0)

SEGMENT_OPEN = "\n\t(segment"
FONTS_MARKER = "\n\t(embedded_fonts"
DRC_TIMEOUT_S = 900
VERSION_TIMEOUT_S = 30


class KiCadCliNotFound(RuntimeError):
    pass


class ProjectFilesError(RuntimeError):
    pass


class BoardTextError(RuntimeError):
    """The board text does not hold the tracks the chain replaces."""


class DrcError(RuntimeError):
    pass


# ---- kicad-cli and project files ----------------------------------------------


class BinaryPathSource(Protocol):
    def get_kicad_binary_path(self, binary_name: str) -> str: ...


def _executable(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_file() and os.access(p, os.X_OK) else None


def locate_kicad_cli(explicit: str | None, kicad: BinaryPathSource | None) -> Path:
    """Find kicad-cli: explicit path, the running KiCad, PATH, known installs."""
    if explicit:
        found = _executable(explicit)
        if found is None:
            raise KiCadCliNotFound(f"--kicad-cli {explicit} is not an executable file")
        return found

    if kicad is not None:
        try:
            found = _executable(kicad.get_kicad_binary_path(KICAD_CLI))
        except (ApiError, KiCadConnectionError) as e:
            logger.debug(f"KiCad did not report a kicad-cli path: {e}")
            found = None
        if found:
            return found

    found = _executable(shutil.which(KICAD_CLI))
    if found:
        return found

    for candidate in KNOWN_KICAD_CLI_PATHS:
        found = _executable(candidate)
        if found:
            return found

    raise KiCadCliNotFound(
        "kicad-cli not found; pass --kicad-cli PATH, put kicad-cli on PATH, "
        "or skip the check with --no-drc"
    )


def kicad_cli_version(kicad_cli: Path) -> str:
    try:
        out = subprocess.run(
            [str(kicad_cli), "version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "unknown (version probe timed out)"
    return out.stdout.strip() or out.stderr.strip() or "unknown"


@dataclass(frozen=True)
class ProjectFiles:
    pro: Path
    dru: Path | None


class Project(Protocol):
    @property
    def path(self) -> str: ...


class ProjectBoard(Protocol):
    @property
    def name(self) -> str: ...

    def get_project(self) -> Project: ...


def project_files(board: ProjectBoard) -> ProjectFiles:
    """On-disk project and rule files of the open board.

    Unsaved rule or netclass edits made in KiCad are not in these files.
    """
    directory = board.get_project().path
    pcb = Path(directory) / board.name if directory else None
    if pcb is None or not pcb.is_file():
        raise ProjectFilesError(
            "The board has not been saved to disk; save it in KiCad first, "
            "since the DRC check reads its project and rule files"
        )
    pro = pcb.with_suffix(".kicad_pro")
    if not pro.is_file():
        raise ProjectFilesError(f"Project file {pro} not found")
    dru = pcb.with_suffix(".kicad_dru")
    return ProjectFiles(pro=pro, dru=dru if dru.is_file() else None)


# ---- the candidate board text ---------------------------------------------------


@dataclass(frozen=True)
class NewTracks:
    """Straight tracks between consecutive `points` (mm)."""

    points: Sequence[Point2]
    width: float
    layer: str
    net: str


def remove_segments(board_text: str, ids: Collection[str]) -> str:
    """Board text without the segments whose uuids are `ids`.

    Everything else is left byte for byte, so that DRC sees no other change.
    """
    spans: set[tuple[int, int]] = set()
    missing = []
    for i in ids:
        span = _segment_span(board_text, i)
        if span is None:
            missing.append(i)
        else:
            spans.add(span)
    if missing:
        raise BoardTextError(
            f"Tracks not found in the board text: {', '.join(missing)}; "
            "select the chain again and rerun"
        )
    out = board_text
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + out[end:]
    return out


def _segment_span(text: str, item_id: str) -> tuple[int, int] | None:
    """Start and end of the top-level segment with this uuid, with its newline."""
    at = text.find(f'(uuid "{item_id}")')
    if at < 0:
        return None
    opening = text.rfind(SEGMENT_OPEN, 0, at)
    if opening < 0:
        return None
    start = opening + 1
    end = _after_closing_paren(text, start + 1)
    if end <= at:
        return None
    return start, end + 1 if text.startswith("\n", end) else end


def _after_closing_paren(text: str, opening: int) -> int:
    """Index just past the parenthesis closing the one at `opening`."""
    depth = 0
    in_string = False
    i = opening
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise BoardTextError("Unbalanced parentheses in the board text")


def _num(v: float) -> str:
    s = f"{v:.{COORD_DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def inject(board_text: str, tracks: NewTracks) -> tuple[str, tuple[str, ...]]:
    """Board text with `tracks` added as segments, and the segments' uuids."""
    ids = tuple(str(uuid.uuid4()) for _ in tracks.points[1:])
    items = "".join(
        f"\t(segment\n\t\t(start {_num(start[0])} {_num(start[1])})\n"
        f"\t\t(end {_num(end[0])} {_num(end[1])})\n"
        f"\t\t(width {_num(tracks.width)})\n\t\t(layer {_quote(tracks.layer)})\n"
        f"\t\t(net {_quote(tracks.net)})\n"
        f'\t\t(uuid "{item_id}")\n\t)\n'
        for (start, end), item_id in zip(pairwise(tracks.points), ids, strict=True)
    )
    if FONTS_MARKER in board_text:
        at = board_text.rindex(FONTS_MARKER) + 1
    else:
        at = board_text.rstrip().rindex(")")
        if board_text[at - 1] != "\n":
            items = "\n" + items
    return board_text[:at] + items + board_text[at:], ids


# ---- reports -------------------------------------------------------------------


class _Position(TypedDict):
    x: float
    y: float


class _ReportItem(TypedDict):
    description: str
    pos: _Position
    uuid: NotRequired[str]


class _ReportViolation(TypedDict, total=False):
    type: str
    description: str
    severity: str
    items: list[_ReportItem]
    excluded: bool


class Report(TypedDict, total=False):
    violations: list[_ReportViolation]
    unconnected_items: list[_ReportViolation]


@dataclass(frozen=True)
class ViolationItem:
    description: str
    position: Point2
    uuid: str = ""


@dataclass(frozen=True)
class Violation:
    type: str
    description: str
    severity: str
    items: tuple[ViolationItem, ...]

    def __str__(self) -> str:
        where = ", ".join(
            f"{i.description} @ ({i.position[0]:.3f}, {i.position[1]:.3f})"
            for i in self.items
        )
        return f"{self.type}: {self.description} [{where}]"


def parse_report(report: Report) -> list[Violation]:
    out = []
    for v in report.get("violations", []) + report.get("unconnected_items", []):
        if v.get("excluded"):
            continue
        out.append(
            Violation(
                type=v.get("type", "unconnected_items"),
                description=v.get("description", ""),
                severity=v.get("severity", ""),
                items=tuple(
                    ViolationItem(
                        i["description"],
                        (i["pos"]["x"], i["pos"]["y"]),
                        i.get("uuid", ""),
                    )
                    for i in v.get("items", [])
                ),
            )
        )
    return out


ViolationKey = tuple[str, tuple[tuple[str, float, float], ...]]


def _key(v: Violation, chain_ids: Collection[str]) -> ViolationKey:
    return (
        v.type,
        tuple(
            sorted(
                CHAIN_ITEM_KEY
                if i.uuid in chain_ids
                else (
                    i.description,
                    round(i.position[0], POSITION_DECIMALS),
                    round(i.position[1], POSITION_DECIMALS),
                )
                for i in v.items
            )
        ),
    )


def violation_nets(v: Violation) -> frozenset[str]:
    """Nets named in the items' descriptions, or UNNAMED_NET if none is."""
    nets = frozenset(
        m.group(1) for i in v.items if (m := NET_TAG.search(i.description))
    )
    return nets or frozenset({UNNAMED_NET})


def _connectivity_by_net(violations: Iterable[Violation]) -> Counter[str]:
    return Counter(
        n for v in violations if v.type in CONNECTIVITY_TYPES for n in violation_nets(v)
    )


def rising_connectivity_nets(
    baseline: Iterable[Violation], candidate: Iterable[Violation]
) -> frozenset[str]:
    """Nets with more connectivity violations in `candidate` than in `baseline`."""
    before = _connectivity_by_net(baseline)
    after = _connectivity_by_net(candidate)
    return frozenset(n for n, c in after.items() if c > before[n])


@dataclass(frozen=True)
class Introduced:
    """Violations the candidate adds; true when there are any.

    `split_nets` are the nets whose connectivity violations rose, as when the
    new path splits a pour; `violations` lists every candidate connectivity
    violation on them, since which one is new can't be told.
    """

    violations: tuple[Violation, ...]
    split_nets: frozenset[str]

    def __bool__(self) -> bool:
        return bool(self.violations or self.split_nets)


def introduced(
    baseline: Sequence[Violation],
    candidate: Sequence[Violation],
    old_ids: Collection[str],
    new_ids: Collection[str],
) -> Introduced:
    """The violations in `candidate` that `baseline` does not have.

    Violations other than connectivity ones are compared as multisets of
    (type, items), items by description and position. An item on the old
    chain in the baseline and one on the new chain in the candidate both
    stand for the chain, so a violation the chain already had against some
    item is not new. Connectivity violations count per net.
    """
    remaining = Counter(
        _key(v, old_ids) for v in baseline if v.type not in CONNECTIVITY_TYPES
    )
    added = []
    for v in candidate:
        if v.type in CONNECTIVITY_TYPES:
            continue
        k = _key(v, new_ids)
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            added.append(v)
    split_nets = rising_connectivity_nets(baseline, candidate)
    splits = [
        v
        for v in candidate
        if v.type in CONNECTIVITY_TYPES and violation_nets(v) & split_nets
    ]
    return Introduced(tuple(added + splits), split_nets)


# ---- running kicad-cli -----------------------------------------------------------


def check(
    kicad_cli: Path,
    project: ProjectFiles,
    board_text: str,
    removed_ids: Collection[str],
    tracks: NewTracks,
    workdir: Path,
) -> Introduced:
    """Violations that replacing `removed_ids` with `tracks` would add.

    The candidate text is built first, so a missing track fails before any
    DRC run.
    """
    candidate_text, new_ids = inject(remove_segments(board_text, removed_ids), tracks)
    baseline = _timed_run(kicad_cli, project, workdir, "baseline", board_text)
    candidate = _timed_run(kicad_cli, project, workdir, "candidate", candidate_text)
    return introduced(baseline, candidate, removed_ids, new_ids)


def _timed_run(
    kicad_cli: Path, project: ProjectFiles, workdir: Path, stem: str, board_text: str
) -> list[Violation]:
    started = time.monotonic()
    violations = _run(kicad_cli, project, workdir, stem, board_text)
    logger.info(
        f"{stem.capitalize()} DRC: {len(violations)} violations in "
        f"{time.monotonic() - started:.1f} s"
    )
    return violations


def _run(
    kicad_cli: Path, project: ProjectFiles, workdir: Path, stem: str, board_text: str
) -> list[Violation]:
    pcb = workdir / f"{stem}.kicad_pcb"
    pcb.write_text(board_text, encoding="utf-8")
    shutil.copyfile(project.pro, pcb.with_suffix(".kicad_pro"))
    if project.dru is not None:
        shutil.copyfile(project.dru, pcb.with_suffix(".kicad_dru"))
    out = pcb.with_suffix(".json")
    cmd = [
        str(kicad_cli),
        "pcb",
        "drc",
        "--refill-zones",
        "--format",
        "json",
        "-o",
        str(out),
        str(pcb),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=DRC_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        raise DrcError(f"kicad-cli DRC timed out after {DRC_TIMEOUT_S} s") from None
    if proc.returncode != 0 or not out.is_file():
        raise DrcError(
            f"kicad-cli DRC failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return parse_report(cast(Report, json.loads(out.read_text(encoding="utf-8"))))

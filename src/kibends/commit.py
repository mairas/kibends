"""Replace the tracks of one or more chains, and move the vias they join, in
KiCad as one undoable commit.

The changed chains' tracks and the moved vias are re-read in one request,
and the run refuses if any changed since it read them; a selected arc,
which the run smoothed as straight segments, is compared as the arc it was
read as. Every created track is straight. After the push, the created
items are added to the editor's selection (`_select`); KiCad deselects the
deleted ones. The selection is outside the undo history, so the run stays
one undo step, and since the commit has landed by then, a failure to
select is logged at debug and the run reports committed. A dry run, or one
with nothing to commit, never gets here and leaves the selection alone.

`commit_label`, `create_checked` and `push_checked` are copied and
trimmed from kistitch's commit.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

from collections.abc import Sequence
from dataclasses import astuple, dataclass
from datetime import datetime
from itertools import pairwise
from typing import cast

import kipy.board
import kipy.board_types as kbt
from kipy.common_types import Commit
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.geometry import Vector2
from kipy.proto.common.commands.editor_commands_pb2 import (
    AddToSelection,
    CreateItems,
    CreateItemsResponse,
    DeleteItems,
    DeleteItemsResponse,
    ItemDeletionStatus,
    ItemStatusCode,
    SelectionResponse,
)
from kipy.proto.common.types.base_types_pb2 import KIID
from kipy.util.proto import pack_any
from kipy.wrapper import Wrapper
from loguru import logger

from kibends.chain import NM_PER_MM, Chain, NmPoint, Point2, Segment
from kibends.energy import FloatArray
from kibends.snapshot import (
    Arc,
    Via,
    arc_from_kipy,
    segment_from_track,
    via_from_kipy,
)

COMMIT_PREFIX = "kibends"
# Rounding to KiCad's grid may move a point of a straight path this far off it.
STRAIGHT_TOLERANCE_NM = 1


class BoardChangedError(RuntimeError):
    """A selected track or via changed in KiCad after it was read."""


class CreateItemsError(RuntimeError):
    pass


class DeleteItemsError(RuntimeError):
    pass


UNCERTAIN_ADVICE = "it may have landed. Check KiCad's undo history before running again"


class CommitUncertainError(RuntimeError):
    """The push timed out or lost the connection; the commit may have landed."""


@dataclass(frozen=True)
class ViaMove:
    via: Via
    """As it was read."""
    position: Point2
    """Where it moves to, mm."""


def commit_label(command: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{COMMIT_PREFIX} {command} {stamp}"


def commit_chains(
    board: kipy.board.Board,
    chains: Sequence[tuple[Chain, FloatArray]],
    selected: Sequence[Segment],
    arcs: Sequence[Arc],
    vias: Sequence[ViaMove] = (),
) -> None:
    """Replace each chain's tracks with straight tracks along its polyline
    (mm), and move each of `vias`.

    All chains and vias go into one KiCad commit, so the run is one undo
    step, and each IPC step is one request for all of them. A via moves as
    a copy at its new position that replaces it. The tracks, the selected
    `arcs` among them, and the vias are re-read first, so one edited since
    the selection was read aborts the run instead of being overwritten.
    Zones are not refilled: a refill would be a second undo step. Empty
    `chains` and `vias` raise ValueError, since they would push an empty
    undo step.
    """
    if not chains and not vias:
        raise ValueError("Nothing to commit")
    # A track split at a junction, or an arc, belongs to each chain that has
    # a part of it, and is deleted once.
    track_ids = list(dict.fromkeys(i for chain, _ in chains for i in chain.segment_ids))
    current, current_vias = _unchanged_items(board, track_ids, selected, vias, arcs)
    new_items: list[Wrapper] = [
        track
        for chain, polyline in chains
        for track in track_items(polyline, chain.width, current[chain.segment_ids[0]])
    ]
    new_items += [_moved(current_vias[m.via.id], m.position) for m in vias]
    ids = [*track_ids, *(m.via.id for m in vias)]
    label = commit_label("smooth")

    commit = board.begin_commit()
    try:
        delete_checked(board, ids)
        created = create_checked(board, new_items)
    except BaseException:
        board.drop_commit(commit)
        raise

    # Outside the try above: once pushed, the commit can't be dropped.
    push_checked(board, commit, label)
    _select(board, created)
    parts: list[str] = []
    if track_ids:
        parts.append(
            f"replaced {_count(len(track_ids), 'track')} with "
            f"{len(new_items) - len(vias)}"
        )
    if vias:
        parts.append(f"moved {_count(len(vias), 'via')}")
    logger.info(f"Committed '{label}': {' and '.join(parts)}")


def _select(board: kipy.board.Board, ids: Sequence[KIID]) -> None:
    """Add the items `ids` to the editor's selection, which is outside the
    undo history. The commit has landed, so a failure here fails nothing.

    kipy's add_to_selection unwraps the whole selection KiCad answers with,
    and fails on any item type it does not know; the answer is not needed.
    """
    command = AddToSelection()
    command.header.document.CopyFrom(board.document)
    command.items.extend(ids)
    try:
        board.client.send(command, SelectionResponse)
    except (ApiError, KiCadConnectionError, KeyboardInterrupt) as e:
        logger.debug(f"Could not select the new tracks and vias: {e!r}")


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _unchanged_items(
    board: kipy.board.Board,
    track_ids: Sequence[str],
    selected: Sequence[Segment],
    vias: Sequence[ViaMove],
    arcs: Sequence[Arc],
) -> tuple[dict[str, kbt.Track | kbt.ArcTrack], dict[str, kbt.Via]]:
    """The tracks and arcs `track_ids` and the vias of `vias` as KiCad has
    them now, by id, in one request; raise if any changed. An arc is
    compared as the arc it was read as, not as its segments."""
    expected_arcs = {a.id: a for a in arcs}
    # An arc's segments are not expected of a straight track, so a track
    # re-read under an arc's id, or the reverse, counts as changed.
    wanted = set(track_ids) - expected_arcs.keys()
    expected = {s.id: s for s in selected if s.id in wanted}
    ids = [*track_ids, *(m.via.id for m in vias)]
    items = board.get_items_by_id([KIID(value=i) for i in ids])
    tracks = {i.id.value: i for i in items if isinstance(i, (kbt.Track, kbt.ArcTrack))}
    found_vias = {i.id.value: i for i in items if isinstance(i, kbt.Via)}

    def unchanged(track_id: str) -> bool:
        match tracks.get(track_id):
            case kbt.ArcTrack() as arc:
                return arc_from_kipy(arc) == expected_arcs.get(track_id)
            case kbt.Track() as track:
                return segment_from_track(track) == expected.get(track_id)
            case _:
                return False

    changed = [i for i in track_ids if not unchanged(i)]
    changed += [
        m.via.id
        for m in vias
        if m.via.id not in found_vias
        or astuple(via_from_kipy(found_vias[m.via.id])) != astuple(m.via)
    ]
    if changed:
        raise BoardChangedError(
            "Selected items changed in KiCad since they were read: "
            f"{', '.join(changed)}; select them again and rerun"
        )
    return tracks, found_vias


def _moved(via: kbt.Via, position: Point2) -> kbt.Via:
    """A copy of `via` at `position` (mm), for KiCad to give a new id."""
    moved = kbt.Via(via.proto)
    moved.proto.ClearField("id")
    # A group lists its members by id, and the old via leaves it when
    # deleted; a copy claiming the group would join it behind KiCad's back.
    moved.proto.ClearField("parent")
    moved.position = Vector2.from_xy(_nm(position[0]), _nm(position[1]))
    return moved


def track_items(
    polyline: FloatArray, width: float, model: kbt.Track | kbt.ArcTrack
) -> list[kbt.Track]:
    """Straight tracks along `polyline`, on the layer and net of `model`."""
    tracks: list[kbt.Track] = []
    for start, end in pairwise(snapped_points(polyline)):
        track = kbt.Track()
        track.start = Vector2.from_xy(_nm(start[0]), _nm(start[1]))
        track.end = Vector2.from_xy(_nm(end[0]), _nm(end[1]))
        track.width = _nm(width)
        track.layer = model.layer
        track.net = model.net
        tracks.append(track)
    return tracks


def snapped_points(polyline: FloatArray) -> list[Point2]:
    """`polyline` (mm) rounded to KiCad's 1 nm grid, without repeated points.

    These are the committed tracks' vertices, in mm.
    """
    points: list[Point2] = []
    for x, y in polyline:
        p = (_nm(x) / NM_PER_MM, _nm(y) / NM_PER_MM)
        if not points or p != points[-1]:
            points.append(p)
    return points


def snapped_path(polyline: FloatArray) -> list[NmPoint]:
    """`polyline` on KiCad's 1 nm grid, less the vertices that do not bend it.

    A vertex goes when it lies between its neighbours, within
    `STRAIGHT_TOLERANCE_NM` of the line through them, so two polylines with
    the same path compare equal however many vertices each has.
    """
    path: list[NmPoint] = []
    for x, y in polyline:
        p = (_nm(x), _nm(y))
        if path and p == path[-1]:
            continue
        if len(path) > 1 and _between(path[-2], path[-1], p):
            path.pop()
        path.append(p)
    return path


def _between(a: NmPoint, q: NmPoint, b: NmPoint) -> bool:
    """Whether `q` lies on the segment from `a` to `b`, within tolerance."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    qx, qy = q[0] - a[0], q[1] - a[1]
    length_sq = dx * dx + dy * dy
    cross = dx * qy - dy * qx
    return (
        0 <= qx * dx + qy * dy <= length_sq
        and cross * cross <= STRAIGHT_TOLERANCE_NM**2 * length_sq
    )


def _nm(v: float) -> int:
    return round(v * NM_PER_MM)


def delete_checked(board: kipy.board.Board, ids: Sequence[str]) -> None:
    """Delete items by id and raise unless KiCad deleted every one.

    kipy's remove_items discards the per-item results, so a refused deletion
    would otherwise leave the old item under the new ones.
    """
    command = DeleteItems()
    command.header.document.CopyFrom(board.document)
    command.item_ids.extend(KIID(value=i) for i in ids)
    response = board.client.send(command, DeleteItemsResponse)
    statuses = {r.id.value: r.status for r in response.deleted_items}
    names = {i: ItemDeletionStatus.Name(s) for i, s in statuses.items()}
    failed = [
        f"{i} ({names.get(i, 'no answer')})"
        for i in ids
        if statuses.get(i) != ItemDeletionStatus.IDS_OK
    ]
    if failed:
        raise DeleteItemsError(
            f"KiCad did not delete {len(failed)} of {len(ids)} items: "
            + ", ".join(failed)
        )


def create_checked(board: kipy.board.Board, items: Sequence[Wrapper]) -> list[KIID]:
    """Create items and raise if KiCad refuses any of them; return the ids
    KiCad gave them, which the push keeps.

    KiCad reports refusals per item, and kipy's create_items discards those
    statuses, so a refused item would otherwise vanish silently.
    """
    command = CreateItems()
    command.header.document.CopyFrom(board.document)
    command.items.extend(pack_any(i.proto) for i in items)
    response = board.client.send(command, CreateItemsResponse)
    refused = [
        r.status.error_message or ItemStatusCode.Name(r.status.code)
        for r in response.created_items
        if r.status.code != ItemStatusCode.ISC_OK
    ]
    if refused or len(response.created_items) != len(items):
        reasons = f": {'; '.join(sorted(set(refused)))}" if refused else ""
        raise CreateItemsError(
            f"KiCad created {len(response.created_items) - len(refused)} of "
            f"{len(items)} items{reasons}"
        )
    return [cast(kbt.BoardItem, kbt.unwrap(r.item)).id for r in response.created_items]


def push_checked(board: kipy.board.Board, commit: Commit, label: str) -> None:
    """Push the commit and raise unless KiCad confirms it landed.

    A lost or timed-out connection, or an interrupt, doesn't mean KiCad
    didn't land it, so the commit is kept and `CommitUncertainError` raised.
    An ApiError is KiCad's answer, so nothing landed and the commit is
    dropped.
    """
    try:
        board.push_commit(commit, label)
    except KiCadConnectionError as e:
        raise CommitUncertainError(
            f"Committing '{label}' timed out or lost the connection to KiCad "
            f"({e}); {UNCERTAIN_ADVICE}"
        ) from None
    except KeyboardInterrupt:
        raise CommitUncertainError(
            f"Interrupted while committing '{label}'; {UNCERTAIN_ADVICE}"
        ) from None
    except ApiError:
        board.drop_commit(commit)
        raise

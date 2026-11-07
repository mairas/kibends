"""Replace the chain's tracks in KiCad as one undoable commit.

`commit_label`, `create_checked` and `push_checked` are copied and
trimmed from kistitch's commit.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise

import kipy.board
import kipy.board_types as kbt
from kipy.common_types import Commit
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.geometry import Vector2
from kipy.proto.common.commands.editor_commands_pb2 import (
    CreateItems,
    CreateItemsResponse,
    DeleteItems,
    DeleteItemsResponse,
    ItemDeletionStatus,
    ItemStatusCode,
)
from kipy.proto.common.types.base_types_pb2 import KIID
from kipy.util.proto import pack_any
from kipy.wrapper import Wrapper
from loguru import logger

from kibends.chain import NM_PER_MM, Chain, Point2, Segment
from kibends.energy import FloatArray
from kibends.snapshot import segment_from_track

COMMIT_PREFIX = "kibends"


class BoardChangedError(RuntimeError):
    """A selected track changed in KiCad after it was read."""


class CreateItemsError(RuntimeError):
    pass


class DeleteItemsError(RuntimeError):
    pass


class CommitUncertainError(RuntimeError):
    """The push timed out or lost the connection; the commit may have landed."""


def commit_label(command: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"{COMMIT_PREFIX} {command} {stamp}"


def commit_chain(
    board: kipy.board.Board,
    chain: Chain,
    selected: Sequence[Segment],
    polyline: FloatArray,
) -> None:
    """Replace the chain's tracks with straight tracks along `polyline` (mm).

    The tracks are re-read first, so a track edited since the selection was
    read aborts the run instead of being overwritten. Zones are not
    refilled: a refill would be a second undo step.
    """
    current = _unchanged_tracks(board, chain, selected)
    new_tracks = track_items(polyline, chain.width, current[0])
    label = commit_label("smooth")

    commit = board.begin_commit()
    try:
        delete_checked(board, chain.segment_ids)
        create_checked(board, new_tracks)
    except BaseException:
        board.drop_commit(commit)
        raise

    # Outside the try above: once pushed, the commit can't be dropped.
    push_checked(board, commit, label)
    logger.info(
        f"Committed '{label}': replaced {len(chain.segment_ids)} tracks "
        f"with {len(new_tracks)}"
    )


def _unchanged_tracks(
    board: kipy.board.Board, chain: Chain, selected: Sequence[Segment]
) -> list[kbt.Track]:
    """The chain's tracks as KiCad has them now; raise if any changed."""
    expected = {s.id: s for s in selected if s.id in set(chain.segment_ids)}
    fetched = {
        item.id.value: item
        for item in board.get_items_by_id([KIID(value=i) for i in chain.segment_ids])
        if isinstance(item, kbt.Track)
    }
    changed = [
        i
        for i in chain.segment_ids
        if i not in fetched or segment_from_track(fetched[i]) != expected[i]
    ]
    if changed:
        raise BoardChangedError(
            "Selected tracks changed in KiCad since they were read: "
            f"{', '.join(changed)}; select the chain again and rerun"
        )
    return [fetched[i] for i in chain.segment_ids]


def track_items(
    polyline: FloatArray, width: float, model: kbt.Track
) -> list[kbt.Track]:
    """Straight tracks along `polyline`, on the layer and net of `model`."""
    tracks = []
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


def _nm(v: float) -> int:
    return round(v * NM_PER_MM)


def delete_checked(board: kipy.board.Board, ids: Sequence[str]) -> None:
    """Delete items by id and raise unless KiCad deleted every one.

    kipy's remove_items discards the per-item results, so a refused deletion
    would otherwise leave the old track under the new ones.
    """
    command = DeleteItems()
    command.header.document.CopyFrom(board.document)
    command.item_ids.extend(KIID(value=i) for i in ids)
    response = board.client.send(command, DeleteItemsResponse)
    statuses = {r.id.value: r.status for r in response.deleted_items}
    failed = [
        f"{i} ({ItemDeletionStatus.Name(statuses[i]) if i in statuses else 'no answer'})"
        for i in ids
        if statuses.get(i) != ItemDeletionStatus.IDS_OK
    ]
    if failed:
        raise DeleteItemsError(
            f"KiCad did not delete {len(failed)} of {len(ids)} tracks: "
            + ", ".join(failed)
        )


def create_checked(board: kipy.board.Board, items: Sequence[Wrapper]) -> list[Wrapper]:
    """Create items and raise if KiCad refuses any of them.

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
        raise CreateItemsError(
            f"KiCad created {len(response.created_items) - len(refused)} of "
            f"{len(items)} tracks: {'; '.join(sorted(set(refused)))}"
        )
    return [kbt.unwrap(r.item) for r in response.created_items]


def push_checked(board: kipy.board.Board, commit: Commit, label: str) -> None:
    """Push the commit and raise unless KiCad confirms it landed.

    A lost or timed-out connection doesn't mean KiCad didn't land it, so the
    commit is kept and `CommitUncertainError` raised. An ApiError is KiCad's
    answer, so nothing landed and the commit is dropped.
    """
    try:
        board.push_commit(commit, label)
    except KiCadConnectionError as e:
        raise CommitUncertainError(
            f"Committing '{label}' timed out or lost the connection to KiCad "
            f"({e}); it may have landed. Check KiCad's undo history before "
            "running again"
        ) from None
    except ApiError:
        board.drop_commit(commit)
        raise

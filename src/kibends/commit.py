"""Replace the chain's tracks in KiCad as one undoable commit.

`commit_label`, `create_checked`, `push_tolerant`, `refill_tolerant`,
`refill_after_commit` and `_busy` are copied and trimmed from kistitch's
commit.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import pairwise
from typing import Protocol

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
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.proto.common.types.base_types_pb2 import KIID
from kipy.util.proto import pack_any
from kipy.wrapper import Wrapper
from loguru import logger

from kibends.chain import NM_PER_MM, Chain, NmPoint, Segment
from kibends.energy import FloatArray
from kibends.snapshot import segment_from_track

REFILL_TIMEOUT_S = 300.0
REFILL_POLL_S = 1.0
COMMIT_PREFIX = "kibends"


class KiCadSession(Protocol):
    """The part of `kipy.KiCad` the refill needs."""

    def ping(self) -> None: ...


class BoardChangedError(RuntimeError):
    """A selected track changed in KiCad after it was read."""


class CreateItemsError(RuntimeError):
    pass


class DeleteItemsError(RuntimeError):
    pass


def commit_label(command: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"{COMMIT_PREFIX} {command} {stamp}"


def commit_chain(
    kicad: KiCadSession,
    board: kipy.board.Board,
    chain: Chain,
    selected: Sequence[Segment],
    polyline: FloatArray,
) -> None:
    """Replace the chain's tracks with straight tracks along `polyline` (mm).

    The tracks are re-read first, so a track edited since the selection was
    read aborts the run instead of being overwritten. Zones are refilled
    after the commit, which KiCad records as a separate undo step.
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
    if push_tolerant(board, commit, label):
        logger.info(
            f"Committed '{label}': replaced {len(chain.segment_ids)} tracks "
            f"with {len(new_tracks)}"
        )
    refill_after_commit(kicad, board)


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
    """Straight tracks along `polyline`, on the layer and net of `model`.

    Vertices are rounded to KiCad's 1 nm grid; consecutive vertices that land
    on the same grid point make no track.
    """
    points: list[NmPoint] = []
    for x, y in polyline:
        p = (_nm(x), _nm(y))
        if not points or p != points[-1]:
            points.append(p)
    tracks = []
    for start, end in pairwise(points):
        track = kbt.Track()
        track.start = Vector2.from_xy(*start)
        track.end = Vector2.from_xy(*end)
        track.width = _nm(width)
        track.layer = model.layer
        track.net = model.net
        tracks.append(track)
    return tracks


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


def push_tolerant(board: kipy.board.Board, commit: Commit, label: str) -> bool:
    """Push the commit; True once KiCad confirms it landed.

    A lost or timed-out connection doesn't mean KiCad didn't land it, so that
    only warns. An ApiError is KiCad's answer, so nothing landed and the
    commit is dropped.
    """
    try:
        board.push_commit(commit, label)
    except KiCadConnectionError as e:
        logger.warning(
            f"Committing '{label}' timed out or lost the connection to KiCad "
            f"({e}); it may have landed. Check KiCad's undo history before "
            "running again"
        )
        return False
    except ApiError:
        board.drop_commit(commit)
        raise
    return True


def refill_after_commit(kicad: KiCadSession, board: kipy.board.Board) -> None:
    """Refill zones; a failure only warns, since the commit has landed."""
    logger.info("Refilling zones; KiCad records the refill as a separate undo step")
    try:
        refilled = refill_tolerant(kicad, board)
    except (ApiError, KiCadConnectionError) as e:
        logger.warning(f"Zone refill after the commit failed ({e})")
        refilled = False
    if not refilled:
        logger.warning("Zone refill not confirmed; press B in KiCad to refill zones")


def _busy(e: Exception) -> bool:
    return isinstance(e, KiCadConnectionError) or (
        isinstance(e, ApiError) and e.code == ApiStatusCode.AS_BUSY
    )


def refill_tolerant(
    kicad: KiCadSession,
    board: kipy.board.Board,
    timeout_s: float = REFILL_TIMEOUT_S,
    poll_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Refill zones and wait, until `timeout_s` has passed, for KiCad to answer.

    kipy's blocking refill never applies its own time limit (its poll loop
    doesn't count), so the refill is started without blocking and the
    deadline is ours. A ping to a KiCad busy filling waits out the client
    timeout, so the deadline is wall-clock, not a poll count. A ping reply
    suggests, but doesn't prove, that the fill finished. AS_BUSY on the
    refill itself means KiCad refused to start it (a tool or dialog is
    active), so it is retried.
    """
    if poll_s is None:
        poll_s = REFILL_POLL_S
    deadline = clock() + timeout_s
    busy_logged = False
    while True:
        try:
            board.refill_zones(block=False)
            break
        except KiCadConnectionError as e:
            logger.warning(f"Zone refill request timed out ({e}); waiting for KiCad")
            break
        except ApiError as e:
            if e.code != ApiStatusCode.AS_BUSY:
                raise
            if not busy_logged:
                logger.warning(
                    "KiCad is busy — a tool or dialog may be open in the PCB "
                    "editor; retrying the zone refill"
                )
                busy_logged = True
            if clock() >= deadline:
                logger.warning("KiCad stayed busy; zones were not refilled")
                return False
            sleep(poll_s)

    # Once KiCad has the refill, ask at least once whether it is done.
    while True:
        sleep(poll_s)
        try:
            kicad.ping()
            return True
        except (ApiError, KiCadConnectionError) as e:
            if not _busy(e):
                raise
        if clock() >= deadline:
            logger.warning(
                f"KiCad did not answer within {timeout_s:.0f} s of the zone refill"
            )
            return False

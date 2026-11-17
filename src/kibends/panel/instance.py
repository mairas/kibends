"""One panel per KiCad instance.

A panel listens on a local server (Qt's `QLocalServer`: a Unix socket in
the temp directory, a named pipe on Windows) named after the user, the
KiCad socket it talks to and KiCad's API token (`server_name`,
`kicad_socket`, `kicad_token`). The socket is the one kipy connects to:
`KICAD_API_SOCKET`, which KiCad sets when it starts the panel, else kipy's
default. The token, `KICAD_API_TOKEN`, is new each time KiCad starts, so a
KiCad restarted on the same socket gets a panel of its own instead of
raising one that holds the old token, which KiCad refuses. A panel started
from a terminal without that token therefore does not share a name with
one KiCad started. A start that reaches a running panel's server
asks it to come forward and exits; a connection is the whole request.
Otherwise it listens itself, after removing a name a crashed panel left
(`decide`, `claim`). Listening on a name a live panel holds fails, so two
starts at once, the second asking before the first listened, still end
with one panel. A panel that can take no name opens anyway, alone.

`bring_forward` shows, raises and activates the window, at start and on
each request. On macOS Qt's raise also activates the application
(`QT_MAC_SET_RAISE_PROCESS`, on by default), which a panel KiCad started
needs, since KiCad brings only the process it starts to the front.
"""

import getpass
import hashlib
import os
import sys
import tempfile
from collections.abc import Callable
from enum import Enum, auto

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QWidget

NAME_PREFIX = "kibends-panel-"
DIGEST_LENGTH = 16
"""Hex digits of the hash in a server name: short enough for a Unix
socket path under macOS's long temp directory."""
CONNECT_TIMEOUT_MS = 1000


class Claim(Enum):
    RAISED = auto()
    """A panel for this KiCad runs and was asked to come forward."""
    SERVING = auto()
    """This panel holds the name."""
    ALONE = auto()
    """No panel answered and the name could not be taken."""


def kicad_socket() -> str:
    """The KiCad socket kipy connects to without arguments.

    kipy's own choice, so the panel keys on the KiCad it will talk to. The
    helper is private to kipy; without it, `KICAD_API_SOCKET`, else
    `default_socket`.
    """
    try:
        from kipy.kicad import (
            _default_socket_path,  # pyright: ignore[reportPrivateUsage]
        )
    except ImportError:
        socket = os.environ.get("KICAD_API_SOCKET")
        if socket is not None:
            return socket
        return default_socket(windows=sys.platform == "win32")
    return _default_socket_path()


def default_socket(*, windows: bool) -> str:
    """KiCad's default API socket, as kipy 0.8.0 gives it, but for the
    Flatpak's socket under the home directory, which kipy prefers when it
    exists."""
    if windows:
        return f"ipc://{tempfile.gettempdir()}\\kicad\\api.sock"
    return "ipc:///tmp/kicad/api.sock"


def kicad_token() -> str:
    """KiCad's API token, `KICAD_API_TOKEN`; empty when KiCad did not start
    the panel."""
    return os.environ.get("KICAD_API_TOKEN", "")


def server_name(socket_path: str, token: str) -> str:
    """The panel's server name for KiCad's socket at `socket_path` and its
    API `token`."""
    key = f"{getpass.getuser()}\0{socket_path}\0{token}".encode()
    return NAME_PREFIX + hashlib.sha256(key).hexdigest()[:DIGEST_LENGTH]


def decide(
    connect: Callable[[], bool], listen: Callable[[], bool], remove: Callable[[], None]
) -> Claim:
    """Ask a running panel to raise, else take the name.

    A failed listen with no panel answering means either a panel that
    started meanwhile, asked once more, or a name a crashed panel left,
    removed before listening again.
    """
    if connect():
        return Claim.RAISED
    if listen():
        return Claim.SERVING
    if connect():
        return Claim.RAISED
    remove()
    return Claim.SERVING if listen() else Claim.ALONE


def _connect(name: str) -> bool:
    client = QLocalSocket()
    client.connectToServer(name)
    connected = client.waitForConnected(CONNECT_TIMEOUT_MS)
    client.abort()
    return connected


def claim(name: str) -> tuple[Claim, QLocalServer]:
    """`decide` on real local sockets; the server listens when serving."""
    # No socket options: with UserAccessOption Qt binds in a private
    # directory and renames over the name, so a second listen succeeds and
    # takes a running panel's name, leaving that panel unreachable.
    server = QLocalServer()

    def remove() -> None:
        QLocalServer.removeServer(name)

    outcome = decide(lambda: _connect(name), lambda: server.listen(name), remove)
    if outcome == Claim.ALONE:
        logger.warning(
            f"Cannot listen as {name} ({server.errorString()}); "
            "another start opens another panel"
        )
    return outcome, server


def serve(server: QLocalServer, on_request: Callable[[], None]) -> None:
    """Call `on_request` for each start that asks this panel to raise."""

    def accept() -> None:
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            connection.abort()
            connection.deleteLater()
        on_request()

    server.newConnection.connect(accept)


def bring_forward(window: QWidget) -> None:
    """Show `window`, unminimised, above other windows, and give it focus."""
    window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
    window.show()
    window.raise_()
    window.activateWindow()

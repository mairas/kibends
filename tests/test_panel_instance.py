"""One panel per KiCad instance: the server name, the decision to raise or
serve, and real local sockets on Qt's offscreen platform; the Qt parts are
skipped without the `gui` extra."""

import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtNetwork")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QDeadlineTimer, Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QWidget

from kibends.panel.instance import (
    Claim,
    bring_forward,
    claim,
    decide,
    kicad_socket,
    kicad_token,
    serve,
    server_name,
)

SOCKET = "ipc:///tmp/kicad/api.sock"
TOKEN = "token-1"
SETTLE_MS = 1000
"""How long a test waits for a raise request to arrive."""
POLL_S = 0.05


@pytest.fixture(scope="session")
def app() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def kicad_path() -> str:
    """A KiCad socket path no other test uses; nothing listens on it."""
    return f"ipc:///tmp/kibends-test-{uuid.uuid4().hex}/api.sock"


@pytest.fixture
def name(app: QApplication) -> Iterator[str]:
    """A server name no other test uses, removed afterwards."""
    made = server_name(kicad_path(), TOKEN)
    yield made
    QLocalServer.removeServer(made)


def until(done: Callable[[], bool]) -> bool:
    """Process events until `done`, for at most `SETTLE_MS`."""
    deadline = QDeadlineTimer(SETTLE_MS)
    while not done() and not deadline.hasExpired():
        QCoreApplication.processEvents()
    return done()


class TestServerName:
    def test_the_same_socket_and_token_give_the_same_name(self) -> None:
        assert server_name(SOCKET, TOKEN) == server_name(SOCKET, TOKEN)

    def test_another_socket_gives_another_name(self) -> None:
        assert server_name(SOCKET, TOKEN) != server_name(
            "ipc:///tmp/kicad/api-2.sock", TOKEN
        )

    def test_another_token_on_the_same_socket_gives_another_name(self) -> None:
        """A KiCad restarted on the same socket has a new token, which the
        panel the old one started cannot use."""
        assert server_name(SOCKET, TOKEN) != server_name(SOCKET, "token-2")

    def test_is_short_and_safe_as_a_file_name(self) -> None:
        made = server_name(
            "ipc:///a very/long path/" + "x" * 300 + "/api.sock", "t" * 300
        )

        assert len(made) <= 40
        assert made.replace("-", "").isalnum()

    def test_keys_on_kicads_socket_when_kicad_started_the_panel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KICAD_API_SOCKET", "ipc:///tmp/kicad/api-3.sock")

        assert kicad_socket() == "ipc:///tmp/kicad/api-3.sock"

    def test_keys_on_kipys_default_when_started_from_a_terminal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("KICAD_API_SOCKET", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

        expected = (
            "ipc:///tmp/kicad/api.sock"
            if sys.platform != "win32"
            else kicad_socket()  # under the temp directory on Windows
        )
        assert kicad_socket() == expected

    def test_the_token_is_kicads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KICAD_API_TOKEN", TOKEN)

        assert kicad_token() == TOKEN

    def test_without_a_token_it_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KICAD_API_TOKEN", raising=False)

        assert kicad_token() == ""


class TestDecide:
    def test_a_running_panel_is_asked_to_raise(self) -> None:
        calls: list[str] = []

        def connect() -> bool:
            calls.append("connect")
            return True

        def listen() -> bool:
            calls.append("listen")
            return True

        assert decide(connect, listen, lambda: calls.append("remove")) == Claim.RAISED
        assert calls == ["connect"]

    def test_without_one_this_panel_serves(self) -> None:
        assert decide(lambda: False, lambda: True, lambda: None) == Claim.SERVING

    def test_a_panel_that_started_meanwhile_is_asked_to_raise(self) -> None:
        answers = iter([False, True])
        removed: list[bool] = []

        assert (
            decide(lambda: next(answers), lambda: False, lambda: removed.append(True))
            == Claim.RAISED
        )
        assert removed == []

    def test_a_stale_name_is_removed_and_taken(self) -> None:
        listens = iter([False, True])
        removed: list[bool] = []

        assert (
            decide(lambda: False, lambda: next(listens), lambda: removed.append(True))
            == Claim.SERVING
        )
        assert removed == [True]

    def test_a_name_it_cannot_take_leaves_the_panel_alone(self) -> None:
        assert decide(lambda: False, lambda: False, lambda: None) == Claim.ALONE


class TestClaim:
    def test_the_first_panel_serves(self, name: str) -> None:
        outcome, server = claim(name)

        assert outcome == Claim.SERVING
        assert server.isListening()
        server.close()

    def test_a_second_start_asks_the_first_to_raise(self, name: str) -> None:
        _, first = claim(name)
        requests: list[bool] = []
        serve(first, lambda: requests.append(True))

        outcome, second = claim(name)

        assert outcome == Claim.RAISED
        assert not second.isListening()
        assert until(lambda: requests == [True])
        first.close()

    def test_another_kicad_gets_its_own_panel(self, app: QApplication) -> None:
        claims = [claim(server_name(kicad_path(), TOKEN)) for _ in "ab"]

        assert [outcome for outcome, _ in claims] == [Claim.SERVING, Claim.SERVING]
        for _, server in claims:
            server.close()

    @pytest.mark.skipif(sys.platform == "win32", reason="named pipes leave no file")
    def test_a_name_left_by_a_crashed_panel_is_taken(self, name: str) -> None:
        probe = QLocalServer()
        assert probe.listen(name)
        path = probe.fullServerName()
        probe.close()
        stale = socket.socket(socket.AF_UNIX)
        stale.bind(path)
        stale.close()
        assert os.path.exists(path)

        outcome, server = claim(name)

        assert outcome == Claim.SERVING
        assert server.isListening()
        server.close()

    def test_a_start_after_the_panel_closed_serves(self, name: str) -> None:
        _, first = claim(name)
        first.close()

        outcome, second = claim(name)

        assert outcome == Claim.SERVING
        second.close()

    def test_the_client_never_lingers(self, name: str) -> None:
        _, first = claim(name)
        connections: list[QLocalSocket] = []
        first.newConnection.connect(
            lambda: connections.append(first.nextPendingConnection())
        )
        unconnected = QLocalSocket.LocalSocketState.UnconnectedState

        claim(name)

        assert until(lambda: len(connections) == 1)
        assert until(lambda: connections[0].state() == unconnected)
        first.close()


class TestBringForward:
    def test_shows_and_activates_a_hidden_window(self, app: QApplication) -> None:
        window = QWidget()

        bring_forward(window)

        assert window.isVisible()
        assert until(window.isActiveWindow)
        window.close()

    def test_restores_a_minimised_window(self, app: QApplication) -> None:
        window = QWidget()
        window.showMinimized()

        bring_forward(window)

        assert not window.windowState() & Qt.WindowState.WindowMinimized
        window.close()

    def test_a_raise_request_brings_the_window_forward(self, name: str) -> None:
        window = QWidget()
        _, server = claim(name)
        serve(server, lambda: bring_forward(window))

        claim(name)

        assert until(window.isVisible)
        server.close()
        window.close()


@pytest.mark.slow
class TestPanelCommand:
    """`kibends-panel` itself, against KiCad sockets nothing listens on."""

    PANEL = (sys.executable, "-c", "from kibends.cli import panel_main; panel_main()")
    START_S = 30

    def start(self, kicad: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            self.PANEL,
            env=os.environ | {"KICAD_API_SOCKET": kicad, "KICAD_API_TOKEN": TOKEN},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def listening(self, kicad: str, panel: subprocess.Popen[bytes]) -> bool:
        """Wait until `panel` listens as `kicad`'s panel."""
        deadline = time.monotonic() + self.START_S
        while panel.poll() is None and time.monotonic() < deadline:
            client = QLocalSocket()
            client.connectToServer(server_name(kicad, TOKEN))
            if client.waitForConnected(SETTLE_MS):
                client.abort()
                return True
            time.sleep(POLL_S)
        return False

    def test_a_second_start_for_the_same_kicad_exits(self, app: QApplication) -> None:
        kicad = kicad_path()
        first = self.start(kicad)
        try:
            assert self.listening(kicad, first)

            second = self.start(kicad)

            assert second.wait(self.START_S) == 0
            assert first.poll() is None
        finally:
            first.kill()
            first.wait()
            QLocalServer.removeServer(server_name(kicad, TOKEN))

    def test_a_start_for_another_kicad_opens_its_own(self, app: QApplication) -> None:
        kicads = [kicad_path() for _ in "ab"]
        panels = [self.start(k) for k in kicads]
        try:
            for kicad, panel in zip(kicads, panels, strict=True):
                assert self.listening(kicad, panel)
        finally:
            for kicad, panel in zip(kicads, panels, strict=True):
                panel.kill()
                panel.wait()
                QLocalServer.removeServer(server_name(kicad, TOKEN))

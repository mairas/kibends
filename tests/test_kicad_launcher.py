import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kibends import cli

LAUNCHER = Path(__file__).parent.parent / "pcm" / "plugins" / "kibends_launcher.py"

if TYPE_CHECKING:
    import kibends_launcher as launcher
else:
    _spec = importlib.util.spec_from_file_location("kibends_launcher", LAUNCHER)
    assert _spec is not None and _spec.loader is not None
    launcher = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = launcher
    _spec.loader.exec_module(launcher)

Completed = launcher.Completed
Choice = launcher.Choice

HOME = "/home/u"
LOCAL_UV = "/home/u/.local/bin/uv"
LOCAL_PANEL = "/home/u/.local/bin/kibends-panel"
SPEC = f"kibends[gui]>={launcher.REQUIRED_VERSION}"
CURRENT = f"kibends {launcher.REQUIRED_VERSION}\n"
OLD = "kibends 0.0.9\n"
_MAJOR, _MINOR, _PATCH = (int(part) for part in launcher.REQUIRED_VERSION.split("."))
CURRENT_VERSIONS = [
    launcher.REQUIRED_VERSION,
    f"{_MAJOR}.{_MINOR}.{_PATCH + 1}",
    # Compared as numbers, not as text.
    f"{_MAJOR}.{_MINOR + 10}.0",
    f"{_MAJOR + 1}.0.0",
]
OK = Completed(0, "", "")
NO_GUI = Completed(
    launcher.GUI_EXTRA_EXIT,
    "",
    "ERROR | The panel needs PySide6 (No module named PySide6)",
)
CRASHED = Completed(1, "", "Traceback ...\nRuntimeError: no Qt platform plugin")

Response = Completed | OSError | Callable[[], Completed]


def posix(env: dict[str, str] | None = None) -> launcher.Host:
    return launcher.Host(windows=False, home=HOME, env=env or {})


class FakeSystem:
    """A machine with the given files, PATH entries and command responses."""

    def __init__(
        self,
        *,
        files: Sequence[str] = (),
        on_path: dict[str, str] | None = None,
        responses: dict[tuple[str, ...], Response] | None = None,
        uv_installer: Callable[[], Completed] | None = None,
    ) -> None:
        self.files = set(files)
        self.on_path = on_path or {}
        self.responses = responses or {}
        self.uv_installer = uv_installer
        self.ran: list[tuple[str, ...]] = []
        self.started: list[str] = []
        self.exits: list[Completed | None] = []
        """What each start gives in turn: an early exit, or None for a
        panel still running; None once used up."""
        self.uv_installs = 0

    def which(self, name: str) -> str | None:
        return self.on_path.get(name)

    def is_file(self, path: str) -> bool:
        return path in self.files

    def run(self, argv: Sequence[str], timeout: float | None = None) -> Completed:
        self.ran.append(tuple(argv))
        response = self.responses.get(tuple(argv))
        if response is None:
            raise FileNotFoundError(argv[0])
        if isinstance(response, OSError):
            raise response
        if callable(response):
            return response()
        return response

    def install_uv(self) -> Completed:
        self.uv_installs += 1
        assert self.uv_installer is not None
        return self.uv_installer()

    def start(self, panel: str) -> Completed | None:
        self.started.append(panel)
        return self.exits.pop(0) if self.exits else None

    def installs(self) -> list[tuple[str, ...]]:
        return [argv for argv in self.ran if "install" in argv]


class FakeUi:
    def __init__(self, choice: launcher.Choice) -> None:
        self.choice = choice
        self.asked: list[tuple[str, str]] = []
        self.instructions: list[str] = []
        self.working_on: list[str] = []
        self.failures: list[tuple[str, str]] = []
        self.successes: list[str] = []

    def ask(self, message: str, commands: str) -> launcher.Choice:
        self.asked.append((message, commands))
        return self.choice

    def show_instructions(self, text: str) -> None:
        self.instructions.append(text)

    def working(
        self, message: str, task: Callable[[], launcher.Completed]
    ) -> launcher.Completed:
        self.working_on.append(message)
        return task()

    def failed(self, message: str, details: str) -> None:
        self.failures.append((message, details))

    def succeeded(self, message: str) -> None:
        self.successes.append(message)


def launch(
    system: FakeSystem,
    ui: FakeUi | None,
    host: launcher.Host | None = None,
) -> tuple[int, str]:
    err = io.StringIO()
    code = launcher.launch(host or posix(), system, lambda: ui, err)
    return code, err.getvalue()


def uv_with_panel(version: Response, uv: str = LOCAL_UV) -> FakeSystem:
    """uv on PATH, its tool bin directory holding kibends-panel."""
    return FakeSystem(
        files=[uv, LOCAL_PANEL],
        on_path={"uv": uv},
        responses={
            (uv, "tool", "dir", "--bin"): Completed(0, "/home/u/.local/bin\n", ""),
            (LOCAL_PANEL, "--version"): version,
        },
    )


def uv_without_panel(uv: str = LOCAL_UV, on_path: bool = True) -> FakeSystem:
    return FakeSystem(
        files=[uv],
        on_path={"uv": uv} if on_path else {},
        responses={
            (uv, "tool", "dir", "--bin"): Completed(0, "/home/u/.local/bin\n", ""),
        },
    )


def installs_panel(system: FakeSystem, uv: str = LOCAL_UV) -> None:
    """`uv tool install` succeeds and leaves a current kibends-panel."""

    def install() -> Completed:
        system.files.add(LOCAL_PANEL)
        system.responses[(LOCAL_PANEL, "--version")] = Completed(0, CURRENT, "")
        return OK

    system.responses[(uv, "tool", "install", "--python", "3.14", SPEC)] = install


class TestStart:
    @pytest.mark.parametrize("version", CURRENT_VERSIONS)
    def test_a_current_panel_starts_without_asking(self, version: str) -> None:
        system = uv_with_panel(Completed(0, f"kibends {version}\n", ""))

        code, _ = launch(system, None)

        assert code == 0
        assert system.started == [LOCAL_PANEL]
        assert system.installs() == []

    def test_without_uv_a_panel_in_local_bin_starts(self) -> None:
        system = FakeSystem(
            files=[LOCAL_PANEL],
            responses={(LOCAL_PANEL, "--version"): Completed(0, CURRENT, "")},
        )

        code, _ = launch(system, None)

        assert (code, system.started) == (0, [LOCAL_PANEL])

    def test_a_failing_start_is_reported_on_stderr(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))

        def refuse(panel: str) -> Completed | None:
            raise PermissionError(13, "Permission denied", panel)

        system.start = refuse

        code, err = launch(system, None)

        assert code != 0
        assert "Permission denied" in err


class TestEarlyExit:
    def test_a_panel_that_exits_at_once_without_error_has_started(self) -> None:
        """It found a panel already open for this KiCad and raised it."""
        system = uv_with_panel(Completed(0, CURRENT, ""))
        system.exits = [OK]
        ui = FakeUi(Choice.CANCEL)

        code, err = launch(system, ui)

        assert (code, err, ui.failures, ui.asked) == (0, "", [], [])

    def test_a_panel_that_fails_at_start_is_reported_in_a_dialog(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))
        system.exits = [CRASHED]
        ui = FakeUi(Choice.CANCEL)

        code, err = launch(system, ui)

        assert code != 0
        [(message, details)] = ui.failures
        assert "exit code 1" in message
        assert details == CRASHED.output
        assert "no Qt platform plugin" in err
        assert (ui.asked, system.installs()) == ([], [])

    def test_without_wx_a_failing_panel_is_reported_on_stderr(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))
        system.exits = [CRASHED]

        code, err = launch(system, None)

        assert code != 0
        assert "no Qt platform plugin" in err

    def test_a_panel_without_the_gui_extra_offers_to_reinstall(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))
        system.exits = [NO_GUI]
        ui = FakeUi(Choice.CANCEL)

        code, _ = launch(system, ui)

        assert code == 0
        [(message, commands)] = ui.asked
        assert "gui extra" in message
        assert commands == f"{LOCAL_UV} tool install --python 3.14 '{SPEC}'"
        assert ui.failures == []

    def test_reinstalling_with_the_gui_extra_starts_the_panel(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))
        installs_panel(system)
        system.exits = [NO_GUI]
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code == 0
        assert system.installs() == [
            (LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)
        ]
        assert ui.successes
        assert system.started == [LOCAL_PANEL, LOCAL_PANEL]

    def test_a_panel_still_without_the_extra_after_reinstalling_fails(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))
        installs_panel(system)
        system.exits = [NO_GUI, NO_GUI]
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code != 0
        [(_, details)] = ui.failures
        assert "PySide6" in details

    def test_without_wx_the_missing_extra_prints_the_reinstall(self) -> None:
        system = uv_with_panel(Completed(0, CURRENT, ""))
        system.exits = [NO_GUI]

        code, err = launch(system, None)

        assert code != 0
        assert "gui" in err
        assert f"{LOCAL_UV} tool install --python 3.14 '{SPEC}'" in err
        assert system.installs() == []

    def test_the_exit_code_is_the_panels(self) -> None:
        """The launcher cannot import kibends, so it keeps a copy."""
        assert launcher.GUI_EXTRA_EXIT == cli.GUI_EXTRA_EXIT


class TestOffer:
    @pytest.mark.parametrize(
        ("version", "found"),
        [
            (Completed(0, OLD, ""), "0.0.9"),
            (Completed(2, "", "unrecognized arguments: --version"), None),
            (Completed(0, "something else\n", ""), None),
            (FileNotFoundError("kibends-panel"), None),
        ],
        ids=["older", "version-fails", "unparsable", "unrunnable"],
    )
    def test_an_outdated_panel_offers_an_upgrade(
        self, version: Response, found: str | None
    ) -> None:
        system = uv_with_panel(version)
        ui = FakeUi(Choice.CANCEL)

        launch(system, ui)

        [(message, commands)] = ui.asked
        assert "upgrade" in message.lower()
        assert launcher.REQUIRED_VERSION in message
        if found is not None:
            assert found in message
        assert commands == f"{LOCAL_UV} tool install --python 3.14 '{SPEC}'"
        assert system.started == []

    def test_uv_without_kibends_offers_installing_kibends(self) -> None:
        ui = FakeUi(Choice.CANCEL)

        launch(uv_without_panel(), ui)

        [(message, commands)] = ui.asked
        assert "not installed" in message
        assert commands == f"{LOCAL_UV} tool install --python 3.14 '{SPEC}'"

    def test_neither_offers_uv_then_kibends(self) -> None:
        ui = FakeUi(Choice.CANCEL)

        launch(FakeSystem(), ui)

        [(message, commands)] = ui.asked
        assert "uv" in message
        assert commands.splitlines() == [
            "Download https://astral.sh/uv/install.sh and run it with sh",
            f"uv tool install --python 3.14 '{SPEC}'",
        ]


class TestFindUv:
    def test_uv_only_in_a_default_location_is_found(self) -> None:
        ui = FakeUi(Choice.CANCEL)

        launch(uv_without_panel(on_path=False), ui)

        [(_, commands)] = ui.asked
        assert commands == f"{LOCAL_UV} tool install --python 3.14 '{SPEC}'"

    @pytest.mark.parametrize(
        "uv",
        ["/xdg/bin/uv", "/home/u/.cargo/bin/uv", "/opt/homebrew/bin/uv"],
    )
    def test_other_default_locations(self, uv: str) -> None:
        ui = FakeUi(Choice.CANCEL)
        host = posix({"XDG_BIN_HOME": "/xdg/bin"})

        launch(uv_without_panel(uv=uv, on_path=False), ui, host)

        [(_, commands)] = ui.asked
        assert commands.startswith(f"{uv} tool install")

    def test_uv_on_path_wins(self) -> None:
        ui = FakeUi(Choice.CANCEL)
        system = uv_without_panel(uv="/somewhere/uv")
        system.files.add(LOCAL_UV)

        launch(system, ui)

        [(_, commands)] = ui.asked
        assert commands.startswith("/somewhere/uv tool install")

    def test_the_panel_is_looked_for_in_uvs_tool_bin_directory(self) -> None:
        panel = "/tools/bin/kibends-panel"
        system = FakeSystem(
            files=[LOCAL_UV, panel],
            on_path={"uv": LOCAL_UV},
            responses={
                (LOCAL_UV, "tool", "dir", "--bin"): Completed(0, "/tools/bin\n", ""),
                (panel, "--version"): Completed(0, CURRENT, ""),
            },
        )

        launch(system, None)

        assert system.started == [panel]


class TestDecline:
    def test_cancel_runs_nothing_and_starts_no_panel(self) -> None:
        system = FakeSystem()

        code, _ = launch(system, FakeUi(Choice.CANCEL))

        assert code == 0
        assert (system.uv_installs, system.installs(), system.started) == (0, [], [])

    def test_instructions_show_the_commands_and_run_nothing(self) -> None:
        system = FakeSystem()
        ui = FakeUi(Choice.INSTRUCTIONS)

        code, _ = launch(system, ui)

        assert code == 0
        [text] = ui.instructions
        assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in text
        assert f"uv tool install --python 3.14 '{SPEC}'" in text
        assert (system.uv_installs, system.installs(), system.started) == (0, [], [])


class TestNoWx:
    def test_instructions_go_to_stderr_with_a_failing_exit(self) -> None:
        system = uv_without_panel()

        code, err = launch(system, None)

        assert code != 0
        assert f"{LOCAL_UV} tool install --python 3.14 '{SPEC}'" in err
        assert (system.installs(), system.started) == ([], [])


class TestInstall:
    def test_installing_kibends_starts_the_panel(self) -> None:
        system = uv_without_panel()
        installs_panel(system)
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code == 0
        assert system.installs() == [
            (LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)
        ]
        assert ui.working_on and ui.successes
        assert system.started == [LOCAL_PANEL]

    def test_upgrading_reinstalls_with_the_minimum_version(self) -> None:
        system = uv_with_panel(Completed(0, OLD, ""))
        installs_panel(system)

        code, _ = launch(system, FakeUi(Choice.INSTALL))

        assert code == 0
        assert system.installs() == [
            (LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)
        ]
        assert system.started == [LOCAL_PANEL]

    def test_installing_uv_then_kibends(self) -> None:
        system = FakeSystem()

        def install_uv() -> Completed:
            system.files.add(LOCAL_UV)
            system.responses[(LOCAL_UV, "tool", "dir", "--bin")] = Completed(
                0, "/home/u/.local/bin\n", ""
            )
            installs_panel(system)
            return OK

        system.uv_installer = install_uv

        code, _ = launch(system, FakeUi(Choice.INSTALL))

        assert code == 0
        assert system.uv_installs == 1
        assert system.installs() == [
            (LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)
        ]
        assert system.started == [LOCAL_PANEL]

    def test_a_failing_install_reports_its_output_and_starts_nothing(self) -> None:
        system = uv_without_panel()
        system.responses[(LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)] = (
            Completed(1, "", "error: No solution found")
        )
        ui = FakeUi(Choice.INSTALL)

        code, err = launch(system, ui)

        assert code != 0
        [(_, details)] = ui.failures
        assert "No solution found" in details
        assert "No solution found" in err
        assert system.started == []

    def test_a_failing_uv_install_stops_before_kibends(self) -> None:
        system = FakeSystem(uv_installer=lambda: Completed(1, "", "no curl"))
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code != 0
        assert "no curl" in ui.failures[0][1]
        assert (system.installs(), system.started) == ([], [])

    def test_uv_not_found_after_installing_it_is_a_failure(self) -> None:
        system = FakeSystem(uv_installer=lambda: OK)
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code != 0
        assert ui.failures
        assert system.started == []

    def test_a_panel_still_outdated_after_installing_is_a_failure(self) -> None:
        system = uv_with_panel(Completed(0, OLD, ""))
        system.responses[(LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)] = OK
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code != 0
        assert ui.failures
        assert system.started == []

    def test_an_install_that_cannot_start_is_a_failure(self) -> None:
        system = uv_without_panel()
        system.responses[(LOCAL_UV, "tool", "install", "--python", "3.14", SPEC)] = (
            PermissionError(13, "Permission denied")
        )
        ui = FakeUi(Choice.INSTALL)

        code, _ = launch(system, ui)

        assert code != 0
        assert "Permission denied" in ui.failures[0][1]


WIN_HOME = "C:\\Users\\u"
WIN_UV = "C:\\Users\\u\\.local\\bin\\uv.exe"
WIN_PANEL = "C:\\Users\\u\\.local\\bin\\kibends-panel.exe"


def windows() -> launcher.Host:
    return launcher.Host(windows=True, home=WIN_HOME, env={})


class TestWindows:
    def test_uv_in_its_default_location_and_the_panel_exe(self) -> None:
        system = FakeSystem(
            files=[WIN_UV, WIN_PANEL],
            responses={
                (WIN_UV, "tool", "dir", "--bin"): Completed(
                    0, "C:\\Users\\u\\.local\\bin\r\n", ""
                ),
                (WIN_PANEL, "--version"): Completed(
                    0, CURRENT.replace("\n", "\r\n"), ""
                ),
            },
        )

        code, _ = launch(system, None, windows())

        assert (code, system.started) == (0, [WIN_PANEL])

    def test_uv_from_cargo(self) -> None:
        uv = "C:\\Users\\u\\.cargo\\bin\\uv.exe"
        ui = FakeUi(Choice.CANCEL)
        system = FakeSystem(
            files=[uv],
            responses={(uv, "tool", "dir", "--bin"): Completed(1, "", "")},
        )

        launch(system, ui, windows())

        [(_, commands)] = ui.asked
        assert commands == f'{uv} tool install --python 3.14 "{SPEC}"'

    def test_without_uv_a_panel_in_local_bin_starts(self) -> None:
        system = FakeSystem(
            files=[WIN_PANEL],
            responses={(WIN_PANEL, "--version"): Completed(0, CURRENT, "")},
        )

        launch(system, None, windows())

        assert system.started == [WIN_PANEL]

    def test_instructions_use_powershell_and_double_quotes(self) -> None:
        ui = FakeUi(Choice.INSTRUCTIONS)

        launch(FakeSystem(), ui, windows())

        [text] = ui.instructions
        assert (
            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1'
            ' | iex"' in text
        )
        assert f'uv tool install --python 3.14 "{SPEC}"' in text

    def test_the_offer_names_the_powershell_installer(self) -> None:
        ui = FakeUi(Choice.CANCEL)

        launch(FakeSystem(), ui, windows())

        [(_, commands)] = ui.asked
        assert commands.splitlines()[0] == (
            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1'
            ' | iex"'
        )


def test_the_plugin_runs_the_launcher() -> None:
    """KiCad fails a python plugin without requirements.txt."""
    plugin = json.loads((LAUNCHER.parent / "plugin.json").read_text())

    [action] = plugin["actions"]
    assert LAUNCHER.parent / action["entrypoint"] == LAUNCHER
    assert (LAUNCHER.parent / "requirements.txt").is_file()


def test_the_installer_download_names_its_client() -> None:
    """Astral's release host answers urllib's default User-Agent with 403."""
    request = launcher.installer_request()

    assert request.full_url == launcher.UV_INSTALL_SH
    agent = request.get_header("User-agent")
    assert agent is not None
    assert not agent.startswith("Python-urllib")


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell script")
class TestRealSystem:
    def test_finds_checks_and_starts_a_panel_in_local_bin(self, tmp_path: Path) -> None:
        """The real probes and a detached start, on a fake kibends-panel."""
        bin_dir = tmp_path / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        started = tmp_path / "started"
        panel = bin_dir / "kibends-panel"
        panel.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = --version ]; then echo "{CURRENT.strip()}"; exit 0; fi\n'
            f'echo "$KICAD_API_SOCKET" > "{started}"\n'
        )
        panel.chmod(0o755)
        host = launcher.Host(
            windows=False,
            home=str(tmp_path),
            # A uv in a system location must not find the host's own panel.
            env={
                "PATH": "",
                "UV_TOOL_BIN_DIR": str(bin_dir),
                "KICAD_API_SOCKET": "ipc:///tmp/kicad/api.sock",
            },
        )

        code = launcher.launch(
            host, launcher.RealSystem(host), lambda: None, io.StringIO()
        )

        assert code == 0
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.read_text() == "ipc:///tmp/kicad/api.sock\n"

    def panel(self, tmp_path: Path, body: str) -> str:
        path = tmp_path / "kibends-panel"
        path.write_text(f"#!/bin/sh\n{body}")
        path.chmod(0o755)
        return str(path)

    def system(self, tmp_path: Path, wait: float) -> launcher.RealSystem:
        host = launcher.Host(
            windows=False, home=str(tmp_path), env={"PATH": os.defpath}
        )
        return launcher.RealSystem(host, start_wait=wait)

    WAIT_S = 0.2
    """The start wait for a panel that outlives it."""

    def test_an_early_exit_gives_its_code_and_stderr(self, tmp_path: Path) -> None:
        panel = self.panel(tmp_path, "echo out; echo 'no PySide6' >&2; exit 3\n")

        exited = self.system(tmp_path, launcher.START_WAIT).start(panel)

        assert exited == Completed(3, "", "no PySide6\n")

    def test_a_running_panel_is_left_running_and_its_stderr_unblocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its stderr goes on into a file deleted with nothing left behind,
        not a pipe, which would block it or end it once the launcher left."""
        temp = tmp_path / "temp"
        temp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp))
        done = tmp_path / "done"
        panel = self.panel(
            tmp_path,
            f"sleep {2 * self.WAIT_S}\n"
            "head -c 200000 /dev/zero | tr '\\0' x >&2\n"
            f'touch "{done}"\n',
        )

        assert self.system(tmp_path, self.WAIT_S).start(panel) is None
        assert list(temp.iterdir()) == []

        deadline = time.monotonic() + 5
        while not done.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert done.exists()

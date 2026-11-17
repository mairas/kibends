"""Start kibends' panel from a toolbar button in KiCad's PCB editor.

KiCad runs this script as the entrypoint of a ``python`` IPC plugin, in a
virtualenv it builds from its own interpreter with ``--system-site-packages``.
That interpreter is one setting for every plugin and is Python 3.9 on macOS,
while kibends needs Python 3.14, so kibends cannot run here. This launcher
finds an installed ``kibends-panel`` instead and starts it as a separate
process. It therefore uses only the standard library and Python 3.9 syntax,
and imports wxPython, which KiCad's Python ships, only to show dialogs.

The panel is looked for in uv's tool bin directory (``uv tool dir --bin``),
or in ``~/.local/bin`` when there is no uv. uv is looked for on PATH and then
in its default install locations, since KiCad started from the macOS GUI has
launchd's minimal PATH. A panel that is missing, or older than
``REQUIRED_VERSION`` (or whose ``--version`` fails), is installed with
``uv tool install``, after installing uv with Astral's installer when needed.
Reinstalling with the minimum version, rather than ``uv tool upgrade``, also
lifts an older pin and adds the ``gui`` extra.

Nothing is installed without the user's approval in a dialog that names the
commands, and which can show them for running by hand instead. Without wx,
the launcher prints the commands to stderr and exits non-zero: KiCad shows an
action's exit code and stderr in the PCB editor's status bar and discards
stdout. A declined dialog exits 0, since the user chose it.

The panel starts detached, with the launcher's environment, so that
``KICAD_API_SOCKET`` and ``KICAD_API_TOKEN`` reach it. Each press starts it
again; the panel itself hands over to one already open.

The decisions in ``launch`` take the machine (``System``) and the dialogs
(``Ui``) as arguments, so that they test without KiCad, wx or subprocesses.
"""

from __future__ import annotations

import enum
import ntpath
import os
import posixpath
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TextIO

REQUIRED_VERSION = "0.1.0"
PACKAGE = f"kibends[gui]>={REQUIRED_VERSION}"

UV_INSTALL_SH = "https://astral.sh/uv/install.sh"
UV_INSTALL_PS1 = (
    "powershell",
    "-ExecutionPolicy",
    "ByPass",
    "-c",
    "irm https://astral.sh/uv/install.ps1 | iex",
)
PROBE_TIMEOUT = 60.0
"""Seconds a probe such as ``--version`` or a download may take."""
PULSE_INTERVAL = 0.1
"""Seconds between updates of the progress dialog."""
TITLE = "kibends"
MARGIN = 12
"""Pixels around the parts of a dialog."""
TEXT_SIZE = (600, 200)
"""Pixels of the field holding commands or output."""

_VERSION_LINE = re.compile(r"kibends (\d+)\.(\d+)\.(\d+)")
_WINDOWS_SAFE = re.compile(r"[\w\-.:\\/]+")


def _version(text: str) -> tuple[int, ...] | None:
    match = _VERSION_LINE.fullmatch(text.strip())
    return tuple(int(part) for part in match.groups()) if match else None


_REQUIRED = _version(f"kibends {REQUIRED_VERSION}")


@dataclass(frozen=True)
class Host:
    """The platform, home directory and environment the launcher runs in."""

    windows: bool
    home: str
    env: Mapping[str, str]

    @classmethod
    def current(cls) -> Host:
        return cls(sys.platform == "win32", os.path.expanduser("~"), dict(os.environ))

    def join(self, *parts: str) -> str:
        return ntpath.join(*parts) if self.windows else posixpath.join(*parts)

    def exe(self, name: str) -> str:
        return f"{name}.exe" if self.windows else name

    def shell_line(self, argv: Sequence[str]) -> str:
        """``argv`` as the user would type it in a terminal."""
        if not self.windows:
            return shlex.join(argv)
        return " ".join(
            arg if _WINDOWS_SAFE.fullmatch(arg) else f'"{arg}"' for arg in argv
        )


@dataclass(frozen=True)
class Completed:
    """A finished command."""

    code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return "\n".join(
            text.strip() for text in (self.stdout, self.stderr) if text.strip()
        )


class Choice(enum.Enum):
    INSTALL = enum.auto()
    INSTRUCTIONS = enum.auto()
    CANCEL = enum.auto()


class System(Protocol):
    """What the launcher does to the machine."""

    def which(self, name: str) -> str | None: ...

    def is_file(self, path: str) -> bool: ...

    def run(self, argv: Sequence[str], timeout: float | None = None) -> Completed:
        """Run ``argv`` to the end; raises ``OSError`` if it cannot."""
        ...

    def install_uv(self) -> Completed:
        """Run Astral's uv installer; raises ``OSError`` if it cannot."""
        ...

    def start(self, panel: str) -> None:
        """Start ``panel`` detached; raises ``OSError`` if it cannot."""
        ...


class Ui(Protocol):
    """The dialogs."""

    def ask(self, message: str, commands: str) -> Choice: ...

    def show_instructions(self, text: str) -> None: ...

    def working(self, message: str, task: Callable[[], Completed]) -> Completed:
        """Run ``task``, showing that it is working."""
        ...

    def failed(self, message: str, details: str) -> None: ...

    def succeeded(self, message: str) -> None: ...


@dataclass(frozen=True)
class Start:
    panel: str
    version: tuple[int, ...]


class Need(enum.Enum):
    UV_AND_KIBENDS = enum.auto()
    KIBENDS = enum.auto()
    UPGRADE = enum.auto()


@dataclass(frozen=True)
class Offer:
    need: Need
    uv: str | None
    found: tuple[int, ...] | None = None


def uv_locations(host: Host) -> list[str]:
    """Where uv's installers put uv, for when it is not on PATH."""
    home = host.home
    if host.windows:
        return [
            host.join(home, ".local", "bin", "uv.exe"),
            host.join(home, ".cargo", "bin", "uv.exe"),
        ]
    xdg_bin = host.env.get("XDG_BIN_HOME")
    return [
        host.join(home, ".local", "bin", "uv"),
        *([host.join(xdg_bin, "uv")] if xdg_bin else []),
        host.join(home, ".cargo", "bin", "uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ]


def find_uv(host: Host, system: System) -> str | None:
    on_path = system.which("uv")
    if on_path:
        return on_path
    return next((uv for uv in uv_locations(host) if system.is_file(uv)), None)


def find_panel(host: Host, system: System, uv: str | None) -> str | None:
    bin_dir = host.join(host.home, ".local", "bin")
    if uv is not None:
        try:
            result = system.run([uv, "tool", "dir", "--bin"], PROBE_TIMEOUT)
        except OSError:
            result = None
        if result is not None and result.code == 0 and result.stdout.strip():
            bin_dir = result.stdout.strip()
    panel = host.join(bin_dir, host.exe("kibends-panel"))
    return panel if system.is_file(panel) else None


def panel_version(system: System, panel: str) -> tuple[int, ...] | None:
    """The panel's version, or None if ``--version`` does not give one."""
    try:
        result = system.run([panel, "--version"], PROBE_TIMEOUT)
    except OSError:
        return None
    return _version(result.stdout) if result.code == 0 else None


def decide(host: Host, system: System) -> Start | Offer:
    uv = find_uv(host, system)
    panel = find_panel(host, system, uv)
    if panel is None:
        return Offer(Need.KIBENDS if uv else Need.UV_AND_KIBENDS, uv)
    version = panel_version(system, panel)
    if version is not None and _REQUIRED is not None and version >= _REQUIRED:
        return Start(panel, version)
    return Offer(Need.UPGRADE, uv, version)


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def status(offer: Offer) -> str:
    if offer.need is Need.UV_AND_KIBENDS:
        return "kibends is not installed, nor is uv, which installs it"
    if offer.need is Need.KIBENDS:
        return "kibends is not installed"
    found = (
        f"kibends {_dotted(offer.found)} is installed"
        if offer.found
        else "the installed kibends does not report its version"
    )
    return f"this plugin needs kibends {REQUIRED_VERSION} or later, and {found}"


def offer_message(offer: Offer) -> str:
    question = {
        Need.UV_AND_KIBENDS: "Install uv with Astral's installer, then kibends?",
        Need.KIBENDS: "Install it with uv?",
        Need.UPGRADE: "Upgrade it with uv?",
    }[offer.need]
    text = status(offer)
    return f"{text[0].upper()}{text[1:]}. {question}"


def _uv_tool_install(host: Host, uv: str) -> str:
    return host.shell_line([uv, "tool", "install", PACKAGE])


def offer_commands(offer: Offer, host: Host) -> str:
    """What installing runs, one line per step."""
    if offer.uv is not None:
        return _uv_tool_install(host, offer.uv)
    install_uv = (
        host.shell_line(UV_INSTALL_PS1)
        if host.windows
        else f"Download {UV_INSTALL_SH} and run it with sh"
    )
    return f"{install_uv}\n{_uv_tool_install(host, 'uv')}"


def instructions(offer: Offer, host: Host) -> str:
    """The commands to install by hand, for this platform."""
    again = "Then press the kibends button in KiCad again."
    if offer.uv is not None:
        return (
            f"Run in a terminal:\n\n    {_uv_tool_install(host, offer.uv)}\n\n{again}"
        )
    install_uv = (
        host.shell_line(UV_INSTALL_PS1)
        if host.windows
        else f"curl -LsSf {UV_INSTALL_SH} | sh"
    )
    return (
        f"Install uv in a terminal:\n\n    {install_uv}\n\n"
        f"then kibends, in a new terminal:\n\n    {_uv_tool_install(host, 'uv')}\n\n"
        f"{again}"
    )


def perform(offer: Offer, host: Host, system: System) -> Completed:
    """Install what ``offer`` needs."""
    uv = offer.uv
    try:
        if uv is None:
            installed = system.install_uv()
            if installed.code != 0:
                return installed
            uv = find_uv(host, system)
            if uv is None:
                places = ", ".join(uv_locations(host))
                return Completed(
                    1, installed.output, f"uv was installed, but not found in {places}"
                )
        return system.run([uv, "tool", "install", PACKAGE])
    except OSError as error:
        return Completed(1, "", str(error))


def _fail(ui: Ui | None, err: TextIO, message: str, details: str) -> int:
    if ui is not None:
        ui.failed(message, details)
    print(f"kibends: {message}\n{details}".rstrip(), file=err)
    return 1


def _start(system: System, panel: str, ui: Ui | None, err: TextIO) -> int:
    try:
        system.start(panel)
    except OSError as error:
        return _fail(ui, err, f"could not start {panel}", str(error))
    return 0


def launch(
    host: Host, system: System, make_ui: Callable[[], Ui | None], err: TextIO
) -> int:
    """Start the panel, installing it first if the user approves; the exit code."""
    target = decide(host, system)
    if isinstance(target, Start):
        return _start(system, target.panel, None, err)
    ui = make_ui()
    if ui is None:
        print(f"kibends: {status(target)}.\n{instructions(target, host)}", file=err)
        return 1
    choice = ui.ask(offer_message(target), offer_commands(target, host))
    if choice is Choice.CANCEL:
        return 0
    if choice is Choice.INSTRUCTIONS:
        ui.show_instructions(instructions(target, host))
        return 0
    doing = "Upgrading" if target.need is Need.UPGRADE else "Installing"
    result = ui.working(
        f"{doing} kibends. This downloads Python and Qt and can take minutes.",
        lambda: perform(target, host, system),
    )
    if result.code != 0:
        return _fail(ui, err, f"{doing} kibends failed.", result.output)
    after = decide(host, system)
    if not isinstance(after, Start):
        return _fail(ui, err, f"After installing, {status(after)}.", "")
    ui.succeeded(f"kibends {_dotted(after.version)} is installed.")
    return _start(system, after.panel, ui, err)


class RealSystem:
    """The machine, through subprocesses and the file system."""

    def __init__(self, host: Host) -> None:
        self.host = host

    def which(self, name: str) -> str | None:
        path = self.host.env.get("PATH")
        return shutil.which(name, path=path) if path else None

    def is_file(self, path: str) -> bool:
        return os.path.isfile(path)

    def run(self, argv: Sequence[str], timeout: float | None = None) -> Completed:
        try:
            result = subprocess.run(
                list(argv),
                env=dict(self.host.env),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as expired:
            raise OSError(f"{argv[0]} did not finish in {timeout:g} s") from expired
        return Completed(result.returncode, result.stdout, result.stderr)

    def install_uv(self) -> Completed:
        if self.host.windows:
            return self.run(UV_INSTALL_PS1)
        with urllib.request.urlopen(
            installer_request(), timeout=PROBE_TIMEOUT, context=_tls()
        ) as response:
            script: bytes = response.read()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "install.sh")
            with open(path, "wb") as file:
                file.write(script)
            return self.run(["/bin/sh", path])

    def start(self, panel: str) -> None:
        subprocess.Popen(
            [panel],
            env=dict(self.host.env),
            # KiCad starts the launcher in the plugin's directory; a panel
            # left there would keep Windows from updating or removing it.
            cwd=self.host.home,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=sys.platform != "win32",
            creationflags=_DETACHED,
        )


if sys.platform == "win32":
    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
    _DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
else:
    _NO_WINDOW = 0
    _DETACHED = 0


def installer_request() -> urllib.request.Request:
    """The download of uv's installer; Astral refuses urllib's default User-Agent."""
    return urllib.request.Request(
        UV_INSTALL_SH, headers={"User-Agent": f"kibends-launcher/{REQUIRED_VERSION}"}
    )


def _tls() -> ssl.SSLContext:
    """KiCad's Python on macOS finds no CA certificates of its own, but has certifi."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class WxUi:
    """The dialogs, in wxPython."""

    def __init__(self) -> None:
        import wx

        self._app = wx.App(False)

    def ask(self, message: str, commands: str) -> Choice:
        import wx

        dialog = wx.MessageDialog(
            None, message, TITLE, wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION
        )
        dialog.SetExtendedMessage(f"This runs:\n\n{commands}")
        dialog.SetYesNoCancelLabels("Install", "Show instructions", "Cancel")
        answer = dialog.ShowModal()
        dialog.Destroy()
        if answer == wx.ID_YES:
            return Choice.INSTALL
        if answer == wx.ID_NO:
            return Choice.INSTRUCTIONS
        return Choice.CANCEL

    def show_instructions(self, text: str) -> None:
        self._text("To install kibends by hand:", text)

    def working(self, message: str, task: Callable[[], Completed]) -> Completed:
        import wx

        done: list[Completed] = []

        def work() -> None:
            try:
                done.append(task())
            except Exception as error:  # reported in the dialog, not lost
                done.append(Completed(1, "", f"{type(error).__name__}: {error}"))

        progress = wx.ProgressDialog(
            TITLE, message, style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME
        )
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        while thread.is_alive():
            progress.Pulse()
            thread.join(PULSE_INTERVAL)
        progress.Destroy()
        return done[0]

    def failed(self, message: str, details: str) -> None:
        self._text(message, details or "(no output)")

    def succeeded(self, message: str) -> None:
        import wx

        wx.MessageBox(message, TITLE, wx.OK | wx.ICON_INFORMATION)

    def _text(self, message: str, text: str) -> None:
        """``message`` above ``text`` in a field that can be selected and copied."""
        import wx

        dialog = wx.Dialog(
            None, title=TITLE, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(dialog, label=message), 0, wx.ALL, MARGIN)
        field = wx.TextCtrl(
            dialog,
            value=text,
            size=TEXT_SIZE,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
        )
        sizer.Add(field, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, MARGIN)
        sizer.Add(
            dialog.CreateStdDialogButtonSizer(wx.OK), 0, wx.EXPAND | wx.ALL, MARGIN
        )
        dialog.SetSizerAndFit(sizer)
        dialog.ShowModal()
        dialog.Destroy()


def make_wx_ui() -> Ui | None:
    """The wx dialogs, or None where wx does not import or start."""
    try:
        return WxUi()
    except Exception:  # any wx failure falls back to stderr
        return None


def main() -> int:
    host = Host.current()
    return launch(host, RealSystem(host), make_wx_ui, sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

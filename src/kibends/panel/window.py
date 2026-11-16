"""The panel's window and Qt application (`main`).

`PanelWindow` lays out one column (fields, Advanced behind a disclosure
collapsed at launch, the scope line, the Run/Stop button, the progress
line, the energy curve and the status) and mirrors its `Controller` on
every change. It keeps no state of its own beyond what the widgets hold:
each edit hands the controller the fields as typed, and each refresh sets
the widgets from the controller. A timer ticks the controller several
times a second, which sends the probe and keeps the elapsed time moving.
Return runs, Esc stops, and a close mid-run waits for the run to end
(`Controller.close`).

The window asks to stay above other windows with Qt's stay-on-top hint, a
normal window rather than a tool window, which macOS hides when the app
loses focus; Wayland ignores the hint.
"""

import sys
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kibends.panel.controller import Controller, RunSmooth, Spawn, State
from kibends.panel.curve import EnergyCurve
from kibends.panel.probe import BoardId, Probe, Prober
from kibends.present import Field, PanelFields
from kibends.run import run_smooth

TICK_MS = 200
STATUS_LINES = 5
"""How often the window ticks its controller."""
INVALID_STYLE = 'QLineEdit[invalid="true"] { border: 2px solid #d32f2f; }'
LABELS = {
    Field.GROUND_NETS: "Ground nets",
    Field.TIME_LIMIT: "Time limit (s)",
    Field.REPULSION: "Repulsion",
    Field.RANGE: "Range (mm)",
    Field.ATTRACTION: "Attraction",
    Field.ATTRACTION_RANGE: "Attraction range (mm)",
    Field.SPACING: "Spacing (mm)",
}
BUTTON_LABELS = ("Run", "Stop", "Stopping…")
ADVANCED = (
    Field.REPULSION,
    Field.RANGE,
    Field.ATTRACTION,
    Field.ATTRACTION_RANGE,
    Field.SPACING,
)


class Poster(QObject):
    """Runs callables posted from any thread on the thread it lives on,
    through a queued signal; once closed, drops them, since a probe may
    answer after the window is gone."""

    _posted = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._posted.connect(self._call, Qt.ConnectionType.QueuedConnection)
        self._lock = threading.Lock()
        self._closed = False

    def post(self, call: Callable[[], None]) -> None:
        # Under the lock, so no post emits once `close` has returned.
        with self._lock:
            if not self._closed:
                self._posted.emit(call)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _call(self, call: Callable[[], None]) -> None:
        call()


def _texts(fields: PanelFields) -> dict[Field, str]:
    return {
        Field.GROUND_NETS: fields.ground_nets,
        Field.TIME_LIMIT: fields.time_limit,
        Field.REPULSION: fields.repulsion,
        Field.RANGE: fields.falloff_range,
        Field.ATTRACTION: fields.attraction,
        Field.ATTRACTION_RANGE: fields.attraction_range,
        Field.SPACING: fields.spacing,
    }


class PanelWindow(QWidget):
    def __init__(
        self,
        probe: Callable[[BoardId | None], Probe],
        run: RunSmooth,
        *,
        spawn: Spawn,
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setStyleSheet(INVALID_STYLE)
        self.poster = Poster(self)
        self.controller = Controller(
            post=self.poster.post,
            spawn=spawn,
            probe=probe,
            run=run,
            notify=self.refresh,
            closable=self._closable,
        )

        self.edits = {f: QLineEdit() for f in Field}
        self.edits[Field.TIME_LIMIT].setPlaceholderText("none")
        self.via_in_pad = QCheckBox("Allow via in pad")
        self.disclosure = QToolButton()
        self.disclosure.setText("Advanced")
        self.disclosure.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.disclosure.setAutoRaise(True)
        self.advanced = QWidget()
        self.scope = QLabel()
        self.button = QPushButton("Run")
        self.progress = QLabel()
        self.curve = EnergyCurve()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Room for a summary or several field messages, so neither grows
        # the window.
        self.status.setMinimumHeight(
            self.status.fontMetrics().lineSpacing() * STATUS_LINES
        )
        self._measured = False
        self._lay_out()

        fields = self.controller.fields
        for f, text in _texts(fields).items():
            self.edits[f].setText(text)
            self.edits[f].textEdited.connect(self._edited)
        self.via_in_pad.setChecked(fields.allow_via_in_pad)
        self.via_in_pad.toggled.connect(self._edited)
        self.disclosure.clicked.connect(
            lambda: self._disclose(self.advanced.isHidden())
        )
        self.button.clicked.connect(self._press)
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            QShortcut(QKeySequence(key), self).activated.connect(self.controller.run)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(
            self.controller.stop
        )
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.refresh()

    def _lay_out(self) -> None:
        form = QFormLayout()
        for f in (Field.GROUND_NETS, Field.TIME_LIMIT):
            form.addRow(LABELS[f], self.edits[f])
        advanced = QFormLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        for f in ADVANCED:
            advanced.addRow(LABELS[f], self.edits[f])
        advanced.addRow(self.via_in_pad)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)

        column = QVBoxLayout(self)
        column.addLayout(form)
        column.addWidget(self.disclosure)
        column.addWidget(self.advanced)
        column.addWidget(rule)
        column.addWidget(self.scope)
        # One width for every label, so the button never moves.
        widths: list[int] = []
        for label in BUTTON_LABELS:
            self.button.setText(label)
            widths.append(self.button.sizeHint().width())
        self.button.setFixedWidth(max(widths))
        column.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(self.progress)
        column.addWidget(self.curve, 1)
        column.addWidget(self.status, 1)

        self._disclose(False)

    def label_of(self, edit: QLineEdit) -> str:
        """The text of the label bound to `edit`."""
        for label in self.findChildren(QLabel):
            if label.buddy() is edit:
                return label.text()
        return ""

    def refresh(self) -> None:
        """Set the widgets from the controller."""
        c = self.controller
        self.setWindowTitle(c.title)
        editable = c.editable
        for f, text in _texts(c.fields).items():
            edit = self.edits[f]
            if edit.text() != text:
                edit.setText(text)
            edit.setReadOnly(not editable)
            invalid = f in c.invalid
            if edit.property("invalid") is not invalid:
                edit.setProperty("invalid", invalid)
                edit.style().unpolish(edit)
                edit.style().polish(edit)
        self.via_in_pad.setChecked(c.fields.allow_via_in_pad)
        self.via_in_pad.setEnabled(editable)
        self.edits[Field.GROUND_NETS].setPlaceholderText(c.ground_placeholder)
        self.scope.setText(c.scope_line)
        self.button.setText(c.button_label)
        self.button.setEnabled(c.button_enabled)
        self.progress.setText(c.progress_line)
        self.curve.set_values(c.energies, c.separators)
        self.status.setText(c.status)

    def _edited(self) -> None:
        text = {f: e.text() for f, e in self.edits.items()}
        self.controller.edit(
            PanelFields(
                ground_nets=text[Field.GROUND_NETS],
                time_limit=text[Field.TIME_LIMIT],
                repulsion=text[Field.REPULSION],
                falloff_range=text[Field.RANGE],
                attraction=text[Field.ATTRACTION],
                attraction_range=text[Field.ATTRACTION_RANGE],
                spacing=text[Field.SPACING],
                allow_via_in_pad=self.via_in_pad.isChecked(),
            )
        )

    def _disclose(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self.disclosure.setArrowType(
            Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow
        )

    def _press(self) -> None:
        if self.controller.state is State.RUNNING:
            self.controller.stop()
        else:
            self.controller.run()

    def _tick(self) -> None:
        self.controller.tick()
        self.refresh()

    def showEvent(self, event: QShowEvent) -> None:
        # The minimum fits Advanced open, so opening it never squeezes the
        # window; measured once shown, when the style's metrics apply.
        if not self._measured:
            self._measured = True
            shown = self.advanced.isVisible()
            self.advanced.setVisible(True)
            # Natural width: narrower, the summary wraps beyond the status.
            self.setMinimumSize(
                self.sizeHint().width(), self.minimumSizeHint().height()
            )
            self.advanced.setVisible(shown)
        super().showEvent(event)

    def _closable(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller.close():
            self._timer.stop()
            self.poster.close()
            event.accept()
        else:
            event.ignore()


def start_thread(work: Callable[[], None], *, daemon: bool) -> None:
    threading.Thread(target=work, daemon=daemon).start()


def main() -> None:
    """Open the panel and run until it is closed."""
    app = QApplication(sys.argv)
    window = PanelWindow(Prober(), run_smooth, spawn=start_thread)
    window.show()
    window.controller.tick()
    app.exec()

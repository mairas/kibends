"""The panel's window and Qt application (`main`).

`PanelWindow` lays out one column: the scope line (what Run touches),
the settings (Ground nets and Time limit, then the rest behind an Advanced
disclosure, collapsed at launch), the activity area (the progress line
with the elapsed time, the graph's header, the graph (`curve`) and the
report area), then Stop and Run. It mirrors its `Controller` on every
change and keeps no state of its own beyond what the widgets hold and
whether the run's warnings are shown: each edit hands the controller the
fields as typed, each field the user leaves is reported (`leave`), and
each refresh sets the widgets from the controller. A timer ticks the
controller ten times a second, which sends the probe and keeps the
elapsed time moving.

Return runs from anywhere, unless a button or the checkbox has focus,
which it then activates; Esc stops and never closes the window. Closing
while the run can still stop asks first; later in a run the window waits
for the run to end (`Controller.close`).

Colours come from the palette (`curve.secondary` for captions), and the
only style sheet is an invalid field's border, in an error red picked for
a light or a dark palette. The window asks to stay above other windows
with Qt's stay-on-top hint, a normal window rather than a tool window,
which macOS hides when the app loses focus; Wayland ignores the hint.
"""

import html
import sys
import threading
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from typing import cast

from PySide6.QtCore import QEvent, QMimeData, QObject, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFocusEvent,
    QFont,
    QGuiApplication,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPalette,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kibends.panel.controller import (
    ADVANCED,
    Controller,
    Icon,
    Line,
    Outcome,
    RunSmooth,
    Spawn,
)
from kibends.panel.curve import LengthGraph, secondary, small
from kibends.panel.probe import BoardId, Probe, Prober
from kibends.present import Field, PanelFields
from kibends.run import run_smooth

TICK_MS = 100
"""How often the window ticks its controller and the elapsed time."""
WIDTH = 380
MIN_WIDTH = 360
MARGINS = (14, 12, 14, 12)
"""Left, top, right, bottom, px."""
BLOCK_SPACING = 8
REPORT_HEIGHT = 138
REPORT_LINES = 7
"""The report area's least height, px, or in lines of the font if more."""
DETAIL_LINES = 4
NET_LINES = 3
"""The ground net list's most lines before it scrolls."""
BUTTON_WIDTH = 88
ERROR_LIGHT = "#C4161C"
ERROR_DARK = "#FF6B62"
DARK = 0.5
"""A palette whose window is darker than this lightness is dark."""
WARNING_ICON = 14
MESSAGE_ICON = 12
HEADLINE_ICON = 16
ARROW = 10
GRID_GAP = 12

ATTRIBUTES = {
    Field.GROUND_NETS: "ground_nets",
    Field.TIME_LIMIT: "time_limit",
    Field.REPULSION: "repulsion",
    Field.RANGE: "falloff_range",
    Field.ATTRACTION: "attraction",
    Field.ATTRACTION_RANGE: "attraction_range",
    Field.SPACING: "spacing",
}
"""The `PanelFields` attribute each field's edit holds."""
DIGITS = {
    Field.TIME_LIMIT: 6,
    Field.REPULSION: 5,
    Field.RANGE: 5,
    Field.ATTRACTION: 5,
    Field.ATTRACTION_RANGE: 5,
    Field.SPACING: 14,
}
"""How many digits wide each line edit is."""
TIPS = {
    Field.GROUND_NETS: "Vias of these nets are return vias: other vias are drawn "
    "toward them, and they spread apart from each other. Separate names with "
    "commas. Override detection when it picked a power net poured on inner layers.",
    Field.TIME_LIMIT: "How long each spacing level may optimize. When it runs out "
    "at a coarse level, the run moves to the next level and the time starts "
    "again; at the final level, optimizing stops and the best result so far is "
    "committed. Empty: no limit.",
    Field.REPULSION: "Weight of keeping tracks and vias away from other nets' "
    "copper, relative to length. It rounds bends and centres tracks in free "
    "space. 0 turns it off: tracks pull tight against obstacles at exactly the "
    "clearance.",
    Field.RANGE: "Distance beyond the required clearance at which the push fades "
    "to zero. Copper farther away has no effect.",
    Field.ATTRACTION: "Weight of the pull between ground vias and other vias, so "
    "that signal vias keep their return vias near. While it is above 0, ground "
    "vias of one net also push each other apart, with Repulsion's range. 0 turns "
    "both off, and Ground nets is then ignored.",
    Field.ATTRACTION_RANGE: "Centre-to-centre distance between a ground via and "
    "another via at which the pull fades to zero.",
    Field.SPACING: "Distances between a chain's vertices that each problem runs "
    "at, coarsest first. Each level hands over to the next when it settles or the "
    "time limit runs out. The last value, at most 0.1, is the final spacing; one "
    "value runs at it alone. At most 8 values, each smaller than the one before.",
}
VIA_IN_PAD_TIP = (
    "Let a moving via settle inside a pad of its own net. KiCad's DRC does not "
    "check via-in-pad, so this is the only guard: use it only where the "
    "fabricator fills and caps vias in pads."
)
ICONS = {
    Icon.SUCCESS: QStyle.StandardPixmap.SP_DialogApplyButton,
    Icon.INFO: QStyle.StandardPixmap.SP_MessageBoxInformation,
    Icon.WARNING: QStyle.StandardPixmap.SP_MessageBoxWarning,
    Icon.CRITICAL: QStyle.StandardPixmap.SP_MessageBoxCritical,
}


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


def is_dark(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Window).lightnessF() < DARK


def error_colour(palette: QPalette) -> QColor:
    return QColor(ERROR_DARK if is_dark(palette) else ERROR_LIGHT)


def rich(line: Line) -> str:
    bold = f"<b>{html.escape(line.bold)}</b>" if line.bold else ""
    return bold + html.escape(line.rest)


def tinted(label: QWidget, colour: QColor) -> None:
    palette = label.palette()
    palette.setColor(QPalette.ColorRole.WindowText, colour)
    palette.setColor(QPalette.ColorRole.Text, colour)
    label.setPalette(palette)


class NetEditor(QPlainTextEdit):
    """A line edit for a list of net names that wraps instead of
    scrolling: one to `NET_LINES` lines high, as its text needs. Return
    goes to the window, and pasted line breaks and tabs become separators.
    `edited` hears the user's changes, `left` the focus leaving."""

    edited = Signal(str)
    left = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._setting = False
        self.textChanged.connect(self._changed)
        self._fit()

    def set_text(self, text: str) -> None:
        """Show `text`, keeping the cursor at its end, as no edit."""
        if self.toPlainText() == text:
            return
        self._setting = True
        self.setPlainText(text)
        self.moveCursor(self.textCursor().MoveOperation.End)
        self._setting = False
        self._fit()

    def _changed(self) -> None:
        self._fit()
        if not self._setting:
            self.edited.emit(self.toPlainText())

    def _fit(self) -> None:
        lines = max(1, min(NET_LINES, round(self.document().size().height())))
        margins = self.contentsMargins()
        height = (
            self.fontMetrics().lineSpacing() * lines
            + 2 * self.document().documentMargin()
            + margins.top()
            + margins.bottom()
            + 2 * self.frameWidth()
        )
        self.setFixedHeight(round(height))

    def resizeEvent(self, e: QResizeEvent) -> None:
        super().resizeEvent(e)
        self._fit()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.ignore()
            return
        super().keyPressEvent(e)

    def insertFromMimeData(self, source: QMimeData) -> None:
        text = source.text().replace("\r", "").replace("\t", ", ").replace("\n", ", ")
        self.insertPlainText(text)

    def focusOutEvent(self, e: QFocusEvent) -> None:
        super().focusOutEvent(e)
        self.left.emit()


class Message(QWidget):
    """An invalid field's message: a warning icon and the text, in the
    error colour."""

    def __init__(self) -> None:
        super().__init__()
        self.icon = QLabel()
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setFont(small(self.text.font()))
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.text, 1)
        self.icon.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            .pixmap(MESSAGE_ICON, MESSAGE_ICON)
        )
        self.hide()


class Row:
    """A settings row's fields, the message for those invalid and the
    caption the message replaces."""

    def __init__(self, fields: tuple[Field, ...], caption: QLabel | None) -> None:
        self.fields = fields
        self.caption = caption
        self.message = Message()

    def show(self, messages: dict[Field, str]) -> None:
        shown = [messages[f] for f in self.fields if f in messages]
        self.message.text.setText("\n".join(shown))
        self.message.setVisible(bool(shown))
        if self.caption is not None:
            self.caption.setVisible(not shown and bool(self.caption.text()))


class ReportArea(QWidget):
    """The run's outcome, or a hint in its place: a headline, a body, a
    grid of figures and a box for a detail or the run's warnings."""

    def __init__(self) -> None:
        super().__init__()
        self.outcome: Outcome | str | None = None
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setFont(small(self.hint.font()))
        self.icon = QLabel()
        self.headline = QLabel()
        self.headline.setWordWrap(True)
        self.headline.setTextFormat(Qt.TextFormat.RichText)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(GRID_GAP)
        self.grid.setVerticalSpacing(2)
        self.grid.setColumnStretch(1, 1)
        self.keys: list[QLabel] = []
        self.values: list[QLabel] = []
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.copy = QPushButton("Copy")
        self.copy.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.detail.toPlainText())
        )
        self.warnings: tuple[str, ...] = ()
        self.warnings_shown = False

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.hint)
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        top.addWidget(self.headline, 1)
        column.addLayout(top)
        column.addWidget(self.body)
        column.addLayout(self.grid)
        column.addWidget(self.detail)
        column.addWidget(self.copy, 0, Qt.AlignmentFlag.AlignRight)
        column.addStretch(1)
        lines = self.fontMetrics().lineSpacing()
        self.detail.setFixedHeight(
            round(
                lines * DETAIL_LINES
                + 2 * self.detail.document().documentMargin()
                + 2 * self.detail.frameWidth()
            )
        )
        self.least = max(REPORT_HEIGHT, lines * REPORT_LINES)
        self.set_outcome("")

    def minimumSizeHint(self) -> QSize:
        # At the width it has, since its wrapped lines need more height
        # than the layout's minimum gives them.
        hint = super().minimumSizeHint()
        layout = self.layout()
        height = hint.height()
        if layout is not None and self.width() > 0:
            height = max(height, layout.totalHeightForWidth(self.width()))
        return QSize(hint.width(), max(height, self.least))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.updateGeometry()

    def set_outcome(self, outcome: Outcome | str) -> None:
        if outcome == self.outcome:
            return
        self.outcome = outcome
        self.warnings_shown = False
        is_hint = isinstance(outcome, str)
        self.hint.setVisible(is_hint)
        for widget in (self.icon, self.headline, self.body):
            widget.setVisible(not is_hint)
        for label in (*self.keys, *self.values):
            self.grid.removeWidget(label)
            label.deleteLater()
        self.keys, self.values = [], []
        if isinstance(outcome, str):
            self.hint.setText(outcome)
            self.warnings = ()
            self._show_detail("")
            return
        self.icon.setPixmap(
            self.style()
            .standardIcon(ICONS[outcome.icon])
            .pixmap(HEADLINE_ICON, HEADLINE_ICON)
        )
        self.headline.setText(rich(Line(outcome.title, outcome.rest)))
        self.body.setText(outcome.body)
        self.body.setVisible(bool(outcome.body))
        rows = list(outcome.rows)
        self.warnings = outcome.warnings
        if outcome.warnings:
            rows.append(("Warnings", ""))
        for i, (key, value) in enumerate(rows):
            k = QLabel(key)
            v = QLabel(value)
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.grid.addWidget(k, i, 0, Qt.AlignmentFlag.AlignTop)
            self.grid.addWidget(v, i, 1, Qt.AlignmentFlag.AlignTop)
            k.show()
            v.show()
            self.keys.append(k)
            self.values.append(v)
        if outcome.warnings:
            toggle = self.values[-1]
            toggle.setTextInteractionFlags(
                Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            toggle.linkActivated.connect(self._link)
            self._label_warnings()
        self.recolour()
        self._show_detail(outcome.detail)
        self.updateGeometry()

    def _link(self, _href: str) -> None:
        self.toggle_warnings()

    def toggle_warnings(self) -> None:
        self.warnings_shown = not self.warnings_shown
        self._label_warnings()
        self._show_detail("\n".join(self.warnings) if self.warnings_shown else "")
        self.updateGeometry()

    def _label_warnings(self) -> None:
        action = "hide" if self.warnings_shown else "show"
        self.values[-1].setText(f'{len(self.warnings)} — <a href="#">{action}</a>')

    def _show_detail(self, text: str) -> None:
        self.detail.setPlainText(text)
        self.detail.setVisible(bool(text))
        self.copy.setVisible(bool(text))

    def recolour(self) -> None:
        muted = secondary(self.palette())
        for label in (self.hint, *self.keys):
            tinted(label, muted)


class PanelWindow(QWidget):
    def __init__(
        self,
        probe: Callable[[BoardId | None], Probe],
        run: RunSmooth,
        *,
        spawn: Spawn,
    ) -> None:
        super().__init__()
        self._built = False
        self._sized = False
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.poster = Poster(self)
        self.controller = Controller(
            post=self.poster.post,
            spawn=spawn,
            probe=probe,
            run=run,
            notify=self.refresh,
            closable=self._closable,
        )
        self.confirm: QMessageBox | None = None
        self._focused = False

        self.scope = QLabel()
        self.scope.setWordWrap(True)
        self.scope.setTextFormat(Qt.TextFormat.RichText)
        self.nets = NetEditor()
        self.edits = {f: QLineEdit() for f in Field if f is not Field.GROUND_NETS}
        self.edits[Field.TIME_LIMIT].setPlaceholderText("none")
        for f, edit in self.edits.items():
            digits = edit.fontMetrics().horizontalAdvance("0" * DIGITS[f])
            edit.setFixedWidth(digits + 16)
        self.ground_caption = QLabel()
        self.use_detected = QPushButton("Use detected")
        self.use_detected.setFlat(True)
        self.use_detected.setCursor(Qt.CursorShape.PointingHandCursor)
        self.via_in_pad = QCheckBox("Allow via in pad")
        self.restore = QPushButton("Restore defaults")
        self.disclosure = QToolButton()
        self.disclosure.setAutoRaise(True)
        self.disclosure.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced = QWidget()
        self.progress = QLabel()
        self.progress.setTextFormat(Qt.TextFormat.RichText)
        self.progress.setWordWrap(True)
        self.elapsed = QLabel()
        self.elapsed.setAlignment(Qt.AlignmentFlag.AlignRight)
        figures = QFont(self.elapsed.font())
        figures.setFeature(QFont.Tag("tnum"), 1)
        self.elapsed.setFont(figures)
        self.graph_title = QLabel()
        self.graph_figure = QLabel()
        self.graph_figure.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.graph = LengthGraph()
        self.report = ReportArea()
        self.stop_button = QPushButton("&Stop")
        self.run_button = QPushButton("&Run")
        self.run_button.setDefault(True)
        self.stop_button.setToolTip(
            "Stop optimizing and commit the best result so far (Esc)"
        )
        self.captions: list[QLabel] = [self.ground_caption]
        self.rows: list[Row] = []
        self._lay_out()
        self._describe()

        fields = self.controller.fields
        for f, edit in self.edits.items():
            edit.setText(cast(str, getattr(fields, ATTRIBUTES[f])))
            edit.textEdited.connect(self._edited)
            edit.editingFinished.connect(partial(self.controller.leave, f))
        self.nets.edited.connect(self.controller.type_ground_nets)
        self.nets.left.connect(lambda: self.controller.leave(Field.GROUND_NETS))
        self.use_detected.clicked.connect(self.controller.use_detected)
        self.via_in_pad.setChecked(fields.allow_via_in_pad)
        self.via_in_pad.toggled.connect(self._edited)
        self.restore.clicked.connect(self.controller.restore_defaults)
        self.disclosure.clicked.connect(
            lambda: self._disclose(not self.advanced.isVisible())
        )
        self.run_button.clicked.connect(self.attempt_run)
        self.stop_button.clicked.connect(self.controller.stop)
        QShortcut(QKeySequence.StandardKey.Close, self).activated.connect(self.close)
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._recolour()
        self._disclose(False)
        self.setFocus()
        self.refresh()
        self._built = True

    def _lay_out(self) -> None:
        column = QVBoxLayout(self)
        column.setContentsMargins(*MARGINS)
        column.setSpacing(BLOCK_SPACING)
        column.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        column.addWidget(self.scope)
        top = rule()
        top.setMinimumWidth(MIN_WIDTH - MARGINS[0] - MARGINS[2])
        column.addWidget(top)

        settings = QGridLayout()
        settings.setHorizontalSpacing(8)
        settings.setVerticalSpacing(3)
        settings.setColumnStretch(1, 1)
        nets_label = QLabel("&Ground nets")
        nets_label.setBuddy(self.nets)
        settings.addWidget(nets_label, 0, 0, Qt.AlignmentFlag.AlignRight)
        settings.addWidget(self.nets, 0, 1)
        under = QHBoxLayout()
        under.setSpacing(4)
        self.ground_caption.setWordWrap(True)
        under.addWidget(self.ground_caption, 1)
        under.addWidget(self.use_detected)
        settings.addLayout(under, 1, 1)
        time_label = QLabel("&Time limit")
        time_label.setBuddy(self.edits[Field.TIME_LIMIT])
        settings.addWidget(
            time_label,
            2,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        settings.addLayout(
            fields_row(self.edits[Field.TIME_LIMIT], QLabel("s per spacing level")),
            2,
            1,
        )
        time = Row((Field.TIME_LIMIT,), None)
        settings.addWidget(time.message, 3, 1)
        self.rows.append(time)
        column.addLayout(settings)

        column.addWidget(self.disclosure, 0, Qt.AlignmentFlag.AlignLeft)
        advanced = QGridLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 4)
        advanced.setHorizontalSpacing(8)
        advanced.setVerticalSpacing(3)
        advanced.setColumnStretch(1, 1)
        line = 0
        for weight, reach, caption in (
            (
                Field.REPULSION,
                Field.RANGE,
                "Pushes tracks and vias away from other nets' copper.",
            ),
            (
                Field.ATTRACTION,
                Field.ATTRACTION_RANGE,
                "Pulls signal vias and ground vias toward each other.",
            ),
        ):
            title = group_title(weight.value, self.edits[weight])
            reach_label = QLabel("range")
            reach_label.setBuddy(self.edits[reach])
            advanced.addWidget(title, line, 0)
            advanced.addLayout(
                fields_row(
                    self.edits[weight],
                    spacer(8),
                    reach_label,
                    self.edits[reach],
                    QLabel("mm"),
                ),
                line,
                1,
            )
            line = self._caption_rows(advanced, line + 1, (weight, reach), caption)
        advanced.addWidget(group_title("Spacing", self.edits[Field.SPACING]), line, 0)
        advanced.addLayout(fields_row(self.edits[Field.SPACING], QLabel("mm")), line, 1)
        line = self._caption_rows(
            advanced,
            line + 1,
            (Field.SPACING,),
            "Coarsest first; the last, at most 0.1, is the final spacing.",
        )
        box = QHBoxLayout()
        box.addWidget(self.via_in_pad)
        box.addStretch(1)
        box.addWidget(self.restore)
        advanced.addLayout(box, line, 0, 1, 2)
        self._caption_rows(
            advanced,
            line + 1,
            (),
            "KiCad's DRC does not check via-in-pad: only for filled, capped vias.",
        )
        column.addWidget(self.advanced)
        column.addWidget(rule())

        progress = QHBoxLayout()
        progress.addWidget(self.progress, 1)
        progress.addWidget(self.elapsed, 0, Qt.AlignmentFlag.AlignTop)
        column.addLayout(progress)
        header = QHBoxLayout()
        header.addWidget(self.graph_title, 1)
        header.addWidget(self.graph_figure)
        self.captions += [self.graph_title, self.graph_figure]
        column.addLayout(header)
        column.addWidget(self.graph)
        column.addWidget(self.report, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for button in (self.stop_button, self.run_button):
            width = max(
                BUTTON_WIDTH, button.sizeHint().width(), self._stopping_width(button)
            )
            button.setMinimumWidth(width)
            buttons.addWidget(button)
        column.addLayout(buttons)
        for label in self.captions:
            label.setFont(small(label.font()))

        order: list[QWidget] = [
            self.nets,
            self.use_detected,
            self.edits[Field.TIME_LIMIT],
            self.disclosure,
            *(self.edits[f] for f in Field if f in ADVANCED),
            self.via_in_pad,
            self.restore,
            self.stop_button,
            self.run_button,
        ]
        for first, second in pairwise(order):
            QWidget.setTabOrder(first, second)

    def _stopping_width(self, button: QPushButton) -> int:
        if button is not self.stop_button:
            return 0
        return button.fontMetrics().horizontalAdvance("Stopping…") + 24

    def _caption_rows(
        self, grid: QGridLayout, line: int, fields: tuple[Field, ...], text: str
    ) -> int:
        """Lay out the message and the caption under a row of `fields`, from
        `line`; the next free line."""
        caption = QLabel(text)
        caption.setWordWrap(True)
        self.captions.append(caption)
        row = Row(fields, caption)
        self.rows.append(row)
        grid.addWidget(row.message, line, 0, 1, 2)
        grid.addWidget(caption, line + 1, 0, 1, 2)
        return line + 2

    def _describe(self) -> None:
        """Every input's accessible name and description."""
        names = {
            Field.REPULSION: "Repulsion",
            Field.RANGE: "Repulsion range",
            Field.ATTRACTION: "Attraction",
            Field.ATTRACTION_RANGE: "Attraction range",
            Field.SPACING: "Spacing",
            Field.TIME_LIMIT: "Time limit",
        }
        for f, edit in self.edits.items():
            edit.setAccessibleName(names[f])
            edit.setToolTip(TIPS[f])
        self.nets.setAccessibleName("Ground nets")
        self.nets.setToolTip(TIPS[Field.GROUND_NETS])
        self.nets.setAccessibleDescription(TIPS[Field.GROUND_NETS])
        self.via_in_pad.setAccessibleName("Allow via in pad")
        self.via_in_pad.setToolTip(VIA_IN_PAD_TIP)
        self.via_in_pad.setAccessibleDescription(VIA_IN_PAD_TIP)

    def refresh(self) -> None:
        """Set the widgets from the controller."""
        c = self.controller
        self.setWindowTitle(c.title)
        self.scope.setText(rich(c.scope_line))
        editable = c.editable
        fields = c.fields
        shown = dict(c.shown_invalid)
        error = error_colour(self.palette()).name()
        for f, edit in self.edits.items():
            text = cast(str, getattr(fields, ATTRIBUTES[f]))
            if edit.text() != text:
                edit.setText(text)
            edit.setReadOnly(not editable)
            message = shown.get(f)
            style = f"QLineEdit {{ border: 2px solid {error}; border-radius: 3px; }}"
            edit.setStyleSheet(style if message else "")
            edit.setAccessibleDescription(message or TIPS[f])
        for row in self.rows:
            row.show(shown)
        ground = c.ground
        self.nets.set_text(ground.text)
        self.nets.setPlaceholderText(ground.placeholder)
        self.nets.setEnabled(ground.enabled)
        self.nets.setReadOnly(not editable)
        self.ground_caption.setText(ground.caption)
        self.ground_caption.setVisible(bool(ground.caption))
        self.use_detected.setVisible(ground.use_detected)
        self.use_detected.setEnabled(editable)
        self.via_in_pad.setChecked(fields.allow_via_in_pad)
        self.via_in_pad.setEnabled(editable)
        self.restore.setEnabled(editable)
        self._label_disclosure()

        line = c.progress_line
        self.progress.setText(rich(line))
        # Named anew only as the stage changes, not at every iteration.
        name = line.bold or line.rest
        if self.progress.accessibleName() != name:
            self.progress.setAccessibleName(name)
        self.elapsed.setText(c.elapsed)
        self.graph.set_graph(c.graph)
        self.graph_title.setText(c.graph_title)
        self.graph_figure.setText(c.graph_figure)
        if self.graph.accessibleDescription() != c.graph_description:
            self.graph.setAccessibleDescription(c.graph_description)
        self.report.set_outcome(c.report)
        self.stop_button.setText("&Stop" if c.stop_label == "Stop" else c.stop_label)
        self._enable_buttons(c.run_enabled, c.stop_enabled)
        self.run_button.setToolTip(c.run_tip)

    def _enable_buttons(self, run: bool, stop: bool) -> None:
        """Enable Run and Stop as given. Focus on a button that turns off
        moves to the other if it is on, else to the window, which takes
        Return and Esc as well, and back to Run once it is on; at launch it
        goes to Run once Run is first on."""
        focus = self.focusWidget()
        parked = focus is self or not self._focused
        for button, on, other, other_on in (
            (self.run_button, run, self.stop_button, stop),
            (self.stop_button, stop, self.run_button, run),
        ):
            if focus is button and not on:
                if other_on:
                    other.setEnabled(True)
                    other.setFocus()
                else:
                    self.setFocus()
        self.run_button.setEnabled(run)
        self.stop_button.setEnabled(stop)
        if run and parked:
            self._focused = True
            self.run_button.setFocus()

    def _label_disclosure(self) -> None:
        fixes = self.controller.advanced_fixes
        warn = fixes > 0 and not self.advanced.isVisible()
        text = "&Advanced"
        if warn:
            text += f" — {fixes} field{' needs' if fixes == 1 else 's need'} a fix"
        self.disclosure.setText(text)
        self.disclosure.setIcon(self._disclosure_icon(self.advanced.isVisible(), warn))

    def _disclosure_icon(self, shown: bool, warn: bool) -> QIcon:
        """The disclosure's arrow, followed by a warning sign if `warn`."""
        ratio = self.devicePixelRatioF()
        width = ARROW + (4 + WARNING_ICON if warn else 0)
        pixmap = QPixmap(QSize(round(width * ratio), round(WARNING_ICON * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        option = QStyleOption()
        option.initFrom(self.disclosure)
        option.rect = QRect(0, (WARNING_ICON - ARROW) // 2, ARROW, ARROW)
        arrow = (
            QStyle.PrimitiveElement.PE_IndicatorArrowDown
            if shown
            else QStyle.PrimitiveElement.PE_IndicatorArrowRight
        )
        self.style().drawPrimitive(arrow, option, painter, self.disclosure)
        if warn:
            sign = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            sign.paint(painter, ARROW + 4, 0, WARNING_ICON, WARNING_ICON)
        painter.end()
        icon = QIcon(pixmap)
        self.disclosure.setIconSize(QSize(width, WARNING_ICON))
        return icon

    def _edited(self) -> None:
        texts = {ATTRIBUTES[f]: e.text() for f, e in self.edits.items()}
        self.controller.edit(
            PanelFields(**texts, allow_via_in_pad=self.via_in_pad.isChecked())
        )

    def _disclose(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self._label_disclosure()
        layout = self.layout()
        if isinstance(layout, QLayout):
            layout.activate()
        self.resize(self.width(), self.sizeHint().height())

    def attempt_run(self) -> None:
        """Run, after taking in the field that has focus; with an invalid
        field, show every message and go to the first."""
        focus = self.focusWidget()
        for f, edit in self.edits.items():
            if focus is edit:
                self.controller.leave(f)
        if focus is self.nets:
            self.controller.leave(Field.GROUND_NETS)
        if self.controller.run():
            return
        first = self.controller.first_invalid
        if first is None:
            return
        if first in ADVANCED:
            self._disclose(True)
        edit = self.edits[first]
        edit.setFocus()
        edit.selectAll()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            focus = self.focusWidget()
            if isinstance(focus, QAbstractButton) and focus is not self.run_button:
                focus.click()
            else:
                self.attempt_run()
            event.accept()
        elif key == Qt.Key.Key_Escape:
            self.controller.stop()
            event.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        # Sized once shown, when the style's metrics apply.
        if not self._sized:
            self._sized = True
            self.resize(WIDTH, self.sizeHint().height())
        super().showEvent(event)

    def _tick(self) -> None:
        self.controller.tick()
        self.elapsed.setText(self.controller.elapsed)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if (
            event.type()
            in (
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
            )
            and self._built
        ):
            self._recolour()
            self.refresh()

    def _recolour(self) -> None:
        palette = self.palette()
        muted = secondary(palette)
        for label in self.captions:
            tinted(label, muted)
        error = error_colour(palette)
        for row in self.rows:
            tinted(row.message.text, error)
        link = self.use_detected.palette()
        link.setColor(
            QPalette.ColorRole.ButtonText, palette.color(QPalette.ColorRole.Link)
        )
        self.use_detected.setPalette(link)
        self.report.recolour()
        self.graph.update()

    def _closable(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        c = self.controller
        if c.close_asks:
            event.ignore()
            self._ask_to_close()
            return
        if c.close():
            self._timer.stop()
            self.poster.close()
            event.accept()
        else:
            event.ignore()
            self.refresh()

    def _ask_to_close(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("A run is in progress")
        box.setText("A run is in progress")
        box.setInformativeText("Stop it and commit the best result so far, then close?")
        stop = box.addButton("Stop and close", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Keep running", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(stop)
        box.setWindowModality(Qt.WindowModality.WindowModal)

        def answered(button: QAbstractButton) -> None:
            self.confirm = None
            if button is stop and self.controller.close():
                self.close()
            self.refresh()

        box.buttonClicked.connect(answered)
        self.confirm = box
        box.open()


def rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def spacer(width: int) -> QWidget:
    gap = QWidget()
    gap.setFixedWidth(width)
    return gap


def fields_row(*widgets: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    for widget in widgets:
        row.addWidget(widget)
    row.addStretch(1)
    return row


def group_title(text: str, buddy: QWidget) -> QLabel:
    title = QLabel(text)
    bold = QFont(title.font())
    bold.setBold(True)
    title.setFont(bold)
    title.setBuddy(buddy)
    return title


def start_thread(work: Callable[[], None], *, daemon: bool) -> None:
    threading.Thread(target=work, daemon=daemon).start()


def main() -> None:
    """Open the panel and run until it is closed."""
    app = QApplication(sys.argv)
    window = PanelWindow(Prober(), run_smooth, spawn=start_thread)
    window.show()
    window.controller.tick()
    app.exec()

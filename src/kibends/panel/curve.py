"""The panel's graph (`LengthGraph`): the change of the summed track length
from the run's start, in mm, against the outer iteration, as the
controller's `Graph` gives it.

The x axis starts at iteration 0, the start, and reaches at least 10; the
y axis always holds 0, the start, and spans at least `floor` below it, so
that a run that hardly moves the tracks shows as a nearly flat line rather
than noise filling the plot. Both take steps of 1, 2 or 5 times a power of ten
(`nice_step`): at most 5 intervals across, 4 up. The y range only grows
during a run, so the line never jumps back as it rescales. A dashed line
marks where each finer spacing level starts, its spacing above it; the line
runs on across it, so the jump a finer spacing makes shows as a step.

Colours come from the palette: the accent for the line, its fill and its
start and end marks, `Base` for the plot area, `Mid` for its frame, and
the window's text mixed into its background (`secondary`) for labels and
marks. Drawn with QPainter, since Qt's chart modules are GPLv3 or
commercial and ship outside PySide6-Essentials. Repaints are coalesced
to one per `REPAINT_MS`.
"""

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from kibends.panel.controller import MINUS, Graph

HEIGHT = 150
"""The graph's fixed height, px."""
LEVEL_BAND = 16
"""The band above the plot that holds the level labels, px."""
TICK = 3
"""How far tick marks reach below the plot, px."""
LABEL_GAP = 6
"""Between the plot and the x labels, px."""
MIN_ITERATIONS = 10
"""The x axis reaches at least this far."""
X_INTERVALS = 5
Y_INTERVALS = 4
FLOOR = 0.1
"""The least the y axis spans below 0, mm."""
FLOOR_SHARE = 0.01
"""...or this share of the starting length, if more."""
REPAINT_MS = 50
SECONDARY = 0.65
"""The share of the window's text colour in captions and labels."""
GRID_ALPHA = 28
ZERO_ALPHA = 150
FILL_ALPHA = 38
LINE_WIDTH = 2.0
MARK_RADIUS = 3.0
SMALLER = 2.0
"""How much smaller than the application font small text is, pt."""


def secondary(palette: QPalette) -> QColor:
    """The window's text colour mixed into its background: captions and
    labels."""
    text = palette.color(QPalette.ColorRole.WindowText)
    back = palette.color(QPalette.ColorRole.Window)
    return QColor.fromRgbF(
        text.redF() * SECONDARY + back.redF() * (1 - SECONDARY),
        text.greenF() * SECONDARY + back.greenF() * (1 - SECONDARY),
        text.blueF() * SECONDARY + back.blueF() * (1 - SECONDARY),
    )


def small(font: QFont) -> QFont:
    smaller = QFont(font)
    smaller.setPointSizeF(font.pointSizeF() - SMALLER)
    return smaller


@dataclass(frozen=True)
class Axis:
    low: float
    high: float
    step: float

    @property
    def ticks(self) -> list[float]:
        count = round((self.high - self.low) / self.step)
        # Rounded to the step, so that no tick carries float noise.
        return [
            round((self.low + k * self.step) / self.step) * self.step + 0.0
            for k in range(count + 1)
        ]


def nice_step(span: float, intervals: int) -> float:
    """The smallest of 1, 2 and 5 times a power of ten that covers `span` in at most
    `intervals` steps."""
    raw = span / intervals
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        step = float(f"{m * magnitude:.12g}")
        if step >= raw * (1 - 1e-9):
            return step
    raise AssertionError("unreachable")


def x_axis(last: int) -> Axis:
    """From iteration 0 to the last, at least `MIN_ITERATIONS`, rounded up
    to a step of whole iterations."""
    reach = max(MIN_ITERATIONS, last)
    step = max(1.0, nice_step(reach, X_INTERVALS))
    return Axis(0.0, math.ceil(reach / step) * step, step)


def floor(start: float) -> float:
    """How far below 0 the y axis reaches at least, for a run starting at
    `start` mm."""
    return max(FLOOR, FLOOR_SHARE * start)


def y_axis(low: float, high: float) -> Axis:
    """An axis that holds `low` to `high` in at most `Y_INTERVALS` steps,
    each end rounded out to a step."""
    step = nice_step(high - low, Y_INTERVALS)
    while True:
        bottom = math.floor(low / step + 1e-9) * step
        top = math.ceil(high / step - 1e-9) * step
        if round((top - bottom) / step) <= Y_INTERVALS:
            return Axis(bottom + 0.0, top + 0.0, step)
        step = nice_step(step * 1.0001 * Y_INTERVALS, Y_INTERVALS)


def decimals(step: float) -> int:
    """The decimals a tick label of `step` needs."""
    return max(0, -math.floor(math.log10(step) + 1e-9))


def tick_label(value: float, step: float) -> str:
    text = f"{value:.{decimals(step)}f}"
    if float(text) == 0:
        return text.lstrip("-")
    return text.replace("-", MINUS)


EMPTY = Graph(0.0, (), (), "")


class LengthGraph(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.graph = EMPTY
        self.across = x_axis(0)
        self.up: Axis | None = None
        self.setFixedHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Track length change graph")
        self._repaint = QTimer(self)
        self._repaint.setSingleShot(True)
        self._repaint.setInterval(REPAINT_MS)
        self._repaint.timeout.connect(self.update)

    def sizeHint(self) -> QSize:
        return QSize(300, HEIGHT)

    def set_graph(self, graph: Graph) -> None:
        """Show `graph`; the axes grow with it, and start over when its
        points do."""
        if graph == self.graph:
            return
        old = self.graph
        self.graph = graph
        points = graph.points
        if not points:
            self.up = None
            self.across = x_axis(0)
        else:
            changes = [change for _, change in points]
            low = min(-floor(graph.start), *changes)
            high = max(0.0, *changes)
            ongoing = (
                self.up is not None
                and old.start == graph.start
                and old.points == points[: len(old.points)]
            )
            if ongoing and self.up is not None:
                low, high = min(low, self.up.low), max(high, self.up.high)
            self.up = y_axis(low, high)
            self.across = x_axis(points[-1][0])
        if not self._repaint.isActive():
            self._repaint.start()

    def plot_rect(self) -> QRectF:
        """Where the plot lies in the widget."""
        metrics = QFontMetricsF(small(self.font()))
        y = self.up
        labels = [tick_label(v, y.step) for v in y.ticks] if y is not None else []
        left = max((metrics.horizontalAdvance(t) for t in labels), default=0.0) + 8
        right = metrics.horizontalAdvance("000") / 2 + 4
        bottom = LABEL_GAP + metrics.height()
        return QRectF(
            left,
            LEVEL_BAND,
            self.width() - left - right,
            self.height() - LEVEL_BAND - bottom,
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        font = small(self.font())
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        muted = secondary(palette)
        plot = self.plot_rect()
        painter.fillRect(plot, palette.color(QPalette.ColorRole.Base))
        y = self.up
        points = self.graph.points
        if y is None or not points:
            self._frame(painter, plot)
            painter.setPen(muted)
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self.graph.placeholder)
            return
        x = self.across

        def at(iteration: float, change: float) -> QPointF:
            return QPointF(
                plot.left() + plot.width() * (iteration - x.low) / (x.high - x.low),
                plot.bottom() - plot.height() * (change - y.low) / (y.high - y.low),
            )

        grid = QColor(palette.color(QPalette.ColorRole.WindowText))
        grid.setAlpha(GRID_ALPHA)
        zero = QColor(muted)
        zero.setAlpha(ZERO_ALPHA)
        for value in y.ticks:
            left, right = at(x.low, value), at(x.high, value)
            if value == 0:
                painter.setPen(QPen(zero, 1, Qt.PenStyle.DotLine))
            else:
                painter.setPen(QPen(grid, 1))
            painter.drawLine(left, right)
            painter.setPen(muted)
            painter.drawText(
                QRectF(
                    0,
                    left.y() - metrics.height() / 2,
                    plot.left() - 5,
                    metrics.height(),
                ),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                tick_label(value, y.step),
            )
        for value in x.ticks:
            base = at(value, y.low)
            painter.setPen(QPen(muted, 1))
            painter.drawLine(base, base + QPointF(0, TICK))
            painter.drawText(
                QRectF(base.x() - 20, plot.bottom() + LABEL_GAP, 40, metrics.height()),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                f"{value:.0f}",
            )
        dashed = QPen(muted, 1, Qt.PenStyle.DashLine)
        for iteration, spacing in self.graph.levels:
            line = at(iteration, y.high)
            if iteration > x.low:
                painter.setPen(dashed)
                painter.drawLine(line, at(iteration, y.low))
            painter.setPen(muted)
            painter.drawText(
                QPointF(line.x() + 3, plot.top() - metrics.descent() - 2),
                f"{spacing:g} mm",
            )

        accent = palette.color(QPalette.ColorRole.Accent)
        path = QPainterPath(at(*points[0]))
        for point in points[1:]:
            path.lineTo(at(*point))
        fill = QPainterPath(path)
        fill.lineTo(at(points[-1][0], 0.0))
        fill.lineTo(at(points[0][0], 0.0))
        shade = QColor(accent)
        shade.setAlpha(FILL_ALPHA)
        painter.fillPath(fill, shade)
        line = QPen(accent, LINE_WIDTH)
        line.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        line.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(line)
        painter.drawPath(path)
        self._frame(painter, plot)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(at(*points[-1]), MARK_RADIUS, MARK_RADIUS)
        painter.setBrush(palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(accent, 1.5))
        painter.drawEllipse(at(*points[0]), MARK_RADIUS, MARK_RADIUS)

    def _frame(self, painter: QPainter, plot: QRectF) -> None:
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(plot)

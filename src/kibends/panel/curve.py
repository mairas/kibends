"""The panel's energy curve: the run's total energy after each outer
iteration as one polyline, scaled to the run's range, without axes or
numbers, in the palette's colours. A dashed vertical line in the frame's
colour parts the values of each level of the spacing schedule from the
next; the curve goes on across it, so the jump the finer spacing makes in
the energy shows. Drawn with QPainter, since Qt's chart modules are GPLv3
or commercial and ship outside PySide6-Essentials."""

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

PEN_WIDTH = 2.0
MARGIN = 2
"""Pixels kept clear inside the frame, so the pen is never clipped."""


class EnergyCurve(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.values: tuple[float, ...] = ()
        self.separators: tuple[int, ...] = ()
        """Each the index of the first value at a level after the first."""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(300, 80)

    def minimumSizeHint(self) -> QSize:
        return QSize(100, 40)

    def set_values(
        self, values: Sequence[float], separators: Sequence[int] = ()
    ) -> None:
        values, separators = tuple(values), tuple(separators)
        if (values, separators) != (self.values, self.separators):
            self.values, self.separators = values, separators
            self.update()

    @staticmethod
    def separator_xs(
        count: int, separators: Sequence[int], width: float
    ) -> list[float]:
        """Where each separator lies across a `width` that `count` values
        span as `scaled` lays them out: halfway between the value before it
        and the first after it."""
        last = max(count - 1, 1)
        return [width * (i - 0.5) / last for i in separators]

    @staticmethod
    def scaled(
        values: Sequence[float], width: float, height: float
    ) -> list[tuple[float, float]]:
        """Each value's point in a `width` by `height` box, first at the
        left, last at the right, the highest at the top and the lowest at
        the bottom; a flat history lies across the middle."""
        low, high = min(values), max(values)
        span = high - low
        last = max(len(values) - 1, 1)
        return [
            (
                width * i / last,
                height / 2 if span == 0 else height * (high - v) / span,
            )
            for i, v in enumerate(values)
        ]

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        palette = self.palette()
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid)))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if not self.values:
            return
        inner = self.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        separator = QPen(palette.color(QPalette.ColorRole.Mid))
        separator.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(separator)
        for x in self.separator_xs(len(self.values), self.separators, inner.width()):
            painter.drawLine(
                QPointF(inner.left() + x, inner.top()),
                QPointF(inner.left() + x, inner.bottom()),
            )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Highlight), PEN_WIDTH))
        points = [
            QPointF(inner.left() + x, inner.top() + y)
            for x, y in self.scaled(self.values, inner.width(), inner.height())
        ]
        if len(points) == 1:
            painter.drawPoint(points[0])
        else:
            painter.drawPolyline(points)

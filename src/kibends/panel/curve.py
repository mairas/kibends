"""The panel's energy curve: the run's total energy after each outer
iteration as one polyline, scaled to the run's range, without axes or
numbers, in the palette's colours. Drawn with QPainter, since Qt's chart
modules are GPLv3 or commercial and ship outside PySide6-Essentials."""

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QSize
from PySide6.QtGui import QPainter, QPaintEvent, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

PEN_WIDTH = 2.0
MARGIN = 2
"""Pixels kept clear inside the frame, so the pen is never clipped."""


class EnergyCurve(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.values: tuple[float, ...] = ()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(300, 80)

    def minimumSizeHint(self) -> QSize:
        return QSize(100, 40)

    def set_values(self, values: Sequence[float]) -> None:
        values = tuple(values)
        if values != self.values:
            self.values = values
            self.update()

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Highlight), PEN_WIDTH))
        inner = self.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        points = [
            QPointF(inner.left() + x, inner.top() + y)
            for x, y in self.scaled(self.values, inner.width(), inner.height())
        ]
        if len(points) == 1:
            painter.drawPoint(points[0])
        else:
            painter.drawPolyline(points)

"""The panel's window on Qt's offscreen platform; skipped without the `gui`
extra."""

import os
import threading
from collections.abc import Callable, Iterator

import pytest

pytest.importorskip("PySide6.QtWidgets")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_panel_controller import LEVELLED, PROGRESS, STOPPED, Scripted
from test_panel_probe import prober, ten_tracks_four_vias

from kibends.panel.controller import State
from kibends.panel.curve import MARGIN, EnergyCurve
from kibends.panel.window import PanelWindow, Poster, start_thread
from kibends.present import Field, PanelFields
from kibends.run import Stage


@pytest.fixture(scope="session")
def app() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def settle() -> None:
    """Deliver what workers posted."""
    QCoreApplication.processEvents()


@pytest.fixture
def window(app: QApplication) -> Iterator[Callable[[Scripted], PanelWindow]]:
    made: list[PanelWindow] = []

    def make(run: Scripted) -> PanelWindow:
        w = PanelWindow(
            prober(ten_tracks_four_vias()), run, spawn=lambda work, daemon: work()
        )
        made.append(w)
        w.show()
        w.controller.tick()
        settle()
        return w

    yield make
    for w in made:
        w.controller.stop()
        settle()
        w.close()
        w.deleteLater()
    settle()


class TestLayout:
    def test_shows_the_board_after_reading_it(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.windowTitle() == "kibends — board.kicad_pcb"
        assert w.scope.text() == "Whole board: 10 tracks, 4 vias"
        assert w.button.text() == "Run"

    def test_advanced_is_collapsed_at_launch(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert not w.advanced.isVisible()
        w.disclosure.click()
        assert w.advanced.isVisible()

    def test_minimum_size_fits_advanced_open(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        w.disclosure.click()

        assert w.minimumSize().height() >= w.minimumSizeHint().height()

    def test_the_summary_fits_at_the_minimum_size(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        w.disclosure.click()
        w.resize(w.minimumSize())
        w.button.click()
        settle()

        status = w.status
        assert status.heightForWidth(status.width()) <= status.height()

    def test_every_field_has_a_label(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert set(w.edits) == set(Field)
        assert all(w.label_of(edit) for edit in w.edits.values())

    def test_stays_on_top(self, window: Callable[[Scripted], PanelWindow]) -> None:
        w = window(Scripted())

        assert w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint

    def test_ground_nets_show_the_detected_nets(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.edits[Field.GROUND_NETS].placeholderText() == "GND (detected)"
        assert w.edits[Field.TIME_LIMIT].placeholderText() == "none"


class TestEditing:
    def test_typing_edits_the_fields(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        QTest.keyClicks(w.edits[Field.TIME_LIMIT], "5")
        w.via_in_pad.click()

        assert w.controller.fields == PanelFields(time_limit="5", allow_via_in_pad=True)

    def test_an_invalid_field_is_marked_and_named(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        edit = w.edits[Field.REPULSION]

        edit.selectAll()
        QTest.keyClicks(edit, "-1")

        assert edit.property("invalid") is True
        assert w.status.text() == "Repulsion: must not be negative"
        assert not w.button.isEnabled()

    def test_fields_are_read_only_while_running(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        read_only: list[bool] = []
        run = Scripted()
        w = window(run)

        def during(_stage: Stage) -> None:
            settle()
            read_only.append(
                all(e.isReadOnly() for e in w.edits.values())
                and not w.via_in_pad.isEnabled()
            )

        run.during = during
        w.button.click()
        settle()

        assert read_only and all(read_only)
        assert not any(e.isReadOnly() for e in w.edits.values())


class TestRunning:
    def test_the_button_runs_and_shows_the_summary(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        w.button.click()
        settle()

        assert w.controller.state is State.DONE
        assert w.status.text().startswith("Settled after")
        assert w.curve.values == tuple(p.energy for p in PROGRESS)
        assert w.curve.separators == ()
        assert w.progress.text().endswith("iteration 3")

    def test_shows_the_level_and_marks_its_changes(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted(progress=LEVELLED))

        w.button.click()
        settle()

        assert w.curve.values == tuple(p.energy for p in LEVELLED)
        assert w.curve.separators == (2, 4)
        assert w.progress.text().endswith("level 3 of 3 · iteration 6")

    def test_return_runs(self, window: Callable[[Scripted], PanelWindow]) -> None:
        run = Scripted()
        w = window(run)

        QTest.keyClick(w.edits[Field.TIME_LIMIT], Qt.Key.Key_Return)
        settle()

        assert len(run.options) == 1

    def test_esc_stops_and_the_button_is_one(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        buttons: list[tuple[str, bool]] = []
        run = Scripted(outcome=STOPPED)
        w = window(run)
        geometry = w.button.geometry()

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                buttons.append((w.button.text(), w.button.isEnabled()))
                QTest.keyClick(w, Qt.Key.Key_Escape)
                settle()
                buttons.append((w.button.text(), w.button.isEnabled()))
                assert w.button.geometry() == geometry

        run.during = during
        QTest.keyClick(w, Qt.Key.Key_Return)
        settle()

        assert buttons == [("Stop", True), ("Stopping…", False)]
        assert run.stop is not None and run.stop()
        assert w.status.text().startswith("Stopped after")

    def test_the_button_stops(self, window: Callable[[Scripted], PanelWindow]) -> None:
        buttons: list[tuple[str, bool]] = []
        run = Scripted(outcome=STOPPED)
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                w.button.click()
                settle()
                buttons.append((w.button.text(), w.button.isEnabled()))

        run.during = during
        w.button.click()
        settle()

        assert buttons == [("Stopping…", False)]
        assert len(run.options) == 1
        assert run.stop is not None and run.stop()
        assert w.status.text().startswith("Stopped after")

    def test_close_mid_run_waits_for_the_commit(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        visible: list[bool] = []
        run = Scripted(outcome=STOPPED)
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                w.close()
                visible.append(w.isVisible())

        run.during = during
        w.button.click()
        settle()

        assert visible == [True]
        assert not w.isVisible()

    def test_a_run_on_a_worker_thread_reports_on_the_gui_thread(
        self, app: QApplication
    ) -> None:
        threads: list[threading.Thread] = []
        seen: list[bool] = []

        class Watched(PanelWindow):
            def refresh(self) -> None:
                seen.append(threading.current_thread() is threading.main_thread())
                super().refresh()

        def spawn(work: Callable[[], None], *, daemon: bool) -> None:
            threads.append(threading.Thread(target=work, daemon=daemon))
            threads[-1].start()

        w = Watched(prober(ten_tracks_four_vias()), Scripted(), spawn=spawn)
        w.controller.tick()
        for t in threads:
            t.join()
        settle()

        w.button.click()
        for t in threads:
            t.join()
        settle()

        assert w.controller.state is State.DONE
        assert seen and all(seen)
        w.deleteLater()


class TestPoster:
    def test_runs_a_callable_posted_from_another_thread_on_its_own(
        self, app: QApplication
    ) -> None:
        poster = Poster()
        ran_on: list[threading.Thread] = []

        worker = threading.Thread(
            target=lambda: poster.post(
                lambda: ran_on.append(threading.current_thread())
            )
        )
        worker.start()
        worker.join()
        settle()

        assert ran_on == [threading.main_thread()]

    def test_drops_what_is_posted_once_closed(self, app: QApplication) -> None:
        poster = Poster()
        ran: list[bool] = []
        poster.close()

        worker = threading.Thread(target=lambda: poster.post(lambda: ran.append(True)))
        worker.start()
        worker.join()
        settle()

        assert ran == []

    def test_the_window_closes_its_poster_as_it_closes(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        ran: list[bool] = []

        w.close()
        w.poster.post(lambda: ran.append(True))
        settle()

        assert ran == []


class TestThreads:
    @pytest.mark.parametrize("daemon", [True, False])
    def test_starts_work_on_a_thread_as_asked(self, daemon: bool) -> None:
        seen: list[bool] = []
        done = threading.Event()

        def work() -> None:
            seen.append(threading.current_thread().daemon)
            done.set()

        start_thread(work, daemon=daemon)

        assert done.wait(timeout=5.0)
        assert seen == [daemon]


class TestCurve:
    def test_paints_the_history(self, app: QApplication) -> None:
        curve = EnergyCurve()
        curve.resize(200, 80)
        blank = curve.grab().toImage()

        curve.set_values((9.0, 4.0, 3.5, 3.4))

        drawn = curve.grab().toImage()
        assert pen_pixels(drawn, curve.palette()) > pen_pixels(blank, curve.palette())

    def test_paints_a_separator_at_each_level_change(self, app: QApplication) -> None:
        curve = EnergyCurve()
        curve.resize(200, 80)
        values = tuple(p.energy for p in LEVELLED)
        curve.set_values(values)
        plain = curve.grab().toImage()

        curve.set_values(values, (2, 4))

        marked = curve.grab().toImage()
        palette = curve.palette()
        assert separator_columns(plain, palette) == []
        inner = curve.width() - 2 * MARGIN
        expected = EnergyCurve.separator_xs(len(values), (2, 4), inner)
        found = separator_columns(marked, palette)
        assert len(found) == len(expected)
        for x, at in zip(found, expected, strict=True):
            assert abs(x - (MARGIN + at)) <= 1
        # The curve continues across them, drawn over them.
        assert pen_pixels(marked, palette) == pen_pixels(plain, palette)

    def test_separators_lie_between_the_values_they_part(
        self, app: QApplication
    ) -> None:
        assert EnergyCurve.separator_xs(6, (2, 4), 100.0) == [30.0, 70.0]

    def test_scales_to_the_runs_range(self, app: QApplication) -> None:
        points = EnergyCurve.scaled((10.0, 5.0, 0.0), 100.0, 50.0)

        assert points == [(0.0, 0.0), (50.0, 25.0), (100.0, 50.0)]

    def test_a_flat_history_lies_in_the_middle(self, app: QApplication) -> None:
        assert EnergyCurve.scaled((3.0, 3.0), 10.0, 20.0) == [
            (0.0, 10.0),
            (10.0, 10.0),
        ]


def separator_columns(image: QImage, palette: QPalette) -> list[int]:
    """The columns inside the frame where the separators' colour fills at
    least a third of the rows, each run of neighbours as its first."""
    colour = QColor(palette.color(QPalette.ColorRole.Mid)).rgb()
    rows = range(MARGIN, image.height() - MARGIN)
    filled = [
        x
        for x in range(MARGIN, image.width() - MARGIN)
        if sum(image.pixel(x, y) == colour for y in rows) * 3 >= len(rows)
    ]
    return [x for x in filled if x - 1 not in filled]


def pen_pixels(image: QImage, palette: QPalette) -> int:
    pen = QColor(palette.color(QPalette.ColorRole.Highlight)).rgb()
    return sum(
        image.pixel(x, y) == pen
        for x in range(image.width())
        for y in range(image.height())
    )

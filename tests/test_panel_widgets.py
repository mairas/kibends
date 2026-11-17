"""The panel's window on Qt's offscreen platform; skipped without the `gui`
extra."""

import os
import threading
from collections.abc import Callable, Iterator

import pytest

pytest.importorskip("PySide6.QtWidgets")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from loguru import logger
from PySide6.QtCore import QCoreApplication, QMimeData, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget
from test_panel_controller import LEVELLED, PROGRESS, STOPPED, Scripted
from test_panel_probe import prober, ten_tracks_four_vias

from kibends.panel.controller import MINUS, Graph, State
from kibends.panel.curve import (
    HEIGHT,
    Axis,
    LengthGraph,
    floor,
    nice_step,
    tick_label,
    x_axis,
    y_axis,
)
from kibends.panel.window import (
    KICAD_PANEL,
    MIN_WIDTH,
    REPORT_HEIGHT,
    WIDTH,
    PanelWindow,
    Poster,
    start_thread,
)
from kibends.present import Field, PanelFields
from kibends.run import Stage
from kibends.snapshot import SelectionError


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
        if w.confirm is not None:
            w.confirm.done(0)
        w.close()
        w.deleteLater()
    settle()


def texts(widget: QWidget) -> list[str]:
    """The text of every visible label in `widget`."""
    return [label.text() for label in widget.findChildren(QLabel) if label.isVisible()]


class TestLayout:
    def test_shows_the_board_after_reading_it(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.windowTitle() == "kibends — board.kicad_pcb"
        assert w.scope.text() == (
            "<b>Whole board</b>: 10 tracks, 4 vias (nothing selected)"
        )
        assert w.progress.text() == "Ready"
        assert w.run_button.isEnabled() and not w.stop_button.isEnabled()

    def test_focus_starts_on_run(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.focusWidget() is w.run_button

    def test_has_its_default_and_least_width(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.width() == WIDTH
        layout = w.layout()
        assert layout is not None
        assert w.minimumSize().width() == MIN_WIDTH
        assert w.minimumSize().height() == layout.minimumSize().height()

    def test_advanced_is_collapsed_at_launch_and_opens_in_height_only(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        width, height = w.width(), w.height()

        assert not w.advanced.isVisible()
        w.disclosure.click()

        assert w.advanced.isVisible()
        assert w.width() == width and w.height() > height
        w.disclosure.click()
        assert w.height() == height

    def test_the_graph_and_the_report_area_hold_their_height(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        heights: list[tuple[int, int]] = []
        run = Scripted()
        w = window(run)

        def during(_stage: Stage) -> None:
            settle()
            heights.append((w.graph.height(), w.report.height()))

        run.during = during
        w.run_button.click()
        settle()

        assert w.graph.height() == HEIGHT
        assert w.report.height() >= REPORT_HEIGHT
        assert len(set(heights)) == 1

    def test_every_input_has_a_label_and_an_accessible_name(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        inputs: list[QWidget] = [w.nets, *w.edits.values()]
        buddies = {label.buddy() for label in w.findChildren(QLabel)}

        assert all(i in buddies for i in inputs)
        assert [i.accessibleName() for i in inputs] == [
            "Ground nets",
            "Time limit",
            "Repulsion",
            "Repulsion range",
            "Attraction",
            "Attraction range",
            "Spacing",
        ]
        assert all(i.toolTip() and i.accessibleDescription() for i in inputs)
        assert w.graph.accessibleName() == "Track length change graph"

    def test_stays_on_top(self, window: Callable[[Scripted], PanelWindow]) -> None:
        w = window(Scripted())

        assert w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint

    def test_run_sits_left_of_stop(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.run_button.x() < w.stop_button.x()
        assert w.restore.nextInFocusChain() is w.run_button
        assert w.run_button.nextInFocusChain() is w.stop_button

    def test_a_light_palette_takes_kicads_panel_colour(
        self, app: QApplication, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        original = app.palette()
        light = QPalette(QColor("#ffffff"))
        dark = QPalette(QColor("#202020"))
        window_role = QPalette.ColorRole.Window
        try:
            for system, expected in (
                (light, KICAD_PANEL),
                (dark, dark.color(window_role)),
                (light, KICAD_PANEL),
            ):
                app.setPalette(system)
                settle()
                assert w.palette().color(window_role) == expected
                assert w.palette().color(QPalette.ColorRole.Base) == system.color(
                    QPalette.ColorRole.Base
                )
        finally:
            app.setPalette(original)
            settle()

    def test_ground_nets_show_the_detected_nets(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        assert w.nets.toPlainText() == "GND"
        assert w.ground_caption.text() == (
            "Detected: GND (no net has zones on two inner layers)"
        )
        assert not w.use_detected.isVisible()
        assert w.edits[Field.TIME_LIMIT].placeholderText() == "none"


class TestEditing:
    def test_typing_edits_the_fields(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        QTest.keyClicks(w.edits[Field.TIME_LIMIT], "5")
        w.via_in_pad.click()

        assert w.controller.fields == PanelFields(time_limit="5", allow_via_in_pad=True)

    def test_each_edit_holds_its_own_field(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        typed = {
            Field.TIME_LIMIT: "5",
            Field.REPULSION: "0.7",
            Field.RANGE: "0.4",
            Field.ATTRACTION: "0.2",
            Field.ATTRACTION_RANGE: "0.3",
            Field.SPACING: "0.5, 0.1",
        }

        for f, text in typed.items():
            w.edits[f].selectAll()
            QTest.keyClicks(w.edits[f], text)

        assert w.controller.fields == PanelFields(
            time_limit="5",
            repulsion="0.7",
            falloff_range="0.4",
            attraction="0.2",
            attraction_range="0.3",
            spacing="0.5, 0.1",
        )
        assert {f: e.text() for f, e in w.edits.items()} == typed

    def test_typing_net_names_makes_them_the_users_list(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        w.nets.selectAll()

        QTest.keyClicks(w.nets, "AGND")

        assert w.controller.fields.ground_nets == "AGND"
        assert w.ground_caption.text() == "Your list, not detection."
        assert w.use_detected.isVisible()
        w.use_detected.click()
        assert w.nets.toPlainText() == "GND"
        assert not w.use_detected.isVisible()

    def test_the_net_list_wraps_up_to_three_lines(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        one = w.nets.height()

        w.nets.set_text(", ".join(f"/Isolated/GND_{i}" for i in range(4)))
        two_or_three = w.nets.height()
        w.nets.set_text(", ".join(f"/Isolated/GND_{i}" for i in range(40)))

        lines = w.nets.fontMetrics().lineSpacing()
        assert one < two_or_three < w.nets.height() == one + 2 * lines

    def test_return_in_the_net_list_runs_instead_of_breaking_the_line(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        run = Scripted()
        w = window(run)
        w.nets.setFocus()

        QTest.keyClick(w.nets, Qt.Key.Key_Return)
        settle()

        assert "\n" not in w.nets.toPlainText()
        assert len(run.options) == 1

    def test_pasted_lines_become_a_list(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        w.nets.selectAll()
        pasted = QMimeData()
        pasted.setText("GND\nAGND\tPGND")

        w.nets.insertFromMimeData(pasted)

        assert w.controller.fields.ground_nets == "GND, AGND, PGND"

    def test_an_invalid_field_is_marked_once_left(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        w.disclosure.click()
        edit = w.edits[Field.REPULSION]
        edit.setFocus()
        edit.selectAll()

        QTest.keyClicks(edit, "-1")
        assert "Repulsion: must not be negative" not in texts(w)
        assert not w.run_button.isEnabled()
        w.edits[Field.RANGE].setFocus()

        assert "Repulsion: must not be negative" in texts(w)
        assert "border" in edit.styleSheet()
        assert edit.accessibleDescription() == "Repulsion: must not be negative"
        assert w.run_button.toolTip() == "Repulsion: must not be negative"
        assert w.progress.text() == "Ready — fix the marked field to run"

    def test_return_with_an_invalid_field_opens_advanced_at_it(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        run = Scripted()
        w = window(run)
        w.controller.edit(PanelFields(spacing="0.4 0.5"))

        assert "Advanced — 1 field needs a fix" in w.disclosure.text()
        QTest.keyClick(w, Qt.Key.Key_Return)

        assert run.options == []
        assert w.advanced.isVisible()
        assert w.focusWidget() is w.edits[Field.SPACING]
        assert "Spacing: must fall strictly" in " ".join(texts(w))

    def test_leaving_the_spacing_lists_it_with_commas(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())
        w.disclosure.click()
        edit = w.edits[Field.SPACING]
        edit.setFocus()
        edit.selectAll()
        QTest.keyClicks(edit, "0.4 0.1")

        w.edits[Field.REPULSION].setFocus()

        assert edit.text() == "0.4, 0.1"

    def test_restore_defaults(self, window: Callable[[Scripted], PanelWindow]) -> None:
        w = window(Scripted())
        w.disclosure.click()
        w.edits[Field.REPULSION].setText("0.9")
        QTest.keyClicks(w.edits[Field.TIME_LIMIT], "5")
        w.edits[Field.REPULSION].textEdited.emit("0.9")

        w.restore.click()

        assert w.edits[Field.REPULSION].text() == "0.5"
        assert w.edits[Field.TIME_LIMIT].text() == "5"

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
                and w.nets.isReadOnly()
                and not w.via_in_pad.isEnabled()
                and not w.restore.isEnabled()
            )

        run.during = during
        w.run_button.click()
        settle()

        assert read_only and all(read_only)
        assert not any(e.isReadOnly() for e in w.edits.values())


class TestRunning:
    def test_run_shows_progress_the_graph_and_the_report(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        seen: list[tuple[str, bool, bool]] = []
        run = Scripted(progress=LEVELLED)
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                seen.append(
                    (
                        w.progress.text(),
                        w.stop_button.isEnabled(),
                        w.run_button.isEnabled(),
                    )
                )

        run.during = during
        w.run_button.click()
        settle()

        assert seen == [
            (
                "<b>Optimizing</b> · level 3 of 3, 0.1 mm · iteration 6",
                True,
                False,
            )
        ]
        assert w.controller.state is State.DONE
        assert w.graph.graph.levels == ((0, 1.6), (3, 0.4), (5, 0.1))
        assert w.graph_title.text() == "Length change before simplifying, mm"
        shown = texts(w.report)
        assert "<b>Settled</b> — committed as one undo step in KiCad" in shown
        assert f"{MINUS}2.50 mm" in shown
        # The run is over: only Run is on.
        assert not w.stop_button.isEnabled() and w.run_button.isEnabled()

    def test_return_runs(self, window: Callable[[Scripted], PanelWindow]) -> None:
        run = Scripted()
        w = window(run)

        QTest.keyClick(w.edits[Field.TIME_LIMIT], Qt.Key.Key_Return)
        settle()

        assert len(run.options) == 1

    def test_a_second_return_while_optimizing_does_not_stop(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        focused: list[QWidget | None] = []
        run = Scripted()
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                focus = w.focusWidget()
                focused.append(focus)
                assert focus is not None
                QTest.keyClick(focus, Qt.Key.Key_Return)
                settle()

        run.during = during
        QTest.keyClick(w.run_button, Qt.Key.Key_Return)
        settle()

        assert focused == [w.stop_button]
        assert run.stop is not None and not run.stop()
        assert len(run.options) == 1
        assert w.controller.state is State.DONE

    def test_return_on_a_button_activates_it(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        run = Scripted()
        w = window(run)
        w.disclosure.setFocus()

        QTest.keyClick(w.disclosure, Qt.Key.Key_Return)

        assert w.advanced.isVisible()
        assert run.options == []

    def test_esc_stops_and_never_closes(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        buttons: list[tuple[str, bool]] = []
        run = Scripted(outcome=STOPPED)
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                buttons.append((w.stop_button.text(), w.stop_button.isEnabled()))
                QTest.keyClick(w, Qt.Key.Key_Escape)
                settle()
                buttons.append((w.stop_button.text(), w.stop_button.isEnabled()))

        run.during = during
        QTest.keyClick(w, Qt.Key.Key_Return)
        settle()
        QTest.keyClick(w, Qt.Key.Key_Escape)

        assert buttons == [("&Stop", True), ("Stopping…", False)]
        assert run.stop is not None and run.stop()
        assert "<b>Stopped</b> — best result so far committed as one undo step" in (
            texts(w.report)
        )
        assert w.isVisible()

    def test_closing_mid_run_asks_and_stop_and_close_closes_after_the_commit(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        visible: list[bool] = []
        run = Scripted(outcome=STOPPED)
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                w.close()
                assert w.confirm is not None
                assert (w.confirm.text(), w.confirm.informativeText()) == (
                    "A run is in progress",
                    "Stop it and commit the best result so far, then close?",
                )
                (stop_close,) = (
                    b for b in w.confirm.buttons() if b.text() == "Stop and close"
                )
                stop_close.click()
                visible.append(w.isVisible())

        run.during = during
        w.run_button.click()
        settle()

        assert visible == [True]
        assert run.stop is not None and run.stop()
        assert not w.isVisible()

    def test_keep_running_keeps_the_run_and_the_window(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        run = Scripted()
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.OPTIMIZING:
                w.close()
                assert w.confirm is not None
                (keep,) = (b for b in w.confirm.buttons() if b.text() == "Keep running")
                keep.click()

        run.during = during
        w.run_button.click()
        settle()

        assert run.stop is not None and not run.stop()
        assert w.isVisible()
        assert w.controller.state is State.DONE

    def test_closing_while_committing_waits_and_says_so(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        lines: list[str] = []
        run = Scripted()
        w = window(run)

        def during(stage: Stage) -> None:
            settle()
            if stage is Stage.COMMITTING:
                w.close()
                lines.append(w.progress.text())

        run.during = during
        w.run_button.click()
        settle()

        assert lines == ["<b>Closing</b> after the commit…"]
        assert w.confirm is None
        assert not w.isVisible()

    def test_shows_the_runs_warnings_on_asking(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        run = Scripted()
        w = window(run)
        run.during = lambda stage: (
            logger.warning("No copper on net GND3") if stage is Stage.READING else None
        )
        w.run_button.click()
        settle()
        assert '1 — <a href="#">show</a>' in texts(w.report)
        assert not w.report.detail.isVisible()

        w.report.toggle_warnings()

        assert w.report.detail.isVisible()
        assert w.report.detail.toPlainText() == "No copper on net GND3"

    def test_a_failure_shows_its_detail_to_copy(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted(outcome=RuntimeError("boom")))

        w.run_button.click()
        settle()
        w.report.copy.click()

        assert w.progress.text() == "<b>Failed</b> while committing"
        assert "<b>Unexpected error</b> — nothing committed" in texts(w.report)
        assert w.report.detail.isVisible()
        clipboard = QGuiApplication.clipboard().text()
        assert clipboard.startswith("RuntimeError: boom")

    def test_a_selection_without_tracks_says_what_to_select(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        error = SelectionError("selection has no tracks or vias")
        w = window(Scripted(outcome=error, stages=()))

        w.run_button.click()
        settle()

        assert w.progress.text() == "<b>Failed</b> while connecting to KiCad"
        assert not w.report.detail.isVisible()

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

        w.run_button.click()
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


def graph_of(progress: tuple[float, ...], start: float = 10.0) -> Graph:
    return Graph(start, tuple(enumerate(progress)), (), "")


class TestScales:
    @pytest.mark.parametrize(
        ("span", "intervals", "step"),
        [(37, 5, 10.0), (87, 5, 20.0), (10, 5, 2.0), (3.08, 4, 1.0), (0.52, 4, 0.2)],
    )
    def test_steps_are_1_2_or_5_times_a_power_of_ten(
        self, span: float, intervals: int, step: float
    ) -> None:
        assert nice_step(span, intervals) == step

    @pytest.mark.parametrize(
        ("last", "axis"),
        [
            (0, Axis(0.0, 10.0, 2.0)),
            (3, Axis(0.0, 10.0, 2.0)),
            (37, Axis(0.0, 40.0, 10.0)),
            (87, Axis(0.0, 100.0, 20.0)),
        ],
    )
    def test_iterations_start_at_0_and_reach_at_least_10(
        self, last: int, axis: Axis
    ) -> None:
        assert x_axis(last) == axis

    def test_the_length_holds_0_and_rounds_out_to_at_most_4_steps(self) -> None:
        axis = y_axis(-3.08, 0.0)

        assert axis == Axis(-4.0, 0.0, 1.0)
        assert axis.ticks == [-4.0, -3.0, -2.0, -1.0, 0.0]

    def test_a_rise_above_the_start_shows(self) -> None:
        axis = y_axis(-0.5, 0.12)

        assert axis.low <= -0.5 and axis.high >= 0.12
        assert len(axis.ticks) <= 5

    @pytest.mark.parametrize(("start", "least"), [(5.0, 0.1), (52.1, 0.521)])
    def test_the_least_span_below_0_is_01_mm_or_1_percent(
        self, start: float, least: float
    ) -> None:
        assert floor(start) == pytest.approx(least)

    def test_a_tiny_change_keeps_the_least_span(self, app: QApplication) -> None:
        graph = LengthGraph()

        graph.set_graph(graph_of((0.0, -0.003, -0.004), start=50.0))

        assert graph.up is not None
        assert graph.up.low <= -0.5 and graph.up.high == 0.0

    @pytest.mark.parametrize(
        ("value", "step", "label"),
        [
            (-1.0, 0.5, f"{MINUS}1.0"),
            (0.0, 0.5, "0.0"),
            (-2.0, 1.0, f"{MINUS}2"),
            (0.15, 0.05, "0.15"),
        ],
    )
    def test_tick_labels_take_the_decimals_the_step_needs(
        self, value: float, step: float, label: str
    ) -> None:
        assert tick_label(value, step) == label

    def test_the_range_only_grows_during_a_run(self, app: QApplication) -> None:
        graph = LengthGraph()
        graph.set_graph(graph_of((0.0, -3.5)))
        wide = graph.up

        graph.set_graph(graph_of((0.0, -3.5, -0.2)))

        assert graph.up == wide

    def test_a_new_run_starts_the_range_again(self, app: QApplication) -> None:
        graph = LengthGraph()
        graph.set_graph(graph_of((0.0, -3.5)))

        graph.set_graph(graph_of((0.0, -0.2), start=10.0))

        assert graph.up == y_axis(-0.2, 0.0)


class TestGraphPaint:
    def test_draws_the_line_in_the_accent_colour(self, app: QApplication) -> None:
        graph = LengthGraph()
        graph.resize(300, HEIGHT)
        blank = graph.grab().toImage()

        graph.set_graph(graph_of((0.0, -1.0, -2.0, -2.5)))

        drawn = graph.grab().toImage()
        assert accent_pixels(drawn, graph) > accent_pixels(blank, graph)

    def test_marks_each_level_start_with_a_dashed_line(self, app: QApplication) -> None:
        graph = LengthGraph()
        graph.resize(300, HEIGHT)
        points = tuple((p.iteration, p.length - 10.0) for p in LEVELLED)
        graph.set_graph(Graph(10.0, points, (), ""))
        plain = graph.grab().toImage()

        graph.set_graph(Graph(10.0, points, ((0, 1.6), (3, 0.4), (5, 0.1)), ""))

        marked = graph.grab().toImage()
        plot = graph.plot_rect()
        across = graph.across
        expected = [
            plot.left() + plot.width() * i / (across.high - across.low) for i in (3, 5)
        ]
        found = changed_columns(plain, marked, graph)
        assert len(found) == len(expected)
        for x, at in zip(found, expected, strict=True):
            assert abs(x - at) <= 1.5

    def test_without_points_says_why_and_draws_no_ticks(
        self, app: QApplication
    ) -> None:
        graph = LengthGraph()
        graph.resize(300, HEIGHT)
        graph.set_graph(Graph(0.0, (), (), "No optimization ran"))

        assert graph.up is None
        assert graph.plot_rect().left() == 8

    def test_the_graph_follows_the_controllers(
        self, window: Callable[[Scripted], PanelWindow]
    ) -> None:
        w = window(Scripted())

        w.run_button.click()
        settle()

        assert w.graph.graph.points == tuple(
            (p.iteration, p.length - PROGRESS[0].length) for p in PROGRESS
        )
        assert w.graph_figure.text() == f"{MINUS}2.50 mm ({MINUS}25.0 %)"


def accent_pixels(image: QImage, graph: LengthGraph) -> int:
    accent = graph.palette().color(graph.palette().ColorRole.Accent).rgb()
    return sum(
        image.pixel(x, y) == accent
        for x in range(image.width())
        for y in range(image.height())
    )


def changed_columns(before: QImage, after: QImage, graph: LengthGraph) -> list[int]:
    """The columns inside the plot where a third of the rows changed, each
    run of neighbours as its first."""
    plot = graph.plot_rect().toRect()
    rows = range(plot.top() + 2, plot.bottom() - 1)
    ratio = before.devicePixelRatio()
    changed = [
        x
        for x in range(plot.left() + 2, plot.right() - 1)
        if sum(
            before.pixel(round(x * ratio), round(y * ratio))
            != after.pixel(round(x * ratio), round(y * ratio))
            for y in rows
        )
        * 3
        >= len(rows)
    ]
    return [x for x in changed if x - 1 not in changed]

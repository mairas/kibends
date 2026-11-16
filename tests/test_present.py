from dataclasses import replace

import pytest
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode

from kibends import cli, present
from kibends.chain import ChainError
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
)
from kibends.coords import UserFrame
from kibends.inside import INSIDE_MESSAGE, InsideClearanceError
from kibends.present import Field, Invalid, PanelFields, SmoothValues
from kibends.results import PolylineStats, Termination
from kibends.run import (
    ABNORMAL_MESSAGE,
    PLOT_EXTRA_MESSAGE,
    ChainReport,
    NoBoardError,
    PlotUnavailableError,
    RunReport,
    Scope,
    ScopeKind,
    ScopeOption,
    ViaReport,
)
from kibends.snapshot import SelectionError

SELECTION = Scope(ScopeKind.SELECTION, 12, 3)


def chain(before: float, after: float, *, changed: bool = True) -> ChainReport:
    return ChainReport(
        "SIG",
        "F.Cu",
        0.2,
        "SIG on F.Cu",
        2,
        PolylineStats(before, 2, float("inf")),
        PolylineStats(after, 5, 1.0),
        1,
        changed,
    )


def via(*, moved: bool) -> ViaReport:
    return ViaReport("v", (0.0, 0.0), (0.5, 0.0) if moved else (0.0, 0.0), 2)


def report(
    chains: tuple[ChainReport, ...] = (),
    vias: tuple[ViaReport, ...] = (),
    termination: Termination = Termination.CONVERGED,
    scope: Scope = SELECTION,
) -> RunReport:
    return RunReport(
        chains,
        vias,
        (),
        termination,
        38,
        False,
        12.44,
        (),
        UserFrame(),
        committed=True,
        refusal=None,
        scope=scope,
    )


class TestSummary:
    def test_sums_the_changed_chains_length_change_and_names_the_scope(
        self,
    ) -> None:
        chains = (chain(10.0, 7.5), chain(4.0, 4.29), chain(3.0, 1.0, changed=False))

        text = present.summary(report(chains, (via(moved=True), via(moved=False))))

        assert text == (
            "Settled after 38 iterations in 12.4 s; 2 chains, 1 via changed; "
            "length -2.21 mm; 7 rejected edge moves; selection of 12 tracks, 3 vias"
        )

    def test_a_longer_result_has_a_plus_sign(self) -> None:
        assert "length +0.50 mm" in present.summary(report((chain(1.0, 1.5),)))

    def test_a_change_rounding_to_zero_has_no_sign(self) -> None:
        assert "length 0.00 mm" in present.summary(report((chain(1.0, 0.999),)))

    def test_names_a_whole_board_run(self) -> None:
        whole = Scope(ScopeKind.BOARD, 1567, 210)

        text = present.summary(report((chain(1.0, 0.5),), scope=whole))

        assert text.endswith("; whole board of 1567 tracks, 210 vias")

    @pytest.mark.parametrize(
        ("termination", "ending"),
        [
            (Termination.CONVERGED, "Settled"),
            (Termination.STOPPED, "Stopped"),
            (Termination.TIME_LIMIT, "Stopped at the time limit"),
            (Termination.ITERATION_CAP, "Stopped at the iteration cap"),
            (Termination.GAVE_UP, "Stopped on abnormal bursts"),
        ],
    )
    def test_begins_with_how_the_run_ended(
        self, termination: Termination, ending: str
    ) -> None:
        text = present.summary(report((chain(1.0, 0.5),), termination=termination))

        assert text.startswith(f"{ending} after 38 iterations")

    def test_without_changes_is_nothing_to_do(self) -> None:
        unchanged = (chain(1.0, 1.0, changed=False),)

        assert present.summary(report(unchanged, (via(moved=False),))) == (
            "Nothing to do"
        )

    def test_a_stop_without_changes_says_nothing_changed(self) -> None:
        stopped = report(termination=Termination.STOPPED)

        assert present.summary(stopped) == "Stopped; nothing changed"

    def test_a_moved_via_alone_is_a_change(self) -> None:
        text = present.summary(report(vias=(via(moved=True),)))

        assert "0 chains, 1 via changed; length 0.00 mm" in text


class TestNormalised:
    @pytest.mark.parametrize(
        ("text", "listed"),
        [
            ("GND  PGND,AGND, ", "GND, PGND, AGND"),
            (" 1.6 0.4,0.1", "1.6, 0.4, 0.1"),
            ("0..5 x", "0..5, x"),
            ("  ", ""),
        ],
    )
    def test_separates_the_items_with_a_comma_and_a_space(
        self, text: str, listed: str
    ) -> None:
        assert present.normalised(text) == listed


class TestFigures:
    def test_count_and_sum_what_the_summary_says(self) -> None:
        chains = (chain(10.0, 7.5), chain(4.0, 4.29), chain(3.0, 1.0, changed=False))

        figures = present.figures(report(chains, (via(moved=True), via(moved=False))))

        assert figures == present.RunFigures(
            ending="Settled",
            iterations=38,
            elapsed=12.44,
            chains=2,
            vias=1,
            length_change=-2.21,
            rejected=7,
            scope=SELECTION,
        )

    def test_a_change_rounding_to_zero_is_zero(self) -> None:
        change = present.figures(report((chain(1.0, 0.999),))).length_change

        assert (change, str(change)) == (0.0, "0.0")


class TestErrorMessage:
    @pytest.mark.parametrize(
        "error",
        [
            SelectionError("The selection holds no tracks or vias"),
            ChainError("no tracks that can be modified"),
            BoardChangedError(("t1",)),
            CommitUncertainError("Lost the connection; it may have landed"),
            CreateItemsError("KiCad created 0 of 2 tracks"),
            DeleteItemsError("KiCad deleted 0 of 2 tracks"),
            InsideClearanceError(()),
            NoBoardError("No board is open in KiCad"),
            PlotUnavailableError(PLOT_EXTRA_MESSAGE),
            present.RunRefusedError(ABNORMAL_MESSAGE),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_an_expected_error_gives_its_own_text(self, error: Exception) -> None:
        assert present.error_message(error) == str(error)

    def test_every_expected_error_is_covered(self) -> None:
        assert len(present.EXPECTED_ERRORS) == 10
        assert present.error_message(InsideClearanceError(())) == INSIDE_MESSAGE

    def test_busy_kicad(self) -> None:
        busy = ApiError("busy", code=ApiStatusCode.AS_BUSY)

        assert present.error_message(busy) == present.BUSY_MESSAGE

    def test_another_refusal_names_kicads_reason(self) -> None:
        assert present.error_message(ApiError("nope")) == (
            "KiCad refused the request: nope"
        )

    def test_unreachable_kicad(self) -> None:
        message = present.error_message(KiCadConnectionError("Failed to connect"))

        assert message.startswith("Cannot reach KiCad (Failed to connect): ")
        assert "not running" in message and "timed out after 30 s" in message

    def test_an_interrupt_commits_nothing(self) -> None:
        assert present.error_message(KeyboardInterrupt()) == (
            "Interrupted: nothing committed"
        )

    def test_an_unexpected_error_names_its_type(self) -> None:
        message = present.error_message(RuntimeError("boom"))

        assert message == "Unexpected error: RuntimeError: boom"


class TestSmoothOptions:
    def test_settled_change_is_half_the_chord_tolerance(self) -> None:
        options = present.smooth_options(SmoothValues(chord_tolerance=0.04))

        assert options.settings.settled_change == 0.02

    def test_a_given_settled_change_wins(self) -> None:
        values = SmoothValues(chord_tolerance=0.04, settled_change=0.003)

        assert present.smooth_options(values).settings.settled_change == 0.003

    def test_the_schedule_ends_at_the_final_spacing(self) -> None:
        options = present.smooth_options(SmoothValues(spacing=(0.4, 0.2, 0.05)))

        assert options.settings.spacing == 0.05
        assert options.settings.coarse_spacings == (0.4, 0.2)

    def test_the_command_lines_defaults_are_the_values_defaults(self) -> None:
        from_cli = cli.smooth_options(cli.parse_args(["smooth"]))

        assert from_cli == present.smooth_options(
            SmoothValues(override_names=cli.CLEARANCE_FLAGS)
        )


def read(**texts: str) -> SmoothValues:
    values = present.read_panel(replace(PanelFields(), **texts))
    assert isinstance(values, SmoothValues)
    return values


def invalid(**texts: str) -> dict[Field, str]:
    result = present.read_panel(replace(PanelFields(), **texts))
    assert isinstance(result, Invalid)
    return dict(result.messages)


class TestReadPanel:
    def test_the_defaults_give_the_command_lines_options_on_the_panels_scope(
        self,
    ) -> None:
        from_cli = cli.smooth_options(cli.parse_args(["smooth"]))

        options = present.smooth_options(read())

        assert options == replace(
            from_cli, scope=ScopeOption.SELECTION_ELSE_BOARD, override_names={}
        )

    def test_reads_every_field(self) -> None:
        fields = PanelFields(
            ground_nets="GND, PGND  AGND",
            time_limit="2.5",
            repulsion="0",
            falloff_range="0.7",
            attraction="0",
            attraction_range="1.5",
            spacing="0.4 0.05",
            allow_via_in_pad=True,
        )

        values = present.read_panel(fields)

        assert values == SmoothValues(
            ground_nets=("GND", "PGND", "AGND"),
            time_limit=2.5,
            repulsion=0.0,
            falloff_range=0.7,
            attraction=0.0,
            attraction_range=1.5,
            spacing=(0.4, 0.05),
            allow_via_in_pad=True,
            scope=ScopeOption.SELECTION_ELSE_BOARD,
        )

    def test_empty_ground_nets_and_time_limit_mean_detection_and_none(self) -> None:
        values = read(ground_nets="  ", time_limit=" ")

        assert values.ground_nets == ()
        assert values.time_limit is None

    def test_a_zero_time_limit_is_refused(self) -> None:
        assert invalid(time_limit="0") == {
            Field.TIME_LIMIT: "Time limit: must be positive"
        }

    @pytest.mark.parametrize(
        ("field", "name"),
        [
            (Field.REPULSION, "repulsion"),
            (Field.ATTRACTION, "attraction"),
            (Field.RANGE, "falloff_range"),
            (Field.ATTRACTION_RANGE, "attraction_range"),
            (Field.TIME_LIMIT, "time_limit"),
        ],
    )
    def test_a_negative_value_is_refused(self, field: Field, name: str) -> None:
        message = invalid(**{name: "-1"})[field]

        assert message.startswith(f"{field.value}: must")

    @pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
    def test_a_value_that_is_not_finite_is_refused(self, text: str) -> None:
        assert invalid(repulsion=text) == {
            Field.REPULSION: "Repulsion: must be a finite number"
        }

    def test_a_value_that_is_not_a_number_is_refused(self) -> None:
        assert invalid(falloff_range="far") == {
            Field.RANGE: "Repulsion range: must be a number, got far"
        }

    @pytest.mark.parametrize(
        ("field", "name"),
        [
            (Field.REPULSION, "repulsion"),
            (Field.RANGE, "falloff_range"),
            (Field.ATTRACTION, "attraction"),
            (Field.ATTRACTION_RANGE, "attraction_range"),
            (Field.SPACING, "spacing"),
        ],
    )
    def test_an_empty_required_field_is_refused(self, field: Field, name: str) -> None:
        assert invalid(**{name: " "}) == {field: f"{field.value}: needs a value"}

    def test_every_invalid_field_has_its_message(self) -> None:
        assert set(invalid(repulsion="-1", time_limit="0", spacing="")) == {
            Field.REPULSION,
            Field.TIME_LIMIT,
            Field.SPACING,
        }

    @pytest.mark.parametrize(
        ("text", "problem"),
        [
            ("0.05 0.1", "must fall strictly, coarsest first, got 0.05, 0.1"),
            ("0.2", "the final spacing, the last value, must be at most 0.1, got 0.2"),
            ("0.4 -0.05", "must be positive"),
            ("0.4 x", "must be a number, got x"),
        ],
    )
    def test_a_bad_spacing_schedule_is_refused_as_on_the_command_line(
        self, text: str, problem: str
    ) -> None:
        assert invalid(spacing=text) == {Field.SPACING: f"Spacing: {problem}"}

    def test_the_default_spacing_is_listed_with_commas(self) -> None:
        assert PanelFields().spacing == "1.6, 0.4, 0.1"

    def test_spacing_values_may_be_separated_by_commas(self) -> None:
        assert read(spacing="0.4, 0.2,0.05").spacing == (0.4, 0.2, 0.05)

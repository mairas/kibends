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
from kibends.inside import INSIDE_MESSAGE, InsideClearanceError
from kibends.present import Field, Invalid, PanelFields, SmoothValues
from kibends.run import (
    ABNORMAL_MESSAGE,
    PLOT_EXTRA_MESSAGE,
    NoBoardError,
    PlotUnavailableError,
    ScopeOption,
)
from kibends.snapshot import SelectionError


class TestErrorMessage:
    @pytest.mark.parametrize(
        "error",
        [
            SelectionError("The selection holds no tracks or vias"),
            ChainError("no tracks that can be modified"),
            BoardChangedError("a track changed in KiCad"),
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

    def test_the_panel_names_the_advanced_fields_for_an_abnormal_end(self) -> None:
        refused = present.RunRefusedError(ABNORMAL_MESSAGE)

        message = present.error_message(refused, panel=True)

        assert message != ABNORMAL_MESSAGE
        assert "--" not in message
        assert "Repulsion" in message and "Range" in message
        assert "under Advanced" in message
        assert present.error_message(refused) == ABNORMAL_MESSAGE

    def test_the_panel_gives_other_errors_the_command_lines_text(self) -> None:
        error = BoardChangedError("a track changed in KiCad")

        assert present.error_message(error, panel=True) == str(error)

    def test_an_unexpected_error_names_its_type(self) -> None:
        message = present.error_message(RuntimeError("boom"), panel=True)

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
            Field.RANGE: "Range: must be a number, got far"
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
            ("0.05 0.1", "must fall strictly, coarsest first, got 0.05 0.1"),
            ("0.2", "the final spacing, the last value, must be at most 0.1, got 0.2"),
            ("0.4 -0.05", "must be positive"),
            ("0.4 x", "must be a number, got x"),
        ],
    )
    def test_a_bad_spacing_schedule_is_refused_as_on_the_command_line(
        self, text: str, problem: str
    ) -> None:
        assert invalid(spacing=text) == {Field.SPACING: f"Spacing: {problem}"}

    def test_spacing_values_may_be_separated_by_commas(self) -> None:
        assert read(spacing="0.4, 0.2,0.05").spacing == (0.4, 0.2, 0.05)

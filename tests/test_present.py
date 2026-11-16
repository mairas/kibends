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
from kibends.present import SmoothValues
from kibends.run import (
    ABNORMAL_MESSAGE,
    PLOT_EXTRA_MESSAGE,
    NoBoardError,
    PlotUnavailableError,
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

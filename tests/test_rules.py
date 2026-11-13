import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import kipy.board
import pytest
from conftest import FakeBoard

from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.rules import Clearances, read_clearances

NAME = "boat"
OVERRIDES = {
    "edge": "--edge-clearance",
    "hole": "--hole-clearance",
    "hole_to_hole": "--hole-to-hole",
}
"""How a caller names its overrides, as the CLI does."""


def project(tmp_path: Path, rules: Mapping[str, object] | None = None) -> FakeBoard:
    """A board whose project file in `tmp_path` holds `rules` as Board
    Setup's constraints; none written when `rules` is None."""
    if rules is not None:
        settings = {"board": {"design_settings": {"rules": rules}}}
        (tmp_path / f"{NAME}.kicad_pro").write_text(json.dumps(settings))
    return FakeBoard(project_path=str(tmp_path), project_name=NAME)


def clearances_of(
    fake: FakeBoard,
    edge: float | None = None,
    hole: float | None = None,
    hole_to_hole: float | None = None,
    override_names: Mapping[str, str] = OVERRIDES,
) -> Clearances:
    return read_clearances(
        cast(kipy.board.Board, fake),
        edge=edge,
        hole=hole,
        hole_to_hole=hole_to_hole,
        override_names=override_names,
    )


BOTH = {"min_copper_edge_clearance": 0.5, "min_hole_clearance": 0.2}
DEFAULTS = Clearances(EDGE_CLEARANCE, HOLE_CLEARANCE, HOLE_TO_HOLE)


class TestReadClearances:
    def test_come_from_the_project_file(self, tmp_path: Path, log: list[str]) -> None:
        assert clearances_of(project(tmp_path, BOTH)) == Clearances(
            0.5, 0.2, HOLE_TO_HOLE
        )
        (line,) = [m for m in log if m.startswith("INFO") and "clearance" in m]
        assert line.rstrip() == (
            "INFO Edge clearance 0.5 mm from boat.kicad_pro, "
            "hole clearance 0.2 mm from boat.kicad_pro, "
            f"hole to hole {HOLE_TO_HOLE:g} mm by default"
        )

    def test_a_value_given_wins_over_the_project_file(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        assert clearances_of(project(tmp_path, BOTH), edge=0.1) == Clearances(
            0.1, 0.2, HOLE_TO_HOLE
        )
        assert any(
            "Edge clearance 0.1 mm as given, hole clearance 0.2 mm from" in m
            for m in log
        )

    def test_values_given_leave_the_project_file_unread(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(tmp_path)
        (tmp_path / f"{NAME}.kicad_pro").write_text("{not json")

        assert clearances_of(fake, edge=0.1, hole=0.3, hole_to_hole=0.4) == Clearances(
            0.1, 0.3, 0.4
        )
        assert not any(m.startswith("WARNING") for m in log)
        assert fake.reads["get_project"] == 0

    def test_a_project_path_naming_the_file_itself_is_read(
        self, tmp_path: Path
    ) -> None:
        fake = project(tmp_path, BOTH)
        fake.project_path = str(tmp_path / f"{NAME}.kicad_pro")
        fake.project_name = ""

        assert clearances_of(fake) == Clearances(0.5, 0.2, HOLE_TO_HOLE)

    @pytest.mark.parametrize(
        ("path", "name"), [("", NAME), ("dir", "")], ids=["no-path", "no-name"]
    )
    def test_a_board_without_a_project_takes_the_defaults(
        self, path: str, name: str, log: list[str]
    ) -> None:
        fake = FakeBoard(project_path=path, project_name=name)

        assert clearances_of(fake) == DEFAULTS
        assert any(
            m.rstrip() == f"INFO Edge clearance {EDGE_CLEARANCE:g} mm by default, "
            f"hole clearance {HOLE_CLEARANCE:g} mm by default, "
            f"hole to hole {HOLE_TO_HOLE:g} mm by default"
            for m in log
        )

    def test_a_missing_project_file_gives_the_defaults(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        assert clearances_of(project(tmp_path)) == DEFAULTS
        assert any("boat.kicad_pro" in m and m.startswith("DEBUG") for m in log)
        assert not any(m.startswith("WARNING") for m in log)

    def test_a_missing_rule_gives_its_default(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(tmp_path, {"min_hole_clearance": 0.2})

        assert clearances_of(fake) == Clearances(EDGE_CLEARANCE, 0.2, HOLE_TO_HOLE)
        assert any(
            f"Edge clearance {EDGE_CLEARANCE:g} mm by default, hole clearance "
            "0.2 mm from boat.kicad_pro" in m
            for m in log
        )

    @pytest.mark.parametrize(
        "value", ["0.5", -0.1, True, None], ids=["text", "negative", "bool", "null"]
    )
    def test_a_rule_that_is_no_clearance_gives_its_default(
        self, tmp_path: Path, value: object, log: list[str]
    ) -> None:
        fake = project(
            tmp_path, {"min_copper_edge_clearance": value, "min_hole_clearance": 0.2}
        )

        assert clearances_of(fake) == Clearances(EDGE_CLEARANCE, 0.2, HOLE_TO_HOLE)
        assert any(
            m.startswith("WARNING") and "min_copper_edge_clearance" in m for m in log
        )

    @pytest.mark.parametrize(
        "text",
        ["{not json", "[1, 2]", '{"board": 3}', b"\xff\xfe".decode("latin-1")],
        ids=["syntax", "list", "no-settings", "binary"],
    )
    def test_an_unreadable_project_file_gives_the_defaults_and_warns(
        self, tmp_path: Path, text: str, log: list[str]
    ) -> None:
        fake = project(tmp_path)
        (tmp_path / f"{NAME}.kicad_pro").write_text(text, encoding="latin-1")

        assert clearances_of(fake) == DEFAULTS
        warnings = [m for m in log if m.startswith("WARNING")]
        assert len(warnings) == 1 and "boat.kicad_pro" in warnings[0]

    def test_a_directory_named_as_the_project_file_gives_the_defaults_and_warns(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(tmp_path)
        (tmp_path / f"{NAME}.kicad_pro").mkdir()

        assert clearances_of(fake) == DEFAULTS
        warnings = [m for m in log if m.startswith("WARNING")]
        assert len(warnings) == 1 and "boat.kicad_pro" in warnings[0]


class TestBelowTheDefault:
    @pytest.mark.parametrize("edge", [0, 0.025])
    def test_a_project_rule_below_the_default_is_kept_and_warns(
        self, tmp_path: Path, edge: float, log: list[str]
    ) -> None:
        fake = project(
            tmp_path, {"min_copper_edge_clearance": edge, "min_hole_clearance": 0}
        )

        assert clearances_of(fake) == Clearances(edge, 0, HOLE_TO_HOLE)
        warnings = [m for m in log if m.startswith("WARNING")]
        assert len(warnings) == 2
        edge_warning, hole_warning = warnings
        assert "min_copper_edge_clearance" in edge_warning
        assert "--edge-clearance" in edge_warning
        assert f"{EDGE_CLEARANCE:g} mm" in edge_warning
        assert "min_hole_clearance" in hole_warning
        assert "--hole-clearance" in hole_warning
        assert f"{HOLE_CLEARANCE:g} mm" in hole_warning

    def test_a_rule_without_an_override_name_warns_without_one(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(tmp_path, {"min_hole_clearance": 0})

        clearances_of(fake, override_names={"edge": "--edge-clearance"})

        (warning,) = [m for m in log if m.startswith("WARNING")]
        assert warning.rstrip().endswith(
            f"sets min_hole_clearance to 0 mm, below the default {HOLE_CLEARANCE:g} mm"
        )

    def test_a_project_rule_at_the_default_does_not_warn(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(
            tmp_path,
            {
                "min_copper_edge_clearance": EDGE_CLEARANCE,
                "min_hole_clearance": HOLE_CLEARANCE + 0.1,
            },
        )

        clearances_of(fake)

        assert not any(m.startswith("WARNING") for m in log)

    def test_a_value_given_below_the_default_does_not_warn(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        clearances_of(project(tmp_path, BOTH), edge=0, hole=0)

        assert not any(m.startswith("WARNING") for m in log)


class TestHoleToHole:
    def test_comes_from_the_project_file(self, tmp_path: Path, log: list[str]) -> None:
        fake = project(tmp_path, {**BOTH, "min_hole_to_hole": 0.5})

        assert clearances_of(fake) == Clearances(0.5, 0.2, 0.5)
        assert any("hole to hole 0.5 mm from boat.kicad_pro" in m for m in log)

    def test_a_value_given_wins_over_the_project_file(self, tmp_path: Path) -> None:
        fake = project(tmp_path, {**BOTH, "min_hole_to_hole": 0.5})

        assert clearances_of(fake, hole_to_hole=0.3).hole_to_hole == 0.3

    def test_below_the_default_is_kept_and_warns(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(tmp_path, {"min_hole_to_hole": 0})

        assert clearances_of(fake).hole_to_hole == 0
        (warning,) = [m for m in log if m.startswith("WARNING")]
        assert "min_hole_to_hole" in warning and "--hole-to-hole" in warning
        assert f"{HOLE_TO_HOLE:g} mm" in warning

    def test_one_that_is_no_clearance_gives_the_default(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        fake = project(tmp_path, {"min_hole_to_hole": "big"})

        assert clearances_of(fake).hole_to_hole == HOLE_TO_HOLE
        assert any(m.startswith("WARNING") and "min_hole_to_hole" in m for m in log)

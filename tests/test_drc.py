"""Tests for the DRC gate.

The s-expression reader, the fake kicad-cli and the kicad-cli lookup and
project-file tests are adapted from kistitch's tests/test_drc.py (MIT,
Copyright (c) 2025 Hat Labs Oy).
"""

import json
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from kipy.errors import ApiError

from kibends import drc
from kibends.coords import UserFrame

FIXTURES = Path(__file__).parent / "fixtures" / "gate"
GATE_BOARD = FIXTURES / "gate.kicad_pcb"
OLD_CHAIN = (
    "00000000-0000-0000-0000-000000000011",
    "00000000-0000-0000-0000-000000000012",
)
FIRST_SEGMENT = """\t(segment
\t\t(start 5 10)
\t\t(end 15 10)
\t\t(width 0.25)
\t\t(layer "F.Cu")
\t\t(net "SIG")
\t\t(uuid "00000000-0000-0000-0000-000000000011")
\t)
"""
SECOND_SEGMENT = """\t(segment
\t\t(start 15 10)
\t\t(end 25 10)
\t\t(width 0.25)
\t\t(layer "F.Cu")
\t\t(net "SIG")
\t\t(uuid "00000000-0000-0000-0000-000000000012")
\t)
"""

VIAS_BOARD = Path(__file__).parent / "fixtures" / "vias" / "vias.kicad_pcb"
SIG_VIA = "00000000-0000-0000-0000-0000000000e1"
"""The SIG via at (10, 10) joining the F.Cu and B.Cu halves of the route."""
VIA_ROUTE = (
    ("00000000-0000-0000-0000-000000000011",),
    ("00000000-0000-0000-0000-000000000012",),
)

BOARD_WITHOUT_FONTS = """(kicad_pcb
\t(version 20260206)
\t(generator "pcbnew")
)
"""


def gate_text() -> str:
    return GATE_BOARD.read_text(encoding="utf-8")


def vias_text() -> str:
    return VIAS_BOARD.read_text(encoding="utf-8")


# A teardrop where the gate's two segments meet, as KiCad 10 writes one.
TEARDROP = """\t(zone
\t\t(net "SIG")
\t\t(layer "F.Cu")
\t\t(uuid "00000000-0000-0000-0000-0000000000e1")
\t\t(hatch none 0.1)
\t\t(priority 30000)
\t\t(attr
\t\t\t(teardrop
\t\t\t\t(type track_end)
\t\t\t)
\t\t)
\t\t(connect_pads yes
\t\t\t(clearance 0)
\t\t)
\t\t(min_thickness 0.0254)
\t\t(filled_areas_thickness no)
\t\t(fill yes
\t\t\t(thermal_gap 0.5)
\t\t\t(thermal_bridge_width 0.5)
\t\t\t(island_removal_mode 1)
\t\t\t(island_area_min 10)
\t\t)
\t\t(polygon
\t\t\t(pts
\t\t\t\t(xy 14 9.875) (xy 15 9.6) (xy 15 10.4) (xy 14 10.125)
\t\t\t)
\t\t)
\t\t(filled_polygon
\t\t\t(layer "F.Cu")
\t\t\t(pts
\t\t\t\t(xy 14 9.875) (xy 15 9.6) (xy 15 10.4) (xy 14 10.125)
\t\t\t)
\t\t)
\t)
"""


def with_teardrop(text: str) -> str:
    fonts = text.index(drc.FONTS_MARKER) + 1
    return text[:fonts] + TEARDROP + text[fonts:]


def tracks(*points: tuple[float, float], net: str = "SIG") -> drc.NewTracks:
    return drc.NewTracks(points=points, width=0.25, layer="F.Cu", net=net)


# ---- minimal s-expression reader for checking injector output -------------

type Sexpr = str | list[Sexpr]


def tokenize(text: str) -> Iterator[str | tuple[str, str]]:
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "()":
            yield c
            i += 1
        elif c.isspace():
            i += 1
        elif c == '"':
            j = i + 1
            buf: list[str] = []
            while text[j] != '"':
                if text[j] == "\\":
                    j += 1
                buf.append(text[j])
                j += 1
            yield ("str", "".join(buf))
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '()"':
                j += 1
            yield text[i:j]
            i = j


def parse_sexpr(text: str) -> list[Sexpr]:
    stack: list[list[Sexpr]] = [[]]
    for tok in tokenize(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            node = stack.pop()
            stack[-1].append(node)
        else:
            stack[-1].append(tok[1] if isinstance(tok, tuple) else tok)
    assert len(stack) == 1, "unbalanced parentheses"
    (root,) = stack[0]
    assert isinstance(root, list), "the text is not one list"
    return root


def children(node: list[Sexpr], name: str) -> list[list[Sexpr]]:
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def field(node: list[Sexpr], name: str) -> list[Sexpr]:
    (f,) = children(node, name)
    return f[1:]


# ---- removing the old chain --------------------------------------------------


class TestRemoveSegments:
    def test_removes_only_the_segments_byte_for_byte(self) -> None:
        text = gate_text()
        assert FIRST_SEGMENT in text and SECOND_SEGMENT in text

        out = drc.remove_segments(text, OLD_CHAIN)

        assert out == text.replace(FIRST_SEGMENT, "").replace(SECOND_SEGMENT, "")

    def test_removes_a_subset(self) -> None:
        text = gate_text()

        out = drc.remove_segments(text, OLD_CHAIN[1:])

        assert out == text.replace(SECOND_SEGMENT, "")

    def test_removes_a_locked_segment(self) -> None:
        locked = SECOND_SEGMENT.replace('\t\t(uuid "', '\t\t(locked yes)\n\t\t(uuid "')
        text = gate_text().replace(SECOND_SEGMENT, locked)

        assert drc.remove_segments(text, OLD_CHAIN[1:]) == gate_text().replace(
            SECOND_SEGMENT, ""
        )

    def test_missing_uuid_is_an_error_naming_it(self) -> None:
        with pytest.raises(drc.BoardTextError, match="no-such-track"):
            drc.remove_segments(gate_text(), (OLD_CHAIN[0], "no-such-track"))

    def test_uuid_of_another_item_is_an_error(self) -> None:
        pad = "00000000-0000-0000-0000-0000000000b3"

        with pytest.raises(drc.BoardTextError, match=pad):
            drc.remove_segments(gate_text(), (pad,))


class TestRemoveTeardrops:
    def test_removes_only_teardrop_zones_byte_for_byte(self) -> None:
        text = gate_text()
        assert "(zone" in text

        assert drc.remove_teardrops(with_teardrop(text)) == text

    def test_a_zone_without_an_outline_before_a_teardrop_is_kept(self) -> None:
        bare = '\t(zone\n\t\t(net "GND")\n\t\t(layer "B.Cu")\n\t)\n'
        fonts = gate_text().index(drc.FONTS_MARKER) + 1
        text = gate_text()[:fonts] + bare + gate_text()[fonts:]

        assert drc.remove_teardrops(with_teardrop(text)) == text

    def test_a_board_without_teardrops_is_unchanged(self) -> None:
        assert drc.remove_teardrops(gate_text()) == gate_text()


# ---- injecting the new chain -------------------------------------------------


class TestInject:
    @pytest.mark.parametrize("board", [gate_text(), BOARD_WITHOUT_FONTS])
    def test_adds_one_segment_per_edge_with_the_chain_attributes(
        self, board: str
    ) -> None:
        new = drc.NewTracks(
            points=((5.0, 10.0), (14.123456, 6.775), (25.0, 10.0)),
            width=0.2,
            layer="In1.Cu",
            net='Odd "net"',
        )
        before = parse_sexpr(board)

        text, (ids,) = drc.inject(board, [new])

        after = parse_sexpr(text)
        added = children(after, "segment")[len(children(before, "segment")) :]
        assert [(field(s, "start"), field(s, "end")) for s in added] == [
            (["5", "10"], ["14.123456", "6.775"]),
            (["14.123456", "6.775"], ["25", "10"]),
        ]
        for s in added:
            assert field(s, "width") == ["0.2"]
            assert field(s, "layer") == ["In1.Cu"]
            assert field(s, "net") == ['Odd "net"']
        assert [field(s, "uuid")[0] for s in added] == list(ids)
        assert len(set(ids)) == 2 and all(len(i) == 36 for i in ids)
        others = [c[0] for c in after[1:] if c[0] != "segment"]
        assert others == [c[0] for c in before[1:] if c[0] != "segment"]

    def test_places_segments_before_embedded_fonts(self) -> None:
        text, _ = drc.inject(gate_text(), [tracks((0, 0), (1, 1))])

        assert text.rindex("(segment") < text.rindex("(embedded_fonts")


# ---- moving vias ---------------------------------------------------------------


class TestMoveVias:
    def test_moves_the_via_under_a_new_uuid_leaving_the_rest(self) -> None:
        board = vias_text()

        text, (new_id,) = drc.move_vias(
            board, [drc.MovedVia(SIG_VIA, (10.5, 11.123456))]
        )

        (moved, fixed) = children(parse_sexpr(text), "via")
        assert field(moved, "at") == ["10.5", "11.123456"]
        assert field(moved, "uuid") == [new_id]
        assert field(fixed, "at") == ["10", "9.4"]
        assert SIG_VIA not in text and len(new_id) == 36
        restored = text.replace(new_id, SIG_VIA).replace(
            "(at 10.5 11.123456)", "(at 10 10)"
        )
        assert restored == board

    def test_missing_via_is_an_error_naming_it(self) -> None:
        with pytest.raises(drc.BoardTextError, match="gone"):
            drc.move_vias(vias_text(), [drc.MovedVia("gone", (0, 0))])

    def test_uuid_of_a_track_is_no_via(self) -> None:
        (track,) = VIA_ROUTE[0]

        with pytest.raises(drc.BoardTextError, match=track):
            drc.move_vias(vias_text(), [drc.MovedVia(track, (0, 0))])


# ---- report parsing ----------------------------------------------------------


def item(desc: str, x: float, y: float, uuid: str = "u") -> drc.ReportItem:
    return {"description": desc, "pos": {"x": x, "y": y}, "uuid": uuid}


def violation(
    vtype: str, *items: drc.ReportItem, excluded: bool = False
) -> drc.ReportViolation:
    v: drc.ReportViolation = {
        "description": f"{vtype} issue",
        "items": list(items),
        "severity": "error",
        "type": vtype,
    }
    if excluded:
        v["excluded"] = True
    return v


def parsed(
    *violations: drc.ReportViolation,
    unconnected: tuple[drc.ReportViolation, ...] = (),
) -> list[drc.Violation]:
    return drc.parse_report(
        {"violations": list(violations), "unconnected_items": list(unconnected)}
    )


class TestParseReport:
    def test_skips_excluded_and_reads_unconnected_items(self) -> None:
        out = parsed(
            violation("clearance", item("Via [GND]", 1, 1), excluded=True),
            unconnected=(violation("unconnected_items", item("Pad 1 [GND]", 3, 3)),),
        )

        assert [v.type for v in out] == ["unconnected_items"]
        assert out[0].items == (drc.ViolationItem("Pad 1 [GND]", (3, 3), "u"),)

    def test_reads_a_kicad_report(self) -> None:
        report: drc.Report = {
            "violations": [
                {
                    "description": "Clearance violation",
                    "items": [
                        {
                            "description": "Pad 3 [GND] of J1 on F.Cu",
                            "pos": {"x": 15.0, "y": 6.0},
                            "uuid": "pad",
                        }
                    ],
                    "severity": "error",
                    "type": "clearance",
                }
            ]
        }

        (v,) = drc.parse_report(report)

        # kicad-cli reports positions in the board frame, whatever the origins.
        assert v.describe(UserFrame((10.0, 2.0))) == (
            "clearance: Clearance violation "
            "[Pad 3 [GND] of J1 on F.Cu @ (5.000, 4.000)]"
        )


# ---- introduced violations ---------------------------------------------------

PAD = item("Pad 3 [GND] of J1 on F.Cu", 15, 6, "pad")
OTHER_PAD = item("Pad 4 [GND] of J1 on F.Cu", 9, 16, "pad4")


def old_track(length: float = 10.0, x: float = 5.0) -> drc.ReportItem:
    return item(f"Track [SIG] on F.Cu, length {length:.4f} mm", x, 10, "old")


def new_track(length: float = 9.1, x: float = 14.0) -> drc.ReportItem:
    return item(f"Track [SIG] on F.Cu, length {length:.4f} mm", x, 6.775, "new")


def introduced(
    baseline: list[drc.Violation], candidate: list[drc.Violation]
) -> drc.Introduced:
    return drc.introduced(
        baseline, candidate, old_ids=[frozenset({"old"})], new_ids=[frozenset({"new"})]
    )


class TestIntroduced:
    def test_one_chains_old_violation_does_not_excuse_anothers(self) -> None:
        def track(net: str, uuid: str, x: float) -> drc.ReportItem:
            return item(f"Track [{net}] on F.Cu, length {x:.4f} mm", x, 1, uuid)

        base = parsed(violation("clearance", track("A", "oldA", 1), PAD))
        candidate = parsed(
            violation("clearance", track("B", "newB", 3), PAD),
            violation("clearance", track("A", "newA", 2), PAD),
        )

        result = drc.introduced(
            base,
            candidate,
            old_ids=[frozenset({"oldA"}), frozenset({"oldB"})],
            new_ids=[frozenset({"newA"}), frozenset({"newB"})],
        )

        assert [drc.violation_nets(v) for v in result.violations] == [{"B", "GND"}]

    @pytest.mark.parametrize("new", ["newA", "newB"])
    def test_chains_sharing_a_split_track_count_as_one(self, new: str) -> None:
        def track(uuid: str, x: float) -> drc.ReportItem:
            return item(f"Track [SIG] on F.Cu, length {x:.4f} mm", x, 1, uuid)

        base = parsed(violation("clearance", track("split", 1), PAD))
        candidate = parsed(violation("clearance", track(new, 3), PAD))

        result = drc.introduced(
            base,
            candidate,
            old_ids=[frozenset({"split"}), frozenset({"split", "stub"})],
            new_ids=[frozenset({"newA"}), frozenset({"newB"})],
        )

        assert not result

    def test_identical_reports_introduce_nothing(self) -> None:
        report = parsed(
            violation("clearance", item("Via [GND]", 1, 1), item("Track [X]", 2, 1)),
            violation("lib_footprint_issues", item("Footprint J1", 0, 0)),
            unconnected=(violation("unconnected_items", PAD),),
        )

        result = introduced(report, report)

        assert result == drc.Introduced(violations=(), split_nets=frozenset())
        assert not result

    def test_an_added_clearance_violation_is_introduced(self) -> None:
        base = parsed(violation("clearance", item("Via [GND]", 1, 1)))
        added = violation("clearance", new_track(), PAD)
        candidate = parsed(violation("clearance", item("Via [GND]", 1, 1)), added)

        result = introduced(base, candidate)

        assert result.violations == tuple(parsed(added))
        assert result

    @pytest.mark.parametrize(
        "vtype",
        [
            "skew_out_of_range",
            "length_out_of_range",
            "too_many_vias",
            "diff_pair_uncoupled_length_too_long",
        ],
    )
    def test_net_wide_violation_on_another_anchor_is_not_introduced(
        self, vtype: str
    ) -> None:
        base = parsed(violation(vtype, item("Pad 1 [/USB_D+] of TP5", 153.2, 73.9)))
        candidate = parsed(
            violation(vtype, item("Pad 14 [/USB_D+] of U2 on F.Cu", 153.8, 87.0))
        )

        assert not introduced(base, candidate)

    def test_net_wide_violation_with_new_values_is_introduced(self) -> None:
        base = violation("skew_out_of_range", item("Pad 1 [/USB_D+] of TP5", 1, 1))
        changed: drc.ReportViolation = {
            **base,
            "description": "Skew out of range (actual -3.9 mm)",
        }

        result = introduced(parsed(base), parsed(changed))

        assert result.violations == tuple(parsed(changed))

    def test_violations_are_compared_as_multisets(self) -> None:
        v = violation("copper_sliver", item("Zone [GND]", 1, 1))

        assert len(introduced(parsed(v), parsed(v, v)).violations) == 1

    def test_positions_below_a_micron_and_item_order_do_not_matter(self) -> None:
        a = violation("clearance", item("Via [GND]", 1.0000001, 1), PAD)
        b = violation("clearance", PAD, item("Via [GND]", 1.0, 1))

        assert not introduced(parsed(a), parsed(b))

    def test_chain_violation_already_in_the_baseline_is_not_introduced(self) -> None:
        base = parsed(violation("clearance", old_track(), PAD))
        candidate = parsed(violation("clearance", new_track(), PAD))

        assert not introduced(base, candidate)

    def test_chain_violation_against_another_item_is_introduced(self) -> None:
        base = parsed(violation("clearance", old_track(), PAD))
        candidate = parsed(violation("clearance", new_track(), OTHER_PAD))

        assert len(introduced(base, candidate).violations) == 1

    def test_chain_violation_of_another_type_is_introduced(self) -> None:
        base = parsed(violation("clearance", old_track(), PAD))
        candidate = parsed(violation("shorting_items", new_track(), PAD))

        assert len(introduced(base, candidate).violations) == 1

    def test_unchanged_track_with_the_same_description_is_not_the_chain(
        self,
    ) -> None:
        # An item outside the chain is compared by description and position.
        base = parsed(violation("clearance", old_track(), PAD))
        candidate = parsed(
            violation("clearance", item(old_track()["description"], 5, 10), PAD)
        )

        assert len(introduced(base, candidate).violations) == 1

    def test_one_baseline_chain_violation_excuses_one_candidate_violation(
        self,
    ) -> None:
        base = parsed(violation("clearance", old_track(), PAD))
        candidate = parsed(
            violation("clearance", new_track(9.1, 14), PAD),
            violation("clearance", new_track(2.0, 16), PAD),
        )

        assert len(introduced(base, candidate).violations) == 1

    def test_moved_connectivity_anchors_with_the_same_count_are_not_introduced(
        self,
    ) -> None:
        base = parsed(
            unconnected=(
                violation("unconnected_items", OTHER_PAD, item("Zone [GND]", 8, 15)),
            )
        )
        candidate = parsed(
            unconnected=(
                violation("unconnected_items", OTHER_PAD, item("Zone [GND]", 12, 17)),
            )
        )

        assert not introduced(base, candidate)

    def test_rising_connectivity_count_is_introduced_naming_the_net(self) -> None:
        moved = violation("unconnected_items", OTHER_PAD, item("Zone [GND]", 12, 17))
        split = violation(
            "unconnected_items",
            item("Pad 5 [GND] of J1 on F.Cu", 21, 16),
            item("Zone [GND]", 16, 17),
        )
        base = parsed(
            unconnected=(
                violation("unconnected_items", OTHER_PAD, item("Zone [GND]", 8, 15)),
            )
        )

        result = introduced(base, parsed(unconnected=(moved, split)))

        assert result.split_nets == {"GND"}
        assert result.violations == tuple(parsed(unconnected=(moved, split)))

    def test_rising_isolated_copper_names_its_net(self) -> None:
        candidate = parsed(
            violation("isolated_copper", item("Zone [GND] on F.Cu", 3, 3))
        )

        assert introduced([], candidate).split_nets == {"GND"}


class TestViolationNets:
    def test_nets_come_from_the_item_tags(self) -> None:
        (v,) = parsed(
            violation(
                "unconnected_items",
                item("Pad 1 [+3V3] of U1", 0, 0),
                item("Zone [GND]", 1, 1),
            )
        )
        (untagged,) = parsed(violation("isolated_copper", item("Zone", 0, 0)))

        assert drc.violation_nets(v) == {"+3V3", "GND"}
        assert drc.violation_nets(untagged) == {drc.UNNAMED_NET}


# ---- kicad-cli lookup --------------------------------------------------------


def make_executable(path: Path, body: str = "") -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class FakeKiCad:
    def __init__(self, path: str = "", error: Exception | None = None) -> None:
        self.path = path
        self.error = error

    def get_kicad_binary_path(self, binary_name: str) -> str:
        if self.error:
            raise self.error
        return self.path


def which_finding(path: Path | None) -> Callable[[str], str | None]:
    """A stand-in for shutil.which that finds `path` whatever is asked."""

    def which(name: str) -> str | None:
        return None if path is None else str(path)

    return which


@pytest.fixture
def no_system_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drc.shutil, "which", which_finding(None))
    monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", ())


class TestLocateKiCadCli:
    def test_explicit_path_wins(self, tmp_path: Path, no_system_cli: None) -> None:
        explicit = make_executable(tmp_path / "explicit")
        other = make_executable(tmp_path / "other")

        assert drc.locate_kicad_cli(str(explicit), FakeKiCad(str(other))) == explicit

    def test_missing_explicit_path_is_an_error(
        self, tmp_path: Path, no_system_cli: None
    ) -> None:
        other = make_executable(tmp_path / "other")

        with pytest.raises(drc.KiCadCliNotFound):
            drc.locate_kicad_cli(str(tmp_path / "nope"), FakeKiCad(str(other)))

    def test_running_kicad_then_path_then_known_installs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from_api = make_executable(tmp_path / "api")
        on_path = make_executable(tmp_path / "onpath")
        known = make_executable(tmp_path / "known")
        monkeypatch.setattr(drc.shutil, "which", which_finding(on_path))
        monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", (str(known),))

        assert drc.locate_kicad_cli(None, FakeKiCad(str(from_api))) == from_api
        assert (
            drc.locate_kicad_cli(None, FakeKiCad(error=ApiError("old KiCad")))
            == on_path
        )
        monkeypatch.setattr(drc.shutil, "which", which_finding(None))
        assert drc.locate_kicad_cli(None, FakeKiCad(str(tmp_path / "gone"))) == known

    def test_not_found_anywhere(self, no_system_cli: None) -> None:
        with pytest.raises(drc.KiCadCliNotFound):
            drc.locate_kicad_cli(None, FakeKiCad())


# ---- project files -----------------------------------------------------------


class FakeProject:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeProjectBoard:
    def __init__(self, directory: str, name: str) -> None:
        self.name = name
        self._dir = directory

    def get_project(self) -> FakeProject:
        return FakeProject(self._dir)


class TestProjectFiles:
    def test_found_next_to_the_saved_board(self, tmp_path: Path) -> None:
        for suffix in (".kicad_pcb", ".kicad_pro", ".kicad_dru"):
            (tmp_path / f"b{suffix}").write_text("")

        files = drc.project_files(FakeProjectBoard(str(tmp_path), "b.kicad_pcb"))

        assert files == drc.ProjectFiles(
            tmp_path / "b.kicad_pro", tmp_path / "b.kicad_dru"
        )

    def test_rules_file_is_optional(self, tmp_path: Path) -> None:
        (tmp_path / "b.kicad_pcb").write_text("")
        (tmp_path / "b.kicad_pro").write_text("{}")

        files = drc.project_files(FakeProjectBoard(str(tmp_path), "b.kicad_pcb"))

        assert files.dru is None

    @pytest.mark.parametrize("directory", ["", "missing"])
    def test_unsaved_board_asks_to_save(self, tmp_path: Path, directory: str) -> None:
        path = str(tmp_path / directory) if directory else ""

        with pytest.raises(drc.ProjectFilesError, match="save it"):
            drc.project_files(FakeProjectBoard(path, "b.kicad_pcb"))

    def test_board_without_project_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "b.kicad_pcb").write_text("")

        with pytest.raises(drc.ProjectFilesError):
            drc.project_files(FakeProjectBoard(str(tmp_path), "b.kicad_pcb"))


# ---- the gate against a fake kicad-cli ------------------------------------------

# Reports a clearance violation on any board that has a segment of the
# candidate's net "NEW", and logs its arguments.
FAKE_CLI = """
import json, pathlib, sys
args = sys.argv[1:]
if args == ["version"]:
    print("10.0.8")
    sys.exit(0)
log = pathlib.Path(sys.argv[0]).with_suffix(".log")
with log.open("a") as f:
    f.write(json.dumps(args) + "\\n")
pcb = pathlib.Path(args[-1])
assert pcb.with_suffix(".kicad_pro").exists(), "project file not copied"
assert pcb.with_suffix(".kicad_dru").exists(), "rules file not copied"
out = pathlib.Path(args[args.index("-o") + 1])
bad = '(net "NEW")' in pcb.read_text(encoding="utf-8")
items = [{"description": "Pad 3 [GND]", "pos": {"x": 15.0, "y": 6.0}, "uuid": "p"}]
violations = [
    {"type": "clearance", "description": "c", "severity": "error", "items": items}
] if bad else []
out.write_text(json.dumps({"violations": violations, "unconnected_items": []}))
"""

# Reports nothing, keeping a copy of each board it checks next to it.
COPYING_FAKE_CLI = """
import json, pathlib, sys
args = sys.argv[1:]
pcb = pathlib.Path(args[-1])
(pathlib.Path(sys.argv[0]).parent / pcb.name).write_text(pcb.read_text())
out = pathlib.Path(args[args.index("-o") + 1])
out.write_text(json.dumps({"violations": [], "unconnected_items": []}))
"""

# Reports a clearance violation on every segment of the net "BAD", naming
# the net in the segment's description as kicad-cli does.
NET_FAKE_CLI = """
import json, pathlib, re, sys
args = sys.argv[1:]
pcb = pathlib.Path(args[-1])
out = pathlib.Path(args[args.index("-o") + 1])
bad = re.findall(r'\\(net "BAD"\\)\\s*\\(uuid "([^"]+)"\\)', pcb.read_text())
violations = [
    {
        "type": "clearance",
        "description": "c",
        "severity": "error",
        "items": [{"description": "Track [BAD]", "pos": {"x": 0, "y": 0}, "uuid": u}],
    }
    for u in bad
]
out.write_text(json.dumps({"violations": violations, "unconnected_items": []}))
"""


# Reports a hole-to-hole violation between every SIG via and the GND via at
# (10, 9.4), as kicad-cli names a via.
VIA_FAKE_CLI = """
import json, pathlib, re, sys
args = sys.argv[1:]
pcb = pathlib.Path(args[-1])
out = pathlib.Path(args[args.index("-o") + 1])
blocks = re.findall(r'\\(via\\s+((?:\\([^()]*\\)\\s*)+)\\)', pcb.read_text())
sig = [b for b in blocks if '(net "SIG")' in b]
vias = [
    (
        re.search(r'\\(uuid "([^"]+)"\\)', b).group(1),
        [float(c) for c in re.search(r"\\(at (\\S+) (\\S+)\\)", b).groups()],
    )
    for b in sig
]
gnd = {"description": "Via [GND]", "pos": {"x": 10, "y": 9.4}, "uuid": "g"}
violations = [
    {
        "type": "hole_to_hole",
        "description": "h",
        "severity": "error",
        "items": [
            {"description": "Via [SIG]", "pos": {"x": x, "y": y}, "uuid": u},
            gnd,
        ],
    }
    for u, (x, y) in vias
]
out.write_text(json.dumps({"violations": violations, "unconnected_items": []}))
"""


def project(tmp_path: Path) -> drc.ProjectFiles:
    directory = tmp_path / "project"
    directory.mkdir()
    for suffix in (".kicad_pro", ".kicad_dru"):
        shutil.copyfile(GATE_BOARD.with_suffix(suffix), directory / f"gate{suffix}")
    return drc.ProjectFiles(directory / "gate.kicad_pro", directory / "gate.kicad_dru")


def cli_calls(cli: Path) -> list[list[str]]:
    log = cli.with_suffix(".log")
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


class TestCheck:
    def test_runs_baseline_and_candidate_and_reports_the_difference(
        self, tmp_path: Path
    ) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)
        files = project(tmp_path)
        work = tmp_path / "work"
        work.mkdir()

        result = drc.check(
            cli,
            files,
            gate_text(),
            [OLD_CHAIN],
            [tracks((5, 10), (25, 10), net="NEW")],
            work,
        )

        assert [v.type for v in result.violations] == ["clearance"]
        calls = cli_calls(cli)
        assert len(calls) == 2
        assert all(c[:3] == ["pcb", "drc", "--refill-zones"] for c in calls)
        assert not list(files.pro.parent.glob("*.kicad_pcb"))

    def test_checks_both_boards_without_teardrops(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", COPYING_FAKE_CLI)

        drc.check(
            cli,
            project(tmp_path),
            with_teardrop(gate_text()),
            [OLD_CHAIN],
            [tracks((5, 10), (25, 10))],
            tmp_path,
        )

        for stem in ("baseline", "candidate"):
            checked = (tmp_path / f"{stem}.kicad_pcb").read_text(encoding="utf-8")
            assert "(teardrop" not in checked
            assert '(net "GND")' in checked

    def test_same_net_candidate_passes(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        result = drc.check(
            cli,
            project(tmp_path),
            gate_text(),
            [OLD_CHAIN],
            [tracks((5, 10), (25, 10))],
            tmp_path,
        )

        assert not result

    def test_missing_uuid_fails_before_any_drc_run(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        with pytest.raises(drc.BoardTextError):
            drc.check(
                cli,
                project(tmp_path),
                gate_text(),
                [("gone",)],
                [tracks((5, 10), (25, 10))],
                tmp_path,
            )

        assert cli_calls(cli) == []

    def test_kicad_cli_failure_is_a_drc_error(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", "import sys; sys.exit(3)")

        with pytest.raises(drc.DrcError, match="3"):
            drc.check(
                cli,
                project(tmp_path),
                gate_text(),
                [OLD_CHAIN],
                [tracks((5, 10), (25, 10))],
                tmp_path,
            )

    def test_timeout_is_a_drc_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(drc, "DRC_TIMEOUT_S", 0.2)
        cli = make_executable(tmp_path / "kicad-cli", "import time; time.sleep(30)")

        with pytest.raises(drc.DrcError, match=r"timed out after 0\.2 s"):
            drc.check(
                cli,
                project(tmp_path),
                gate_text(),
                [OLD_CHAIN],
                [tracks((5, 10), (25, 10))],
                tmp_path,
            )

    def test_logs_the_wall_time_of_both_runs(
        self, tmp_path: Path, log: list[str]
    ) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        drc.check(
            cli,
            project(tmp_path),
            gate_text(),
            [OLD_CHAIN],
            [tracks((5, 10), (25, 10))],
            tmp_path,
        )

        timed = [line for line in log if " s" in line and "DRC" in line]
        assert any("Baseline" in line for line in timed)
        assert any("Candidate" in line for line in timed)

    def test_several_chains_share_one_candidate_run(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        result = drc.check(
            cli,
            project(tmp_path),
            gate_text(),
            [OLD_CHAIN[:1], OLD_CHAIN[1:]],
            [tracks((5, 10), (15, 11)), tracks((15, 11), (25, 10), net="SIG2")],
            tmp_path,
        )

        assert not result
        assert len(cli_calls(cli)) == 2
        candidate = (tmp_path / "candidate.kicad_pcb").read_text(encoding="utf-8")
        assert not any(i in candidate for i in OLD_CHAIN)
        segments = children(parse_sexpr(candidate), "segment")
        added = [(field(s, "net"), field(s, "end")) for s in segments[-2:]]
        assert added == [(["SIG"], ["15", "11"]), (["SIG2"], ["25", "10"])]

    def test_a_violation_on_the_second_chain_names_its_net(
        self, tmp_path: Path
    ) -> None:
        cli = make_executable(tmp_path / "kicad-cli", NET_FAKE_CLI)

        result = drc.check(
            cli,
            project(tmp_path),
            gate_text(),
            [OLD_CHAIN[:1], OLD_CHAIN[1:]],
            [tracks((5, 10), (15, 11)), tracks((15, 11), (25, 10), net="BAD")],
            tmp_path,
        )

        assert result
        assert {n for v in result.violations for n in drc.violation_nets(v)} == {"BAD"}

    def test_moves_vias_in_the_candidate_with_the_tracks(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        drc.check(
            cli,
            project(tmp_path),
            vias_text(),
            VIA_ROUTE,
            via_route_to((10, 10.5)),
            tmp_path,
            [drc.MovedVia(SIG_VIA, (10, 10.5))],
        )

        candidate = (tmp_path / "candidate.kicad_pcb").read_text(encoding="utf-8")
        (moved, _) = children(parse_sexpr(candidate), "via")
        assert field(moved, "at") == ["10", "10.5"]
        assert SIG_VIA not in candidate

    def test_a_violation_the_via_had_before_it_moved_is_not_introduced(
        self, tmp_path: Path
    ) -> None:
        cli = make_executable(tmp_path / "kicad-cli", VIA_FAKE_CLI)

        result = drc.check(
            cli,
            project(tmp_path),
            vias_text(),
            VIA_ROUTE,
            via_route_to((10, 10.1)),
            tmp_path,
            [drc.MovedVia(SIG_VIA, (10, 10.1))],
        )

        assert not result

    def test_missing_via_fails_before_any_drc_run(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        with pytest.raises(drc.BoardTextError, match="gone"):
            drc.check(
                cli,
                project(tmp_path),
                vias_text(),
                VIA_ROUTE,
                via_route_to((10, 10)),
                tmp_path,
                [drc.MovedVia("gone", (10, 10))],
            )

        assert cli_calls(cli) == []

    def test_no_chains_is_an_error(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        with pytest.raises(ValueError, match="No chains"):
            drc.check(cli, project(tmp_path), gate_text(), [], [], tmp_path)

        assert cli_calls(cli) == []


def via_route_to(at: tuple[float, float]) -> list[drc.NewTracks]:
    """The vias fixture's SIG route, F.Cu then B.Cu, through a via at `at`."""
    return [
        drc.NewTracks(points=((2, 10), at), width=0.25, layer="F.Cu", net="SIG"),
        drc.NewTracks(points=(at, (18, 10)), width=0.25, layer="B.Cu", net="SIG"),
    ]


def test_kicad_cli_version(tmp_path: Path) -> None:
    cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

    assert drc.kicad_cli_version(cli) == "10.0.8"


# ---- the gate against the real kicad-cli -----------------------------------------


def installed_kicad_cli() -> Path | None:
    try:
        return drc.locate_kicad_cli(None, None)
    except drc.KiCadCliNotFound:
        return None


KICAD_CLI = installed_kicad_cli()
STRAIGHT = tracks((5, 10), (15, 10), (25, 10))
# The GND pad at (15, 6) is 1 mm square; its lower edge is at y = 6.5. The
# netclass asks for 0.127 mm, the project's pad_clearance rule for 0.17 mm.
TRACK_HALF_WIDTH = 0.125
PAD_EDGE_Y = 6.5


def past_the_pad(gap: float) -> drc.NewTracks:
    """A detour whose copper passes `gap` mm below the GND pad."""
    y = PAD_EDGE_Y + gap + TRACK_HALF_WIDTH
    return tracks((5, 10), (14, y), (16, y), (25, 10))


# Crosses the GND strip between pads 4 and 5 twice, cutting the pour apart.
THROUGH_THE_POUR = tracks((5, 10), (13, 18), (17, 18), (25, 10))


@pytest.mark.slow
@pytest.mark.kicad_cli
@pytest.mark.skipif(KICAD_CLI is None, reason="kicad-cli is not installed")
class TestKiCadCli:
    def gate(
        self,
        tmp_path: Path,
        candidate: drc.NewTracks,
        board_text: str | None = None,
        removed: tuple[str, ...] = OLD_CHAIN,
        rules: bool = True,
    ) -> drc.Introduced:
        assert KICAD_CLI is not None
        files = project(tmp_path)
        if not rules:
            files = drc.ProjectFiles(files.pro, None)
        return drc.check(
            KICAD_CLI,
            files,
            board_text or gate_text(),
            [removed],
            [candidate],
            tmp_path,
        )

    def test_the_same_path_introduces_nothing(self, tmp_path: Path) -> None:
        assert not self.gate(tmp_path, STRAIGHT)

    def test_a_board_with_a_teardrop_loads_and_checks_clean(
        self, tmp_path: Path
    ) -> None:
        board = with_teardrop(gate_text())

        assert not self.gate(tmp_path, STRAIGHT, board_text=board)

    def test_a_path_within_the_custom_pad_rule_is_refused(self, tmp_path: Path) -> None:
        result = self.gate(tmp_path, past_the_pad(0.15))

        assert [v.type for v in result.violations] == ["clearance"]
        (v,) = result.violations
        assert any(i.description.startswith("Pad 3 [GND]") for i in v.items)
        assert result.split_nets == set()

    def test_the_same_path_passes_without_the_custom_rule(self, tmp_path: Path) -> None:
        assert not self.gate(tmp_path, past_the_pad(0.15), rules=False)

    def test_a_path_well_clear_of_the_pad_passes(self, tmp_path: Path) -> None:
        assert not self.gate(tmp_path, past_the_pad(1.0))

    def test_a_path_through_the_pour_names_its_net(self, tmp_path: Path) -> None:
        result = self.gate(tmp_path, THROUGH_THE_POUR)

        assert result.split_nets == {"GND"}

    def test_a_violation_the_old_chain_had_is_not_introduced(
        self, tmp_path: Path
    ) -> None:
        tight, (ids,) = drc.inject(
            drc.remove_segments(gate_text(), OLD_CHAIN), [past_the_pad(0.15)]
        )

        result = self.gate(tmp_path, past_the_pad(0.14), tight, ids)

        assert not result


@pytest.mark.slow
@pytest.mark.kicad_cli
@pytest.mark.skipif(KICAD_CLI is None, reason="kicad-cli is not installed")
class TestKiCadCliVias:
    """The SIG via at (10, 10) on a four-layer board: an OTHER track runs on
    In1.Cu at y = 12, and a GND via at (10, 9.4) is already too close."""

    def gate(self, tmp_path: Path, at: tuple[float, float]) -> drc.Introduced:
        assert KICAD_CLI is not None
        directory = tmp_path / "project"
        directory.mkdir()
        pro = directory / "vias.kicad_pro"
        shutil.copyfile(VIAS_BOARD.with_suffix(".kicad_pro"), pro)
        return drc.check(
            KICAD_CLI,
            drc.ProjectFiles(pro, None),
            vias_text(),
            VIA_ROUTE,
            via_route_to(at),
            tmp_path,
            [drc.MovedVia(SIG_VIA, at)],
        )

    def test_a_via_kept_in_place_introduces_nothing(self, tmp_path: Path) -> None:
        assert not self.gate(tmp_path, (10, 10))

    def test_a_via_moved_too_near_an_inner_layer_track_is_refused(
        self, tmp_path: Path
    ) -> None:
        # Ring 0.3 + clearance 0.127 + half width 0.125 needs 0.552 mm.
        result = self.gate(tmp_path, (10, 11.5))

        assert [v.type for v in result.violations] == ["clearance"]
        (v,) = result.violations
        assert any(
            i.description == "Track [OTHER] on In1.Cu, length 16.0000 mm"
            for i in v.items
        )

    def test_a_clearance_the_via_had_before_it_moved_is_not_introduced(
        self, tmp_path: Path
    ) -> None:
        assert not self.gate(tmp_path, (10, 10.02))

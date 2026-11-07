"""Tests for the DRC gate.

The s-expression reader, the fake kicad-cli and the kicad-cli lookup and
project-file tests are adapted from kistitch's tests/test_drc.py (MIT,
Copyright (c) 2025 Hat Labs Oy).
"""

import json
import shutil
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from kipy.errors import ApiError

from kibends import drc

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

BOARD_WITHOUT_FONTS = """(kicad_pcb
\t(version 20260206)
\t(generator "pcbnew")
)
"""


def gate_text() -> str:
    return GATE_BOARD.read_text(encoding="utf-8")


def tracks(*points: tuple[float, float], net: str = "SIG") -> drc.NewTracks:
    return drc.NewTracks(points=points, width=0.25, layer="F.Cu", net=net)


# ---- minimal s-expression reader for checking injector output -------------


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
            j, buf = i + 1, []
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


def parse_sexpr(text: str) -> list[Any]:
    stack: list[list[Any]] = [[]]
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
    return root  # type: ignore[no-any-return]


def children(node: list[Any], name: str) -> list[list[Any]]:
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def field(node: list[Any], name: str) -> list[Any]:
    (f,) = children(node, name)
    return f[1:]  # type: ignore[no-any-return]


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

        text, ids = drc.inject(board, new)

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
        text, _ = drc.inject(gate_text(), tracks((0, 0), (1, 1)))

        assert text.rindex("(segment") < text.rindex("(embedded_fonts")


# ---- report parsing ----------------------------------------------------------


def item(desc: str, x: float, y: float, uuid: str = "u") -> dict[str, Any]:
    return {"description": desc, "pos": {"x": x, "y": y}, "uuid": uuid}


def violation(
    vtype: str, *items: dict[str, Any], excluded: bool = False
) -> dict[str, Any]:
    v: dict[str, Any] = {
        "description": f"{vtype} issue",
        "items": list(items),
        "severity": "error",
        "type": vtype,
    }
    if excluded:
        v["excluded"] = True
    return v


def parsed(
    *violations: dict[str, Any], unconnected: tuple[dict[str, Any], ...] = ()
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
        report = {
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

        assert str(v) == (
            "clearance: Clearance violation "
            "[Pad 3 [GND] of J1 on F.Cu @ (15.000, 6.000)]"
        )


# ---- introduced violations ---------------------------------------------------

PAD = item("Pad 3 [GND] of J1 on F.Cu", 15, 6, "pad")
OTHER_PAD = item("Pad 4 [GND] of J1 on F.Cu", 9, 16, "pad4")


def old_track(length: float = 10.0, x: float = 5.0) -> dict[str, Any]:
    return item(f"Track [SIG] on F.Cu, length {length:.4f} mm", x, 10, "old")


def new_track(length: float = 9.1, x: float = 14.0) -> dict[str, Any]:
    return item(f"Track [SIG] on F.Cu, length {length:.4f} mm", x, 6.775, "new")


def introduced(
    baseline: list[drc.Violation], candidate: list[drc.Violation]
) -> drc.Introduced:
    return drc.introduced(
        baseline, candidate, old_ids=frozenset({"old"}), new_ids=frozenset({"new"})
    )


class TestIntroduced:
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


@pytest.fixture
def no_system_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drc.shutil, "which", lambda name: None)
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
        monkeypatch.setattr(drc.shutil, "which", lambda name: str(on_path))
        monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", (str(known),))

        assert drc.locate_kicad_cli(None, FakeKiCad(str(from_api))) == from_api
        assert (
            drc.locate_kicad_cli(None, FakeKiCad(error=ApiError("old KiCad")))
            == on_path
        )
        monkeypatch.setattr(drc.shutil, "which", lambda name: None)
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
            OLD_CHAIN,
            tracks((5, 10), (25, 10), net="NEW"),
            work,
        )

        assert [v.type for v in result.violations] == ["clearance"]
        calls = cli_calls(cli)
        assert len(calls) == 2
        assert all(c[:3] == ["pcb", "drc", "--refill-zones"] for c in calls)
        assert not list(files.pro.parent.glob("*.kicad_pcb"))

    def test_same_net_candidate_passes(self, tmp_path: Path) -> None:
        cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)

        result = drc.check(
            cli,
            project(tmp_path),
            gate_text(),
            OLD_CHAIN,
            tracks((5, 10), (25, 10)),
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
                ("gone",),
                tracks((5, 10), (25, 10)),
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
                OLD_CHAIN,
                tracks((5, 10), (25, 10)),
                tmp_path,
            )

    def test_timeout_is_a_drc_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(drc, "DRC_TIMEOUT_S", 0.2)
        cli = make_executable(tmp_path / "kicad-cli", "import time; time.sleep(30)")

        with pytest.raises(drc.DrcError, match="timed out after 0.2 s"):
            drc.check(
                cli,
                project(tmp_path),
                gate_text(),
                OLD_CHAIN,
                tracks((5, 10), (25, 10)),
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
            OLD_CHAIN,
            tracks((5, 10), (25, 10)),
            tmp_path,
        )

        timed = [line for line in log if " s" in line and "DRC" in line]
        assert any("Baseline" in line for line in timed)
        assert any("Candidate" in line for line in timed)


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
            removed,
            candidate,
            tmp_path,
        )

    def test_the_same_path_introduces_nothing(self, tmp_path: Path) -> None:
        assert not self.gate(tmp_path, STRAIGHT)

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
        tight, ids = drc.inject(
            drc.remove_segments(gate_text(), OLD_CHAIN), past_the_pad(0.15)
        )

        result = self.gate(tmp_path, past_the_pad(0.14), tight, ids)

        assert not result

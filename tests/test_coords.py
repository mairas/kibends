from conftest import BOARD_FRAME

from kibends.coords import UserFrame


def test_board_frame_prints_board_coordinates() -> None:
    assert BOARD_FRAME.point((12.5, 3.25)) == "(12.500, 3.250)"


def test_point_is_relative_to_the_origin() -> None:
    assert UserFrame((10.0, 20.0)).point((12.5, 13.25)) == "(2.500, -6.750)"


def test_zero_prints_without_a_sign() -> None:
    assert UserFrame((1.0, 2.0)).point((1.0, 1.9998)) == "(0.000, 0.000)"

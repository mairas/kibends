"""The extension module `kibends._core` is built and current."""

from importlib.metadata import version

from kibends import _core


def test_core_reports_the_package_version() -> None:
    assert _core.version() == version("kibends")

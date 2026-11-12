"""The extension module `kibends._core` imports and reports the version of
the installed `kibends` package."""

from importlib.metadata import version

from kibends import _core


def test_core_reports_the_package_version() -> None:
    assert _core.version() == version("kibends")

"""Build kibends' KiCad Plugin and Content Manager (PCM) repository files.

Writes four files for one stable release, which ``release.yml`` attaches to
it:

- ``kibends-pcm-<VERSION>.zip``, the package: ``metadata.json`` with this
  version, ``plugins/`` and ``resources/icon.png``;
- ``packages.json``, the package with its one version and where to download
  it;
- ``kibends-pcm-resources.zip``, the PCM's resources archive:
  ``<identifier>/icon.png``;
- ``repository.json``, which points at the other two by this release's URLs.

Users add ``https://github.com/mairas/kibends/releases/latest/download/
repository.json`` to the PCM. GitHub redirects that to the newest stable
release's asset, so ``packages.json`` lists only the latest version.

The package metadata lives in ``pcm/metadata.json``; the version entry comes
from ``VERSION`` here. Usage:

    .venv/bin/python scripts/pcm_package.py v<VERSION>+<N> [--out dist/pcm]
"""

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
IDENTIFIER = "io.github.mairas.kibends"
RELEASES = "https://github.com/mairas/kibends/releases/download"
# The stable tag that release-version.yml writes: v<VERSION>+<N>.
STABLE_TAG = re.compile(r"v(?P<version>[0-9][0-9A-Za-z.~-]*)\+[0-9]+")

KICAD_VERSION = "10.0"
PLATFORMS = ["macos", "linux", "windows"]

PACKAGES = "packages.json"
RESOURCES = "kibends-pcm-resources.zip"
REPOSITORY = "repository.json"
REPOSITORY_SCHEMA_VERSION = 2

# Entries get one fixed time, so the archives depend only on their contents.
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SKIPPED = {"__pycache__"}

type Json = dict[str, object]


@dataclass(frozen=True)
class Built:
    package: Path
    packages: Path
    resources: Path
    repository: Path


def check_tag(tag: str, version: str) -> None:
    """Refuse a tag that is not ``version``'s stable release tag."""
    match = STABLE_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"{tag} is not a v<VERSION>+<N> stable release tag")
    if match["version"] != version:
        raise ValueError(f"{tag} is for {match['version']}, but VERSION is {version}")


def _files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not SKIPPED.intersection(path.parts)
    )


def _write_zip(path: Path, entries: Sequence[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


def _dump(data: Json) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource(url: str, path: Path, now: datetime) -> Json:
    return {
        "url": url,
        "sha256": _sha256(path),
        "update_timestamp": int(now.timestamp()),
        "update_time_utc": now.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    }


def build(source: Path, version: str, tag: str, out: Path, now: datetime) -> Built:
    """Write the release's PCM files from the ``pcm`` directory ``source``."""
    check_tag(tag, version)
    release = f"{RELEASES}/{quote(tag, safe='')}"
    metadata = cast(Json, json.loads((source / "metadata.json").read_text()))
    entry: Json = {
        "version": version,
        "status": "stable",
        "kicad_version": KICAD_VERSION,
        "runtime": "ipc",
        "platforms": PLATFORMS,
    }
    out.mkdir(parents=True, exist_ok=True)

    icon = source / "resources" / "icon.png"
    contents = [
        ("metadata.json", _dump({**metadata, "versions": [entry]})),
        *(
            (path.relative_to(source).as_posix(), path.read_bytes())
            for path in _files(source / "plugins")
        ),
        ("resources/icon.png", icon.read_bytes()),
    ]
    package = out / f"kibends-pcm-{version}.zip"
    _write_zip(package, contents)

    download: Json = {
        "download_url": f"{release}/{package.name}",
        "download_sha256": _sha256(package),
        "download_size": package.stat().st_size,
        "install_size": sum(len(data) for _, data in contents),
    }
    packages = out / PACKAGES
    packages.write_bytes(
        _dump({"packages": [{**metadata, "versions": [{**entry, **download}]}]})
    )

    resources = out / RESOURCES
    _write_zip(resources, [(f"{IDENTIFIER}/icon.png", icon.read_bytes())])

    repository = out / REPOSITORY
    author = cast(Json, metadata["author"])
    repository.write_bytes(
        _dump(
            {
                "$schema": "https://go.kicad.org/pcm/schemas/v2#/definitions/Repository",
                "schema_version": REPOSITORY_SCHEMA_VERSION,
                "name": "kibends",
                "maintainer": author,
                "packages": _resource(f"{release}/{PACKAGES}", packages, now),
                "resources": _resource(f"{release}/{RESOURCES}", resources, now),
            }
        )
    )
    return Built(package, packages, resources, repository)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the PCM package and repository files of a release"
    )
    parser.add_argument("tag", help="the stable release tag, v<VERSION>+<N>")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "dist" / "pcm", help="output directory"
    )
    args = parser.parse_args(argv)
    tag = cast(str, args.tag)
    out = cast(Path, args.out)
    version = (ROOT / "VERSION").read_text().strip()
    try:
        built = build(ROOT / "pcm", version, tag, out, datetime.now(UTC))
    except ValueError as error:
        print(f"pcm_package: {error}", file=sys.stderr)
        return 1
    for path in (built.package, built.packages, built.resources, built.repository):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

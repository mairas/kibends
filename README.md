# kibends

Smooths KiCad track corners. Each track is split into short segments whose
vertices are then moved to minimize total track length, keeping clearance to
other copper. The result is plain straight segments, not KiCad native arcs, so
the tracks stay editable.

Uses the KiCad IPC API through [kicad-python](https://pypi.org/project/kicad-python/).

Status: early development, nothing usable yet.

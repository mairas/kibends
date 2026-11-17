# The panel

`kibends-panel` opens a small window for selecting, running, looking at
the result in KiCad and running again. It comes with the `gui` extra (see
[Install](../README.md#install)). The window stays above other windows,
except on Wayland, which does not allow it.

## Scope line

The top line says what Run will touch, and is refreshed about once a
second:

- **Selection**: 12 of 1567 tracks, 3 of 210 vias. Arcs and the members
  of selected groups count as tracks.
- **Whole board**: 1567 tracks, 210 vias (nothing selected).

It also says when KiCad cannot be reached, has no board open, or is busy
with an open tool or dialog. Run is disabled until the panel can run.

## Fields

The fields start at their defaults each time the panel opens and keep
their values while it stays open. Lists take commas or spaces. An invalid
value disables Run and shows a message under its field.

| Field | Default | What it does |
|---|---|---|
| Ground nets | detected | The nets whose vias are return vias for the others ([Return vias](return-vias.md)). |
| Time limit | none | Seconds each spacing level may optimize. At the last level, the run stops and commits the best result so far. |

The caption under Ground nets says how the nets were found. Typing in the
field makes the list yours; Use detected, or emptying the field, goes back
to detection, and so does opening another board. The field is ignored
while Attraction is 0.

Under Advanced:

| Field | Default | What it does |
|---|---|---|
| Repulsion | 0.5 | How strongly tracks and vias keep away from other nets' copper, relative to length. It rounds bends and centres tracks in free space. 0 pulls tracks tight against obstacles at exactly the clearance. |
| range | 1 mm | How far beyond the clearance that push reaches. |
| Attraction | 1.15 | How strongly ground vias and other vias pull each other. 0 turns it off. |
| range | 3 mm | Centre distance at which that pull fades out. |
| Spacing | 1.6, 0.4, 0.1 mm | The distances between track points that the run works at, coarsest first. The coarse levels straighten long tracks quickly, the last adds the detail around copper. The last value is at most 0.1 mm; one value runs at it alone. |
| Allow via in pad | off | Let a moving via settle inside a pad of its own net. KiCad's DRC does not check via-in-pad: use it only where the fabricator fills and caps vias in pads. |

Restore defaults resets the fields under Advanced.

With the defaults, a 90° corner around a pad gets a bend radius of about
1.3 mm. More Repulsion or a longer range widens bends and spreads tracks
further from their neighbours where there is room.

The edge, hole and hole-to-hole clearances come from the project file (see
[Clearances](clearances.md)); the panel has no fields for them.

## Running

Run, or Return, starts a run. The progress line names the stage: reading,
building obstacles, optimizing (with the spacing level and the
iteration), simplifying and committing, with the time since Run.

The graph below plots how much the selected tracks' summed length has
changed since the start, by iteration. A dashed line marks where each
finer spacing level starts. The graph shows the length before the final
simplification, so it differs slightly from the figure in the report.

Stop, or Esc, ends optimizing and commits the best result so far. A stop
before optimizing commits nothing. Once the run simplifies the result, it
can no longer stop.

Closing the window during a run asks whether to stop the run and close
once the commit is done.

## Report

When a run ends, the report says how it ended and what it committed:

- **Length**: the summed length change of the chains it changed.
- **Changed**: how many chains and vias it changed.
- **Optimized**: its iterations and time.
- **Rejected**: the moves it refused because they would have broken a
  clearance.
- **Scope**: the selection or whole board it read.
- **Warnings**: the warnings the run logged, if any, behind a show link.

The headline names the ending:

- **Settled**: the run finished.
- **Stopped**: you pressed Stop.
- **Stopped at the time limit** or **at the iteration cap**: the run
  committed the best result it had.
- **Stopped on abnormal bursts**: the optimizer could not improve the
  result further and committed the best it had, which may fall short of
  the smoothest.
- **Nothing to do**: the tracks are already as short and smooth as these
  settings allow, and nothing was committed.

A run is one undo step in KiCad. A failure says what failed and what to
do, with any long detail in a box with a Copy button; Run tries again.
"KiCad is busy" means a tool or dialog is open in the PCB editor: close
it and run again.

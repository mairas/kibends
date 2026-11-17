# Command line

`kibends smooth` runs the same smoothing as the panel, without a window.
It smooths the tracks and vias selected in KiCad's PCB editor, or the
whole board with `--all`.

```sh
kibends smooth --dry-run --plot chain.png   # look first
kibends smooth                              # then commit
```

`kibends smooth --help` lists the options with their defaults.

## Output

The command prints when it starts optimizing, its progress every few
seconds while a run lasts longer than that, and one line at the end: how
the run ended, its iterations and time, the chains and vias it changed,
their summed length change, the moves it rejected and its scope. The
endings are those of the [panel's report](panel.md#report).

Every position it prints is in millimetres from the board's grid origin,
as described under [Clearances](clearances.md#the-start-check).

## Stopping

The first Ctrl-C while the run reads the board, builds its obstacles or
optimizes stops it. Stopped while optimizing, the run commits the best
result so far; stopped before, it commits nothing. A second Ctrl-C, or one
after optimizing, ends the run with "nothing committed". A Ctrl-C while
KiCad applies the commit may leave it landed, and the run says to check
KiCad's undo history.

## Options

### Scope and output

| Option | Effect |
|---|---|
| `--all` | Smooth every track, arc and via on the board, whatever is selected, shown or filtered in KiCad. Group members stay as drawn. The run adds nothing to KiCad's selection. |
| `--dry-run`, `-n` | Optimize and report without changing the board. |
| `--plot FILE` | Draw each original and smoothed chain over the area it must avoid, one subplot per chain titled with its length before and after; the format follows the extension. Needs the `plot` extra: `uv tool install 'kibends[gui,plot] @ git+https://github.com/mairas/kibends'`. |
| `--verbose`, `-v` | Show progress after every iteration and the debug lines, such as why a selected via stays where it is. Overrides the `LOGURU_LEVEL` environment variable. |

### Shape

These are the panel's fields of the same names.

| Option | Default | Effect |
|---|---|---|
| `--repulsion WEIGHT` | 0.5 | How strongly tracks and vias keep away from other copper, relative to length. 0 pulls tracks tight at exactly the clearance. |
| `--range MM` | 1 | How far beyond the clearance that push reaches. |
| `--attraction WEIGHT` | 1.15 | How strongly ground vias and other vias pull each other ([Return vias](return-vias.md)). 0 turns it off. |
| `--attraction-range MM` | 3 | Centre distance at which that pull fades out. |
| `--ground-nets NET ...` | detected | The ground nets, instead of the detected ones. Ignored with `--attraction 0`. |
| `--allow-via-in-pad` | | Let a moving via settle inside a pad of its own net. KiCad's DRC does not check via-in-pad: use it only where the fabricator fills and caps vias in pads. |

### Clearances

| Option | Default | Effect |
|---|---|---|
| `--edge-clearance MM` | project, else 0.3 | Clearance to the board edge. |
| `--hole-clearance MM` | project, else 0.25 | Clearance between drilled holes and other nets' copper. |
| `--hole-to-hole MM` | project, else 0.25 | Clearance between a moving via's hole and other holes. |

Without these options, the values come from the project file
([Clearances](clearances.md)).

### Tuning

The defaults suit most boards. These options trade run time against how
close the result gets to the optimum.

| Option | Default | Effect, and when to touch it |
|---|---|---|
| `--time-limit SECONDS` | none | How long each spacing level may optimize. When the time runs out at the last level, the run commits its best result so far. Use it to cap a run on a large board. |
| `--chord-tolerance MM` | 0.01 | How far the committed straight tracks may depart from the smooth curve. Raise it for fewer, longer track segments. |
| `--spacing MM ...` | 1.6 0.4 0.1 | The distances between track points that the run works at, coarsest first. The last is at most 0.1; at most 8 values, each smaller than the one before. `--spacing 0.1` skips the coarse levels, which rarely helps. |
| `--max-iterations N` | 200 | The most iterations a run takes before it commits its best result. Raise it if runs end at the iteration cap. |
| `--settled-change MM` | half `--chord-tolerance` | The tracks count as settled when an iteration moves none by this much. Rarely worth changing. |
| `--stall-tolerance MM` | 0.001 | The tracks also count as settled when they shorten by less than this over `--stall-window` iterations. Raising it saves little time. |
| `--stall-window N` | 3 | The iterations the stall check looks back over. |
| `--burst-iterations N` | 50 | Optimizer steps per iteration. Fewer are faster but may settle short of a smooth shape. |
| `--lbfgs-ftol TOL` | 1e-9 | Optimizer stopping threshold on the energy. Larger values shorten runs but can change the result a lot. |
| `--lbfgs-gtol TOL` | 1e-10 | Optimizer stopping threshold on the gradient. Rarely matters. |

A run uses every CPU core. Set the environment variable
`RAYON_NUM_THREADS=N` to use N threads instead; the result is the same.

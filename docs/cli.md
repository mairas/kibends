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

The command logs when it starts optimizing and its progress every few
seconds while a run lasts longer than that. When optimizing ends it
prints one line:

```
Optimizer: converged after 82 iterations in 299.0 s, 3 rejected edge moves
```

The ending is one of `converged`, `nothing to do`, `iteration cap
reached`, `time limit reached`, `stopped` or `stopped on abnormal bursts`
([the panel's report](panel.md#report) explains each). An edge move is a
step of a track segment the run refused because it would have entered a
clearance. Any warnings follow, then a line saying what was committed
(`Committed 'smooth': replaced 1578 tracks with 3445 and moved 318
vias`), or `Dry run: nothing committed`, or `Nothing to do`.

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
| `--plot FILE` | Draw each original and smoothed chain over the area it must avoid, one subplot per chain titled with its length before and after; the format follows the extension. Needs the `plot` extra: `uv tool install --python 3.14 'kibends[gui,plot]'`. |
| `--verbose`, `-v` | Show progress after every iteration and the debug lines, such as why a selected via stays where it is. Overrides the `LOGURU_LEVEL` environment variable. |

### Shape

These set the panel's fields; [The panel](panel.md#fields) describes
each.

| Option | Default | Panel field |
|---|---|---|
| `--ground-nets NET ...` | detected | Ground nets |
| `--repulsion WEIGHT` | 0.5 | Repulsion |
| `--range MM` | 1 | Repulsion's range |
| `--attraction WEIGHT` | 1.15 | Attraction |
| `--attraction-range MM` | 3 | Attraction's range |
| `--spacing MM ...` | 1.6 0.4 0.1 | Spacing |
| `--time-limit SECONDS` | none | Time limit |
| `--allow-via-in-pad` | off | Allow via in pad |

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
| `--chord-tolerance MM` | 0.01 | How far the committed straight tracks may depart from the smooth curve. Raise it for fewer, longer track segments. |
| `--max-iterations N` | 200 | The most iterations a run takes before it commits its best result. Raise it if runs end at the iteration cap. |
| `--settled-change MM` | half `--chord-tolerance` | The tracks count as settled when an iteration moves none by this much. Rarely worth changing. |
| `--stall-tolerance MM` | 0.001 | The tracks also count as settled when they shorten by less than this over `--stall-window` iterations. Raising it saves little time. |
| `--stall-window N` | 3 | The iterations the stall check looks back over. |
| `--burst-iterations N` | 50 | Optimizer steps per iteration. Fewer are faster but may settle short of a smooth shape. |
| `--lbfgs-ftol TOL` | 1e-9 | Optimizer stopping threshold on the energy. Larger values shorten runs but can change the result a lot. |
| `--lbfgs-gtol TOL` | 1e-10 | Optimizer stopping threshold on the gradient. Rarely matters. |

A run uses every CPU core. Set the environment variable
`RAYON_NUM_THREADS=N` to use N threads instead; the result is the same.

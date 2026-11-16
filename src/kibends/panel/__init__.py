"""The `kibends panel` window: `run.run_smooth` on KiCad's selection, or on
the whole board when nothing is selected, with its progress shown and a
stop that keeps the result so far.

`controller` holds the panel's state and decides everything the window
shows; it imports no Qt. Work that talks to KiCad runs on worker threads,
each kind with its own KiCad connection, since kipy's client has no lock:
a run worker runs `run_smooth`, and a probe (`probe.Prober`) reads the
board's identity and selection about once a second between runs, and the
board's totals and ground nets when it sees a board it has not read. The
GUI thread never calls KiCad. Every worker callback reaches the
controller through a "post to the controller's thread" callable: in the
window (`window`) a queued Qt signal (`window.Poster`), in tests a plain
call.

`curve` draws the energy history; `window` lays out the widgets and runs
the Qt application. Only `window` and `curve` import PySide6, which the
`gui` extra installs.
"""

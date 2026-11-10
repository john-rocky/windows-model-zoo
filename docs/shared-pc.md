# Sharing the PC with other jobs

The words used here are explained in [reading-results.md](reading-results.md#words-used-in-the-files).

The PC tools run on a PC that other jobs share. The shared files on the PC are in `%USERPROFILE%\pcup\`: `busy.txt`
(the lock, one holder at a time), `active\<lane>*.txt` (what a job runs, since when) and `yield\<lane>.txt` (the job
waits for the lock). Set `[zoo] lane` in `zoo.toml` to the zoo's own name there.

`tools/pc.sh` reads the shared lock feed on the Mac (`~/code/standup/tools/winpc/busy_now.sh`, no ssh) before every
connection but `wait`, and refuses one (exit 3) during another job's timing or when the feed is older than 3 min; a
Mac without that feed runs it with `ZOO_FEED=none`. Untimed steps and first passes run without the lock: each is
killed when another job's lock appears and runs again from its start once the lock has been free 60 s (a suspended
process would keep its memory). A step over 8 GB (`lock = true`) takes the lock as "(not a timing)", at most 30 min.
Each list and chain keeps its note in `active\` while it runs. In the PC's night (`[zoo] night` in `zoo.toml`,
`"20:45-05:30"` here, when other jobs' timings run), untimed steps and first passes start nothing: a step or run in
flight is killed, its list ends `FAIL <step> (night 20:45-05:30)` or its chain `FAIL the night (20:45-05:30): ...`, the
rest is not run, and the lane's notes in `active\` are removed. `night = "none"` (or no `night`) means no night; timed
sittings are not stopped by it.

A timed sitting (`python3 tools/plan.py timed ...`, `tools/pc.sh push`, then `tools/sitting.sh <name>`) waits on the
Mac until that feed shows the lock free for 60 s, takes it, passes the lane's quiet gate and starts the chain, which
settles 180 s before its first row. When the feed is missing or stale at the start, `tools/sitting.sh` refuses the
sitting (exit 3); on a Mac without the feed, `ZOO_FEED=none tools/sitting.sh <name>` skips the feed, and the take then
waits on the PC, in one ssh connection, until the lock has been free 60 s. Before each run the chain waits for a quiet PC: CPU, NVIDIA memory, free RAM and, for a cell
with `engines_gate` (by default the Intel GPU and RTX cells), the GPU engines' load; each run's `.meta.json` keeps
these reads (`gate`). Control rows bracket the cells, each in its own folder (`runs\<name>\control-<row>\`): an
uncounted warm-up, a control head that stops the sitting when its first run is off the known value by more than 5 %,
the cells, and a control tail. `control.json` holds the verdict, and `tools/results.py` counts the sitting as a timing
only when it is a PASS. Give the known value (`--known`) from several of the control cell's runs on the same PC and
power mode: without it, the chain takes the median of the head's own runs, and stops the sitting when they spread more
than 5 % ((max − min) / median) or are fewer than three. While another job waits, a sitting gives the lock back for
16.5 min (a window) before a run that would end past 30 min of holding it, unless the plan is `--no-windows`; `--slots`
gives it back at fixed times. With `--end-by`, the chain starts no cell that would not end,
with the control tail, before that time; it counts a run at the cell's `est_sec`, else at its
`timeout_min` (no recipe here sets `est_sec`: a cell with the default 20 min and three runs needs over 65 min). The
records come to `raw/<name>/` before the lock goes back. No cell of this repo's `task-*` lines is timed.

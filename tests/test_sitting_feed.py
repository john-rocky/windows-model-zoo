"""tools/sitting.sh and the shared lock feed (Mac, no PC). In a scratch copy with a stand-in tools/pc.sh (it records each
call and answers like the PC: TAKEN, PASS, STARTED, a DONE line, the lane's lock in status, RELEASED) and a stand-in
feed reader (HOME/code/standup/tools/winpc/busy_now.sh, the shared PC's format: exit 0 when the last line is under
3 min old, else 2):
  missing feed            -> exit 3, "REFUSED (lock feed stale or missing)", no call to tools/pc.sh
  stale feed              -> exit 3, the same
  ZOO_FEED=none (no feed) -> the sitting runs: lock take (ZOO_FEED=none passed on), gate, launch, wait, pull, release
  fresh feed, lock free   -> the sitting runs the same calls, with no ZOO_FEED=none line

    python3 tests/test_sitting_feed.py
"""
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

PC_SH = r"""#!/bin/bash
# stand-in tools/pc.sh: one line per call in calls.log, answers like the PC
K="$(cd "$(dirname "$0")/.." && pwd)"
echo "$* | ZOO_FEED=${ZOO_FEED:-} ZOO_FORCE=${ZOO_FORCE:-}" >> "$K/calls.log"
lane=$(python3 -c "import tomllib; print(tomllib.load(open('$K/zoo.toml','rb'))['zoo']['lane'])")
case "$1 ${2:-}" in
  "lock take") echo "TAKEN 2026-11-10T20:00:00+09:00" ;;
  "lock gate") echo "PASS 2026-11-10T20:00:15+09:00" ;;
  "lock status") echo "busy: $lane timing: sitting t done (0 min)" ;;
  "lock release") echo "RELEASED 2026-11-10T20:09:00+09:00" ;;
  "launch "*) echo "STARTED 1234 (chain.ps1 running 20 s after the start)" ;;
  "wait "*) echo "DONE OK 2026-11-10T20:08:00+09:00 (chain.done)" ;;
  "pull "*) mkdir -p "$K/$3/$(basename "$2")" ;;
  "ps"*) cat > /dev/null; echo "no yield note" ;;
  *) echo "stand-in pc.sh: unknown call $*" >&2; exit 2 ;;
esac
"""

BUSY_NOW = r"""#!/bin/bash
# stand-in of the shared PC's feed reader (the form of its output, and its exit codes)
LOG=${BUSY_FEED_LOG:-$HOME/.winpc/busy_feed.log}
[ -f "$LOG" ] || { echo "no feed log at $LOG"; exit 2; }
awk '$3=="BUSY"||$3=="WINDOW"||$3=="ACTIVE"{a[$3]=$0} END{for(k in a) print a[k]}' "$LOG" | sort -k3,3
last=$(tail -1 "$LOG"); ts=${last%% *}
now=$(date +%s); then=$(python3 -c "import datetime,sys; print(int(datetime.datetime.strptime(sys.argv[1], '%Y-%m-%dT%H:%M:%S%z').timestamp()))" "$ts" 2>/dev/null || echo 0)
age=$(( now - then ))
echo "feed age ${age}s (last line: ${last:0:120})"
if [ "$then" -eq 0 ] || [ "$age" -gt 180 ]; then echo "STALE: no fresh feed line"; exit 2; fi
exit 0
"""


def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_case(name, env_more, feed_lines, want_rc, want_calls, want_log, not_log):
    with tempfile.TemporaryDirectory() as td:
        k = pathlib.Path(td) / "zoo"
        (k / "tools").mkdir(parents=True)
        shutil.copy2(ROOT / "tools" / "sitting.sh", k / "tools" / "sitting.sh")
        shutil.copy2(ROOT / "zoo.toml", k / "zoo.toml")
        (k / "tools" / "pc.sh").write_text(PC_SH)
        os.chmod(k / "tools" / "pc.sh", 0o755)
        (k / "pc" / "plans").mkdir(parents=True)
        unit = {"cell": "m/c", "timeout_min": 1, "runs": 1}
        (k / "pc" / "plans" / "t.json").write_text(json.dumps({"sitting": "t", "mode": "timed", "units": [unit]}))
        home = pathlib.Path(td) / "home"
        (home / "code/standup/tools/winpc").mkdir(parents=True)
        (home / "code/standup/tools/winpc/busy_now.sh").write_text(BUSY_NOW)
        feed = pathlib.Path(td) / "busy_feed.log"
        if feed_lines is not None:
            feed.write_text("".join(x + "\n" for x in feed_lines))
        env = dict(os.environ, HOME=str(home), BUSY_FEED_LOG=str(feed))
        env.pop("ZOO_FEED", None)
        env.pop("ZOO_FORCE", None)
        env.update(env_more)
        r = subprocess.run(["bash", str(k / "tools" / "sitting.sh"), "t"], env=env, capture_output=True, text=True,
                           timeout=120)
        log = (k / "raw" / "t.sitting.log").read_text() if (k / "raw" / "t.sitting.log").exists() else ""
        calls = (k / "calls.log").read_text().splitlines() if (k / "calls.log").exists() else []
    verbs = [" ".join(c.split(" | ")[0].split()[:2 if c.startswith("lock ") else 1]) for c in calls]
    bad = []
    if r.returncode != want_rc:
        bad.append(f"exit {r.returncode}, want {want_rc}")
    if verbs != want_calls:
        bad.append(f"calls {verbs}, want {want_calls}")
    for w in want_log:
        if w not in log:
            bad.append(f"log has no {w!r}")
    for w in not_log:
        if w in log:
            bad.append(f"log has {w!r}")
    if env_more.get("ZOO_FEED") == "none" and calls and not all("ZOO_FEED=none" in c for c in calls):
        bad.append("ZOO_FEED=none not passed on to every tools/pc.sh call")
    print(f"{'ok  ' if not bad else 'FAIL'} {name}" + ("" if not bad else ": " + "; ".join(bad)))
    if bad:
        print("  log:", log.strip().replace("\n", "\n       "))
    return not bad


def main():
    now = dt.datetime.now().astimezone()
    free = [f"{stamp(now - dt.timedelta(minutes=5))} 00:00:00 BUSY none",
            f"{stamp(now - dt.timedelta(seconds=20))} 00:00:00 ALIVE busy=none active=none"]
    stale = [f"{stamp(now - dt.timedelta(minutes=40))} 00:00:00 BUSY none"]
    run = ["ps", "lock take", "lock gate", "launch", "wait", "lock status", "pull", "lock release"]
    cases = [
        ("missing feed: refused", {}, None, 3, [], ["REFUSED (lock feed stale or missing): no feed log"], []),
        ("stale feed: refused", {}, stale, 3, [], ["REFUSED (lock feed stale or missing): STALE"], []),
        ("ZOO_FEED=none without a feed: the sitting runs", {"ZOO_FEED": "none"}, None, 0, run,
         ["ZOO_FEED=none: no feed", "lock: TAKEN", "gate: PASS", "sitting t end: DONE OK"], ["REFUSED"]),
        ("fresh feed, the lock free: the sitting runs", {}, free, 0, run,
         ["lock: TAKEN", "gate: PASS", "sitting t end: DONE OK"], ["REFUSED", "ZOO_FEED=none"]),
    ]
    ok = sum(run_case(*c) for c in cases)
    print(f"test_sitting_feed: {ok} of {len(cases)} cases as expected")
    sys.exit(0 if ok == len(cases) else 1)


if __name__ == "__main__":
    main()

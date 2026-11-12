#!/bin/bash
# One timed sitting on the PC, from the Mac (the audio zoo's tools/sitting.sh; the lane and the PC folder from zoo.toml):
#   1. wait on the Mac (the lock feed: the command in PC_FEED_CMD, as in tools/pc.sh's header; every 10 s, no ssh) until
#      busy.txt has been free for 60 s in a row. PC_FEED_CMD not set, or a feed that is missing or stale at the start,
#      refuses the sitting (exit 3). A Mac without a lock feed: ZOO_FEED=none skips the feed (as in tools/pc.sh); the
#      take then waits on the PC, in one ssh connection, until busy.txt has been free for 60 s (pc\lock.ps1 take)
#   2. the lane's lock ("<lane> timing: sitting <plan> ... until HH:MM") and the lane's quiet gate (3 x 5 s under 15 % CPU,
#      RTX empty, build tools quiet; at most 10 min, else the lock goes back and the sitting waits 10 min)
#   3. pc/chain.ps1 -Mode timed on the plan, detached: control-warm, control-head (stops the sitting when off the known
#      value by more than 5 %), the cells, control-tail; duty gaps and slots inside. This side waits in one long
#      connection (pc/wait_done.ps1), pulls runs/<plan> into raw/ while the lock is still ours (the chain holds it
#      20 min for that), then releases it.
#
#   tools/sitting.sh <plan name> [--resume] [--wait-hours N]
# --resume: the chain still runs and this side's wait ended or was stopped: wait for it again, then pull.
# Log: raw/<plan>.sitting.log. Exit 1 when the lock, the gate or the chain failed (the lock is released on every exit);
# exit 3 when PC_FEED_CMD is not set or the feed is missing or stale, and ZOO_FEED is not none.
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
plan="$1"; shift || true
resume=0; waith=12
while (( $# )); do
  case "$1" in --resume) resume=1 ;; --wait-hours) waith="$2"; shift ;; esac
  shift
done
mkdir -p "$K/raw"
log="$K/raw/$plan.sitting.log"
pcsh="$K/tools/pc.sh"
conf() { python3 -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['zoo'][sys.argv[2]])" "$K/zoo.toml" "$1"; }
LANE="$(conf lane)"
R="${ZOO_PC_ROOT:-$(conf pc_root)}"   # the PC folder (the same as tools/pc.sh)
say() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$log"; }
release() { say "release: $(ZOO_FORCE=1 "$pcsh" lock release timing "${1:-0}" 2>&1 | tail -1)"; }
until_hm=$(python3 -c "
import json, datetime as d
p = json.load(open('$K/pc/plans/$plan.json'))
s = sum((u.get('est_sec') or 60 * u.get('timeout_min', 10)) + 90 for u in p['units'] for _ in range(u.get('runs', 1)))
t = d.datetime.now().astimezone() + d.timedelta(seconds=s + 300 + 180)
if p.get('end_by'):   # the chain starts no cell that would end past it (its tail row included): the lock goes back by then
    t = min(t, d.datetime.fromisoformat(p['end_by']))
print(t.strftime('%H:%M'))")

free_for() {  # seconds busy.txt has been free in the feed, 0 when held (or the feed is stale)
  local f; f=$(bash -c "$PC_FEED_CMD" 2>/dev/null) || { echo 0; return; }
  printf '%s\n' "$f" | awk '$3=="BUSY"' | tail -n 1 | python3 -c "
import sys, datetime as d
l = sys.stdin.read().split()
if len(l) < 4 or l[3] != 'none': print(0)
else: print(int((d.datetime.now(d.timezone.utc) - d.datetime.strptime(l[0], '%Y-%m-%dT%H:%M:%S%z')).total_seconds()))"
}

ours_now() {  # the feed's lock is the lane's own (a first-pass block of its slot may run right before the sitting)
  bash -c "$PC_FEED_CMD" 2>/dev/null | awk '$3=="BUSY"' | tail -n 1 | grep -q " BUSY $LANE "
}

drop_dead_yield_note() {  # the lane's yield\<lane>.txt with no wrapper running reads as "this lane waits" to others
  local r
  r=$(ZOO_FORCE=1 "$pcsh" ps 2>&1 <<EOF
\$f = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'pcup\\yield\\$LANE.txt'
if (Test-Path \$f) {
  \$w = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { "\$(\$_.CommandLine)" -match 'yield_run\.ps1' -and "\$(\$_.CommandLine)" -match '-Lane $LANE( |\$)' })
  if (\$w.Count) { 'yield note kept: its wrapper runs' } else { Remove-Item \$f -Force; 'yield note removed: no wrapper ran' }
} else { 'no yield note' }
EOF
)
  say "own yield note: $(printf '%s\n' "$r" | tr -d '\r' | tail -n 1)"
}

take() {
  local r n
  while true; do
    n=0
    if [[ "${ZOO_FEED:-}" == "none" ]]; then say "ZOO_FEED=none: no feed; the take waits on the PC until busy.txt has been free 60 s"
    else while ! ours_now && (( $(free_for) < 60 )); do (( n++ % 30 == 0 )) && say "waiting for busy.txt to be free 60 s (feed)"; sleep 10; done; fi
    drop_dead_yield_note
    r=$("$pcsh" lock take "timing: sitting $plan until $until_hm" 2>&1 | tail -1); say "lock: $r"
    [[ "$r" == TAKEN* || "$r" == OURS* || "$r" == STALE* ]] || return 1
    r=$(ZOO_FORCE=1 "$pcsh" lock gate 2>&1 | tail -1); say "gate: $r"
    [[ "$r" == PASS* ]] && return 0
    release 600; say "gate did not open: waiting 10 min before taking the lock again (window.txt)"; sleep 600
  done
}

if [[ "${ZOO_FEED:-}" != "none" && -z "${PC_FEED_CMD:-}" ]]; then
  say "REFUSED (PC_FEED_CMD is not set: the command that prints the PC lock feed, tools/pc.sh's header); a Mac without a lock feed: ZOO_FEED=none"
  exit 3
fi
if [[ "${ZOO_FEED:-}" != "none" ]] && ! feed=$(bash -c "$PC_FEED_CMD" 2>&1); then
  say "REFUSED (lock feed stale or missing): $(printf '%s\n' "$feed" | tail -n 1); a Mac without a lock feed: ZOO_FEED=none"
  exit 3
fi
if (( resume )); then
  say "sitting $plan resume: waiting for its chain again"
else
  say "sitting $plan start (expected end about $until_hm)"
  # stopped while taking: the PC-side PowerShell of `lock take` outlives its ssh and may still take the lock (11-09)
  stop_take() { say "stopped during the lock take: reading the lock"; sleep 5
    local st; st=$(ZOO_FORCE=1 "$pcsh" lock status 2>&1 | head -1); say "lock: $st"
    [[ "$st" == "busy: $LANE "* ]] && release; exit 1; }
  trap stop_take INT TERM
  take || { trap - INT TERM; say "no lock"; exit 1; }
  trap - INT TERM
  trap 'release' EXIT
  r=$(ZOO_FORCE=1 "$pcsh" launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\$plan.json -Mode timed" 2>&1 | tail -1); say "chain: $r"
  [[ "$r" == STARTED* ]] || exit 1
fi
ZOO_FORCE=1 "$pcsh" wait "$R\\runs\\$plan\\chain.done" "$R\\runs\\$plan\\chain.log" "$waith" >> "$log" 2>&1
cw=$(tail -1 "$log"); say "chain wait: $cw"
if [[ "$cw" != DONE* ]]; then
  trap - EXIT
  say "the wait ended before the chain's done file (the chain holds the lock): nothing released; tools/sitting.sh $plan --resume"
  exit 1
fi
st=$(ZOO_FORCE=1 "$pcsh" lock status 2>&1 | head -1); say "lock after the chain: $st"
if [[ "$st" == "busy: $LANE "* ]]; then
  ZOO_FORCE=1 "$pcsh" pull "runs/$plan" raw && say "pulled runs\\$plan -> raw/$plan"
else say "the lock is not ours after the chain: no pull now (pull runs/$plan in the next own lock or free time)"; fi
release
trap - EXIT
say "sitting $plan end: $cw"
[[ "$cw" == "DONE OK"* ]]

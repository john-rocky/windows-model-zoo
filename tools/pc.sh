#!/bin/bash
# Mac-side calls to the PC (ssh <zoo.toml pc_ssh>; its default shell is PowerShell), one SSH connection per call.
# From the audio zoo's tools/pc.sh (windows-audio-zoo), with the lane, the ssh host and the PC folder read from zoo.toml.
# Before every connection it reads the shared lock feed on the Mac (~/code/standup/tools/winpc/busy_now.sh, no ssh):
# another lane's "timing:" lock, or a feed older than 3 min, refuses the call (exit 3). ZOO_FORCE=1 skips the read
# (tools/sitting.sh inside the lane's own lock; `wait` never reads it: it is the one long connection of a running job).
# A Mac without that feed (another shared-PC set-up): ZOO_FEED=none skips the read.
#
#   pc.sh status                      busy.txt, the active\ and yield\ notes, free disk
#   pc.sh ps                          run the PowerShell read from stdin (-EncodedCommand)
#   pc.sh push                        pc/, suites/ and zoo.toml -> the PC folder (ZOO_PC_ROOT, default zoo.toml pc_root)
#   pc.sh pushfix <S150 dir> [<S220 dir>]   the clip sets -> <PC folder>\fixtures (S220 under fixtures\s220)
#   pc.sh pull <rel path> <K dest>    <PC folder>\<rel path> -> K/<K dest>/<its last part> (one tar, one connection;
#                                     without the cache\ and wav\ folders)
#   pc.sh py <pc\script.py> [args]   that script with the folder's .venv-ovg python (-I), in the foreground;
#                                     ZOO_VENV=<name> picks .venv-<name>
#   pc.sh lock take|gate|release|status|slot|note ["what"] [open s]   pc\lock.ps1 (the waits run on the PC)
#   pc.sh launch <script> "<args>" [yield]          pc\launch.ps1 (detached through WMI, parse check first)
#   pc.sh wait <done file> <log file> [max hours]  pc\wait_done.ps1 (one long connection, the log's new lines)
set -euo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
conf() { python3 -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['zoo'][sys.argv[2]])" "$K/zoo.toml" "$1"; }
LANE="$(conf lane)"
HOST="$(conf pc_ssh)"
R="${ZOO_PC_ROOT:-$(conf pc_root)}"   # the PC folder (README walk: another empty folder)
RF="${R//\\//}"                        # the same with forward slashes (ssh tar)

if [[ -z "${ZOO_FORCE:-}" && "${ZOO_FEED:-}" != "none" && "${1:-}" != "wait" ]]; then
  feed=$(bash ~/code/standup/tools/winpc/busy_now.sh 2>&1) || { echo "REFUSED (lock feed stale or missing): $feed" >&2; exit 3; }
  busy=$(printf '%s\n' "$feed" | awk '$3=="BUSY"' | tail -n 1)
  if [[ "$busy" == *"timing:"* && "$busy" != *" BUSY $LANE "* ]]; then
    echo "REFUSED (another lane's timing lock): ${busy}" >&2
    exit 3
  fi
fi

ps_run() {  # stdin: PowerShell text
  local b64
  b64=$( (printf '%s\n' "\$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8"; cat) \
    | iconv -f UTF-8 -t UTF-16LE | base64)
  ssh "$HOST" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $b64"
}

case "${1:-}" in
  status)
    ps_run <<'EOF'
"now $(Get-Date -Format o)"
$p = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'pcup'
if (Test-Path "$p\busy.txt") { "busy: $((Get-Content "$p\busy.txt" -Raw).Trim()) (written $([int]((Get-Date) - (Get-Item "$p\busy.txt").LastWriteTime).TotalMinutes) min ago)" } else { "busy: none" }
Get-ChildItem "$p\active\*.txt", "$p\yield\*.txt" -ErrorAction SilentlyContinue | ForEach-Object { "$($_.Directory.Name)\$($_.Name): $(((Get-Content $_.FullName -Raw).Trim()) -replace '\s+', ' ')" }
"free C: $([math]::Round((Get-PSDrive C).Free / 1GB, 1)) GB"
EOF
    ;;
  ps) ps_run ;;
  py)
    script="$2"; shift 2
    ps_run <<EOF
\$env:ORT_DISABLE_TELEMETRY = '1'
& '$R\\.venv-${ZOO_VENV:-ovg}\\Scripts\\python.exe' -I '$R\\$script' $* 2>&1 | ForEach-Object { "\$_" }
"exit \$LASTEXITCODE"
EOF
    ;;
  push)
    COPYFILE_DISABLE=1 tar -cf - -C "$K" --exclude __pycache__ pc suites zoo.toml \
      | ssh "$HOST" "New-Item -ItemType Directory -Force $RF | Out-Null; tar -xf - -C $RF"
    # then every pc/*.ps1 parsed on the PC with the Mac's sha, and zlib loaded once (the audio zoo's push_code, 11-09)
    out=$(ps_run <<EOF
Get-ChildItem '$R\\pc\\*.ps1' | ForEach-Object {
    \$e = \$null
    [void][System.Management.Automation.Language.Parser]::ParseFile(\$_.FullName, [ref]\$null, [ref]\$e)
    \$n = if (\$e) { \$e.Count } else { 0 }
    'ps1 ' + \$_.Name + ' ' + \$n + ' ' + (Get-FileHash \$_.FullName -Algorithm SHA256).Hash.Substring(0, 16)
}
try { . '$R\\pc\\zlib.ps1'; \$o = @(Get-OthersWaiting); 'zlib loaded, lane ' + \$Label + ', others waiting ' + \$o.Count }
catch { 'zlib FAILED ' + \$_.Exception.Message }
EOF
)
    got=$(printf '%s\n' "$out" | tr -d '\r' | grep '^ps1 ' | sort)
    want=$(for f in "$K"/pc/*.ps1; do printf 'ps1 %s 0 %s\n' "${f##*/}" "$(shasum -a 256 "$f" | cut -c1-16 | tr 'a-f' 'A-F')"; done | sort)
    miss=$(comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$got"))
    zl=$(printf '%s\n' "$out" | tr -d '\r' | grep '^zlib ' | tail -n 1)
    if [[ -n "$want" && -z "$miss" && "$zl" == "zlib loaded"* ]]; then
      echo "pushed pc/, suites/, zoo.toml -> $R: $(grep -c . <<< "$want") .ps1 files parse on the PC with the Mac's sha; $zl"
    else
      echo "PUSH CHECK FAILED: a .ps1 not parsed or not the Mac's, or zlib did not load:" >&2
      printf '%s\n' "$miss" "--- PC:" "$got" "${zl:-no zlib line}" >&2
      exit 1
    fi
    ;;
  pushfix)
    s150="$2"; s220="${3:-}"
    COPYFILE_DISABLE=1 tar -cf - -C "$s150" meta.json fleurs examples \
      | ssh "$HOST" "New-Item -ItemType Directory -Force $RF/fixtures | Out-Null; tar -xf - -C $RF/fixtures"
    if [[ -n "$s220" ]]; then
      COPYFILE_DISABLE=1 tar -cf - -C "$s220" clips.json clips \
        | ssh "$HOST" "New-Item -ItemType Directory -Force $RF/fixtures/s220 | Out-Null; tar -xf - -C $RF/fixtures/s220"
    fi
    echo "pushed the clip sets -> $R\\fixtures"
    ;;
  pull)
    mkdir -p "$K/$3"
    # not the runtimes' compile caches ({RUN}\cache: GPU / NPU blobs) nor the TTS waves (the write-back reads them on the PC)
    ssh "$HOST" "tar -cf - --exclude cache --exclude wav -C $RF/$(dirname "$2") $(basename "$2")" | tar -xf - -C "$K/$3"
    ;;
  lock)
    what="${3-timing}"   # an empty "what" stays empty (lock note "" removes the note)
    ps_run <<EOF
& '$R\\pc\\lock.ps1' -Action $2 -What '$what' -OpenSec ${4:-0}
EOF
    ;;
  launch)
    y=""; [[ "${4:-}" == "yield" ]] && y="-Yield"
    ps_run <<EOF
& '$R\\pc\\launch.ps1' -Script '$R\\$2' -ArgLine '$3' $y
EOF
    ;;
  wait)
    ps_run <<EOF
& '$R\\pc\\wait_done.ps1' -Done '$2' -Log '$3' -MaxHours ${4:-12}
EOF
    ;;
  *) sed -n 2,22p "$0"; exit 2 ;;
esac

<#
The README walk's check, on the PC after the walk (Windows PowerShell 5.1): every file the README's steps should leave,
as "OK <what> <path>" or "MISSING <what> <path>", each step list's and chain's done line, and per sitting the run
records against its plan. Exit 1 when a file is missing, a done line is absent or does not say OK, or a unit's run
record does not say OK.

  powershell -NoProfile -ExecutionPolicy Bypass -File pc\walk_check.ps1 -Lists setup,fetch,<model>,... -Sittings walk-a

A done line that says FAIL, and a unit run whose status is not OK, count as failed (the last line gives both counts).
A cell that the chain left for a later sitting (--end-by: left_cells in runs\<sitting>\control.json) is listed as LEFT and
counts as neither; the last line then adds "L left by end_by".
#>
param([string[]]$Lists = @(), [string[]]$Sittings = @())   # "a,b" from -File, or a,b (an array) from & inside PowerShell
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Lists = @($Lists | ForEach-Object { "$_".Split(",") } | Where-Object { $_ })
$Sittings = @($Sittings | ForEach-Object { "$_".Split(",") } | Where-Object { $_ })
$script:miss = 0
$script:failed = 0
$script:leftN = 0
function Want([string]$What, [string]$Rel) {
    $p = Join-Path $Root $Rel
    if (-not (Test-Path -LiteralPath $p)) { "MISSING $What $Rel"; $script:miss++; return }
    "OK      $What $Rel"
}
function Done([string]$What, [string]$Rel) {
    $p = Join-Path $Root $Rel
    if (-not (Test-Path -LiteralPath $p)) { "MISSING $What $Rel"; $script:miss++; return }
    $line = ([IO.File]::ReadAllText($p)).Trim()
    if ($line -notmatch '^OK\b') { $script:failed++ }
    "DONE    $What $Rel = $line"
}

# push (and pushfix when a clip set is used)
foreach ($f in "zoo.toml", "pc\chain.ps1", "pc\steps.ps1", "pc\zlib.ps1", "pc\lock.ps1", "pc\launch.ps1", "pc\wait_done.ps1",
               "pc\zoolib.py", "pc\fetch.py", "pc\hub_manifest.json") { Want "push" $f }
if (Test-Path (Join-Path $Root "fixtures\meta.json")) { Want "pushfix" "fixtures\meta.json" }

# step lists: their done lines; fetch: every manifest file matched its hash
foreach ($l in $Lists) { Done "steps $l" "logs\$l.done" }
$rep = Join-Path $Root "hub\fetch_report.json"
if ($Lists -contains "fetch") {
    if (Test-Path $rep) {
        $r = @(Get-Content $rep -Raw -Encoding UTF8 | ConvertFrom-Json)
        $r = @($r | ForEach-Object { $_ })
        $bad = @($r | Where-Object { -not $_.ok })
        if ($bad.Count) { "MISSING fetch $($bad.Count) of $($r.Count) files not matching their hash"; $script:miss++ }
        else { "OK      fetch $($r.Count) files, every hash matching (hub\fetch_report.json)" }
    } else { "MISSING fetch hub\fetch_report.json"; $script:miss++ }
}

# sittings: the chain's done line, one runner JSON and one chain meta per unit run, the "after" list
foreach ($s in $Sittings) {
    Done "chain $s" "runs\$s\chain.done"
    $pp = Join-Path $Root "pc\plans\$s.json"
    if (-not (Test-Path $pp)) { "MISSING plan pc\plans\$s.json"; $script:miss++; continue }
    $plan = Get-Content $pp -Raw -Encoding UTF8 | ConvertFrom-Json
    $leftCells = @(); $leftBy = ""
    $cj = Join-Path $Root "runs\$s\control.json"
    if (Test-Path $cj) {
        $ctlJson = Get-Content $cj -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($ctlJson.left_cells) { $leftCells = @($ctlJson.left_cells); $leftBy = "$($ctlJson.end_by)" }
    }
    foreach ($u in @($plan.units)) {
        if ($leftCells -contains $u.cell) {
            "LEFT    unit $($u.cell): not started, left by end_by $leftBy (runs\$s\control.json)"
            $script:leftN++
            continue
        }
        $runs = if ($u.runs) { [int]$u.runs } else { 1 }
        foreach ($k in 1..$runs) {
            $d = "runs\$s\$($u.cell -replace '/', '\')"
            Want "unit $($u.cell)" "$d\run$k.json"
            Want "unit $($u.cell)" "$d\run$k.meta.json"
            $rj = Join-Path $Root "$d\run$k.json"
            if (Test-Path $rj) {
                $st = (Get-Content $rj -Raw -Encoding UTF8 | ConvertFrom-Json).status
                "STATUS  unit $($u.cell) run $k = $st"
                if ("$st" -ne "OK") { $script:failed++ }
            }
        }
    }
    $al = Join-Path $Root "pc\steps\after-$s.json"
    if (Test-Path $al) {
        $n = @(@(Get-Content $al -Raw -Encoding UTF8 | ConvertFrom-Json) | ForEach-Object { $_ }).Count
        if ($n) { Done "after $s" "logs\after-$s.done" }
    }
}
"walk check: $script:miss missing, $script:failed failed$(if ($script:leftN) { ", $script:leftN left by end_by" })"
exit $(if ($script:miss -or $script:failed) { 1 } else { 0 })

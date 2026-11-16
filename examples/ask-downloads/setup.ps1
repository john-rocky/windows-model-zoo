# Sets up examples\ask-downloads on a Windows PC (PowerShell 5.1 or 7, uv on PATH):
#   1 two Python 3.12 venvs with uv: venv-ovg (OpenVINO GenAI; it also runs the window and the llama.cpp card) and
#     venv-ref (PyTorch + Transformers for the original model), pinned like the zoo's runs (req-ovg.txt, req-ref.txt)
#   2 the Hub files of hub_manifest.json at their pinned revisions (the zoo's pc\fetch.py: every file checked against the
#     Hub's sha256, resumable; about 8 GB)
#   3 llama.cpp b11721 for the CPU (the release zip, checked by sha256, then every .exe / .dll by llama_cpu.json)
# Into -Root (default: .\work next to this script), the folder ask_downloads.py reads as {ROOT}. Run it again to finish
# what is missing; done steps are skipped.     powershell -ExecutionPolicy Bypass -File setup.ps1 [-Root D:\ask-downloads]
param([string]$Root = "$PSScriptRoot\work")
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot
$Zoo = (Resolve-Path "$Here\..\..").Path
$utf8 = New-Object Text.UTF8Encoding($false)
function Say($s) { [Console]::Out.WriteLine("$(Get-Date -Format HH:mm:ss) $s") }
function Sha($p) { return (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() }
$uv = (Get-Command uv.exe -ErrorAction SilentlyContinue).Source
if (-not $uv -and (Test-Path "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe")) { $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe" }
if (-not $uv) { Say "uv.exe not found: install it first (winget install astral-sh.uv), then run this again"; exit 1 }
New-Item -ItemType Directory -Force $Root, "$Root\hub", "$Root\engines" | Out-Null

# 1 venvs
foreach ($v in @(@{ n = "venv-ovg"; req = "$Here\req-ovg.txt" }, @{ n = "venv-ref"; req = "$Here\req-ref.txt" })) {
  $py = "$Root\$($v.n)\Scripts\python.exe"
  $mark = "$Root\$($v.n)\.setup-ok"
  if ((Test-Path $mark) -and ((Get-Content $mark -Raw) -eq (Get-Content $v.req -Raw))) { Say "$($v.n): there"; continue }
  if (-not (Test-Path $py)) { & $uv venv --python 3.12 "$Root\$($v.n)" 2>&1 | ForEach-Object { Say "  $_" } }
  & $uv pip install --python $py -r $v.req 2>&1 | ForEach-Object { Say "  $_" }
  if ($LASTEXITCODE -ne 0) { Say "$($v.n): uv pip install failed (exit $LASTEXITCODE)"; exit 1 }
  [IO.File]::WriteAllText($mark, (Get-Content $v.req -Raw), $utf8)
  Say "$($v.n): made"
}
$PY = "$Root\venv-ovg\Scripts\python.exe"

# 2 Hub files
& $PY "$Zoo\pc\fetch.py" --manifest "$Here\hub_manifest.json" --dest "$Root\hub" --root $Root 2>&1 | ForEach-Object { Say "  $_" }
if ($LASTEXITCODE -ne 0) { Say "downloads: some file does not match its pinned hash (see $Root\hub\fetch_report.json)"; exit 1 }

# 3 llama.cpp b11721 CPU
$lc = Get-Content "$Here\llama_cpu.json" -Raw | ConvertFrom-Json
$ld = "$Root\engines\llama-cpu"
function BadFiles { $b = @(); foreach ($p in $lc.files.PSObject.Properties) { $f = Join-Path $ld $p.Name; if (-not (Test-Path $f) -or (Sha $f) -ne $p.Value) { $b += $p.Name } }; return ,$b }
$bad = BadFiles
if ($bad.Count -gt 0) {
  $zip = "$Root\engines\" + [IO.Path]::GetFileName($lc.zip.url)
  if (-not (Test-Path $zip) -or (Sha $zip) -ne $lc.zip.sha256) {
    & curl.exe -sSL --fail -C - --speed-limit 102400 --speed-time 60 --retry 3 -o $zip $lc.zip.url
  }
  if ((Sha $zip) -ne $lc.zip.sha256) { Say "llama.cpp: the zip's sha256 differs from llama_cpu.json"; exit 1 }
  if (Test-Path $ld) { Remove-Item $ld -Recurse -Force }
  Expand-Archive -Path $zip -DestinationPath $ld -Force
  $top = @(Get-ChildItem $ld)
  if ($top.Count -eq 1 -and $top[0].PSIsContainer) { Get-ChildItem $top[0].FullName | Move-Item -Destination $ld; Remove-Item $top[0].FullName -Recurse -Force }
  Remove-Item $zip
  $bad = BadFiles
}
if ($bad.Count -gt 0) { Say "llama.cpp: files differ from llama_cpu.json: $($bad -join ', ')"; exit 1 }
Say "llama.cpp: $(@($lc.files.PSObject.Properties).Count) files match"
Say "setup done: $Root"
exit 0

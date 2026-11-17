# Create the extension worker environment (extensions\env) without touching
# .venv or mmaction2_demo\env. Re-running is safe.
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Repo = Split-Path $Root -Parent
$Env = Join-Path $Root 'env'
$Python = Join-Path $Env 'Scripts\python.exe'

function Run-Checked {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Setup failed ($LASTEXITCODE): $Executable $CommandArguments" }
}

if (-not (Test-Path -LiteralPath $Python)) {
    # Python 3.12 (torch 2.6 wheels exist for it). Prefer the py launcher, else uv's managed copy.
    $Base = $null
    try { $Base = (& py -3.12 -c "import sys; print(sys.executable)") 2>$null } catch {}
    if (-not $Base) {
        $Candidate = Get-ChildItem "$env:APPDATA\uv\python" -Filter 'python.exe' -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match 'cpython-3\.12' } | Select-Object -First 1
        if ($Candidate) { $Base = $Candidate.FullName }
    }
    if (-not $Base) { throw 'Python 3.12 not found. Install it (py launcher or uv) and run again.' }
    Run-Checked $Base @('-m', 'venv', $Env)
}

Run-Checked $Python @('-m', 'pip', 'install', '--upgrade', 'pip==25.1.1')
# Ultralytics must be installed with --no-deps (its own deps would add opencv-python next to the pinned
# headless build); its runtime dependencies are pinned in the same file.
$Requirements = Join-Path $Repo 'requirements-extensions.txt'
$Filtered = Join-Path $env:TEMP 'requirements-extensions.no-ultralytics.txt'
Get-Content -LiteralPath $Requirements | Where-Object { $_ -notmatch '^\s*ultralytics==' } | Set-Content -LiteralPath $Filtered -Encoding utf8
Run-Checked $Python @('-m', 'pip', 'install', '-r', $Filtered)
$Ultralytics = (Get-Content -LiteralPath $Requirements | Where-Object { $_ -match '^\s*ultralytics==' } | Select-Object -First 1).Trim()
Run-Checked $Python @('-m', 'pip', 'install', '--no-deps', $Ultralytics)
Run-Checked $Python @('-c', "import torch, numpy, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
# Smoke test: the worker starts and answers with the dependency-free selftest extension.
Run-Checked $Python @((Join-Path $Root 'check_worker.py'))
Write-Host 'Extension environment ready:' $Python

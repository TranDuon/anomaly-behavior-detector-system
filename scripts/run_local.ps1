[CmdletBinding()]
param(
    [ValidateSet("serve", "video", "webcam")]
    [string]$Mode = "serve",

    [string]$Source,

    [string]$BundlePath,

    [switch]$Reinstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$SetupStatePath = Join-Path $VenvDir ".sentinel-setup-state"
$ModelsDir = Join-Path $ProjectRoot "models"
$RequiredArtifacts = @(
    "lstm_autoencoder.pt",
    "feature_scaler.npz",
    "threshold.json"
)

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[SENTINEL] $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

function Test-RuntimeArtifacts {
    foreach ($Name in $RequiredArtifacts) {
        $Path = Join-Path $ModelsDir $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            return $false
        }
        if ((Get-Item -LiteralPath $Path).Length -le 0) {
            return $false
        }
    }
    return $true
}

function Test-PythonCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & $Executable @Arguments 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}

function New-VirtualEnvironment {
    Write-Step "Creating Python virtual environment"
    $Created = $false

    $SystemPython = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $SystemPython -and (Test-PythonCommand $SystemPython.Source @(
        "-c", "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13)"
    ))) {
        Write-Host "Using $($SystemPython.Source)"
        Invoke-Checked $SystemPython.Source @("-m", "venv", $VenvDir)
        $Created = $true
    }

    if (-not $Created) {
        $PyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
        if ($null -ne $PyLauncher) {
            foreach ($Version in @("3.11", "3.12", "3.13")) {
                if (Test-PythonCommand $PyLauncher.Source @("-$Version", "-c", "import sys")) {
                    Write-Host "Using py -$Version"
                    Invoke-Checked $PyLauncher.Source @("-$Version", "-m", "venv", $VenvDir)
                    $Created = $true
                    break
                }
            }
        }
    }

    if (-not $Created) {
        throw "Python 3.11, 3.12, or 3.13 was not found. Install Python from python.org, enable Add Python to PATH, then run again."
    }

    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Virtual environment creation failed: $PythonExe was not created."
    }
}

function Install-Dependencies {
    $RequirementsHash = (Get-FileHash -LiteralPath $RequirementsPath -Algorithm SHA256).Hash
    $ExpectedState = "$RequirementsHash|cpu-v1"
    $CurrentState = ""
    if (Test-Path -LiteralPath $SetupStatePath -PathType Leaf) {
        $CurrentState = (Get-Content -LiteralPath $SetupStatePath -Raw).Trim()
    }

    if (-not $Reinstall -and $CurrentState -eq $ExpectedState) {
        Write-Step "Python dependencies are already installed"
        return
    }

    Write-Step "Installing CPU-only PyTorch and project dependencies"
    Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $PythonExe @(
        "-m", "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    )
    Invoke-Checked $PythonExe @("-m", "pip", "install", "-r", $RequirementsPath)
    Set-Content -LiteralPath $SetupStatePath -Value $ExpectedState -Encoding UTF8
}

function Resolve-RuntimeBundle {
    if ($BundlePath) {
        return (Resolve-Path -LiteralPath $BundlePath).Path
    }

    $DownloadsDir = Join-Path $env:USERPROFILE "Downloads"
    if (-not (Test-Path -LiteralPath $DownloadsDir -PathType Container)) {
        return $null
    }

    $Candidate = Get-ChildItem -LiteralPath $DownloadsDir -File |
        Where-Object { $_.Name -like "sentinel_ucsd_ped2_runtime*.zip" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $Candidate) {
        return $null
    }
    return $Candidate.FullName
}

function Install-RuntimeArtifacts {
    $ResolvedBundle = Resolve-RuntimeBundle
    if ($null -ne $ResolvedBundle) {
        Write-Step "Installing runtime bundle: $ResolvedBundle"
        Expand-Archive -LiteralPath $ResolvedBundle -DestinationPath $ProjectRoot -Force
    } elseif (-not (Test-RuntimeArtifacts)) {
        throw @"
Runtime artifacts were not found.
Download sentinel_ucsd_ped2_runtime.zip from the final Colab cell and place it in:
$env:USERPROFILE\Downloads
Then run run_local.cmd again.
"@
    }

    if (-not (Test-RuntimeArtifacts)) {
        throw "The runtime bundle is missing model, scaler, or threshold files."
    }
}

function Set-CpuEnvironment {
    $EnvPath = Join-Path $ProjectRoot ".env"
    $ExamplePath = Join-Path $ProjectRoot ".env.example"
    if (-not (Test-Path -LiteralPath $EnvPath -PathType Leaf)) {
        Copy-Item -LiteralPath $ExamplePath -Destination $EnvPath
    }

    $Lines = @(Get-Content -LiteralPath $EnvPath)
    $FoundDevice = $false
    $UpdatedLines = foreach ($Line in $Lines) {
        if ($Line -match "^\s*YOLO_DEVICE\s*=") {
            $FoundDevice = $true
            "YOLO_DEVICE=cpu"
        } else {
            $Line
        }
    }
    if (-not $FoundDevice) {
        $UpdatedLines += "YOLO_DEVICE=cpu"
    }
    Set-Content -LiteralPath $EnvPath -Value $UpdatedLines -Encoding UTF8
}

function Test-RuntimeModel {
    Write-Step "Verifying checkpoint, scaler, and threshold on CPU"
    Invoke-Checked $PythonExe @("-m", "scripts.verify_runtime")
}

Set-Location $ProjectRoot

Write-Step "Project: $ProjectRoot"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    New-VirtualEnvironment
}
Install-Dependencies
Install-RuntimeArtifacts
Set-CpuEnvironment
Test-RuntimeModel

switch ($Mode) {
    "serve" {
        Write-Step "Starting dashboard at http://localhost:8000"
        Invoke-Checked $PythonExe @("main.py", "serve")
    }
    "video" {
        if (-not $Source) {
            throw "Mode video requires -Source with a video file path."
        }
        $ResolvedSource = (Resolve-Path -LiteralPath $Source).Path
        Write-Step "Processing video on CPU: $ResolvedSource"
        Invoke-Checked $PythonExe @("main.py", "run", "--source", $ResolvedSource)
    }
    "webcam" {
        Write-Step "Starting webcam 0 on CPU"
        Invoke-Checked $PythonExe @("main.py", "run", "--source", "0")
    }
}

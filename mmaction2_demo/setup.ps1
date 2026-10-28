$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root 'env\python.exe'
$Repo = Join-Path $Root 'repo'
function Run-Checked {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Setup failed ($LASTEXITCODE): $Executable" }
}
if (-not (Test-Path -LiteralPath $Python)) {
    $Downloads = Join-Path $Root 'downloads'
    New-Item -ItemType Directory -Force $Downloads | Out-Null
    $Archive = Join-Path $Downloads 'python310.zip'
    if (-not (Test-Path -LiteralPath (Join-Path $Downloads 'python310\tools\python.exe'))) {
        if (-not (Test-Path -LiteralPath $Archive)) { New-Item -ItemType File $Archive | Out-Null }
        Run-Checked 'curl.exe' @('-fL', '-C', '-', '--retry', '2', '--connect-timeout', '15', '--max-time', '300', 'https://api.nuget.org/v3-flatcontainer/python/3.10.11/python.3.10.11.nupkg', '-o', $Archive)
        Expand-Archive -LiteralPath $Archive -DestinationPath (Join-Path $Downloads 'python310') -Force
    }
    New-Item -ItemType Directory -Force (Join-Path $Root 'env') | Out-Null
    Copy-Item -Path (Join-Path $Downloads 'python310\tools\*') -Destination (Join-Path $Root 'env') -Recurse -Force
}
$env:PATH = "$Root\env;$Root\env\Library\bin;$Root\env\Scripts;$env:PATH"
if (-not (Test-Path -LiteralPath $Repo)) {
    Run-Checked 'git' @('clone', '--depth', '1', '--branch', 'v1.2.0', 'https://github.com/open-mmlab/mmaction2.git', $Repo)
}
$Revision = & git -C $Repo rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $Revision -ne '4d6c93474730cad2f25e51109adcf96824efc7a3') { throw 'Expected official MMAction2 v1.2.0; refusing to reset another checkout.' }
Run-Checked $Python @('-m', 'ensurepip')
Run-Checked $Python @('-m', 'pip', 'install', 'setuptools==69.5.1', 'wheel', 'numpy==1.26.4')
$Downloads = Join-Path $Root 'downloads'
New-Item -ItemType Directory -Force $Downloads | Out-Null
foreach ($Wheel in @('torch-2.0.1+cu118-cp310-cp310-win_amd64.whl', 'torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl')) {
    $Target = Join-Path $Downloads $Wheel
    $Partial = "$Target.partial"
    if (-not (Test-Path -LiteralPath $Target)) {
        Run-Checked $Python @((Join-Path $Root 'download_file.py'), "https://download.pytorch.org/whl/cu118/$($Wheel.Replace('+','%2B'))", $Partial)
        Move-Item -LiteralPath $Partial -Destination $Target
    }
    Run-Checked $Python @('-m', 'pip', 'install', $Target)
}
Run-Checked $Python @('-m', 'pip', 'install', 'mmcv==2.1.0', '--only-binary=mmcv', '-f', 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html', 'mmengine==0.10.7', 'mmdet==3.2.0', 'opencv-python==4.8.1.78')
Run-Checked $Python @('-m', 'pip', 'install', 'decord==0.6.0', 'einops', 'importlib-metadata', 'scikit-learn==1.3.2', 'matplotlib', 'Pillow', 'scipy', 'opencv-contrib-python==4.8.1.78')
Run-Checked $Python @('-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', $Repo)
Run-Checked $Python @((Join-Path $Root 'download_weights.py'))
Run-Checked $Python @((Join-Path $Root 'verify_install.py'))
Write-Host 'Ready. Use run_video.cmd or run_camera.cmd. Original project unchanged.'

@echo off
setlocal
if not exist "%~dp0results\install_verified.json" (
  echo Installation is not verified yet. Run setup.cmd first.
  pause
  exit /b 1
)
set "PATH=%~dp0env;%~dp0env\Library\bin;%~dp0env\Scripts;%PATH%"
cd /d "%~dp0"
"%~dp0env\python.exe" "%~dp0launch.py" %*
if errorlevel 1 pause

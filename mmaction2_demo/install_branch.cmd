@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag mmaction_interaction_pilot.zip onto this file. Stop web recognition first.
  pause
  exit /b 1
)
set "PATH=%~dp0env;%~dp0env\Library\bin;%~dp0env\Scripts;%PATH%"
"%~dp0env\python.exe" "%~dp0install_branch.py" "%~1"
pause

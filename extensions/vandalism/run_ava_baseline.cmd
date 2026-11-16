@echo off
rem V1 zero-shot AVA baseline. Uses the core GPU environment read-only.
setlocal
set ROOT=%~dp0..\..
set PYTHONPATH=%ROOT%\mmaction2_demo\repo;%ROOT%
set TORCH_HOME=%ROOT%\mmaction2_demo\weights\torch-cache
set CUDA_MODULE_LOADING=LAZY
cd /d "%ROOT%\mmaction2_demo"
"%ROOT%\mmaction2_demo\env\python.exe" "%~dp0ava_baseline.py" %*

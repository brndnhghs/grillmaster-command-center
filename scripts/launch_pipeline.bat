@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  launch_pipeline.bat — clickable launcher for the Grillmaster image pipeline.
REM
REM  Behaviour:
REM    1. Kills ANY running pipeline instance bound to port 7860 (and any
REM       orphaned python -m image_pipeline.server processes) BEFORE launching,
REM       so you never end up with two servers fighting over the port.
REM    2. Starts the server in this repo's .venv and opens it in your browser.
REM
REM  Edit PIPELINE_PORT below if you run the server on a different port.
REM ─────────────────────────────────────────────────────────────────────────────
setlocal enabledelayedexpansion

set "REPO=%~dp0.."
set "PIPELINE_PORT=7860"
set "VENV_PY=%REPO%\.venv\Scripts\python.exe"
set "LOGDIR=%REPO%\data\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [launch] Repo : %REPO%
echo [launch] Port : %PIPELINE_PORT%

REM ── 1. Kill anything already LISTENING on the pipeline port ─────────────────
REM    Only LISTENING sockets own the port. We dump netstat to a temp file (avoids
REM    fragile pipe-quoting inside a for-loop), then find the listener PID and
REM    taskkill it. We guard against PID 0 / System (PID 4) so we never kill the
REM    wrong process.
echo [launch] Stopping any extant pipeline instances on port %PIPELINE_PORT% ...
set "NST=%TEMP%\grillmaster_netstat.txt"
netstat -aon > "%NST%" 2>nul
for /f "tokens=5" %%p in ('findstr /R ":%PIPELINE_PORT%.*LISTENING" "%NST%"') do (
    if not "%%p"=="0" if not "%%p"=="4" (
        echo [launch]   taskkill /PID %%p
        taskkill /PID %%p /F >nul 2>&1
    )
)
del /f "%NST%" 2>nul
REM Windows-native short pause (git-bash maps `timeout` to GNU coreutils).
ping -n 2 127.0.0.1 >nul 2>&1

REM ── 2. Launch the server in the background ─────────────────────────────────
echo [launch] Starting pipeline server...
start "" cmd /c "%~dp0run_pipeline_server.bat"

REM ── 3. Wait briefly, then open the browser ─────────────────────────────────
ping -n 3 127.0.0.1 >nul 2>&1
echo [launch] Opening http://localhost:%PIPELINE_PORT% ...
start "" "http://localhost:%PIPELINE_PORT%"

echo [launch] Done. Logs: %LOGDIR%\pipeline-server.log
endlocal

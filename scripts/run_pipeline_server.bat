@echo off
REM Internal helper: launch the pipeline server. Called by launch_pipeline.bat.
REM Kept separate so the launcher can background it via `start` without nested quoting.
set "REPO=%~dp0.."
set "VENV_PY=%REPO%\.venv\Scripts\python.exe"
set "LOGDIR=%REPO%\data\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
REM Clear inherited PYTHONPATH (e.g. from a Hermes/git-bash shell) so the repo
REM venv's pinned packages are never shadowed by another Python's site-packages.
set "PYTHONPATH="
set "_VIRTUAL_ENV="
"%VENV_PY%" -m image_pipeline.server --port 7860 >> "%LOGDIR%\pipeline-server.log" 2>&1

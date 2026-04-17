@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem  traiNNer-redux-webui launcher (Windows)
rem
rem  Edit the two variables below if you already have your own Python install
rem  and/or a traiNNer-redux checkout somewhere else on your system.
rem
rem    PYTHON_BIN    Path to the python.exe to run the web UI with.
rem                  Must have the packages in webui\requirements.txt installed
rem                  (pip install -r webui\requirements.txt).
rem                  Defaults to the bundled python\python.exe.
rem
rem    TRAINNER_DIR  Path to your traiNNer-redux checkout
rem                  (the folder that contains train.py, options\, etc.).
rem                  Defaults to the traiNNer-redux\ folder next to this script.
rem ============================================================================

set "PYTHON_BIN=%~dp0python\python.exe"
set "TRAINNER_DIR=%~dp0traiNNer-redux"

rem ── Nothing below this line normally needs editing ──────────────────────────

if not exist "!PYTHON_BIN!" (
    echo [launch.bat] ERROR: Python binary not found at:
    echo     !PYTHON_BIN!
    echo Edit PYTHON_BIN at the top of this script to point at your python.exe.
    pause
    exit /b 1
)

if not exist "!TRAINNER_DIR!\" (
    echo [launch.bat] ERROR: traiNNer-redux directory not found at:
    echo     !TRAINNER_DIR!
    echo Edit TRAINNER_DIR at the top of this script to point at your checkout.
    pause
    exit /b 1
)

set "TRAINNER_REDUX_DIR=!TRAINNER_DIR!"

cd /d "%~dp0webui"
"!PYTHON_BIN!" server.py %*
pause
endlocal

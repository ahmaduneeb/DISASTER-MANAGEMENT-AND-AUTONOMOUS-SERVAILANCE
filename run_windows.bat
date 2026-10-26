@echo off
setlocal enabledelayedexpansion
title AEGIS - UAV Strategy Engine
cd /d "%~dp0"

echo ============================================================
echo   AEGIS - UAV Strategy Engine
echo   Disaster Management ^& Autonomous Surveillance
echo ============================================================
echo.

REM ------------------------------------------------------------------
REM 1. Find a Python 3.10+ interpreter
REM ------------------------------------------------------------------
set "PY_CMD="
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"
if not defined PY_CMD (
  python -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo [ERROR] Python was not found on this computer.
  echo.
  echo Install Python 3.10 or newer from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add Python to PATH" during installation.
  echo.
  pause
  exit /b 1
)

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.10 or newer is required.
  %PY_CMD% --version
  echo.
  echo Install a newer Python from https://www.python.org/downloads/
  echo.
  pause
  exit /b 1
)

REM ------------------------------------------------------------------
REM 2. Create / reuse the virtual environment
REM ------------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment in .venv ...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :failed
) else (
  echo [1/3] Reusing existing virtual environment.
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] The virtual environment looks broken.
  echo Delete the .venv folder and run this file again.
  goto :failed
)

REM ------------------------------------------------------------------
REM 3. Install dependencies
REM ------------------------------------------------------------------
echo [2/3] Installing dependencies from requirements.txt ...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>nul
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [ERROR] Dependency installation failed.
  echo Check your internet connection and try again.
  goto :failed
)

REM ------------------------------------------------------------------
REM 4. Launch the dashboard
REM ------------------------------------------------------------------
echo [3/3] Starting the dashboard ...
echo.
echo   URL:  http://127.0.0.1:5000
echo   The browser should open automatically.
echo   Press CTRL+C or close this window to stop the server.
echo.
echo   Extra options (optional): run_windows.bat --port 8000 --no-browser
echo.

"%VENV_PY%" app.py %*
echo.
echo Dashboard stopped.
exit /b 0

:failed
echo.
echo [ERROR] Setup or startup failed. See the messages above.
echo.
pause
exit /b 1

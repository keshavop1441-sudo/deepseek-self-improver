@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DeepSeek Self-Improver setup

rem ---- find a Python >= 3.10 ------------------------------------------------
set "BASEPY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASEPY=py -3"
if not defined BASEPY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
  if not errorlevel 1 set "BASEPY=python"
)
if not defined BASEPY goto :no_python

rem ---- virtual environment ----------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  %BASEPY% -m venv .venv
  if errorlevel 1 goto :venv_failed
)
set "PY=.venv\Scripts\python.exe"
echo Installing dependencies ...
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed

if not exist data mkdir data
if not exist reports mkdir reports
if not exist logs mkdir logs

"%PY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 echo [WARNING] tkinter is missing: reinstall Python with "tcl/tk and IDLE" for the GUI. The CLI still works.

rem ---- Ollama + model (optional convenience) ---------------------------------
where ollama >nul 2>&1
if errorlevel 1 goto :no_ollama
"%PY%" -m app.main ping >nul 2>&1
if errorlevel 1 (
  echo Starting Ollama server to check the model ...
  start "Ollama Server" /MIN ollama serve
  ping -n 8 127.0.0.1 >nul
)
ollama list 2>nul | findstr /C:"deepseek-r1:1.5b" >nul
if errorlevel 1 (
  echo Pulling model deepseek-r1:1.5b ^(about 1.1 GB^) ...
  ollama pull deepseek-r1:1.5b
)
echo.
"%PY%" -m app.main doctor
echo.
echo Setup finished. Start the app with START_SELF_IMPROVER.bat
pause
exit /b 0

:no_python
echo [ERROR] Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
echo         ^(tick "Add python.exe to PATH" and keep "tcl/tk" selected^), then run SETUP.bat again.
goto :fail
:venv_failed
echo [ERROR] Could not create the virtual environment.
goto :fail
:pip_failed
echo [ERROR] pip install failed. Check your internet connection and try again.
goto :fail
:no_ollama
echo [WARNING] Ollama not found. Install from https://ollama.com/download then run:  ollama pull deepseek-r1:1.5b
echo.
echo Setup finished ^(without Ollama^).
pause
exit /b 0

:fail
echo.
pause
exit /b 1

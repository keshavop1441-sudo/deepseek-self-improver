@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DeepSeek Self-Improver launcher

rem ---- 1. project Python environment ---------------------------------------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY goto :no_env

rem ---- 2. Ollama installed? ------------------------------------------------
where ollama >nul 2>&1
if errorlevel 1 goto :no_ollama

rem ---- 3. Ollama server running? Start it if not ---------------------------
"%PY%" -m app.main ping >nul 2>&1
if not errorlevel 1 goto :server_ok
echo Ollama server is not running. Starting "ollama serve" ...
start "Ollama Server" /MIN ollama serve
set /a TRIES=0
:wait_server
set /a TRIES+=1
ping -n 2 127.0.0.1 >nul
"%PY%" -m app.main ping >nul 2>&1
if not errorlevel 1 goto :server_ok
if %TRIES% LSS 30 goto :wait_server
goto :server_timeout

:server_ok
rem ---- 4. model installed? --------------------------------------------------
ollama list 2>nul | findstr /C:"deepseek-r1:1.5b" >nul
if errorlevel 1 goto :no_model

rem ---- 5. start the GUI -----------------------------------------------------
echo Starting GUI ...
"%PY%" -m app.main gui
if errorlevel 1 goto :gui_failed
exit /b 0

:no_env
echo [ERROR] Project Python environment not found ^(.venv^).
echo         Double-click SETUP.bat first.
goto :fail
:no_ollama
echo [ERROR] Ollama was not found on PATH.
echo         Install it from https://ollama.com/download and run this launcher again.
goto :fail
:server_timeout
echo [ERROR] Ollama did not respond within 30 seconds.
echo         Try running "ollama serve" in a separate window and look at its output.
goto :fail
:no_model
echo [ERROR] Model deepseek-r1:1.5b is not installed.
echo         Run:  ollama pull deepseek-r1:1.5b
goto :fail
:gui_failed
echo [ERROR] The GUI exited with an error. Run "%PY%" -m app.main doctor for diagnostics,
echo         and see logs\self_improver.log
goto :fail

:fail
echo.
pause
exit /b 1

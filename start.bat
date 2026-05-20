@echo off
setlocal enabledelayedexpansion
title TachiDUBB Studio - AI Video Dubbing

cd /d "%~dp0"

:: ── Check venv ──────────────────────────────────────────────
if not exist venv\Scripts\activate.bat (
    echo.
    echo  [!] Virtual environment not found.
    echo      Please run install.bat first.
    echo.
    pause
    exit /b 1
)

:: ── Add bundled bin (ffmpeg) to PATH ────────────────────────
if exist "%CD%\bin\ffmpeg.exe" set "PATH=%CD%\bin;%PATH%"

:: ── Activate venv ──────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Load .env (skip comments, strip surrounding quotes) ─────
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        set "VAL=%%~b"
        if not "%%a"=="" if not "!VAL!"=="" set "%%a=!VAL!"
    )
)

:: ── Start Ollama if needed (flat, no nested errorlevel) ─────
curl -s -o NUL http://localhost:11434/api/tags
if errorlevel 1 goto :start_ollama
goto :ollama_ready

:start_ollama
where ollama >nul 2>&1
if errorlevel 1 goto :ollama_missing
echo Starting Ollama service...
start /b "" ollama serve >nul 2>&1

:: Wait up to 10s for Ollama
set /a RETRIES=0
:ollama_wait
timeout /t 1 /nobreak >nul
curl -s -o NUL http://localhost:11434/api/tags
if not errorlevel 1 goto :ollama_ready
set /a RETRIES+=1
if !RETRIES! lss 10 goto :ollama_wait
echo   (Ollama did not respond - translation may not work until started)
goto :ollama_ready

:ollama_missing
echo   (Ollama not found - install from https://ollama.com/download/windows)

:ollama_ready

:: ── Launch ─────────────────────────────────────────────────
echo.
echo  ==============================================
echo   TachiDUBB Studio  -  http://localhost:8910
echo   Browser will open automatically
echo   Press Ctrl+C in this window to stop
echo  ==============================================
echo.

python server.py

echo.
echo TachiDUBB Studio has stopped.
pause

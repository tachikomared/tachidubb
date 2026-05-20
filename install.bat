@echo off
setlocal enabledelayedexpansion
title TachiDUBB Studio Installer

:: ════════════════════════════════════════════════════════════════
::  TachiDUBB Studio — Windows One-Click Installer
::  Created by TachikomaRed and smolemaru
::  Installs: Python packages, FFmpeg, yt-dlp, Ollama, VoxCPM2
:: ════════════════════════════════════════════════════════════════

cd /d "%~dp0"

echo.
echo  ============================================================
echo   TachiDUBB Studio Installer - Plug-and-Play AI Video Dubbing
echo  ============================================================
echo.

:: ── Check Python ────────────────────────────────────────────────
echo [1/7] Checking Python...
:: Detect Python - skip Microsoft Store stub, fall back to 'py' launcher
set "PY_CMD="

:: Try python.exe first, but skip if it's the Store stub
where python >nul 2>&1
if not errorlevel 1 (
    :: Store stub responds to --version with a "Python was not found" message.
    :: Real Python responds with "Python X.Y.Z". We detect real Python by checking output.
    for /f "tokens=1,2" %%a in ('python --version 2^>^&1') do (
        if "%%a"=="Python" set "PY_CMD=python"
    )
)

:: Fall back to py launcher if python isn't real
if "%PY_CMD%"=="" (
    where py >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=1,2" %%a in ('py -3 --version 2^>^&1') do (
            if "%%a"=="Python" set "PY_CMD=py -3"
        )
    )
)

if "%PY_CMD%"=="" (
    echo.
    echo  [!] Python not found (or Microsoft Store stub detected).
    echo      Please install Python 3.10, 3.11, or 3.12 from:
    echo      https://www.python.org/downloads/
    echo.
    echo      [IMPORTANT] During install, check:
    echo         - "Add Python to PATH"
    echo         - "Install for all users" (optional but recommended)
    echo      And DISABLE the "python.exe was not found" Microsoft Store redirector:
    echo         Settings ^> Apps ^> App Execution Aliases ^> turn off python.exe
    echo.
    start https://www.python.org/downloads/release/python-3120/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PY_CMD% --version 2^>^&1') do set PYVER=%%v
echo      Python !PYVER! found  (using: %PY_CMD%)

:: Validate Python version (3.10-3.12)
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)
if !PYMAJ! NEQ 3 goto bad_python
if !PYMIN! LSS 10 goto bad_python
if !PYMIN! GTR 12 goto bad_python
goto python_ok

:bad_python
echo.
echo  [!] Python !PYVER! is not supported. Need 3.10, 3.11, or 3.12.
echo      VoxCPM2 requires Python 3.10 - 3.12.
echo.
echo      Download compatible version: https://www.python.org/downloads/release/python-3120/
pause
exit /b 1

:python_ok

:: ── FFmpeg ──────────────────────────────────────────────────────
echo.
echo [2/7] Checking FFmpeg...
where ffmpeg >nul 2>&1
if not errorlevel 1 goto :ffmpeg_ok
if exist "bin\ffmpeg.exe" goto :ffmpeg_ok

echo      FFmpeg not found. Installing to .\bin ...
if not exist bin mkdir bin
echo      Downloading ffmpeg-release-essentials.zip (~90 MB)...
powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'bin\ffmpeg.zip' -UseBasicParsing } catch { exit 1 }"
if errorlevel 1 (
    echo  [!] FFmpeg download failed. Install manually: https://ffmpeg.org/download.html
    pause
    exit /b 1
)

echo      Extracting...
powershell -NoProfile -Command "Expand-Archive -Path 'bin\ffmpeg.zip' -DestinationPath 'bin' -Force"
del /q bin\ffmpeg.zip

:: Move ffmpeg/ffprobe to bin root (flattened - no parens)
for /d %%d in (bin\ffmpeg-*) do call :move_ffmpeg "%%d"
goto :ffmpeg_done

:move_ffmpeg
move /y "%~1\bin\ffmpeg.exe" bin\ >nul 2>&1
move /y "%~1\bin\ffprobe.exe" bin\ >nul 2>&1
rmdir /s /q "%~1" 2>nul
exit /b 0

:ffmpeg_done
echo      FFmpeg installed to .\bin
goto :after_ffmpeg

:ffmpeg_ok
echo      FFmpeg found

:after_ffmpeg

:: ── Virtual environment ─────────────────────────────────────────
echo.
echo [3/7] Creating Python virtual environment...
if not exist venv (
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo  [!] venv creation failed
        pause
        exit /b 1
    )
    echo      venv created
) else (
    echo      venv exists
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel setuptools --quiet

:: ── Core Python packages ────────────────────────────────────────
echo.
echo [4/7] Installing core Python packages...
echo      (this takes 2-5 minutes, please wait)

pip install --quiet ^
    fastapi ^
    "uvicorn[standard]" ^
    python-multipart ^
    httpx ^
    soundfile ^
    numpy ^
    pydub ^
    nltk ^
    yt-dlp ^
    edge-tts
if errorlevel 1 (
    echo  [!] Core package install failed
    pause
    exit /b 1
)
echo      Core packages OK

:: ── PyTorch with CUDA ───────────────────────────────────────────
echo.
echo [5/7] Installing PyTorch (CUDA 12.1)...
python -c "import torch; print(torch.__version__)" >nul 2>&1
if errorlevel 1 (
    echo      Installing PyTorch 2.5+ with CUDA 12.1...
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo      PyTorch already installed
)

:: Whisper
echo      Installing faster-whisper...
pip install --quiet faster-whisper
if errorlevel 1 (
    echo      Fallback to openai-whisper...
    pip install --quiet openai-whisper
)
echo      Whisper OK

:: ── VoxCPM2 ─────────────────────────────────────────────────────
echo.
echo [6/7] Installing VoxCPM2 (voice cloning)...
echo      (downloads ~5GB model on first synthesis)
pip install --quiet voxcpm
if errorlevel 1 (
    echo  [!] VoxCPM2 install failed - falling back to edge-tts only
    echo      You can try manually: pip install voxcpm
) else (
    echo      VoxCPM2 OK
)

:: ── Speaker diarization (optional) ──────────────────────────────
echo.
echo [+] Optional: Speaker diarization (multi-speaker voice cloning)
echo     Needs a free HuggingFace token from https://huggingface.co/settings/tokens
echo     and acceptance of https://huggingface.co/pyannote/speaker-diarization-3.1
echo.
set "DIAR_CHOICE="
set /p DIAR_CHOICE="Install pyannote.audio for multi-speaker support? [y/N]: "
if /i "%DIAR_CHOICE%"=="y" (
    echo      Installing pyannote.audio...
    pip install --quiet pyannote.audio
    if errorlevel 1 (
        echo  [!] pyannote install failed - single-speaker mode will still work
    ) else (
        echo      pyannote installed
        set "HF_INPUT="
        set /p HF_INPUT="Enter your HuggingFace token (or press Enter to skip): "
        if not "!HF_INPUT!"=="" (
            echo HF_TOKEN=!HF_INPUT! > .env
            echo      Token saved to .env
        )
    )
)

:: ── Ollama ──────────────────────────────────────────────────────
echo.
echo [7/7] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo      Ollama not installed.
    echo.
    echo      Opening Ollama download page...
    start https://ollama.com/download/windows
    echo.
    echo  [!] Install Ollama, then re-run this installer to pull a model.
    echo      You can also pull models from the UI's System tab.
    goto :nltk_setup
)

echo      Ollama installed

:: Start Ollama if not running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo      Starting Ollama service...
    start /b "" ollama serve >nul 2>&1
    timeout /t 4 /nobreak >nul
)

echo.
echo  ========================================
echo   Pulling default translation model
echo  ========================================
echo  Choose a model:
echo    1) qwen3:8b  - Recommended  ^(5GB,  ~6GB VRAM^)
echo    2) qwen3:14b - Best quality ^(8GB, ~10GB VRAM^)
echo    3) qwen3:4b  - Lightweight  ^(3GB,  ~3GB VRAM^)
echo    4) Skip ^(pull later from UI^)
echo.
set "MODEL_CHOICE="
set /p MODEL_CHOICE="Select [1-4, default=1]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"

if "%MODEL_CHOICE%"=="1" ollama pull qwen3:8b
if "%MODEL_CHOICE%"=="2" ollama pull qwen3:14b
if "%MODEL_CHOICE%"=="3" ollama pull qwen3:4b

:nltk_setup
:: ── NLTK data ───────────────────────────────────────────────────
echo.
echo Downloading NLTK sentence tokenizer data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

:: ── Create folders ──────────────────────────────────────────────
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs
if not exist jobs_db mkdir jobs_db

:: ── Done ────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Installation Complete!
echo  ============================================================
echo.
echo   Launch with:  start.bat
echo   Or:           double-click start.bat
echo.
echo   The app will open automatically at http://localhost:8910
echo.
echo   First synthesis downloads VoxCPM2 weights (~5GB) - one time.
echo.
pause

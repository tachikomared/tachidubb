@echo off
echo === TachiDUBB Studio Diagnostic ===
echo.
echo Current directory: %CD%
echo.
echo Step 1: Check Python
where python
echo Return code: %errorlevel%
echo.
echo Step 2: Python version
python --version
echo Return code: %errorlevel%
echo.
echo Step 3: Check py launcher
where py
echo Return code: %errorlevel%
echo.
echo Step 4: Check curl
where curl
echo Return code: %errorlevel%
echo.
echo Step 5: Check PowerShell
where powershell
echo Return code: %errorlevel%
echo.
echo === End of diagnostic ===
echo.
echo Press any key to close...
pause >nul

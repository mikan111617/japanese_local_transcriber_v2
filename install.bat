@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ==========================================
echo Japanese Local Transcriber - Setup
echo ==========================================
echo.
echo Folder:
echo %CD%
echo.

where py >nul 2>&1
if not errorlevel 1 goto USE_PY

where python >nul 2>&1
if not errorlevel 1 goto USE_PYTHON

echo [ERROR] Python was not found.
echo.
echo Please install Python 3.10 - 3.12, then run this file again.
echo If Python is already installed, enable "Add Python to PATH".
echo.
goto ERROR_END

:USE_PY
set "PYTHON_CMD=py -3"
goto PYTHON_FOUND

:USE_PYTHON
set "PYTHON_CMD=python"
goto PYTHON_FOUND

:PYTHON_FOUND
echo Python:
%PYTHON_CMD% --version
if errorlevel 1 goto ERROR_END
echo.

if exist ".venv\Scripts\python.exe" goto VENV_READY

echo [1/3] Creating virtual environment...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto ERROR_END

:VENV_READY
echo [2/3] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto ERROR_END

echo [3/3] Installing libraries...
python -m pip install --upgrade pip
if errorlevel 1 goto ERROR_END

python -m pip install -r requirements.txt
if errorlevel 1 goto ERROR_END

echo.
echo ==========================================
echo SETUP COMPLETE
echo ==========================================
echo.
echo Next: double-click run.bat
echo.
pause
exit /b 0

:ERROR_END
echo.
echo ==========================================
echo SETUP FAILED
echo ==========================================
echo.
echo The window will stay open.
echo Please take a screenshot of the error above if you need help.
echo.
pause
exit /b 1

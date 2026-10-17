@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ==========================================
echo Japanese Local Transcriber v2
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" goto NO_VENV

echo Existing environment found.
echo Starting app...
call ".venv\Scripts\activate.bat"
python app.py
if errorlevel 1 goto ERROR_END
exit /b 0

:NO_VENV
echo Setup has not been completed yet.
echo Running installer...
call install.bat
if errorlevel 1 goto ERROR_END
call ".venv\Scripts\activate.bat"
python app.py
if errorlevel 1 goto ERROR_END
exit /b 0

:ERROR_END
echo.
echo The app stopped with an error.
pause
exit /b 1

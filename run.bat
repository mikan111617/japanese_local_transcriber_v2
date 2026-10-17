@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ==========================================
echo Japanese Local Transcriber
echo ==========================================
echo.

if exist ".venv\Scripts\python.exe" goto RUN_APP

echo [ERROR] Setup has not been completed.
echo Please run START_HERE.bat first.
echo.
pause
exit /b 1

:RUN_APP
call ".venv\Scripts\activate.bat"
python app.py

if errorlevel 1 goto APP_ERROR
exit /b 0

:APP_ERROR
echo.
echo [ERROR] The app exited with an error.
echo Please take a screenshot of the error above.
echo.
pause
exit /b 1

@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ==========================================
echo NVIDIA GPU Setup - CUDA 12 / cuDNN 9
echo ==========================================
echo Close the transcriber before installing.
echo Runtime libraries may require several GB of disk space.
echo An NVIDIA GPU and a compatible NVIDIA driver are required.
echo.
if exist ".venv\Scripts\python.exe" goto INSTALL_GPU
call install.bat
if errorlevel 1 goto ERROR_END

:INSTALL_GPU
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-gpu.txt
if errorlevel 1 goto ERROR_END
".venv\Scripts\python.exe" gpu_runtime.py
if errorlevel 1 goto CHECK_FAILED
echo.
echo GPU setup complete. Start run.bat and select NVIDIA GPU.
pause
exit /b 0

:CHECK_FAILED
echo.
echo Libraries installed, but GPU validation did not pass.
echo Check the NVIDIA driver and the diagnostic message above.
echo You can still select Auto or CPU in the app.
pause
exit /b 1

:ERROR_END
echo.
echo GPU setup failed. Check the error above.
pause
exit /b 1

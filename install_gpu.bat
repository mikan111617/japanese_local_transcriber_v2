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
pause
if exist ".venv\Scripts\python.exe" goto INSTALL_GPU
call install.bat
if errorlevel 1 goto ERROR_END

:INSTALL_GPU
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-gpu.txt
if errorlevel 1 goto ERROR_END
".venv\Scripts\python.exe" gpu_runtime.py
if not errorlevel 1 goto DONE
if not errorlevel 2 goto CHECK_FAILED

echo.
echo Some NVIDIA DLL files are missing. Reinstalling the NVIDIA runtime...
".venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps -r requirements-nvidia.txt
if errorlevel 1 goto ERROR_END
".venv\Scripts\python.exe" gpu_runtime.py
if errorlevel 1 goto CHECK_FAILED

:DONE
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
echo If the transcriber is open, close it and run install_gpu.bat again.
echo Also check that several GB of disk space are free.
pause
exit /b 1

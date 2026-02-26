@echo off
setlocal
cd /d "%~dp0"

REM =============================
REM Runtime folders (gitignore .runtime/)
REM =============================
set "RUNTIME_DIR=.runtime"
set "VENV_DIR=%RUNTIME_DIR%\venv"
set "ULTRA_CFG_DIR=%RUNTIME_DIR%\ultralytics_config"
set "WEIGHTS_DIR=%RUNTIME_DIR%\weights"
set "RUNS_DIR=%RUNTIME_DIR%\runs"
set "DATASETS_DIR=%RUNTIME_DIR%\datasets"
set "PIP_CACHE_DIR=%RUNTIME_DIR%\pip-cache"

REM Defaults are handled inside webcam.py (preset 'yolo'):
REM   CPU -> yolov8n + yolov8n-pose
REM   GPU -> yolo26x + yolo26x-pose


REM =============================
REM Preconditions: winget + python
REM =============================
where winget >nul 2>&1
if errorlevel 1 goto ERR_WINGET

where python >nul 2>&1
if errorlevel 1 (
  echo WARNING: Python not found. Installing via winget...
  winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
)

where python >nul 2>&1
if errorlevel 1 goto ERR_PYTHON

REM =============================
REM Create runtime dirs
REM =============================
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
if not exist "%ULTRA_CFG_DIR%" mkdir "%ULTRA_CFG_DIR%"
if not exist "%WEIGHTS_DIR%" mkdir "%WEIGHTS_DIR%"
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"
if not exist "%DATASETS_DIR%" mkdir "%DATASETS_DIR%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"

REM Move old root-downloaded weights into .runtime\weights (if they exist)
if exist "%CD%\yolo26x.pt" move /Y "%CD%\yolo26x.pt" "%WEIGHTS_DIR%\yolo26x.pt" >nul
if exist "%CD%\yolo26x-pose.pt" move /Y "%CD%\yolo26x-pose.pt" "%WEIGHTS_DIR%\yolo26x-pose.pt" >nul
if exist "%CD%\sam32.pt" move /Y "%CD%\sam32.pt" "%WEIGHTS_DIR%\sam32.pt" >nul
if exist "%CD%\sam32-pose.pt" move /Y "%CD%\sam32-pose.pt" "%WEIGHTS_DIR%\sam32-pose.pt" >nul

REM =============================
REM Create venv inside .runtime
REM =============================
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv in %VENV_DIR% ...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 goto ERR_VENV
)

REM Ensure venv python/pip are used
set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
set "PIP_CACHE_DIR=%CD%\%PIP_CACHE_DIR%"
set "YOLO_CONFIG_DIR=%CD%\%ULTRA_CFG_DIR%"

python -m pip install --upgrade pip
if errorlevel 1 goto ERR_PIP

REM =============================
REM Torch: only (re)install CUDA wheels if CUDA not available
REM =============================
set "NEED_TORCH=0"
python -c "import torch; print('ok')" > "%RUNTIME_DIR%\torch_check.txt" 2>nul
if errorlevel 1 set "NEED_TORCH=1"

if "%NEED_TORCH%"=="0" (
  python -c "import torch; print('1' if torch.cuda.is_available() else '0')" > "%RUNTIME_DIR%\cuda_check.txt" 2>nul
  if errorlevel 1 set "NEED_TORCH=1"
)

if "%NEED_TORCH%"=="0" (
  set /p CUDA_OK=<"%RUNTIME_DIR%\cuda_check.txt"
  if "%CUDA_OK%"=="1" goto TORCH_DONE
)

echo Installing PyTorch CUDA wheels (best-effort)...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1

echo   Trying cu128...
python -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo   cu128 failed, trying cu126...
  python -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
)
if errorlevel 1 (
  echo   cu126 failed, trying cu118...
  python -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118
)
if errorlevel 1 (
  echo WARNING: CUDA wheels install failed. Installing CPU torch/torchvision fallback...
  python -m pip install --upgrade torch torchvision
)
:TORCH_DONE

REM =============================
REM Ultralytics + deps
REM =============================
python -m pip install ultralytics opencv-python numpy "lap>=0.5.12"
if errorlevel 1 goto ERR_DEPS

REM Configure Ultralytics dirs to stay in .runtime
python -c "from ultralytics import settings; settings.update({'weights_dir': r'%CD%\%WEIGHTS_DIR%','runs_dir': r'%CD%\%RUNS_DIR%','datasets_dir': r'%CD%\%DATASETS_DIR%','sync': False});"
if errorlevel 1 goto ERR_ULTRA_SETTINGS

REM =============================
REM Choose device: GPU(0) if CUDA works, else CPU
REM =============================
python -c "import torch; print('1' if torch.cuda.is_available() else '0')" > "%RUNTIME_DIR%\cuda_check2.txt" 2>nul
set "ULTRA_DEVICE=cpu"
set /p CUDA_OK2=<"%RUNTIME_DIR%\cuda_check2.txt"
if "%CUDA_OK2%"=="1" set "ULTRA_DEVICE=0"

echo torch.cuda.is_available(): %CUDA_OK2%  (using --device %ULTRA_DEVICE%)

REM =============================
REM Virtual camera testing note
REM =============================
REM Easiest option: OBS Studio (open source) -> Start Virtual Camera.
REM Then pick the OBS Virtual Camera device in Windows Camera settings/apps.
REM (This script can't create kernel-level virtual cams on Windows by itself.)

REM Optional NVIDIA info
where nvidia-smi >nul 2>&1
if not errorlevel 1 nvidia-smi

REM List cameras (Windows)
echo Camera devices (Windows):
powershell -NoProfile -Command "Get-PnpDevice -PresentOnly | ?{ $_.Class -match 'Camera|Image' } | Select-Object Status,Class,FriendlyName | Format-Table -AutoSize"

if not exist "webcam.py" goto ERR_WEBCAM

REM =============================
REM Run (weights auto-download into .runtime\weights because paths point there)
REM =============================
REM =============================
REM Build defaults only if user didn't pass them
REM =============================
set "ARGS=%*"

set "DEF_SOURCE=--source 0"
echo %ARGS% | findstr /c:"--source" >nul && set "DEF_SOURCE="

set "DEF_DEVICE=--device %ULTRA_DEVICE%"
echo %ARGS% | findstr /c:"--device" >nul && set "DEF_DEVICE="

set "DEF_POSE=--use-pose"
echo %ARGS% | findstr /c:"--use-pose" >nul && set "DEF_POSE="
echo %ARGS% | findstr /c:"--no-pose" >nul && set "DEF_POSE="

set "DEF_FPS=--max-fps 120"
echo %ARGS% | findstr /c:"--max-fps" >nul && set "DEF_FPS="

echo Running...
python webcam.py %DEF_SOURCE% %DEF_DEVICE% %DEF_POSE% %DEF_FPS% %*
exit /b %ERRORLEVEL%

REM =============================
REM Errors
REM =============================
:ERR_WINGET
echo ERROR: winget not found. Install/update "App Installer" from Microsoft Store.
exit /b 1

:ERR_PYTHON
echo ERROR: Python not found. Open a NEW terminal (PATH refresh) and run again.
exit /b 1

:ERR_VENV
echo ERROR: Failed to create venv.
exit /b 1

:ERR_PIP
echo ERROR: pip failed inside the venv.
exit /b 1

:ERR_DEPS
echo ERROR: Failed installing ultralytics/opencv/numpy/lap.
exit /b 1

:ERR_ULTRA_SETTINGS
echo ERROR: Failed to apply Ultralytics settings.
exit /b 1

:ERR_WEBCAM
echo ERROR: webcam.py not found in %CD%
exit /b 1

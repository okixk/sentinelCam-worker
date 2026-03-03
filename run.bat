@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CFG_FILE=%~dp0webcam.properties"
if not exist "%CFG_FILE%" goto ERR_CFG

for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%CFG_FILE%") do set "%%A=%%B"

if not defined RUNTIME_DIR set "RUNTIME_DIR=.runtime"
if not defined VENV_SUBDIR set "VENV_SUBDIR=venv"
if not defined ULTRA_CFG_SUBDIR set "ULTRA_CFG_SUBDIR=ultralytics_config"
if not defined WEIGHTS_SUBDIR set "WEIGHTS_SUBDIR=weights"
if not defined RUNS_SUBDIR set "RUNS_SUBDIR=runs"
if not defined DATASETS_SUBDIR set "DATASETS_SUBDIR=datasets"
if not defined PIP_CACHE_SUBDIR set "PIP_CACHE_SUBDIR=pip-cache"

if not defined DEFAULT_SOURCE_WINDOWS set "DEFAULT_SOURCE_WINDOWS=0"
if not defined DEFAULT_DEVICE set "DEFAULT_DEVICE=auto"
if not defined DEFAULT_USE_POSE set "DEFAULT_USE_POSE=1"
if not defined DEFAULT_MAX_FPS set "DEFAULT_MAX_FPS=120"

REM =============================
REM Runtime folders (gitignore .runtime/)
REM =============================
set "VENV_DIR=%RUNTIME_DIR%\%VENV_SUBDIR%"
set "ULTRA_CFG_DIR=%RUNTIME_DIR%\%ULTRA_CFG_SUBDIR%"
set "WEIGHTS_DIR=%RUNTIME_DIR%\%WEIGHTS_SUBDIR%"
set "RUNS_DIR=%RUNTIME_DIR%\%RUNS_SUBDIR%"
set "DATASETS_DIR=%RUNTIME_DIR%\%DATASETS_SUBDIR%"
set "PIP_CACHE_DIR=%RUNTIME_DIR%\%PIP_CACHE_SUBDIR%"

REM =============================
REM Preconditions: winget + python
REM =============================
where winget >nul 2>&1 || goto ERR_WINGET

where python >nul 2>&1
if errorlevel 1 (
  echo WARNING: Python not found. Installing via winget...
  winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
)

where python >nul 2>&1 || goto ERR_PYTHON

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
if exist "%CD%\yolo26x.pt"      move /Y "%CD%\yolo26x.pt"      "%WEIGHTS_DIR%\yolo26x.pt" >nul
if exist "%CD%\yolo26x-pose.pt" move /Y "%CD%\yolo26x-pose.pt" "%WEIGHTS_DIR%\yolo26x-pose.pt" >nul
if exist "%CD%\sam32.pt"        move /Y "%CD%\sam32.pt"        "%WEIGHTS_DIR%\sam32.pt" >nul
if exist "%CD%\sam32-pose.pt"   move /Y "%CD%\sam32-pose.pt"   "%WEIGHTS_DIR%\sam32-pose.pt" >nul

REM =============================
REM Create venv inside .runtime (only if missing)
REM =============================
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv in %VENV_DIR% ...
  python -m venv "%VENV_DIR%" || goto ERR_VENV
)

REM Use venv python/pip
set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
set "PIP_CACHE_DIR=%CD%\%PIP_CACHE_DIR%"
set "YOLO_CONFIG_DIR=%CD%\%ULTRA_CFG_DIR%"

REM pip upgrade only if pip is old/broken (safe to run, but not "everything")
python -m pip --version >nul 2>&1 || goto ERR_PIP
python -m pip install --upgrade pip >nul 2>&1

REM =============================
REM Detect NVIDIA presence (best-effort)
REM =============================
set "HAVE_NVIDIA=0"
where nvidia-smi >nul 2>&1
if not errorlevel 1 set "HAVE_NVIDIA=1"

REM =============================
REM Torch check: install ONLY if missing OR (NVIDIA present and CUDA not available)
REM =============================
set "NEED_TORCH=0"
python -c "import torch, torchvision" >nul 2>&1
if errorlevel 1 set "NEED_TORCH=1"

if "%NEED_TORCH%"=="0" (
  if "%HAVE_NVIDIA%"=="1" (
    python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 set "NEED_TORCH=1"
  )
)

if "%NEED_TORCH%"=="1" goto INSTALL_TORCH
echo Torch OK - skipping install.
goto AFTER_TORCH

:INSTALL_TORCH
echo Installing PyTorch (only because missing/broken or CUDA not working)...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1

if "%HAVE_NVIDIA%"=="1" (
  echo   NVIDIA detected: trying CUDA wheels...
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
    echo WARNING: CUDA wheels failed. Installing CPU fallback...
    python -m pip install --upgrade torch torchvision || goto ERR_TORCH
  )
) else (
  echo   No NVIDIA detected: installing CPU torch...
  python -m pip install --upgrade torch torchvision || goto ERR_TORCH
)

:AFTER_TORCH

REM =============================
REM Ultralytics + deps check (install ONLY if missing)
REM =============================
set "NEED_DEPS=0"
python -c "import ultralytics, cv2, numpy, lap" >nul 2>&1
if errorlevel 1 set "NEED_DEPS=1"

if "%NEED_DEPS%"=="1" goto INSTALL_DEPS
echo Ultralytics/OpenCV/Numpy/LAP OK - skipping install.
goto AFTER_DEPS

:INSTALL_DEPS
echo Installing Ultralytics + deps (only because missing)...
python -m pip install ultralytics opencv-python numpy "lap>=0.5.12" || goto ERR_DEPS

:AFTER_DEPS

REM =============================
REM Configure Ultralytics dirs to stay in .runtime (no cloud sync)
REM (write tiny python file to avoid CMD quoting issues)
REM =============================
set "CFG_PY=%RUNTIME_DIR%\ultra_cfg.py"
echo from ultralytics import settings> "%CFG_PY%"
echo settings.update({>> "%CFG_PY%"
echo     "weights_dir": r"%CD%\%WEIGHTS_DIR%",>> "%CFG_PY%"
echo     "runs_dir": r"%CD%\%RUNS_DIR%",>> "%CFG_PY%"
echo     "datasets_dir": r"%CD%\%DATASETS_DIR%",>> "%CFG_PY%"
echo     "sync": False>> "%CFG_PY%"
echo })>> "%CFG_PY%"
python "%CFG_PY%" || goto ERR_ULTRA_SETTINGS

REM =============================
REM Choose device: GPU(0) if CUDA works, else CPU
REM =============================
set "ULTRA_DEVICE=cpu"
python -c "import torch; print('1' if torch.cuda.is_available() else '0')" > "%RUNTIME_DIR%\cuda_ok.txt" 2>nul
set /p CUDA_OK=<"%RUNTIME_DIR%\cuda_ok.txt"
if "%CUDA_OK%"=="1" set "ULTRA_DEVICE=0"

echo torch.cuda.is_available(): %CUDA_OK%  (using --device %ULTRA_DEVICE%)

REM Optional NVIDIA info
where nvidia-smi >nul 2>&1
if not errorlevel 1 nvidia-smi

REM List cameras (optional; comment out if you want)
echo Camera devices (Windows):
powershell -NoProfile -Command "Get-PnpDevice -PresentOnly | Where-Object { $_.Class -match 'Camera|Image' } | Select-Object Status,Class,FriendlyName | Format-Table -AutoSize"

if not exist "webcam.py" goto ERR_WEBCAM

REM =============================
REM Run with defaults only if user didn't pass them
REM =============================
set "ARGS=%*"

REM IMPORTANT: NO single quotes around RTSP URL in CMD!
set "DEF_SOURCE=--source %DEFAULT_SOURCE_WINDOWS%"
echo %ARGS% | findstr /c:"--source" >nul
if not errorlevel 1 set "DEF_SOURCE="

set "DEF_DEVICE=--device %DEFAULT_DEVICE%"
echo %ARGS% | findstr /c:"--device" >nul
if not errorlevel 1 set "DEF_DEVICE="

set "DEF_POSE="
if "%DEFAULT_USE_POSE%"=="1" set "DEF_POSE=--use-pose"
echo %ARGS% | findstr /c:"--use-pose" >nul
if not errorlevel 1 set "DEF_POSE="
echo %ARGS% | findstr /c:"--no-pose" >nul
if not errorlevel 1 set "DEF_POSE="

set "DEF_FPS=--max-fps %DEFAULT_MAX_FPS%"
echo %ARGS% | findstr /c:"--max-fps" >nul
if not errorlevel 1 set "DEF_FPS="

echo Running...
python webcam.py %DEF_SOURCE% %DEF_DEVICE% %DEF_POSE% %DEF_FPS% %*
exit /b %ERRORLEVEL%

REM =============================
REM Errors
REM =============================
:ERR_CFG
echo ERROR: Shared config not found: %CFG_FILE%
exit /b 1

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

:ERR_TORCH
echo ERROR: Failed installing torch/torchvision.
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

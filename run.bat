@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "CFG_FILE=%~dp0webcam.properties"
if not exist "%CFG_FILE%" goto ERR_CFG

for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%CFG_FILE%") do set "%%A=%%B"

REM =============================
REM Defaults (if missing in webcam.properties)
REM =============================
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

if not defined DEFAULT_PRESET_CPU set "DEFAULT_PRESET_CPU=yolov8n"
if not defined DEFAULT_PRESET_ACCEL set "DEFAULT_PRESET_ACCEL=yolo26x"

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
REM Interactive options
REM   -s / --silent     : no prompts, always defaults
REM   --install         : force install step
REM   --no-install      : skip install step
REM =============================
set "SILENT=0"
REM ASK|FORCE|SKIP
set "INSTALL_MODE=ASK"
set "SHOW_HELP=0"

for %%A in (%*) do (
  if /i "%%~A"=="-s" set "SILENT=1"
  if /i "%%~A"=="--silent" set "SILENT=1"
  if /i "%%~A"=="--install" set "INSTALL_MODE=FORCE"
  if /i "%%~A"=="--no-install" set "INSTALL_MODE=SKIP"
  if /i "%%~A"=="-h" set "SHOW_HELP=1"
  if /i "%%~A"=="--help" set "SHOW_HELP=1"
)

if "%SHOW_HELP%"=="1" goto HELP

set "DO_INSTALL=1"
if /i "%INSTALL_MODE%"=="SKIP" set "DO_INSTALL=0"
if /i "%INSTALL_MODE%"=="FORCE" set "DO_INSTALL=1"
if /i "%INSTALL_MODE%"=="ASK" (
  if "%SILENT%"=="0" (
    set /p "INSTALL_ANS=Run setup/install step (venv + pip deps)? [Y/n]: "
    if "!INSTALL_ANS!"=="" set "INSTALL_ANS=Y"
    if /i "!INSTALL_ANS!"=="N" set "DO_INSTALL=0"
  )
)

REM =============================
REM Build forwarded args (strip run.bat-only flags)
REM =============================
set "FWD_ARGS="
call :BUILD_FWD %*

REM =============================
REM Create runtime dirs (always)
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
REM Install / setup (optional)
REM =============================
if "%DO_INSTALL%"=="0" goto SKIP_INSTALL

where winget >nul 2>&1 || goto ERR_WINGET

where python >nul 2>&1
if errorlevel 1 (
  echo WARNING: Python not found. Installing via winget...
  winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
)

where python >nul 2>&1 || goto ERR_PYTHON

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv in %VENV_DIR% ...
  python -m venv "%VENV_DIR%" || goto ERR_VENV
)

set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
set "PIP_CACHE_DIR=%CD%\%PIP_CACHE_DIR%"
set "YOLO_CONFIG_DIR=%CD%\%ULTRA_CFG_DIR%"

python -m pip --version >nul 2>&1 || goto ERR_PIP
python -m pip install --upgrade pip >nul 2>&1

set "HAVE_NVIDIA=0"
where nvidia-smi >nul 2>&1
if not errorlevel 1 set "HAVE_NVIDIA=1"

REM Torch check: install only if missing OR (NVIDIA present and CUDA not available)
python -c "import torch, torchvision" >nul 2>&1
if errorlevel 1 goto INSTALL_TORCH
if "%HAVE_NVIDIA%"=="1" (
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
  if errorlevel 1 goto INSTALL_TORCH
)

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

REM Ultralytics + deps check (install only if missing)
python -c "import ultralytics, cv2, numpy, lap" >nul 2>&1
if errorlevel 1 (
  echo Installing Ultralytics + deps (only because missing)...
  python -m pip install ultralytics opencv-python numpy "lap>=0.5.12" || goto ERR_DEPS
) else (
  echo Ultralytics/OpenCV/Numpy/LAP OK - skipping install.
)

call :CONFIG_ULTRA_STRICT || goto ERR_ULTRA_SETTINGS

goto AFTER_INSTALL

:SKIP_INSTALL
echo Skipping setup/install step (--no-install). Assuming python + deps already exist.
where python >nul 2>&1 || goto ERR_PYTHON
if exist "%VENV_DIR%\Scripts\python.exe" (
  set "PATH=%CD%\%VENV_DIR%\Scripts;%PATH%"
)
set "YOLO_CONFIG_DIR=%CD%\%ULTRA_CFG_DIR%"
call :CONFIG_ULTRA_BESTEFFORT

:AFTER_INSTALL

REM =============================
REM Choose device: GPU(0) if CUDA works, else CPU
REM =============================
set "ULTRA_DEVICE=cpu"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 set "ULTRA_DEVICE=0"

REM Optional NVIDIA info
where nvidia-smi >nul 2>&1
if not errorlevel 1 nvidia-smi

REM =============================
REM Interactive: camera selection + preset
REM =============================
set "CHOSEN_SOURCE=%DEFAULT_SOURCE_WINDOWS%"
echo %FWD_ARGS% | findstr /i /c:"--source" >nul
if errorlevel 1 (
  if "%SILENT%"=="0" (
    echo.
    echo Choose camera source (press Enter for default):
    echo   - number like 0, 1, 2 ... (DirectShow index)
    echo   - or URL like rtsp://... / http://... / file path
    set /p SRC_IN=Camera source [!CHOSEN_SOURCE!]: 
    if not "!SRC_IN!"=="" set "CHOSEN_SOURCE=!SRC_IN!"
  )
)

set "CHOSEN_PRESET=%DEFAULT_PRESET_CPU%"
if "%ULTRA_DEVICE%"=="0" set "CHOSEN_PRESET=%DEFAULT_PRESET_ACCEL%"

echo %FWD_ARGS% | findstr /i /c:"--preset" >nul
if errorlevel 1 (
  if "%SILENT%"=="0" (
    echo.
    echo Choose model preset (press Enter for default). Tip: 'yolo' = auto CPU/GPU.
    python webcam.py --list-presets 2>nul
    set /p PRESET_IN=Preset [!CHOSEN_PRESET!]: 
    if not "!PRESET_IN!"=="" set "CHOSEN_PRESET=!PRESET_IN!"
  )
)

REM =============================
REM Run with defaults only if user didn't pass them
REM =============================
set "DEF_SOURCE=--source %CHOSEN_SOURCE%"
echo %FWD_ARGS% | findstr /i /c:"--source" >nul
if not errorlevel 1 set "DEF_SOURCE="

set "DEF_DEVICE=--device %DEFAULT_DEVICE%"
echo %FWD_ARGS% | findstr /i /c:"--device" >nul
if not errorlevel 1 set "DEF_DEVICE="

set "DEF_PRESET=--preset %CHOSEN_PRESET%"
echo %FWD_ARGS% | findstr /i /c:"--preset" >nul
if not errorlevel 1 set "DEF_PRESET="

set "DEF_POSE=--use-pose"
echo %FWD_ARGS% | findstr /i /c:"--use-pose" >nul
if not errorlevel 1 set "DEF_POSE="
echo %FWD_ARGS% | findstr /i /c:"--no-pose" >nul
if not errorlevel 1 set "DEF_POSE="
if "%DEFAULT_USE_POSE%"=="0" set "DEF_POSE="

set "DEF_FPS=--max-fps %DEFAULT_MAX_FPS%"
echo %FWD_ARGS% | findstr /i /c:"--max-fps" >nul
if not errorlevel 1 set "DEF_FPS="

echo Running...
python webcam.py %DEF_SOURCE% %DEF_DEVICE% %DEF_PRESET% %DEF_POSE% %DEF_FPS% %FWD_ARGS%
exit /b %ERRORLEVEL%

REM =============================
REM Subroutines
REM =============================
:BUILD_FWD
if "%~1"=="" goto :eof
if /i "%~1"=="-s" (shift & goto BUILD_FWD)
if /i "%~1"=="--silent" (shift & goto BUILD_FWD)
if /i "%~1"=="--install" (shift & goto BUILD_FWD)
if /i "%~1"=="--no-install" (shift & goto BUILD_FWD)
if /i "%~1"=="-h" (shift & goto BUILD_FWD)
if /i "%~1"=="--help" (shift & goto BUILD_FWD)
set "FWD_ARGS=%FWD_ARGS% %~1"
shift
goto BUILD_FWD

:CONFIG_ULTRA_STRICT
set "CFG_PY=%RUNTIME_DIR%\ultra_cfg.py"
> "%CFG_PY%" echo from ultralytics import settings
>> "%CFG_PY%" echo settings.update({
>> "%CFG_PY%" echo     "weights_dir": r"%CD%\%WEIGHTS_DIR%",
>> "%CFG_PY%" echo     "runs_dir": r"%CD%\%RUNS_DIR%",
>> "%CFG_PY%" echo     "datasets_dir": r"%CD%\%DATASETS_DIR%",
>> "%CFG_PY%" echo     "sync": False
>> "%CFG_PY%" echo })
python "%CFG_PY%" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:CONFIG_ULTRA_BESTEFFORT
set "CFG_PY=%RUNTIME_DIR%\ultra_cfg.py"
> "%CFG_PY%" echo from ultralytics import settings
>> "%CFG_PY%" echo settings.update({
>> "%CFG_PY%" echo     "weights_dir": r"%CD%\%WEIGHTS_DIR%",
>> "%CFG_PY%" echo     "runs_dir": r"%CD%\%RUNS_DIR%",
>> "%CFG_PY%" echo     "datasets_dir": r"%CD%\%DATASETS_DIR%",
>> "%CFG_PY%" echo     "sync": False
>> "%CFG_PY%" echo })
python "%CFG_PY%" >nul 2>&1
if errorlevel 1 (
  echo WARNING: Could not apply Ultralytics settings (continuing).
)
exit /b 0

:HELP
echo Usage:
echo   run.bat [-s^|--silent] [--install^|--no-install] [webcam.py args...]
echo.
echo Examples:
echo   run.bat                 ^(interactive prompts, Enter = defaults^)
echo   run.bat -s              ^(silent: always defaults^)
echo   run.bat --no-install    ^(skip setup/install step^)
echo   run.bat --source 0 --preset yolo26x
echo.
echo Note: run.bat-only flags are stripped before forwarding args to webcam.py.
exit /b 0

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

@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM sentinelCam-worker launcher (run.bat)
REM - creates/uses venv in .runtime\venv
REM - installs python deps via inline pip list (NO requirements.txt)
REM - runs webcam.py
REM Web streaming is provided by THIS worker repo (webstream.py).

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "CFG_FILE=%SCRIPT_DIR%webcam.properties"

REM Defaults
set "RUNTIME_DIR=.runtime"
set "VENV_SUBDIR=venv"
set "ULTRA_CFG_SUBDIR=ultralytics_config"
set "WEIGHTS_SUBDIR=weights"
set "RUNS_SUBDIR=runs"
set "DATASETS_SUBDIR=datasets"
set "PIP_CACHE_SUBDIR=pip-cache"

set "DEFAULT_SOURCE_WINDOWS=0"
set "DEFAULT_DEVICE=auto"
set "DEFAULT_USE_POSE=1"
set "DEFAULT_MAX_FPS=120"

REM Read webcam.properties (KEY=VALUE, ignore # comments)
if exist "%CFG_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%CFG_FILE%") do (
    set "K=%%A"
    set "V=%%B"
    if not "!K!"=="" (
      if "!K:~0,1!" NEQ "#" (
        for /f "tokens=* delims= " %%K in ("!K!") do set "K2=%%K"
        for /f "tokens=* delims= " %%V in ("!V!") do set "V2=%%V"
        if not "!K2!"=="" set "!K2!=!V2!"
      )
    )
  )
)

set "VENV_DIR=%RUNTIME_DIR%\%VENV_SUBDIR%"
set "ULTRA_CFG_DIR=%RUNTIME_DIR%\%ULTRA_CFG_SUBDIR%"
set "WEIGHTS_DIR=%RUNTIME_DIR%\%WEIGHTS_SUBDIR%"
set "RUNS_DIR=%RUNTIME_DIR%\%RUNS_SUBDIR%"
set "DATASETS_DIR=%RUNTIME_DIR%\%DATASETS_SUBDIR%"
set "PIP_CACHE_DIR_LOCAL=%RUNTIME_DIR%\%PIP_CACHE_SUBDIR%"

REM ---------------------------
REM Parse launcher args
REM ---------------------------
set "SILENT=0"
set "INSTALL_MODE=ask"
REM ask/force/skip
set "FWD_ARGS="

:parse
if "%~1"=="" goto after_parse
if /i "%~1"=="-s"        set "SILENT=1" & shift & goto parse
if /i "%~1"=="--silent"  set "SILENT=1" & shift & goto parse
if /i "%~1"=="--install" set "INSTALL_MODE=force" & shift & goto parse
if /i "%~1"=="--no-install" set "INSTALL_MODE=skip" & shift & goto parse
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help

REM forward arg
set "FWD_ARGS=%FWD_ARGS% %~1"
shift
goto parse

:help
echo Usage:
echo   run.bat [-s^|--silent] [--install^|--no-install] [webcam.py args...]
echo.
echo Notes:
echo   - web server is the default (webcam.py defaults web=True)
echo   - window is optional: add --window
echo   - disable web server: --no-web
exit /b 0

:after_parse

REM Decide install
set "DO_INSTALL=1"
if /i "%INSTALL_MODE%"=="skip"  set "DO_INSTALL=0"
if /i "%INSTALL_MODE%"=="force" set "DO_INSTALL=1"

if /i "%INSTALL_MODE%"=="ask" (
  if "%SILENT%"=="1" (
    set "DO_INSTALL=1"
  ) else (
    set /p "REPLY=Run setup/install step (venv + pip deps)? [Y/n]: "
    if "!REPLY!"=="" set "REPLY=Y"
    if /i "!REPLY!"=="n"  set "DO_INSTALL=0"
    if /i "!REPLY!"=="no" set "DO_INSTALL=0"
  )
)

REM Runtime dirs
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%" >nul 2>&1
if not exist "%ULTRA_CFG_DIR%" mkdir "%ULTRA_CFG_DIR%" >nul 2>&1
if not exist "%WEIGHTS_DIR%" mkdir "%WEIGHTS_DIR%" >nul 2>&1
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%" >nul 2>&1
if not exist "%DATASETS_DIR%" mkdir "%DATASETS_DIR%" >nul 2>&1
if not exist "%PIP_CACHE_DIR_LOCAL%" mkdir "%PIP_CACHE_DIR_LOCAL%" >nul 2>&1

REM venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv: %VENV_DIR%
  py -3 -m venv "%VENV_DIR%" >nul 2>&1
  if errorlevel 1 (
    python -m venv "%VENV_DIR%" || (
      echo ERROR: Failed to create venv. Install Python 3 and try again.
      exit /b 1
    )
  )
)

call "%VENV_DIR%\Scripts\activate.bat"

set "PIP_CACHE_DIR=%SCRIPT_DIR%%PIP_CACHE_DIR_LOCAL%"
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

if "%DO_INSTALL%"=="1" (
  echo Installing python deps...
  python -m pip install ultralytics opencv-python numpy "lap>=0.5.12" || exit /b 1
) else (
  echo Skipping install ^(--no-install^) . Assuming deps already exist.
)

if not exist "webcam.py" (
  echo ERROR: webcam.py not found in %SCRIPT_DIR%
  exit /b 1
)

REM Configure Ultralytics runtime dirs
set "YOLO_CONFIG_DIR=%SCRIPT_DIR%%ULTRA_CFG_DIR%"
set "SC_WEIGHTS_DIR=%SCRIPT_DIR%%WEIGHTS_DIR%"
set "SC_RUNS_DIR=%SCRIPT_DIR%%RUNS_DIR%"
set "SC_DATASETS_DIR=%SCRIPT_DIR%%DATASETS_DIR%"

REM Add defaults if not provided
echo %FWD_ARGS% | findstr /i /c:"--source" >nul
if errorlevel 1 (
  set "FWD_ARGS=--source %DEFAULT_SOURCE_WINDOWS% %FWD_ARGS%"
)

echo %FWD_ARGS% | findstr /i /c:"--device" >nul
if errorlevel 1 (
  set "FWD_ARGS=--device %DEFAULT_DEVICE% %FWD_ARGS%"
)

echo %FWD_ARGS% | findstr /i /c:"--max-fps" >nul
if errorlevel 1 (
  set "FWD_ARGS=%FWD_ARGS% --max-fps %DEFAULT_MAX_FPS%"
)

if "%DEFAULT_USE_POSE%"=="1" (
  echo %FWD_ARGS% | findstr /i /c:"--no-pose" >nul
  if errorlevel 1 (
    echo %FWD_ARGS% | findstr /i /c:"--use-pose" >nul
    if errorlevel 1 (
      set "FWD_ARGS=%FWD_ARGS% --use-pose"
    )
  )
)

REM Run
python webcam.py %FWD_ARGS%
set "RC=%ERRORLEVEL%"
exit /b %RC%
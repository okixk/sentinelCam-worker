@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM sentinelCam-worker launcher (run.bat)
REM - creates/uses venv in .runtime\venv
REM - installs python deps via inline pip list (NO requirements.txt)
REM - asks for the desired camera/stream source if none was passed
REM - validates the selected source before starting webcam.py

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
set "DEFAULT_WEB_HOST=127.0.0.1"

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
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM ---------------------------
REM Parse launcher args
REM ---------------------------
set "SILENT=0"
set "INSTALL_MODE=ask"
set "FWD_ARGS="
set "HAS_SOURCE=0"
set "HAS_DEVICE=0"
set "HAS_HOST=0"
set "HAS_MAXFPS=0"
set "HAS_USE_POSE=0"
set "HAS_NO_POSE=0"
set "EXPECT_SOURCE_VALUE=0"
set "FINAL_SOURCE="
set "SELECTED_HOST=%DEFAULT_WEB_HOST%"

:parse
if "%~1"=="" goto after_parse

if "%EXPECT_SOURCE_VALUE%"=="1" (
  set "FINAL_SOURCE=%~1"
  set "EXPECT_SOURCE_VALUE=0"
)

if /i "%~1"=="-s"             set "SILENT=1" & shift & goto parse
if /i "%~1"=="--silent"       set "SILENT=1" & shift & goto parse
if /i "%~1"=="--install"      set "INSTALL_MODE=force" & shift & goto parse
if /i "%~1"=="--no-install"   set "INSTALL_MODE=skip" & shift & goto parse
if /i "%~1"=="--help"         goto help
if /i "%~1"=="-h"             goto help
if /i "%~1"=="--help-web"     goto help_web

set "ARG=%~1"

if /i "%~1"=="--source" set "HAS_SOURCE=1" & set "EXPECT_SOURCE_VALUE=1"
if /i "%~1"=="--cam"    set "HAS_SOURCE=1" & set "EXPECT_SOURCE_VALUE=1"
if /i "%ARG:~0,9%"=="--source=" set "HAS_SOURCE=1" & set "FINAL_SOURCE=%ARG:~9%"
if /i "%ARG:~0,6%"=="--cam="    set "HAS_SOURCE=1" & set "FINAL_SOURCE=%ARG:~6%"

if /i "%~1"=="--device" set "HAS_DEVICE=1"
if /i "%ARG:~0,9%"=="--device=" set "HAS_DEVICE=1"

if /i "%~1"=="--host" set "HAS_HOST=1"
if /i "%ARG:~0,7%"=="--host=" set "HAS_HOST=1"

if /i "%~1"=="--max-fps" set "HAS_MAXFPS=1"
if /i "%ARG:~0,10%"=="--max-fps=" set "HAS_MAXFPS=1"

if /i "%~1"=="--use-pose" set "HAS_USE_POSE=1"
if /i "%~1"=="--no-pose"  set "HAS_NO_POSE=1"

set "FWD_ARGS=%FWD_ARGS% %~1"
shift
goto parse

:help
echo Usage:
echo   run.bat [-s^|--silent] [--install^|--no-install] [webcam.py args...]
echo.
echo Notes:
echo   - web server is the default (webcam.py defaults web=True)
echo   - default bind host is 127.0.0.1 (localhost only)
echo   - if --host is omitted, the script asks: 1=localhost, 2=0.0.0.0
echo   - window is optional: add --window
echo   - disable web server: --no-web
echo.
echo See also:
echo   run.bat --help-web
exit /b 0

:help_web
echo sentinelCam-worker launcher (run.bat)
echo.
echo This script ONLY manages the worker repo:
echo   - creates/uses a venv in .runtime\venv
echo   - installs python deps via an inline pip list (NO requirements.txt)
echo   - starts webcam.py
echo.
echo The web repo simply displays http://WORKER_IP:8080/stream.mjpg.
echo By default the worker binds only to 127.0.0.1.
echo If --host is omitted, choose 1 for localhost or 2 for 0.0.0.0.
echo Change DEFAULT_WEB_HOST in webcam.properties or pass --host 0.0.0.0 for LAN access.
echo.
echo Examples:
echo   run.bat
echo   run.bat --no-install
echo   run.bat --no-web
echo   run.bat --window
echo   run.bat --host 0.0.0.0 --port 8080
exit /b 0

:after_parse

REM Decide install
set "DO_INSTALL=1"
if /i "%INSTALL_MODE%"=="skip" set "DO_INSTALL=0"
if /i "%INSTALL_MODE%"=="force" set "DO_INSTALL=1"

if /i "%INSTALL_MODE%"=="ask" (
  if "%SILENT%"=="1" (
    set "DO_INSTALL=1"
  ) else (
    set /p "REPLY=Run setup/install step (venv + pip deps)? [Y/n]: "
    if "!REPLY!"=="" set "REPLY=Y"
    if /i "!REPLY!"=="n" set "DO_INSTALL=0"
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
if not exist "%PYTHON_EXE%" (
  echo Creating venv: %VENV_DIR%
  py -3 -m venv "%VENV_DIR%" >nul 2>&1
  if errorlevel 1 (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
      echo ERROR: Failed to create venv. Install Python 3 and try again.
      exit /b 1
    )
  )
)

if not exist "%PYTHON_EXE%" (
  echo ERROR: Python venv executable not found: %PYTHON_EXE%
  exit /b 1
)

set "PIP_CACHE_DIR=%SCRIPT_DIR%%PIP_CACHE_DIR_LOCAL%"

if "%DO_INSTALL%"=="1" (
  "%PYTHON_EXE%" -m pip install --upgrade pip wheel setuptools >nul 2>&1
  "%PYTHON_EXE%" -m pip install ultralytics opencv-python numpy "lap>=0.5.12"
  if errorlevel 1 exit /b 1
) else (
  echo Skipping install ^(--no-install^). Assuming venv + deps already exist.
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

REM Ask for source if none was passed
if "%HAS_SOURCE%"=="0" (
  set "SELECTED_SOURCE=%DEFAULT_SOURCE_WINDOWS%"
  if not "%SILENT%"=="1" (
    set /p "SELECTED_SOURCE=Which cam index or stream URL/path should YOLO use? [%DEFAULT_SOURCE_WINDOWS%]: "
    if "!SELECTED_SOURCE!"=="" set "SELECTED_SOURCE=%DEFAULT_SOURCE_WINDOWS%"
  )
  set "FINAL_SOURCE=!SELECTED_SOURCE!"
  set FWD_ARGS=--source "!SELECTED_SOURCE!" %FWD_ARGS%
)

REM Add defaults if not already provided
if "%HAS_DEVICE%"=="0" set FWD_ARGS=--device "%DEFAULT_DEVICE%" %FWD_ARGS%
if "%HAS_HOST%"=="0" (
  if /i "%DEFAULT_WEB_HOST%"=="0.0.0.0" (
    set "DEFAULT_HOST_CHOICE=2"
  ) else (
    set "DEFAULT_HOST_CHOICE=1"
  )
  if not "%SILENT%"=="1" (
    call :prompt_host_choice "%DEFAULT_HOST_CHOICE%"
  )
  set FWD_ARGS=--host "%SELECTED_HOST%" %FWD_ARGS%
)
if "%HAS_MAXFPS%"=="0" set FWD_ARGS=%FWD_ARGS% --max-fps "%DEFAULT_MAX_FPS%"
if "%DEFAULT_USE_POSE%"=="1" if "%HAS_NO_POSE%"=="0" if "%HAS_USE_POSE%"=="0" set "FWD_ARGS=%FWD_ARGS% --use-pose"

REM Validate final source before starting webcam.py
if not "%FINAL_SOURCE%"=="" (
  call :validate_source "%FINAL_SOURCE%"
  if errorlevel 1 (
    call :is_numeric "%FINAL_SOURCE%"
    if not errorlevel 1 (
      echo ERROR: Selected camera '%FINAL_SOURCE%' is not available.
    ) else (
      echo ERROR: Selected source '%FINAL_SOURCE%' could not be opened.
    )
    exit /b 1
  )
)

"%PYTHON_EXE%" webcam.py %FWD_ARGS%
set "RC=%ERRORLEVEL%"
exit /b %RC%

:is_numeric
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.argv[1].isdigit() else 1)" "%~1" >nul 2>&1
exit /b %ERRORLEVEL%

:validate_source
set "SRC_TO_VALIDATE=%~1"
set "VALIDATE_PY=%TEMP%\sentinelcam_validate_source_%RANDOM%%RANDOM%.py"

> "%VALIDATE_PY%" (
  echo import sys
  echo import cv2
  echo src = sys.argv[1]
  echo ok = False
  echo try:
  echo     if isinstance(src, str^) and src.isdigit(^):
  echo         idx = int(src^)
  echo         backends = [None]
  echo         if sys.platform.startswith("win"^) and hasattr(cv2, "CAP_DSHOW"^):
  echo             backends = [cv2.CAP_DSHOW, None]
  echo         elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"^):
  echo             backends = [cv2.CAP_AVFOUNDATION, None]
  echo         for backend in backends:
  echo             cap = cv2.VideoCapture(idx, backend^) if backend is not None else cv2.VideoCapture(idx^)
  echo             try:
  echo                 if cap is not None and cap.isOpened(^):
  echo                     ret, _frame = cap.read(^)
  echo                     if ret:
  echo                         ok = True
  echo                         break
  echo             finally:
  echo                 try:
  echo                     if cap is not None:
  echo                         cap.release(^)
  echo                 except Exception:
  echo                     pass
  echo     else:
  echo         cap = cv2.VideoCapture(src^)
  echo         try:
  echo             if cap is not None and cap.isOpened(^):
  echo                 ret, _frame = cap.read(^)
  echo                 ok = bool(ret^)
  echo         finally:
  echo             try:
  echo                 if cap is not None:
  echo                     cap.release(^)
  echo             except Exception:
  echo                 pass
  echo except Exception:
  echo     ok = False
  echo raise SystemExit(0 if ok else 1^)
)

"%PYTHON_EXE%" "%VALIDATE_PY%" "%SRC_TO_VALIDATE%" >nul 2>&1
set "RC=%ERRORLEVEL%"
del "%VALIDATE_PY%" >nul 2>&1
exit /b %RC%

:prompt_host_choice
set "HOST_CHOICE="
set "DEFAULT_HOST_CHOICE=%~1"
:prompt_host_choice_loop
set /p "HOST_CHOICE=Stream host waehlen [1=localhost/127.0.0.1, 2=alle Interfaces/0.0.0.0] [%DEFAULT_HOST_CHOICE%]: "
if "!HOST_CHOICE!"=="" set "HOST_CHOICE=%DEFAULT_HOST_CHOICE%"
if "!HOST_CHOICE!"=="1" (
  set "SELECTED_HOST=127.0.0.1"
  exit /b 0
)
if "!HOST_CHOICE!"=="2" (
  set "SELECTED_HOST=0.0.0.0"
  exit /b 0
)
echo Bitte nur 1 oder 2 eingeben.
goto prompt_host_choice_loop

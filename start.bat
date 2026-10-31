@echo off
REM ============================================================
REM  Cute Meow Assistant - startup script for Windows (double-click me)
REM
REM    start.bat              build if needed, then launch + open browser
REM    start.bat --port 8010  use another port
REM    start.bat --rebuild    force rebuild frontend
REM    start.bat --no-build   use existing web\dist
REM    start.bat --check      health check only, do not launch
REM
REM  NOTE: keep this file ASCII-only. cmd.exe parses .bat with the
REM  OEM codepage; non-ASCII comments corrupt command boundaries.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PORT=8000"
set "HOST=127.0.0.1"
set "BUILD=auto"
set "CHECKONLY=0"

:parse
if "%~1"=="" goto after_parse
if /i "%~1"=="--port"     ( set "PORT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--rebuild"  ( set "BUILD=force" & shift & goto parse )
if /i "%~1"=="--no-build" ( set "BUILD=never" & shift & goto parse )
if /i "%~1"=="--check"    ( set "CHECKONLY=1" & shift & goto parse )
if /i "%~1"=="-h"         goto usage
if /i "%~1"=="--help"     goto usage
echo Unknown option: %~1
goto usage

:usage
echo Usage: start.bat [--port N] [--rebuild] [--no-build] [--check]
exit /b 2

:after_parse
echo Cute Meow Assistant - startup check
echo.

REM ---------- 1. Python ----------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo   [X] Python not found. Create a venv first:  python -m venv .venv
  exit /b 1
)
for /f "delims=" %%v in ('%PY% -V 2^>^&1') do echo   [OK] Python: %PY% ^(%%v^)

%PY% -c "import importlib.util,sys; m=[x for x in ('torch','diffusers','transformers','fastapi','uvicorn') if importlib.util.find_spec(x) is None]; print('Missing: '+', '.join(m)) if m else None; sys.exit(1 if m else 0)"
if errorlevel 1 (
  echo   [X] Missing deps. Run:  %PY% -m pip install -r requirements.txt
  exit /b 1
)
echo   [OK] Dependencies present

%PY% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo   [!] CUDA not available, will run on CPU ^(very slow^)
) else (
  for /f "delims=" %%g in ('%PY% -c "import torch;print(torch.cuda.get_device_name(0))"') do echo   [OK] CUDA: %%g
)

REM ---------- 2. Weights ----------
REM  Only Qwen-Image-2.1 is used now; SD weights live in models\_archive\sd\.
if exist "models\checkpoints\qwen-image-2.1\model_index.json" (
  echo   [OK] Qwen-Image-2.1 weights found
) else (
  echo   [X] Qwen-Image-2.1 weights missing, generation will fail
  echo       expected: models\checkpoints\qwen-image-2.1\model_index.json
  echo       see README "Install" for the download layout
)

REM ---------- 3. Frontend ----------
if /i "%BUILD%"=="force" goto do_build
if /i "%BUILD%"=="never" goto skip_build
if not exist "web\dist\index.html" (
  echo   [!] No frontend build found, building
  goto do_build
)
set "STALE=0"
for /f %%i in ('powershell -NoProfile -Command "$d=(Get-Item -LiteralPath 'web\dist\index.html').LastWriteTime; $n=(Get-ChildItem -LiteralPath 'web\src' -Recurse -File ^| Where-Object { $_.LastWriteTime -gt $d }); if($n){'1'}else{'0'}"') do set "STALE=%%i"
if "%STALE%"=="1" (
  echo   [!] Frontend sources changed, rebuilding
  goto do_build
)
goto skip_build

:do_build
where npm >nul 2>&1
if errorlevel 1 (
  echo   [X] npm not found, cannot build frontend. Install Node 18+ or use --no-build
  exit /b 1
)
echo   ... building frontend
pushd web
if not exist "node_modules" call npm ci --no-audit --no-fund
call npm run build
popd
if errorlevel 1 (
  echo   [X] Frontend build failed
  exit /b 1
)
echo   [OK] Frontend built
goto build_done

:skip_build
echo   [OK] Using existing frontend build

:build_done

if "%CHECKONLY%"=="1" (
  echo.
  echo Check complete.
  exit /b 0
)

REM ---------- 4. Port ----------
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo   [X] Port %PORT% already in use. Try:  --port 8010
  exit /b 1
)

REM ---------- 5. Launch ----------
echo.
echo Serving: http://%HOST%:%PORT%
echo Close this window to stop.
echo.

start "" "http://%HOST%:%PORT%"
%PY% -m uvicorn app:app --host %HOST% --port %PORT% --log-level info

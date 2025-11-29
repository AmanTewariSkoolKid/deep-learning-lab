@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===============================
REM Transfer Learning Setup & Run
REM ===============================

REM 1. Move to script directory
pushd %~dp0

REM 2. Check Python
where python >nul 2>nul
IF ERRORLEVEL 1 (
  echo [ERROR] Python not found in PATH.
  echo Install Python 3.10+ and re-run.
  pause
  exit /b 1
)

echo.
echo === Creating virtual environment (.venv) if missing ===
IF NOT EXIST ..\..\.venv (
  python -m venv ..\..\.venv
)

REM 3. Activate venv
call ..\..\.venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
  echo [ERROR] Failed to activate virtual environment.
  pause
  exit /b 1
)

echo.
echo === Upgrading pip ===
python -m pip install --upgrade pip >nul

echo.
echo === Installing dependencies ===
python -m pip install -r requirements.txt
IF ERRORLEVEL 1 (
  echo [ERROR] Dependency installation failed.
  pause
  exit /b 1
)

REM 4. Check dataset presence
set DATA_PATH=data\flowers
IF NOT EXIST %DATA_PATH% (
  echo.
  echo [INFO] Dataset folder %DATA_PATH% not found. Attempting automatic download...
  python download_dataset.py
)

IF NOT EXIST %DATA_PATH% (
  echo [ERROR] Dataset still missing after attempted download. Aborting.
  pause
  exit /b 1
)

echo.
echo === Starting training (feature extraction + fine-tune) ===
python train_transfer.py
IF ERRORLEVEL 1 (
  echo [ERROR] Training script failed.
  pause
  exit /b 1
)

echo.
echo === DONE ===
echo Model and outputs (if configured) saved.
echo If you used a different dataset path, update dataset.path in config_transfer.json.

popd
endlocal

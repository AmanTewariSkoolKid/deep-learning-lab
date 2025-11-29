@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===============================
REM CNN MNIST Tutorial Setup & Run
REM ===============================

pushd %~dp0

echo.
echo === Checking Python ===
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

echo.
echo === Activating virtual environment ===
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

echo.
echo === Setup Complete ===
echo To run the notebook:
echo   1. Activate venv: ..\..\.venv\Scripts\activate.bat
echo   2. Launch Jupyter: jupyter notebook cnn_tutorial.ipynb
echo.
echo Or install Jupyter and run:
echo   python -m pip install jupyter
echo   jupyter notebook cnn_tutorial.ipynb
echo.

pause
popd
endlocal

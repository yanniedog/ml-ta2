@echo off
cls

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    ML-TA - Machine Learning Technical Analysis               ║
echo ║                              Starting Application                            ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

:: Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Check Python version using a different approach
echo Checking Python version...
for /f "tokens=2,3 delims=. " %%a in ('python -c "import sys; print('{0} {1}'.format(sys.version_info.major, sys.version_info.minor))" 2^>^&1') do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if %PY_MAJOR% LSS 3 (
    echo.
    echo ERROR: Python 3.10 or later is required. Found version %PY_MAJOR%.%PY_MINOR%
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
) else if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo.
    echo ERROR: Python 3.10 or later is required. Found version %PY_MAJOR%.%PY_MINOR%
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo ✓ Python version %PY_MAJOR%.%PY_MINOR% is compatible
echo.

:: Use the existing .venv if it exists, otherwise create a new venv
echo Checking virtual environment...
if exist ".venv" (
    echo ✓ Using existing virtual environment
    set VENV_PATH=.venv
) else (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
    set VENV_PATH=.venv
)

echo.
:: Activate the virtual environment
echo Activating virtual environment...
call %VENV_PATH%\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    echo.
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

:: Install dependencies if not already installed
echo Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
if exist "requirements.txt" (
    echo Installing from requirements.txt...
    pip install -r requirements.txt >nul 2>&1
)
if exist "requirements-dev.txt" (
    echo Installing from requirements-dev.txt...
    pip install -r requirements-dev.txt >nul 2>&1
)

echo ✓ Dependencies installed
echo.

:: Run the application
echo Starting ML-TA application...
echo The web interface will be available at: http://localhost:5000
echo.
echo NOTE: To stop the application, close this window.
echo.
start "" http://localhost:5000
timeout /t 3 /nobreak >nul
python -m src.web.app

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start ML-TA application.
    echo Please check the error messages above.
    echo.
    echo You can try running the following commands manually:
    echo   call %VENV_PATH%\Scripts\activate
    echo   python -m src.web.app
    echo.
    pause
    exit /b 1
)

pause

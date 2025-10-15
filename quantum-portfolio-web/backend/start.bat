@echo off
echo Starting QEPO Web API Server...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Check if MLflow server is running
echo Checking MLflow connection...
python -c "import requests; requests.get('http://localhost:5000', timeout=2)" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: MLflow server is not running on localhost:5000
    echo To start MLflow server, run in a separate terminal:
    echo   python start_mlflow.py
    echo.
    echo The web interface will show empty results without MLflow.
    echo.
    pause
)

REM Start the server
echo Starting server on http://localhost:8000
python server.py

pause

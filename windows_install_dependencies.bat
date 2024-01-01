@echo off
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH.
    echo Opening the Python download website...
    start https://www.python.org/downloads/
    pause
    exit
) else (
    python install_dependencies.py
    pause
)

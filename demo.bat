@echo off
echo =======================================================
echo   Adversarial Stress-Testing Framework - Live Demo
echo =======================================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching the Audit Dashboard...
cd IDS_project
streamlit run dashboard.py

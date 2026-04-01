@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it first with:
  echo   py -3.12 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

echo [INFO] Starting Streamlit dashboard on http://localhost:8502
".venv\Scripts\python.exe" -m streamlit run dashboard_app.py --server.headless true --server.port 8502


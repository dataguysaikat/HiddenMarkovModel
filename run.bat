@echo off
title HiddenMarkov
REM Activate venv and launch the Streamlit dashboard

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found. Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

call .venv\Scripts\activate.bat

REM Load .env if present
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

REM One-time Schwab OAuth: python -m src.broker auth

REM Update strategy + alert policy from closed trade history (background, non-blocking)
start "Policy Update" /min cmd /c ".venv\Scripts\python.exe -m src.retrain_policy"

REM Generate live option recommendations in a separate window (requires ThetaData terminal)
start "HMM Recommendations" cmd /k ".venv\Scripts\python.exe -m src.recommend"

REM Launch dashboard immediately
streamlit run src/dashboard.py

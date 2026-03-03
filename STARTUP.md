# HMM Regime Dashboard — Start & Stop Guide

**GitHub:** https://github.com/dataguysaikat/HiddenMarkovModel

---

## New Machine Setup

```bat
git clone https://github.com/dataguysaikat/HiddenMarkovModel.git
cd HiddenMarkovModel
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Then follow the **Daily Workflow** below. On first launch, click **[Fit HMM]** in the dashboard sidebar to build the model cache.

---

## Starting the Dashboard

### Option A — Double-click (easiest)
1. Double-click **`run.bat`** in the project folder
2. A terminal window opens and the dashboard launches
3. Browser opens automatically at **http://localhost:8501**

### Option B — Terminal
```bat
cd C:\Users\saika\OneDrive\Documents\Saikat\Agents\HiddenMarkovModel
run.bat
```

### Option C — Manual (if run.bat fails)
```bat
cd C:\Users\saika\OneDrive\Documents\Saikat\Agents\HiddenMarkovModel
.venv\Scripts\activate
streamlit run src/dashboard.py
```

---

## Daily Workflow

### 1. Start ThetaData Terminal first
The ThetaData Terminal must be running at **http://127.0.0.1:25503** before launching.

### 2. Run `run.bat`
`run.bat` does everything in sequence:
1. Activates the virtual environment
2. Runs `python -m src.recommend` — fits HMM, fetches live option chains, prints recommended strikes, saves trades to `data/tracked_trades.json`
3. Launches the Streamlit dashboard at **http://localhost:8501**

### 3. Refresh trade prices (dashboard)
- Open the **Trade Tracker** tab on the dashboard
- Click **Refresh Prices** to fetch current option mid-prices and update P&L

### 4. Automatic updates (while dashboard is running)
- **Hourly at :30 ET** (Mon-Fri 9:30–15:30): yfinance data fetch + HMM refit
- **Daily at 16:05 ET** (Mon-Fri): EOD option price refresh for all open trades

---

## Stopping the Dashboard

### Option A — Terminal (cleanest)
Press **Ctrl+C** in the terminal window where the dashboard is running.

### Option B — Find and kill the process
Open PowerShell and run:
```powershell
# Find the Streamlit PID
netstat -ano | findstr :8501

# Kill it (replace 12345 with the actual PID)
Stop-Process -Id 12345 -Force
```

### Option C — Kill all dashboard processes at once
```powershell
Get-Process python | Where-Object { $_.CPU -gt 100 } | Stop-Process -Force
```
> **Caution:** This kills any heavy Python process. Check PIDs first if you have other Python projects running.

---

## Checking What's Running

```powershell
# See all Python processes with CPU and memory
Get-Process python | Select-Object Id, CPU, WorkingSet, StartTime | Format-Table -AutoSize

# Check what's on port 8501
netstat -ano | findstr :8501
```

### Known co-existing processes (don't kill these)
| Project | Script |
|---------|--------|
| silentfacts | `src/app.py`, `src/subsidiary_scraper.py` |
| earnings_research_pipeline | `scripts/run_dc_pipeline.py` |

---

## Deduplicating Tracked Trades
If `recommend.py` is run multiple times in a day, duplicate trades accumulate. Clean up with:
```bat
.venv\Scripts\activate
python -c "
from src.trade_tracker import load_trades, _save_all
latest = {}
for tr in load_trades():
    if not latest.get(tr.ticker) or tr.recommended_at > latest[tr.ticker].recommended_at:
        latest[tr.ticker] = tr
_save_all(list(latest.values()))
print(f'Kept {len(latest)} trades.')
"
```

---

## ThetaData Terminal
The recommend script requires the ThetaData Terminal to be running locally.
- URL: **http://127.0.0.1:25503**
- If not running, `recommend.py` will print `ThetaData terminal: UNAVAILABLE` and exit
- Start ThetaData Terminal first, then run `python -m src.recommend`

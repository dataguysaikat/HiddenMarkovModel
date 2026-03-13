# Standard Operating Procedure — HMM Regime Trading Dashboard

**Version:** 1.3
**Last updated:** 2026-03-13
**Application:** Hidden Markov Model Regime Detection + Options Auto-Trading
**Universe:** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA
**GitHub:** https://github.com/dataguysaikat/HiddenMarkovModel

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [System Overview](#2-system-overview)
3. [Setting Up on a New Machine](#3-setting-up-on-a-new-machine)
4. [Prerequisites & One-Time Setup (existing machine)](#4-prerequisites--one-time-setup-existing-machine)
5. [Daily Operating Procedure](#5-daily-operating-procedure)
6. [Dashboard Guide](#6-dashboard-guide)
7. [Weekly Maintenance](#7-weekly-maintenance)
8. [Pushing Updates to GitHub](#8-pushing-updates-to-github)
9. [Adding or Removing Tickers](#9-adding-or-removing-tickers)
10. [Paper Trading vs Live Trading](#10-paper-trading-vs-live-trading)
11. [Adjusting Strategy Parameters](#11-adjusting-strategy-parameters)
12. [Stopping the Application](#12-stopping-the-application)
13. [Troubleshooting](#13-troubleshooting)
14. [Configuration Reference](#14-configuration-reference)

---

## 1. Purpose & Scope

This SOP covers day-to-day operation of the HMM Regime Dashboard — a quantitative system that:

- Fits a 4-state Gaussian Hidden Markov Model to 1-hour RTH bar data for each ticker
- Classifies the current market regime as **Bull, Bear, Vol Expansion, or Mean Reverting**
- Selects an appropriate options strategy for each regime
- Fetches live option chains from ThetaData Terminal and selects specific strikes
- Tracks open trades and reports daily mark-to-market P&L

Intended audience: the operator of the system (the person running `run.bat` each morning).

---

## 2. System Overview

```
ThetaData Terminal (must be running)
        │
        ▼
run.bat
  ├─ [Background window] recommend.py
  │     Fits HMM → selects strikes → saves to tracked_trades.json
  │
  └─ [Main window] streamlit dashboard  →  http://localhost:8501
        ├─ Tab 1: Regime Overview
        ├─ Tab 2: Price Charts
        ├─ Tab 3: Options & Trading
        ├─ Tab 4: Model Diagnostics
        └─ Tab 5: Trade Tracker (P&L)
```

**Automated background jobs (while dashboard is running):**

| Time | Job |
|------|-----|
| Mon–Fri, :30 past each hour, 09:30–15:30 ET | yfinance data refresh + HMM refit |
| Mon–Fri, 16:05 ET | End-of-day option price update for all open trades |

---

## 3. Setting Up on a New Machine

The full codebase and historical bar data are on GitHub. Clone and run in four steps.

**GitHub repository:** https://github.com/dataguysaikat/HiddenMarkovModel

### Step 1 — Clone the repository

```bat
git clone https://github.com/dataguysaikat/HiddenMarkovModel.git
cd HiddenMarkovModel
```

### Step 2 — Create a virtual environment and install dependencies

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

> **Note:** `hmmlearn` is not in `requirements.txt` (requires C++ Build Tools on Windows). The system works without it via the built-in pure-NumPy fallback. See Section 4 for optional install.

### Step 3 — Configure credentials

```bat
copy .env.example .env
```

Open `.env` and fill in your Schwab API keys if using live trading. Leave `TRADE_MODE=paper` for paper trading — no keys needed.

### Step 4 — Start ThetaData Terminal, then run

```bat
run.bat
```

The parquet files (`data/*.parquet`) are included in the repo so bar data is immediately available. Click **[Fit HMM]** in the dashboard sidebar on the first launch to build the model cache, then the system is fully operational.

---

## 4. Prerequisites & One-Time Setup (existing machine)

These steps are completed once and do not need to be repeated on daily starts.

### 4.1 Install Python dependencies

```bat
cd C:\Users\saika\OneDrive\Documents\Saikat\Agents\HiddenMarkovModel
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `hmmlearn` is not in `requirements.txt` because it requires C++ Build Tools on Windows. The system runs without it using the built-in pure-NumPy fallback. If you want faster HMM fitting, install the C++ Build Tools from Microsoft and then run `pip install hmmlearn`.

### 4.2 Import historical CSV data

For each ticker, download the Barchart 1-hour intraday historical CSV and drop it in the `data/` folder. File names must start with the ticker symbol in lowercase (e.g., `aapl_intraday-60min_...csv`).

Then import via dashboard:
1. Start the dashboard (run `run.bat` or `streamlit run src/dashboard.py`)
2. In the sidebar, click **[Import CSVs]**
3. Verify 7 `.parquet` files appear in `data/`

Or via terminal:
```bat
.venv\Scripts\activate
python -c "from src.data_loader import import_csv_to_parquet; import_csv_to_parquet('AAPL')"
```

### 4.3 Fit the initial HMM models

On first run the `data/hmm_cache.pkl` file won't exist. Either:

- Start the dashboard and click **[Fit HMM]** in the sidebar, **or**
- Run `run.bat` and let the background recommend window fit from scratch (takes several minutes without hmmlearn)

Once the cache exists, subsequent runs complete in under 30 seconds.

### 4.4 (Optional) Schwab live trading authentication

For live order execution only — skip if using paper trading mode.

```bat
.venv\Scripts\activate
python -m src.broker auth
```

Follow the OAuth prompts. The resulting token file path should be set in the `SCHWAB_TOKEN_PATH` environment variable (add to `.env` file in the project root).

---

## 5. Daily Operating Procedure

### Step 1 — Start ThetaData Terminal

Open the ThetaData Terminal application and confirm it shows connected at:
`http://127.0.0.1:25503`

> **Do not skip this step.** Without ThetaData, `recommend.py` will exit without saving new recommendations, and the dashboard will display a warning banner (see below).

**If ThetaData is not running**, a yellow warning banner appears at the top of the dashboard above all tabs:

> ⚠️ **ThetaData Terminal is not running.** Option recommendations and live P&L price updates are unavailable. Start ThetaData Terminal at http://127.0.0.1:25503, then the warning will clear automatically.

The banner disappears automatically within 60 seconds of ThetaData coming back up — no manual refresh needed. The rest of the dashboard (regime overview, price charts, model diagnostics) remains fully functional without ThetaData.

### Step 2 — Launch the application

Double-click **`run.bat`** in the project folder, or run from terminal:

```bat
cd C:\Users\saika\OneDrive\Documents\Saikat\Agents\HiddenMarkovModel
run.bat
```

What happens:
1. The virtual environment is activated
2. A **separate window** opens running `recommend.py` in the background
3. The Streamlit dashboard launches immediately at **http://localhost:8501**

### Step 3 — Review recommendations in the background window

The "HMM Recommendations" window shows output like:

```
Loading market data...
Using cached HMM models — re-predicting on fresh bars...
ThetaData terminal: connected

AAPL  price=$262.07  expiry=2026-03-27 (24 DTE)  regime=Mean Rev  conf=100%
  Strategy: IRON CONDOR  |  Confidence: 100%
  SELL  2026-03-27 $280.00C  bid=$1.84  delta=+0.19
  BUY   2026-03-27 $290.00C  ask=$0.62  delta=+0.08
  SELL  2026-03-27 $240.00P  bid=$2.40  delta=-0.17
  BUY   2026-03-27 $230.00P  ask=$1.45  delta=-0.10
  Net credit $2.27  |  Max profit $2.27  |  Max loss $12.27
  [saved to tracked_trades.json]
...
```

Review each recommendation. Trades are automatically saved to `data/tracked_trades.json`.

### Step 4 — Open the Trade Tracker tab

In the dashboard, go to the **Trade Tracker** tab (5th tab) to see:
- **Recommendations table:** all newly saved trades with regime, strategy, and confidence
- **Live P&L table:** current mark-to-market values for all open trades
- **Per-trade cards:** expandable cards showing leg details, daily P&L chart, and underlying price chart

### Step 5 — Refresh prices (if needed)

Click **[Refresh Prices]** at the top of the Trade Tracker tab to fetch current option mid-prices from ThetaData for all open trades. The EOD scheduler does this automatically at 16:05 ET, but you can trigger it manually at any time.

### Step 6 — (Optional) Execute trades

Go to the **Options & Trading** tab (Tab 3):
- Review the proposed orders table
- In **paper mode** (default): click **[Execute]** to log a paper trade to `data/paper_trades.json`
- In **live mode**: trades are routed through the Schwab API (see Section 8)

---

## 6. Dashboard Guide

### Tab 1 — Regime Overview

| Element | Description |
|---------|-------------|
| Summary table | All tickers, current regime, confidence, strategy |
| Expandable ticker cards | Close price chart with regime colour overlay, posterior probability chart |
| Sidebar → [Fetch from yfinance] | Manually top up bar data beyond last parquet date |
| Sidebar → [Fit HMM] | Manually refit all HMM models and save to cache |

### Tab 2 — Price Charts

Full history candlestick or line chart with regime-shaded background. Select a ticker from the dropdown. Includes a regime statistics table (mean return, mean vol, % time in regime per state).

### Tab 3 — Options & Trading

| Element | Description |
|---------|-------------|
| Proposed orders table | Latest recommended legs with strikes, price type, and net credit/debit |
| [Execute] button | Places a paper trade (or live trade if TRADE_MODE=live) |
| Paper trade log | History of all executed paper trades from `data/paper_trades.json` |

### Tab 4 — Model Diagnostics

Transition matrix heatmap, state means (log-ret and vol), and log-return distribution by regime. Useful for verifying the HMM is finding meaningful regimes.

### Tab 5 — Trade Tracker

| Element | Description |
|---------|-------------|
| [Refresh Prices] | Fetches current option mids from ThetaData, updates P&L for all open trades |
| Recommendations table | All saved trades: date, regime, strategy, entry price, confidence, expiry, status |
| Live P&L table | Entry $, Current $, P&L $ (green/red), P&L %, DTE remaining |
| Per-trade cards | Click to expand: full leg table, daily P&L line chart, underlying price chart |

**Card layout (per trade):**

```
TICKER | strategy | exp YYYY-MM-DD | P&L $X (X%) | status | Open YYYY-MM-DD
Row 1: Entry net  |  Current mid  |  P&L $  |  DTE left  |  Days held  |  Confidence
Row 2: Entry price  |  Current price / Close price  |  Max profit  |  Max loss  |  Opened HH:MM
[Leg table]
[Daily P&L chart]
[Underlying price chart]
```

---

## 7. Weekly Maintenance

### Deduplicate tracked trades (if recommend.py was run multiple times)

If you ran recommend more than once on the same day and want to keep only the latest recommendation per ticker:

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

### Refresh historical bar data

The dashboard's hourly scheduler keeps data current during market hours. If the dashboard was not running for several days:
1. Open the dashboard
2. In the sidebar, click **[Fetch from yfinance]**
3. Then click **[Fit HMM]** to refresh the regime model with the new bars

### Verify HMM cache freshness

The recommend script uses the cache if it is less than 7 days old, then falls back to re-fitting. After the dashboard runs the hourly job, the cache is refreshed automatically. If the cache is stale and you need fresh recommendations immediately, click **[Fit HMM]** in the sidebar first, then run `python -m src.recommend` separately.

---

## 8. Pushing Updates to GitHub

Whenever you make changes (new tickers, config tweaks, code fixes), push them to GitHub so the repo stays current and other machines can pull the latest.

```bat
cd C:\Users\saika\OneDrive\Documents\Saikat\Agents\HiddenMarkovModel
git add -p                        REM review and stage changes interactively
REM  — or stage everything at once:
git add config.json src/ data/*.parquet data/tracked_trades.json
git commit -m "brief description of what changed"
git push
```

On first push from a new machine you may be prompted for credentials. Enter your GitHub username (`dataguysaikat`) and a Personal Access Token (PAT) as the password — GitHub no longer accepts plain passwords over HTTPS.

### Pulling updates on another machine

```bat
cd HiddenMarkovModel
git pull
```

This brings in any code, config, or parquet changes committed from the primary machine.

### What is and isn't tracked

| Tracked (in repo) | Not tracked (in .gitignore) |
|-------------------|-----------------------------|
| `src/*.py` — all source code | `.venv/` — recreate with `pip install -r requirements.txt` |
| `data/*.parquet` — bar history | `data/*.csv` — Barchart source files |
| `data/tracked_trades.json` | `data/hmm_cache.pkl` — regenerated on first run |
| `data/paper_trades.json` | `.env` — contains secrets, set up manually |
| `config.json`, `requirements.txt`, `run.bat` | |
| All `.md` documentation | |

---

## 9. Adding or Removing Tickers

### Adding a ticker

**Step 1 — Download Barchart CSV**

- Go to barchart.com → search the ticker → download 1-hour intraday historical data
- Save to `data/` with a filename starting with the ticker symbol in lowercase:
  e.g., `data/spy_intraday-60min_historical-data.csv`

**Step 2 — Add to config.json**

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY"]
}
```

This is the **only code change required.** Everything else is automatic.

**Step 3 — Import the CSV**

In the dashboard sidebar, click **[Import CSVs]**, or from terminal:

```bat
.venv\Scripts\activate
python -c "from src.data_loader import import_csv_to_parquet; import_csv_to_parquet('SPY')"
```

**Step 4 — Refit HMM**

Click **[Fit HMM]** in the dashboard sidebar to include the new ticker in the model cache.

**Requirements per ticker:**

| Requirement | Minimum |
|-------------|---------|
| Historical bars | 500 RTH 1h bars (~1 month); 3+ months recommended |
| ThetaData coverage | Must be on the ThetaData Value plan |
| Options liquidity | Tight bid/ask spreads for reliable IV/delta calculation |

**Practical limits:**

| Range | Performance |
|-------|-------------|
| 1–10 tickers | Fast — recommend + dashboard load in under 30 seconds |
| 10–20 tickers | Good — slight slowdown on startup |
| 20+ tickers | Slow dashboard rendering; ThetaData chain fetch becomes bottleneck |

### Removing a ticker

1. Remove the ticker from `config.json`
2. Optionally delete `data/TICKER_1h.parquet` to free disk space
3. No other changes needed — the ticker disappears from all views on next restart

---

## 10. Paper Trading vs Live Trading

### Paper mode (default)

No configuration needed. All trades are logged to `data/paper_trades.json`. No real orders are sent.

To verify you are in paper mode:
```bat
echo %TRADE_MODE%
```
If nothing prints (empty), the system defaults to paper mode.

### Switching to live mode

**Prerequisites:**
- Schwab account with options trading enabled
- One-time OAuth (see Section 3.4)
- Token file path set in `SCHWAB_TOKEN_PATH`

**To enable:**
Create a `.env` file in the project root:
```
TRADE_MODE=live
SCHWAB_TOKEN_PATH=C:\path\to\schwab_token.json
```

`run.bat` loads this file automatically on startup.

> **Warning:** Live mode sends real orders to your Schwab account. Verify each recommendation in the Trade Tracker tab before enabling live mode. Always start with paper mode for new tickers or strategy parameter changes.

### Switching back to paper mode

Either remove `TRADE_MODE=live` from `.env`, or temporarily override from the terminal:

```bat
set TRADE_MODE=paper
streamlit run src/dashboard.py
```

---

## 11. Adjusting Strategy Parameters

### Via the dashboard (recommended)

1. Open the **sidebar** on any tab
2. Scroll to **Strategy Config → Option strategy parameters**
3. Adjust sliders and number inputs
4. Click **[Save to config.json]**
5. Changes take effect on the next `recommend.py` run

### Via direct file edit

Edit `config.json` in the project root:

```json
{
  "option_strategy": {
    "target_dte":   21,
    "dte_min":      14,
    "dte_max":      45,
    "delta_vert":   0.40,
    "delta_wing":   0.16,
    "otm_pct":      0.03,
    "strike_range": 20
  }
}
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `target_dte` | 21 | Aim for this many days to expiry when picking an expiration |
| `dte_min` | 14 | Skip expirations shorter than this |
| `dte_max` | 45 | Skip expirations longer than this |
| `delta_vert` | 0.40 | Delta of the long leg for Bull Call / Bear Put spreads |
| `delta_wing` | 0.16 | Delta of the short leg (wings) for Iron Condors |
| `otm_pct` | 0.03 | % out-of-the-money for Long Strangle legs (3% = 3% above/below spot) |
| `strike_range` | 20 | Number of strikes fetched above and below ATM from ThetaData |

---

## 12. Stopping the Application

### Normal stop
Press **Ctrl+C** in the Streamlit terminal window.

### If the process is stuck

```powershell
# Find the PID on port 8501
netstat -ano | findstr :8501

# Kill it (replace 12345 with actual PID)
Stop-Process -Id 12345 -Force
```

### Kill all HMM dashboard processes at once

```powershell
Get-NetTCPConnection -LocalPort 8501 |
  Select-Object -ExpandProperty OwningProcess |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

> **Caution:** Do not use `Get-Process python | Stop-Process` — this will also kill unrelated Python projects (silentfacts, earnings_research_pipeline) running on the same machine.

---

## 13. Troubleshooting

### Dashboard opens but shows no regime data

**Cause:** HMM has not been fitted yet.
**Fix:** Click **[Fit HMM]** in the sidebar. The dashboard needs at least one run to create `data/hmm_cache.pkl`.

---

### "ThetaData terminal: UNAVAILABLE" in the recommendations window / yellow banner on dashboard

**Cause:** ThetaData Terminal is not running.
**Fix:** Start ThetaData Terminal and wait for it to show connected at `http://127.0.0.1:25503`.
- The dashboard warning banner clears automatically within 60 seconds.
- To get fresh recommendations, re-run `recommend.py` after ThetaData is up:
```bat
.venv\Scripts\activate
python -m src.recommend
```

---

### recommend.py shows "no expiry found in 14–45 DTE range"

**Cause:** No option expiration falls within the configured DTE window.
**Fix:** Widen `dte_min` / `dte_max` in `config.json`, or wait until a qualifying expiry is listed.

---

### Recommend is very slow (several minutes)

**Cause:** `hmmlearn` is not installed; the pure-NumPy fallback is being used for re-fitting.
**Fix (option A):** Click **[Fit HMM]** in the dashboard to populate the cache, then re-run `run.bat`. Subsequent runs use the cache and complete in ~10 seconds.
**Fix (option B):** Install C++ Build Tools and `pip install hmmlearn` for faster fitting.

---

### Duplicate trades in Trade Tracker

**Cause:** `recommend.py` was run more than once on the same day.
**Fix:** Run the deduplication snippet from Section 6.

---

### Dashboard port 8501 already in use

**Cause:** A previous Streamlit session is still running.
**Fix:**
```powershell
Get-NetTCPConnection -LocalPort 8501 |
  Select-Object -ExpandProperty OwningProcess |
  ForEach-Object { Stop-Process -Id $_ -Force }
```
Then re-run `run.bat`.

---

### "No module named X" error

**Cause:** A dependency is missing from the virtual environment.
**Fix:**
```bat
.venv\Scripts\activate
pip install -r requirements.txt
```
For `hmmlearn` specifically, see Section 3.1.

---

### "No module named 'sklearn.utils'" (or other sklearn submodule)

**Cause:** `scikit-learn` was installed outside of pip (e.g. via conda or a system installer) and is missing its `utils` subpackage. The RECORD file is absent, so `pip install --force-reinstall` cannot remove it automatically.

**Fix:**

1. Find the broken sklearn directory:
```powershell
python -c "import site; print(site.getsitepackages())"
```

2. Manually delete the `sklearn` folder from that path, e.g.:
```powershell
Remove-Item -Recurse -Force "C:\Users\saika\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn"
```

3. Also delete any orphaned dist-info:
```powershell
Remove-Item -Recurse -Force "C:\Users\saika\AppData\Local\Programs\Python\Python314\Lib\site-packages\scikit_learn-*.dist-info"
```

4. Reinstall cleanly:
```bat
pip install scikit-learn
```

This installs the latest version (1.8.0 as of 2026-03-13) with a proper RECORD file.

---

### yfinance fetch returns no data

**Cause:** yfinance has a ~730-day lookback limit for 1h bars.
**Fix:** Import fresh Barchart CSVs to extend history, then let yfinance top up from the parquet's last bar forward.

---

### Trade P&L shows stale prices

**Cause:** The EOD scheduler has not run yet, or the dashboard was not open at 16:05 ET.
**Fix:** Go to Trade Tracker tab and click **[Refresh Prices]**.

---

## 14. Configuration Reference

### Files

| File | Purpose |
|------|---------|
| `config.json` | Tickers list and all option strategy parameters |
| `run.bat` | Entry point — activates venv, starts recommend, launches dashboard |
| `data/*.parquet` | UTC-indexed RTH OHLCV bars per ticker |
| `data/hmm_cache.pkl` | Cached HMM fit results (reused for up to 7 days by recommend.py) |
| `data/tracked_trades.json` | All tracked trades with daily P&L history |
| `data/paper_trades.json` | Paper execution log |
| `.env` | Optional: `TRADE_MODE`, `SCHWAB_TOKEN_PATH` |

### Source modules

| Module | Role |
|--------|------|
| `src/data_loader.py` | CSV parse, yfinance fetch, parquet read/write, RTH filter |
| `src/hmm_model.py` | GaussianHMM fitting, regime labelling, forecasting |
| `src/_hmm_pure.py` | Pure NumPy HMM fallback (used when hmmlearn is not installed) |
| `src/thetadata.py` | ThetaData REST client, Black-Scholes IV and delta |
| `src/recommend.py` | Strike selection, trade construction, recommendation output |
| `src/trade_tracker.py` | TrackedTrade persistence, daily price updates, P&L |
| `src/scheduler.py` | APScheduler: hourly data refresh + 16:05 ET EOD update |
| `src/options.py` | Strategy map and order building |
| `src/broker.py` | Paper log and Schwab live execution |
| `src/dashboard.py` | Streamlit 5-tab UI |

### Regime → strategy mapping

| HMM Regime | Condition | Strategy |
|-----------|-----------|----------|
| `directional_bull` | mean log-ret > +0.0003/hr | Bull Call Vertical Spread |
| `directional_bear` | mean log-ret < −0.0003/hr | Bear Put Vertical Spread |
| `vol_expansion` | mean 20-bar vol > 0.008 and not trending | Long Strangle |
| `mean_reverting` | low vol, low trend | Iron Condor |

### Known co-existing Python processes (do not kill)

| Project | Process |
|---------|---------|
| silentfacts | `src/app.py`, `src/subsidiary_scraper.py` |
| earnings_research_pipeline | `scripts/run_dc_pipeline.py` |

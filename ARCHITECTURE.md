# HMM Regime Dashboard — Architecture & Implementation Logic

## Overview

A multi-ticker quantitative trading system that uses Hidden Markov Models (HMM) to detect market regimes across 7 large-cap US equities and automatically recommends options strategies aligned to each regime. Recommendations are tracked daily with live P&L via ThetaData real-time quotes.

**Universe:** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA
**Data:** 1-hour RTH bars (2021 → present)
**Entry point:** `run.bat` → `src/recommend.py` → `src/dashboard.py`

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                               │
│                                                                     │
│   Barchart CSVs          yfinance API         ThetaData Terminal    │
│   (2021–Feb 2026)        (last 730 days)      (localhost:25503)     │
│   Historical 1h bars     Recent 1h bars       Live option quotes    │
└──────────┬───────────────────┬────────────────────────┬────────────┘
           │                   │                        │
           ▼                   ▼                        ▼
┌──────────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│   data_loader.py     │  │  data_loader.py  │  │   thetadata.py       │
│                      │  │                  │  │                      │
│  _parse_csv_file()   │  │ fetch_yfinance() │  │  get_chain()         │
│  import_csv_to_      │  │ update_with_     │  │  get_quotes()        │
│    parquet()         │  │   yfinance()     │  │  find_expiry()       │
│  filter_rth()        │  │  filter_rth()    │  │  BS IV + delta       │
└──────────┬───────────┘  └────────┬────────┘  └──────────┬───────────┘
           │                       │                       │
           └──────────┬────────────┘                       │
                      ▼                                     │
           ┌──────────────────────┐                        │
           │  data/*.parquet      │                        │
           │  UTC-indexed OHLCV   │                        │
           │  RTH filtered        │                        │
           │  ~7,500 bars/ticker  │                        │
           └──────────┬───────────┘                        │
                      │                                     │
                      ▼                                     │
           ┌──────────────────────┐                        │
           │    hmm_model.py      │                        │
           │                      │                        │
           │  make_features()     │                        │
           │  fit_hmm()           │                        │
           │  characterize_       │                        │
           │    regimes()         │                        │
           │  regime_forecast()   │                        │
           │  run_all_tickers()   │                        │
           └──────────┬───────────┘                        │
                      │                                     │
                      │   TickerResult per ticker           │
                      │   (regime, confidence, forecast)    │
                      │                                     │
                      ▼                                     │
           ┌──────────────────────────────────────────────┐│
           │              recommend.py                     ││
           │                                              │◄┘
           │  For each ticker:                            │
           │    regime_type → strategy selector           │
           │    ThetaData chain → strike selection        │
           │    BS delta/IV → leg construction            │
           │    build_trade() → save_trade()              │
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
           ┌──────────────────────────────────────────────┐
           │         data/tracked_trades.json              │
           │                                              │
           │  TrackedTrade per ticker:                    │
           │  entry price, legs, daily_prices[], P&L     │
           └──────────────────────┬───────────────────────┘
                                  │
                      ┌───────────┴───────────┐
                      │                       │
                      ▼                       ▼
           ┌──────────────────┐   ┌──────────────────────┐
           │   dashboard.py   │   │    scheduler.py       │
           │                  │   │                       │
           │  Streamlit UI    │   │  Hourly HMM refresh   │
           │  5 tabs          │   │  16:05 ET EOD P&L     │
           │  localhost:8501  │   │  update               │
           └──────────────────┘   └──────────────────────┘
```

---

## Component Detail

### 1. `src/data_loader.py` — Data Ingestion

**Purpose:** Ingest historical and live price bars, store in parquet, serve to HMM.

**Key logic:**

**CSV parsing (`_parse_csv_file`)**
- Reads Barchart export: `Time, Open, High, Low, Latest, Change, %Change, Volume`
- Strips footer line starting with `"Downloaded"`
- Renames `Latest` → `close`
- Timestamps have no timezone → localised to `America/New_York` → converted to UTC
- RTH filter applied: keeps bars where NY time is in `[09:30, 16:00)`
- Sorted ascending, duplicates removed (keep last)

**yfinance fetch (`fetch_yfinance`)**
- Downloads 1h bars using `yf.download(tickers=ticker, interval="1h")`
- Handles MultiIndex columns from yfinance (flattens to lowercase)
- Converts to UTC, applies RTH filter
- Max lookback: ~730 days

**Incremental update (`update_with_yfinance`)**
- If parquet exists: `start = last_bar - 5h` (small overlap to avoid gaps)
- If no parquet: `start = now - 725d` (full backfill)
- Concatenates with existing, deduplicates (keep last on overlap)

**Parquet schema**
```
timestamp (UTC, index)  |  open  |  high  |  low  |  close  |  volume
```

---

### 2. `src/hmm_model.py` — HMM Regime Detection

**Purpose:** Fit Gaussian HMM to price features, label regimes, forecast transitions.

**Feature engineering (`make_features`)**
Three features per bar:
```
log_ret  = log(close_t / close_{t-1})      # hourly log return
vol_20   = rolling(20).std(log_ret)        # short-term volatility
log_vol  = log(volume)                     # volume signal
```
Features standardised with `StandardScaler` before fitting.

**HMM fitting (`fit_hmm`)**
- `GaussianHMM(n_components=4, covariance_type="full", n_iter=300)`
- Uses hmmlearn if installed; falls back to pure NumPy implementation (`_hmm_pure.py`)
- Baum-Welch EM algorithm learns transition matrix and emission distributions

**State relabelling (`_relabel_states`)**
HMM states are arbitrary after fitting. To make labels deterministic across re-fits:
1. Invert scaled state means back to original log-ret space
2. Sort states by mean log-ret ascending
3. Remap: label 0 = most bearish … label 3 = most bullish
4. Reorder `transmat_`, `means_`, `covars_`, `startprob_` to match new ordering

**Regime characterisation (`characterize_regimes`)**
Each regime is classified using two thresholds:
```
TREND_THRESHOLD = 0.0003   # hourly mean log-ret (~7% annualised)
VOL_THRESHOLD   = 0.008    # mean 20-bar rolling vol
```

| Condition | Regime type | Strategy |
|-----------|------------|----------|
| `mean_ret > TREND_THRESHOLD` | `directional_bull` | Bull Call Spread |
| `mean_ret < -TREND_THRESHOLD` | `directional_bear` | Bear Put Spread |
| `mean_vol > VOL_THRESHOLD` and not trending | `vol_expansion` | Long Strangle |
| otherwise | `mean_reverting` | Iron Condor |

**Forecasting (`regime_forecast`)**
- Propagates current posterior through transition matrix: `future = posterior @ P^k`
- Returns probability of regime change within 24-hour horizon

**Output: `TickerResult`**
```python
ticker, model, scaler, df_reg, df_prices,
characteristics: dict[int, RegimeCharacteristics],
forecast: dict,
current_regime: int,
n_states: int,
error: str | None
```

---

### 3. `src/thetadata.py` — Live Option Quotes

**Purpose:** Fetch real-time bid/ask from ThetaData Terminal, enrich with Greeks.

**ThetaData Value subscription endpoints used:**
- `GET /v3/option/list/expirations` — all expiries for a symbol
- `GET /v3/option/snapshot/quote` — real-time bid/ask for all strikes

**Greeks (computed locally — Standard/Pro plan required for ThetaData greeks):**

*Implied Volatility* — Brent root-finding via `scipy.optimize.brentq`:
```
solve: BS_price(S, K, T, r, σ) = market_mid
```

*Delta* — Black-Scholes analytic:
```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
delta_call = N(d1)
delta_put  = N(d1) - 1
```

**`get_chain()`** — returns enriched DataFrame:
```
strike | right | bid | ask | mid | bid_size | ask_size | iv | delta | timestamp
```

**Expiry selection (`find_expiry`):**
- Fetches all expiries from today onward
- Returns expiry closest to `TARGET_DTE=21` within `[DTE_MIN=14, DTE_MAX=45]`

---

### 4. `src/recommend.py` — Strategy Recommendation Engine

**Purpose:** Combine HMM regime with live option chain to output specific trade recommendations.

**Flow for each ticker:**
```
1. Get current regime type from HMM result
2. Call td.find_expiry() → target ~21 DTE expiry
3. Call td.get_chain() → enriched option chain with IV + delta
4. Apply strike selector based on regime type
5. Print recommendation with legs, prices, Greeks, confidence
6. Call build_trade() + save_trade() → persist to tracked_trades.json
```

**Strike selection logic:**

| Strategy | Long leg | Short leg |
|----------|----------|-----------|
| Bull Call Spread | `nearest_delta(calls, 0.40)` | next strike above long |
| Bear Put Spread | `nearest_delta(puts, -0.40)` | next strike below long |
| Long Strangle | `nearest_strike(calls, S × 1.03)` | `nearest_strike(puts, S × 0.97)` |
| Iron Condor | Short: `nearest_delta(calls, 0.16)` / `nearest_delta(puts, -0.16)` | Long: 2 strikes beyond short |

**Net price calculation:**
- Debit spreads: `net = max(long_mid - short_mid, 0.05)`
- Credit spreads: `net = max((short_call_mid + short_put_mid) - (long_call_mid + long_put_mid), 0.05)`

---

### 5. `src/trade_tracker.py` — Trade Persistence & P&L Tracking

**Purpose:** Persist trade recommendations and track daily mark-to-market P&L.

**Data model:**
```python
TrackedTrade:
  id                  # ticker-expiry-strategy-date-uuid4[:4]
  date_recommended    # ISO date
  recommended_at      # "YYYY-MM-DD HH:MM:SS UTC" — full datetime
  ticker, expiry, dte_at_entry, underlying_at_entry
  strategy, regime_type, regime_name
  price_type          # "debit" | "credit"
  entry_net           # always positive
  max_profit, max_loss
  legs: list[dict]    # {action, right, strike, entry_mid, entry_ask, entry_bid}
  daily_prices: list[dict]  # {date, underlying, net_mid, pnl_dollars}
  status              # "open" | "closed" | "expired"
  confidence          # HMM regime confidence at entry (0.0–1.0)
```

**P&L convention (per 1 contract = ×100 multiplier):**
```
debit  trade:  pnl = (current_net_mid - entry_net) × 100
credit trade:  pnl = (entry_net - current_net_mid) × 100
```

**Daily price update (`update_trade_prices`):**
1. Fetch current underlying price via yfinance
2. For each unique (right, strike) pair, fetch current mid from ThetaData
3. Compute net mid: `Σ (sign × mid)` where BUY=+1, SELL=-1
4. Compute P&L, append `DailyPrice` entry (idempotent — replaces if same date)

**Auto-expiry (`mark_expired`):**
- Sets `status="expired"` for any open trade whose `expiry < today`

---

### 6. `src/scheduler.py` — Background Automation

**Purpose:** Run data refresh and P&L updates on a schedule without user intervention.

**Jobs (APScheduler BackgroundScheduler):**

| Job | Schedule | Action |
|-----|----------|--------|
| `hourly_refresh` | Mon-Fri, :30 past each hour 9:30–15:30 ET | Fetch yfinance → refit HMM → save `hmm_cache.pkl` |
| `eod_price_update` | Mon-Fri, 16:05 ET | Fetch ThetaData mids → update all open trade P&L |

**Cache (`data/hmm_cache.pkl`):**
- Pickled dict: `{results, proposed, updated_at}`
- Dashboard loads cache on session start so HMM results survive page refreshes
- `@st.fragment(run_every="5m")` checks cache mtime every 5 minutes and reloads if updated

---

### 7. `src/dashboard.py` — Streamlit UI

**Purpose:** Unified multi-ticker dashboard with 5 tabs.

| Tab | Content |
|-----|---------|
| **Regime Overview** | Summary table (all tickers), per-ticker expandable cards with price + posterior charts |
| **Price Charts** | Full history price chart with regime shading, regime stats table |
| **Options & Trading** | Proposed orders table, Execute button, paper trade log |
| **Model Diagnostics** | Transition matrix heatmap, state means, log-ret distribution by regime |
| **Trade Tracker** | Recommendations table, Live P&L table, per-trade cards with daily P&L charts |

**Trade Tracker tab structure:**
```
[Refresh Prices button]

Recommendations table:
  Rec. Date/Time | Ticker | Regime | Price at Entry | Strategy |
  Entry (debit/credit $X) | Confidence | Expiry | Status

Live P&L table:
  Ticker | Strategy | Entry $ | Current $ | P&L $ | P&L % | DTE left | Status

Per-trade expandable cards:
  Metrics: Entry / Current / P&L $ / DTE / Days held
  Leg details table
  Daily P&L line chart (fill above/below zero)
  Underlying price chart
```

---

## Data Flow Summary

```
run.bat
  │
  ├─ python -m src.recommend
  │     │
  │     ├─ load_all_tickers()         ← data/*.parquet
  │     ├─ run_all_tickers()          → TickerResult per ticker
  │     ├─ td.find_expiry()           ← ThetaData /v3/option/list/expirations
  │     ├─ td.get_chain()             ← ThetaData /v3/option/snapshot/quote
  │     │   └─ BS IV + delta computed locally
  │     ├─ strike selection logic
  │     ├─ build_trade()
  │     └─ save_trade()               → data/tracked_trades.json
  │
  └─ streamlit run src/dashboard.py  → http://localhost:8501
        │
        ├─ load_cache()               ← data/hmm_cache.pkl (if exists)
        ├─ get_scheduler()            → background thread
        │     ├─ :30 ET hourly:  update_with_yfinance() + run_all_tickers()
        │     └─ 16:05 ET daily: update_all_open_trades()
        │
        └─ Trade Tracker tab
              ├─ load_trades()        ← data/tracked_trades.json
              └─ [Refresh Prices] → update_all_open_trades()
                    ├─ td.get_quotes() per leg   ← ThetaData
                    ├─ compute net_mid
                    └─ append DailyPrice entry
```

---

## File Reference

| File | Role |
|------|------|
| `config.json` | User configuration: tickers list, option strategy parameters |
| `run.bat` | Entry point: runs recommend.py then launches dashboard |
| `src/data_loader.py` | CSV parsing, yfinance fetch, parquet I/O, RTH filter |
| `src/hmm_model.py` | GaussianHMM fitting, regime labelling, forecasting |
| `src/_hmm_pure.py` | Pure NumPy HMM fallback (for Python 3.14, no hmmlearn wheel) |
| `src/thetadata.py` | ThetaData REST client, Black-Scholes IV + delta |
| `src/recommend.py` | Strike selection, trade building, recommendation output |
| `src/trade_tracker.py` | TrackedTrade dataclass, daily P&L updates, JSON persistence |
| `src/scheduler.py` | APScheduler background jobs (hourly + EOD) |
| `src/options.py` | Strategy map, order building for paper/live execution |
| `src/broker.py` | Paper trade log + Schwab live execution |
| `src/dashboard.py` | Streamlit 5-tab UI |
| `src/hmm.py` | Legacy single-ticker Alpha Vantage app (retired, reference only) |
| `data/*.parquet` | UTC-indexed RTH OHLCV bars per ticker |
| `data/tracked_trades.json` | Open/closed trade records with daily P&L history |
| `data/paper_trades.json` | Paper execution log from broker.py |
| `data/hmm_cache.pkl` | Pickled HMM results for dashboard session persistence |
| `STARTUP.md` | Start/stop instructions |
| `ARCHITECTURE.md` | This document |

---

## Adding New Tickers

### Step-by-step

**Step 1 — Get historical data from Barchart**
- Go to [barchart.com](https://www.barchart.com), search for the ticker
- Download the 1-hour intraday historical CSV
- Required format: `Time, Open, High, Low, Latest, Change, %Change, Volume`
  - Rows must be descending time order
  - Last line must be the Barchart footer (`"Downloaded from Barchart..."`)
- Save to `data/` with a filename starting with the ticker in lowercase:
  ```
  data/spy_intraday-60min_historical-data.csv
  ```

**Step 2 — Add the ticker to `config.json`**
```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY"]
}
```
This is the **only change required**. Everything downstream — HMM fitting, ThetaData chain fetch, dashboard, scheduler — reads from `config.json` automatically via `data_loader.TICKERS`.

**Step 3 — Import the CSV**
Either use the dashboard sidebar button **[Import CSVs]**, or run from terminal:
```bat
.venv\Scripts\activate
python -c "from src.data_loader import import_csv_to_parquet; _, msg = import_csv_to_parquet('SPY'); print(msg)"
```
This creates `data/SPY_1h.parquet`.

**Step 4 — Run as normal**
Run `run.bat`. The new ticker will be:
- Topped up with recent bars from yfinance automatically
- Fitted with GaussianHMM alongside all other tickers
- Included in ThetaData option chain fetch and recommendation output
- Shown in all dashboard tabs

---

### Requirements per ticker

| Requirement | Detail |
|-------------|--------|
| **Min bars** | 500 RTH bars after feature engineering (~1 month of hourly data minimum; 3+ months recommended for stable HMM) |
| **CSV format** | Barchart 1h intraday export (see above) |
| **ThetaData coverage** | Ticker must have listed options on ThetaData Value plan (most liquid US equities and ETFs qualify) |
| **yfinance coverage** | Ticker must be available on yfinance for incremental updates (virtually all US equities) |
| **Options liquidity** | Recommend selecting tickers with tight bid/ask spreads and active open interest — wide spreads make delta/IV calculations unreliable |

---

### How many tickers can you add?

There is no hard-coded limit. Practical constraints are:

| Factor | Guideline |
|--------|-----------|
| **HMM fitting time** | Each ticker takes ~2–5 seconds to fit. 7 tickers ≈ 30s. 20 tickers ≈ 90s. 50 tickers would be ~4 minutes on startup. |
| **ThetaData API rate** | The Value plan has no documented rate limit for snapshot endpoints, but fetching chains for 20+ tickers sequentially may be slow (~1–2s per ticker). |
| **Dashboard performance** | Streamlit renders all tickers in the Regime Overview tab. Beyond ~20 tickers the page becomes slow to load. |
| **Memory** | Each parquet is ~5–10 MB in memory. 50 tickers ≈ 250–500 MB RAM — fine on a modern machine. |
| **Recommended sweet spot** | **10–20 tickers** — fast enough to run on startup, dashboard stays responsive. |

---

### Removing a ticker

1. Remove the ticker from `config.json`
2. Optionally delete `data/TICKER_1h.parquet` to free disk space
3. The ticker will no longer appear anywhere in the system

---

## Key Configuration Constants

All user-tunable parameters live in **`config.json`** at the project root. No code edits needed.

### Editing parameters

**Option strategy parameters are editable directly from the dashboard:**
- Open the sidebar → **Strategy Config** → **Option strategy parameters**
- Adjust any value using the sliders and number inputs
- Click **Save to config.json** — takes effect on the next `recommend.py` run

`config.json` is also directly editable as a plain text file if preferred.

### `config.json` reference

```json
{
  "tickers": ["AAPL", "MSFT", ...],

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

| Key | Default | Dashboard control | Purpose |
|-----|---------|-------------------|---------|
| `tickers` | 7 large-caps | Add to list + Import CSVs | Universe of tickers to analyse and trade |
| `option_strategy.target_dte` | 21 | Number input | Target days-to-expiry when selecting an expiration |
| `option_strategy.dte_min` | 14 | Number input | Minimum acceptable DTE |
| `option_strategy.dte_max` | 45 | Number input | Maximum acceptable DTE |
| `option_strategy.delta_vert` | 0.40 | Slider 0.10–0.70 | Long leg delta for Bull Call / Bear Put verticals |
| `option_strategy.delta_wing` | 0.16 | Slider 0.05–0.40 | Short leg delta for Iron Condor wings |
| `option_strategy.otm_pct` | 0.03 | Slider 0.01–0.15 | How far OTM (as % of spot) to place Long Strangle legs |
| `option_strategy.strike_range` | 20 | Number input | Number of strikes above/below ATM to fetch from ThetaData |

### Internal constants (in source code)

These are less frequently changed and remain in source:

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `TREND_THRESHOLD` | 0.0003 | `hmm_model.py` | Min hourly log-ret to classify regime as trending (~7% annualised) |
| `VOL_THRESHOLD` | 0.008 | `hmm_model.py` | Min rolling vol to classify regime as high-volatility |
| `RISK_FREE` | 0.053 | `thetadata.py` | Risk-free rate for Black-Scholes IV and delta calculation |
| `BASE_URL` | localhost:25503/v3 | `thetadata.py` | ThetaData Terminal endpoint |

"""
scheduler.py — Background hourly refresh during market hours.

Runs as a daemon thread (APScheduler BackgroundScheduler).
Every hour Mon-Fri 10:00-15:00 ET it:
  1. Fetches latest 1h bars from yfinance for all tickers
  2. Refits GaussianHMM
  3. Executes paper trades
  4. Saves results to data/hmm_cache.pkl for the dashboard to read

Start once per process via get_scheduler().
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import pytz

CACHE_PATH = Path("data/hmm_cache.pkl")
NY_TZ = pytz.timezone("America/New_York")

_scheduler_lock = threading.Lock()
_scheduler_started = False


def _get_yfinance_price(ticker: str, fallback: float) -> float:
    """Return the latest yfinance spot price, or fallback if unavailable."""
    try:
        import yfinance as yf
        price = float(yf.Ticker(ticker).fast_info.last_price)
        return price if price > 0 else fallback
    except Exception:
        return fallback


def _option_strategy_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("option_strategy", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _eod_price_update_job() -> None:
    """Fetch current option mid-prices for all open tracked trades and record daily P&L."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.trade_tracker import update_all_open_trades
    print(f"[scheduler] EOD price update started at {datetime.now(NY_TZ).strftime('%H:%M ET')}")
    try:
        updated = update_all_open_trades()
        open_count = sum(1 for t in updated if t.status == "open")
        print(f"[scheduler] EOD price update complete — {open_count} open trades updated")
    except Exception as e:
        print(f"[scheduler] EOD price update error: {e}")


def _refresh_job(n_states: int = 4, trade_mode: str = "paper") -> None:
    """Fetch yfinance, refit HMM, execute paper trades, save cache."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loader import load_all_tickers, update_with_yfinance, TICKERS
    from src.hmm_model import run_all_tickers
    from src.options import build_order_from_thetadata_chain, option_chain_unavailable_meta
    from src.broker import execute_order
    from src.trade_tracker import load_trades, check_regime_alerts, update_trade_prices, _save_all
    from src import thetadata as td

    print(f"[scheduler] refresh started at {datetime.now(NY_TZ).strftime('%H:%M ET')}")

    # 1. Fetch yfinance for all tickers
    for t in TICKERS:
        try:
            _, msg = update_with_yfinance(t)
            print(f"[scheduler] {msg}")
        except Exception as e:
            print(f"[scheduler] yfinance error {t}: {e}")

    # 2. Fit HMM
    bars = load_all_tickers()
    results = run_all_tickers(bars, n_states=n_states)

    # 3. Check regime-change alerts for open tracked trades
    try:
        current_regime_types = {
            ticker: res.characteristics[res.current_regime].regime_type
            for ticker, res in results.items()
            if not res.error and res.characteristics and res.current_regime in res.characteristics
        }
        trades = load_trades()
        trades = check_regime_alerts(trades, current_regime_types)

        # Auto-close trades where regime has changed to incompatible type
        closed_tickers = []
        for tr in trades:
            if tr.status == "open" and tr.regime_alert:
                try:
                    tr = update_trade_prices(tr)
                except Exception:
                    pass  # close with last known price
                tr.status = "closed"
                # Set underlying_at_close from latest daily snapshot
                if tr.daily_prices:
                    tr.underlying_at_close = tr.daily_prices[-1].get("underlying", tr.underlying_at_entry)
                else:
                    tr.underlying_at_close = tr.underlying_at_entry
                new_regime = current_regime_types.get(tr.ticker, "unknown")
                closed_tickers.append(f"{tr.ticker}({tr.strategy}->{new_regime})")

        _save_all(trades)
        if closed_tickers:
            print(f"[scheduler] AUTO-CLOSED: {', '.join(closed_tickers)}")
        alerts = [t for t in trades if t.regime_alert and t.status == "open"]
        if alerts:
            print(f"[scheduler] REGIME ALERTS (still open): {', '.join(t.ticker for t in alerts)}")
    except Exception as e:
        print(f"[scheduler] regime alert check error: {e}")

    # 4. Build orders from ThetaData option quotes and execute paper/live trades.
    opt_cfg = _option_strategy_config()
    target_dte = opt_cfg.get("target_dte", 21)
    dte_min = opt_cfg.get("dte_min", 14)
    dte_max = opt_cfg.get("dte_max", 45)
    delta_vert = opt_cfg.get("delta_vert", 0.40)
    delta_wing = opt_cfg.get("delta_wing", 0.16)
    otm_pct = opt_cfg.get("otm_pct", 0.03)
    strike_range = opt_cfg.get("strike_range", 20)
    td_up = td.is_available()
    print(f"[scheduler] ThetaData terminal: {'connected' if td_up else 'unavailable - option orders disabled'}")

    proposed = []
    for t, res in results.items():
        if res.error or not res.characteristics:
            continue
        rc = res.characteristics.get(res.current_regime)
        if rc is None:
            continue
        last_close = float(res.df_prices["close"].iloc[-1])
        spot = _get_yfinance_price(t, last_close)

        if not td_up:
            order = None
            meta = option_chain_unavailable_meta(
                t, rc.regime_type, spot, "ThetaData unavailable - option chain required"
            )
        else:
            exp = td.find_expiry(t, target_dte, dte_min, dte_max)
            if exp is None:
                order = None
                meta = option_chain_unavailable_meta(
                    t, rc.regime_type, spot, f"No expiry in {dte_min}-{dte_max} DTE range"
                )
            else:
                chain_df = td.get_chain(t, exp, spot, strike_range=strike_range)
                if chain_df.empty:
                    order = None
                    meta = option_chain_unavailable_meta(
                        t, rc.regime_type, spot, "Empty chain from ThetaData", expiry=exp
                    )
                else:
                    calls = td.get_calls(chain_df)
                    puts = td.get_puts(chain_df)
                    order, meta = build_order_from_thetadata_chain(
                        ticker=t,
                        regime_type=rc.regime_type,
                        underlying_price=spot,
                        expiry=exp,
                        calls=calls,
                        puts=puts,
                        delta_vert=delta_vert,
                        delta_wing=delta_wing,
                        otm_pct=otm_pct,
                    )

        if order is None:
            proposed.append({"ticker": t, "order": order, "meta": meta, "rc": rc, "record": None})
            print(f"[scheduler] {t}: skipped ({meta.get('error') or rc.regime_type})")
            continue
        rec = execute_order(order, meta, mode=trade_mode)
        proposed.append({"ticker": t, "order": order, "meta": meta, "rc": rc, "record": rec})
        print(f"[scheduler] {t}: {rec.status} ({rc.regime_type})")

    # 4. Save cache
    cache = {
        "results": results,
        "proposed": proposed,
        "updated_at": datetime.now(NY_TZ),
    }
    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp, CACHE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    print(f"[scheduler] refresh complete — {len(results)} tickers")


# ---------------------------------------------------------------------------
# Market-hours check
# ---------------------------------------------------------------------------

def _is_market_hours() -> bool:
    """True if current NY time is Mon-Fri 09:30-16:00 ET."""
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t = now.time()
    from datetime import time
    return time(9, 30) <= t < time(16, 0)


def _is_refresh_window() -> bool:
    """True if current NY time is Mon-Fri 10:00-15:00 ET."""
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t = now.time()
    from datetime import time
    return time(10, 0) <= t <= time(15, 0)


def _market_hours_job(n_states: int, trade_mode: str) -> None:
    """Wrapper that skips the job outside the refresh window."""
    if _is_refresh_window():
        _refresh_job(n_states=n_states, trade_mode=trade_mode)
    else:
        print(f"[scheduler] outside refresh window, skipping ({datetime.now(NY_TZ).strftime('%H:%M ET')})")


# ---------------------------------------------------------------------------
# Singleton scheduler
# ---------------------------------------------------------------------------

def get_scheduler(n_states: int = 4, trade_mode: str = "paper"):
    """
    Start (or return the already-running) APScheduler BackgroundScheduler.
    Runs _market_hours_job every 30 minutes from 10:00-15:00 ET.
    Safe to call multiple times — only starts once per process.
    """
    global _scheduler_started

    with _scheduler_lock:
        if _scheduler_started:
            return

        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler(timezone=NY_TZ)
        scheduler.add_job(
            func=_market_hours_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="10-14",
                minute="0,30",
                timezone=NY_TZ,
            ),
            kwargs={"n_states": n_states, "trade_mode": trade_mode},
            id="half_hour_refresh",
            name="Half-hour HMM refresh",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            func=_market_hours_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=15,
                minute=0,
                timezone=NY_TZ,
            ),
            kwargs={"n_states": n_states, "trade_mode": trade_mode},
            id="close_refresh",
            name="Close HMM refresh",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            func=_eod_price_update_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=5,           # 5 min after close — options still quoted briefly
                timezone=NY_TZ,
            ),
            id="eod_price_update",
            name="EOD trade price update",
            max_instances=1,
            coalesce=True,
        )
        # Daily supervisor report — runs after EOD price update completes
        from src.supervisor import eod_supervisor_job
        scheduler.add_job(
            func=eod_supervisor_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=30,          # 25 min after price update — time for data to settle
                timezone=NY_TZ,
            ),
            id="eod_supervisor",
            name="EOD supervisor report",
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        _scheduler_started = True
        print(f"[scheduler] started — HMM refresh Mon-Fri every 30 min 10:00-15:00 ET, EOD price 16:05 ET, supervisor 16:30 ET")
        return scheduler


# ---------------------------------------------------------------------------
# Cache reader
# ---------------------------------------------------------------------------

def load_cache() -> dict | None:
    """Load the latest results from the cache file. Returns None if not found."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_mtime() -> float:
    """Modification time of cache file, or 0 if missing."""
    return CACHE_PATH.stat().st_mtime if CACHE_PATH.exists() else 0.0

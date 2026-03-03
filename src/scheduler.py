"""
scheduler.py — Background hourly refresh during market hours.

Runs as a daemon thread (APScheduler BackgroundScheduler).
Every hour Mon-Fri 10:00-16:00 ET it:
  1. Fetches latest 1h bars from yfinance for all tickers
  2. Refits GaussianHMM
  3. Executes paper trades
  4. Saves results to data/hmm_cache.pkl for the dashboard to read

Start once per process via get_scheduler().
"""
from __future__ import annotations

import pickle
import threading
from datetime import datetime
from pathlib import Path

import pytz

CACHE_PATH = Path("data/hmm_cache.pkl")
NY_TZ = pytz.timezone("America/New_York")

_scheduler_lock = threading.Lock()
_scheduler_started = False


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
    from src.options import select_and_build_order
    from src.broker import execute_order

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

    # 3. Execute paper trades
    proposed = []
    for t, res in results.items():
        if res.error or not res.characteristics:
            continue
        rc = res.characteristics.get(res.current_regime)
        if rc is None:
            continue
        last_close = float(res.df_prices["close"].iloc[-1])
        order, meta = select_and_build_order(t, rc.regime_type, {}, last_close)
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
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"[scheduler] refresh complete — {len(results)} tickers")


# ---------------------------------------------------------------------------
# Market-hours check
# ---------------------------------------------------------------------------

def _is_market_hours() -> bool:
    """True if current NY time is Mon-Fri 09:30-16:00."""
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t = now.time()
    from datetime import time
    return time(9, 30) <= t < time(16, 0)


def _market_hours_job(n_states: int, trade_mode: str) -> None:
    """Wrapper that skips the job outside market hours."""
    if _is_market_hours():
        _refresh_job(n_states=n_states, trade_mode=trade_mode)
    else:
        print(f"[scheduler] outside market hours, skipping ({datetime.now(NY_TZ).strftime('%H:%M ET')})")


# ---------------------------------------------------------------------------
# Singleton scheduler
# ---------------------------------------------------------------------------

def get_scheduler(n_states: int = 4, trade_mode: str = "paper"):
    """
    Start (or return the already-running) APScheduler BackgroundScheduler.
    Runs _market_hours_job every hour on the hour.
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
                hour="9-15",
                minute=30,          # fire at :30 past each hour (first bar complete)
                timezone=NY_TZ,
            ),
            kwargs={"n_states": n_states, "trade_mode": trade_mode},
            id="hourly_refresh",
            name="Hourly HMM refresh",
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
        scheduler.start()
        _scheduler_started = True
        print(f"[scheduler] started — HMM refresh Mon-Fri :30 ET, EOD price update 16:05 ET")
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

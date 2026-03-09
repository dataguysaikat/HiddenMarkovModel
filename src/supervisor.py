"""
supervisor.py — End-of-day watchdog that monitors models, trades, and data quality.

Runs automatically at 16:30 ET via scheduler, or manually:
    python -m src.supervisor              # report only
    python -m src.supervisor --shutdown   # report + kill Streamlit

Produces a daily text report in data/reports/YYYY-MM-DD.txt and prints to console.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path

import pytz

NY_TZ      = pytz.timezone("America/New_York")
DATA_DIR   = Path("data")
CACHE_PATH = DATA_DIR / "hmm_cache.pkl"
REPORT_DIR = DATA_DIR / "reports"
CONFIG_PATH = Path("config.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_et() -> datetime:
    return datetime.now(NY_TZ)


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        return {}


def _load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section generators — each returns a list of report lines
# ---------------------------------------------------------------------------

def _section_model_health(cache: dict | None, tickers: list[str]) -> list[str]:
    """HMM model status per ticker."""
    lines = ["", "MODEL HEALTH", "-" * 60]
    if cache is None:
        lines.append("  No HMM cache found — models have not been fit today.")
        return lines

    results = cache.get("results", {})
    updated = cache.get("updated_at")
    if updated:
        lines.append(f"  Last HMM fit: {updated.strftime('%Y-%m-%d %H:%M %Z')}")

    ok, errors = 0, 0
    for ticker in tickers:
        res = results.get(ticker)
        if res is None:
            lines.append(f"  {ticker:<6}  MISSING — not in cache")
            errors += 1
            continue
        if res.error:
            lines.append(f"  {ticker:<6}  ERROR — {res.error}")
            errors += 1
            continue

        rc   = res.characteristics.get(res.current_regime)
        name = rc.name if rc else "unknown"
        rtype = rc.regime_type if rc else "unknown"
        conf = res.forecast.get("current_confidence", 0)
        bars = len(res.df_prices) if res.df_prices is not None else 0
        lines.append(f"  {ticker:<6}  OK  regime={rtype:<20} conf={conf:.0%}  bars={bars}")
        ok += 1

    lines.append(f"  Summary: {ok} OK, {errors} errors out of {len(tickers)} tickers")
    return lines


def _section_trade_summary() -> list[str]:
    """Open/closed trade P&L summary."""
    from src.trade_tracker import load_trades, latest_pnl, pnl_pct, dte_remaining

    lines = ["", "TRADE SUMMARY", "-" * 60]
    trades = load_trades()

    if not trades:
        lines.append("  No tracked trades.")
        return lines

    open_trades   = [t for t in trades if t.status == "open"]
    closed_trades = [t for t in trades if t.status in ("closed", "expired")]
    alerted       = [t for t in open_trades if getattr(t, "regime_alert", "")]

    # Open trades
    lines.append(f"  Open:    {len(open_trades)} trades")
    total_pnl = 0.0
    for t in open_trades:
        _, pnl = latest_pnl(t)
        pct = pnl_pct(t)
        dte = dte_remaining(t)
        total_pnl += pnl
        lines.append(f"    {t.ticker:<6} {t.strategy:<22} P&L ${pnl:>+8.2f} ({pct:>+6.1%})  DTE {dte}")
    if open_trades:
        lines.append(f"    {'':40} Total ${total_pnl:>+8.2f}")

    # Alerts
    if alerted:
        lines.append(f"  Alerts:  {', '.join(t.ticker for t in alerted)}")
        for t in alerted:
            lines.append(f"    {t.ticker}: {t.regime_alert}")

    # Closed trades
    lines.append(f"  Closed/Expired: {len(closed_trades)} trades")
    if closed_trades:
        wins  = sum(1 for t in closed_trades if (latest_pnl(t)[1]) > 0)
        total = sum(latest_pnl(t)[1] for t in closed_trades)
        wr    = wins / len(closed_trades)
        lines.append(f"    Win rate: {wr:.0%}  ({wins}/{len(closed_trades)})  Total P&L: ${total:>+.2f}")

    # Expired today
    today_str = date.today().isoformat()
    expired_today = [t for t in closed_trades if t.expiry == today_str or
                     (t.status == "expired" and t.daily_prices and t.daily_prices[-1]["date"] == today_str)]
    if expired_today:
        lines.append(f"  Expired today: {len(expired_today)}")
        for t in expired_today:
            _, pnl = latest_pnl(t)
            lines.append(f"    {t.ticker:<6} {t.strategy:<22} Final P&L ${pnl:>+8.2f}")

    return lines


def _section_data_quality(tickers: list[str]) -> list[str]:
    """Check parquet freshness and ThetaData connectivity."""
    lines = ["", "DATA QUALITY", "-" * 60]

    # Parquet freshness
    for ticker in tickers:
        parquet = DATA_DIR / f"{ticker}_1h.parquet"
        if not parquet.exists():
            lines.append(f"  {ticker:<6}  MISSING parquet")
            continue
        try:
            import pandas as pd
            df = pd.read_parquet(parquet)
            last_ts = df["timestamp"].max() if "timestamp" in df.columns else df.index.max()
            lines.append(f"  {ticker:<6}  last bar: {last_ts}")
        except Exception as e:
            lines.append(f"  {ticker:<6}  READ ERROR: {e}")

    # ThetaData
    try:
        from src import thetadata as td
        td_up = td.is_available()
        lines.append(f"  ThetaData: {'connected' if td_up else 'OFFLINE'}")
    except Exception:
        lines.append(f"  ThetaData: could not check (import error)")

    return lines


def _section_policy_status() -> list[str]:
    """Current retrain_policy state from config.json."""
    lines = ["", "POLICY STATUS", "-" * 60]
    cfg = _load_config()

    policy = cfg.get("learned_policy", {})
    if not policy:
        lines.append("  No learned policy yet — retrain_policy.py has not produced results.")
    else:
        skip    = policy.get("skip_regimes", [])
        caution = policy.get("caution_regimes", [])
        min_c   = policy.get("min_confidence", 0)
        lines.append(f"  Min confidence : {min_c:.0%}" if min_c else "  Min confidence : not calibrated")
        lines.append(f"  Skip regimes   : {', '.join(skip) or 'none'}")
        lines.append(f"  Caution regimes: {', '.join(caution) or 'none'}")

    alert_pol = cfg.get("regime_alert_policy", {})
    confirmed = alert_pol.get("confirmed_incompatible", [])
    cleared   = alert_pol.get("cleared_incompatible", [])
    if confirmed or cleared:
        lines.append(f"  Confirmed harmful: {', '.join(f'{c[0]}->{c[1]}' for c in confirmed) or 'none'}")
        lines.append(f"  Cleared (false+) : {', '.join(f'{c[0]}->{c[1]}' for c in cleared) or 'none'}")

    perf = cfg.get("strategy_performance", {})
    if perf:
        lines.append("  Per-regime performance:")
        for regime, s in perf.items():
            lines.append(f"    {regime:<24} N={s['count']}  WR={s['win_rate']:.0%}  "
                         f"avg=${s['avg_pnl']:+.2f}  [{s['status']}]")

    return lines


def _section_improvements(cache: dict | None) -> list[str]:
    """Identify actionable improvement opportunities."""
    from src.trade_tracker import load_trades

    lines = ["", "IMPROVEMENT OPPORTUNITIES", "-" * 60]
    findings = []

    trades = load_trades()
    closed = [t for t in trades if t.status in ("closed", "expired")]
    open_t = [t for t in trades if t.status == "open"]

    # 1. Regime with consistently bad performance
    cfg  = _load_config()
    perf = cfg.get("strategy_performance", {})
    for regime, s in perf.items():
        if s["count"] >= 3 and s["win_rate"] < 0.35:
            findings.append(f"Regime '{regime}' has {s['win_rate']:.0%} win rate across "
                            f"{s['count']} trades — consider adding to skip list")

    # 2. Tickers with frequent HMM errors
    if cache:
        for ticker, res in cache.get("results", {}).items():
            if res.error:
                findings.append(f"{ticker}: HMM fitting failed — {res.error}")

    # 3. Trades approaching expiry with large losses
    for t in open_t:
        from src.trade_tracker import latest_pnl, dte_remaining
        _, pnl = latest_pnl(t)
        dte = dte_remaining(t)
        if dte <= 3 and pnl < -50:
            findings.append(f"{t.ticker} {t.strategy}: ${pnl:+.2f} with only {dte} DTE — "
                            f"consider closing to limit loss")

    # 4. Alerted trades that have been alerted for >2 days
    for t in open_t:
        alert_log = getattr(t, "alert_log", [])
        if alert_log:
            first_alert = alert_log[0].get("timestamp", "")
            if first_alert:
                try:
                    alert_dt = datetime.strptime(first_alert, "%Y-%m-%d %H:%M UTC")
                    days_alerted = (datetime.utcnow() - alert_dt).days
                    if days_alerted >= 2:
                        findings.append(f"{t.ticker}: regime alert active for {days_alerted} days — "
                                        f"original thesis may be invalid")
                except ValueError:
                    pass

    # 5. Sufficient data to run retrain_policy
    if len(closed) >= 3 and not perf:
        findings.append(f"{len(closed)} closed trades available — run "
                        f"`python -m src.retrain_policy` to generate policy")

    if not findings:
        findings.append("No actionable improvements identified at this time.")

    for f in findings:
        lines.append(f"  * {f}")

    return lines


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def generate_report() -> str:
    """Build the full daily supervisor report."""
    now = _now_et()
    cfg = _load_config()
    tickers = cfg.get("tickers", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"])
    cache = _load_cache()

    header = [
        "",
        "=" * 60,
        f"  DAILY SUPERVISOR REPORT",
        f"  {now.strftime('%Y-%m-%d %H:%M %Z')}",
        "=" * 60,
    ]

    sections = (
        _section_model_health(cache, tickers)
        + _section_trade_summary()
        + _section_data_quality(tickers)
        + _section_policy_status()
        + _section_improvements(cache)
    )

    footer = [
        "",
        "=" * 60,
        f"  Report generated at {now.strftime('%H:%M:%S %Z')}",
        "=" * 60,
        "",
    ]

    return "\n".join(header + sections + footer)


def save_report(report: str) -> Path:
    """Save report to data/reports/YYYY-MM-DD.txt. Returns the file path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{date.today().isoformat()}.txt"
    path.write_text(report, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _find_streamlit_pids() -> list[int]:
    """Find PIDs of Streamlit processes on port 8501 (Windows)."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue "
             "| Select-Object -ExpandProperty OwningProcess -Unique"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(pid) for pid in result.stdout.strip().split("\n") if pid.strip().isdigit()]
    except Exception:
        pass
    return []


def shutdown_streamlit() -> list[int]:
    """Kill Streamlit processes on port 8501. Returns list of killed PIDs."""
    pids = _find_streamlit_pids()
    killed = []
    for pid in pids:
        try:
            subprocess.run(
                ["powershell", "-Command", f"Stop-Process -Id {pid} -Force"],
                capture_output=True, timeout=10,
            )
            killed.append(pid)
        except Exception:
            pass
    return killed


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

def eod_supervisor_job() -> None:
    """Scheduled job: generate report, save, print summary."""
    print(f"[supervisor] generating daily report at {_now_et().strftime('%H:%M ET')}...")
    try:
        # Run retrain_policy first to update stats
        from src.retrain_policy import main as retrain_main
        try:
            retrain_main()
        except Exception as e:
            print(f"[supervisor] retrain_policy error: {e}")

        report = generate_report()
        path   = save_report(report)
        print(report)
        print(f"[supervisor] report saved to {path}")
    except Exception as e:
        print(f"[supervisor] report generation error: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    shutdown = "--shutdown" in sys.argv

    report = generate_report()
    path   = save_report(report)
    print(report)
    print(f"Report saved to {path}")

    if shutdown:
        print("\nShutting down Streamlit...")
        killed = shutdown_streamlit()
        if killed:
            print(f"Killed Streamlit PIDs: {killed}")
        else:
            print("No Streamlit processes found on port 8501.")
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

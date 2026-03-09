"""
data_loader.py — CSV import, yfinance fetch, parquet read/write for 7 tickers.

Barchart CSV format:
  Time, Open, High, Low, Latest (=close), Change, %Change, Volume
  Rows are descending time; last line is a footer "Downloaded from Barchart.com..."
  Timestamps have no timezone → localize to America/New_York → convert to UTC
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: {_CONFIG_PATH} is malformed — using defaults.")
    return {}

_config = _load_config()
TICKERS: list[str] = _config.get("tickers", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"])

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

NY_TZ = pytz.timezone("America/New_York")
UTC = pytz.UTC

_RTH_START = datetime(2000, 1, 1, 9, 30).time()
_RTH_END = datetime(2000, 1, 1, 16, 0).time()


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _parquet_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}_1h.parquet"


def load_local_bars(ticker: str) -> pd.DataFrame | None:
    p = _parquet_path(ticker)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def save_local_bars(ticker: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = df.reset_index(names="timestamp")
    out.to_parquet(_parquet_path(ticker), index=False)


# ---------------------------------------------------------------------------
# RTH filter
# ---------------------------------------------------------------------------

def filter_rth(df_utc: pd.DataFrame) -> pd.DataFrame:
    """Keep bars where New York local time is in [09:30, 16:00)."""
    if df_utc.empty:
        return df_utc
    idx_ny = df_utc.index.tz_convert(NY_TZ)
    t = idx_ny.time
    keep = (t >= _RTH_START) & (t < _RTH_END)
    return df_utc.loc[keep].copy()


# ---------------------------------------------------------------------------
# Barchart CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv_file(csv_path: Path) -> pd.DataFrame:
    """
    Parse a Barchart intraday CSV export.

    Columns expected: Time, Open, High, Low, Latest, Change, %Change, Volume
    Rows are newest-first; last row is a footer ('Downloaded from Barchart...').
    """
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    # Strip footer
    lines = [ln for ln in lines if not ln.startswith('"Downloaded') and not ln.startswith("Downloaded")]

    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines)))

    # Rename columns
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "Time": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Latest": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Parse timestamps — no tz in source
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Localize NY → UTC
    df["timestamp"] = df["timestamp"].dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    df["timestamp"] = df["timestamp"].dt.tz_convert(UTC)

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # RTH filter
    df = filter_rth(df)

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])

    return df


def find_csv_for_ticker(ticker: str) -> Path | None:
    """Find a Barchart CSV in data/ whose stem starts with ticker.lower()."""
    matches = sorted(DATA_DIR.glob(f"{ticker.lower()}*.csv"))
    return matches[0] if matches else None


def import_csv_to_parquet(ticker: str) -> tuple[pd.DataFrame, str]:
    """
    Parse the Barchart CSV for ticker, merge with any existing parquet
    (parquet wins on overlapping bars), and save.

    Returns (df, status_message).
    """
    csv_path = find_csv_for_ticker(ticker)
    if csv_path is None:
        return pd.DataFrame(), f"{ticker}: no CSV found in data/"

    df_csv = _parse_csv_file(csv_path)
    if df_csv.empty:
        return pd.DataFrame(), f"{ticker}: CSV parsed to 0 bars"

    existing = load_local_bars(ticker)
    if existing is not None and not existing.empty:
        # Parquet wins: concat and keep existing on overlap
        combined = pd.concat([df_csv, existing])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df_csv

    save_local_bars(ticker, combined)
    msg = f"{ticker}: {len(combined):,} bars saved ({csv_path.name})"
    return combined, msg


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def fetch_yfinance(ticker: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
    """
    Download 1h bars from yfinance (max lookback ~730 days).
    Returns UTC-indexed DataFrame with [open, high, low, close, volume].
    """
    import yfinance as yf

    kw: dict = dict(
        tickers=ticker,
        interval="1h",
        start=start,
        prepost=False,
        auto_adjust=True,
        progress=False,
    )
    if end is not None:
        kw["end"] = end

    raw = yf.download(**kw)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance MultiIndex columns when downloading single ticker via download()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    col_map = {"adj close": "close", "adj_close": "close"}
    raw = raw.rename(columns=col_map)

    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    raw = raw[needed].copy()

    # Ensure UTC index
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize(UTC)
    else:
        raw.index = raw.index.tz_convert(UTC)
    raw.index.name = "timestamp"

    raw = filter_rth(raw)
    raw = raw.dropna(subset=["close"])
    return raw.sort_index()


def update_with_yfinance(ticker: str) -> tuple[pd.DataFrame, str]:
    """
    Append recent 1h bars from yfinance to the local parquet.
    - If parquet exists: start = last_bar - 5h (small overlap)
    - If parquet missing: start = now - 725d (yfinance 730-day limit, leave margin)
    """
    existing = load_local_bars(ticker)
    now_utc = datetime.now(UTC)

    if existing is not None and not existing.empty:
        start = existing.index.max().to_pydatetime() - timedelta(hours=5)
        label = "incremental"
    else:
        start = now_utc - timedelta(days=725)
        label = "full backfill"

    df_new = fetch_yfinance(ticker, start=start)
    if df_new.empty:
        msg = f"{ticker}: yfinance returned 0 bars ({label})"
        return existing if existing is not None else pd.DataFrame(), msg

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df_new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df_new

    save_local_bars(ticker, combined)
    added = len(combined) - (len(existing) if existing is not None else 0)
    msg = f"{ticker}: {len(combined):,} bars total, +{max(added,0)} new ({label})"
    return combined, msg


# ---------------------------------------------------------------------------
# Bulk loader
# ---------------------------------------------------------------------------

def load_all_tickers(tickers: list[str] = TICKERS) -> dict[str, pd.DataFrame]:
    """Load all tickers from local parquet files. Returns dict of non-empty DataFrames."""
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = load_local_bars(t)
        if df is not None and not df.empty:
            result[t] = df
    return result

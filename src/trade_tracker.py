"""
trade_tracker.py — Persist recommended trades and track their daily P&L.

Storage: data/tracked_trades.json
Each trade stores entry prices, all legs, and a daily_prices list that
grows by one entry each time update_trade_prices() is called.

P&L convention (per 1 contract = 100 shares):
  debit  trade:  pnl = (current_net - entry_net) * 100
  credit trade:  pnl = (entry_net   - current_net) * 100
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

TRACKED_PATH = Path("data/tracked_trades.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeLeg:
    action: str        # "BUY" | "SELL"
    right:  str        # "CALL" | "PUT"
    strike: float
    entry_mid: float
    entry_ask: float = 0.0
    entry_bid: float = 0.0


@dataclass
class DailyPrice:
    date:       str    # ISO date
    underlying: float
    net_mid:    float
    pnl_dollars: float  # per 1 contract (×100 multiplier)


@dataclass
class TrackedTrade:
    id:                   str
    date_recommended:     str          # ISO date
    ticker:               str
    expiry:               str          # "YYYY-MM-DD"
    dte_at_entry:         int
    underlying_at_entry:  float
    strategy:             str          # "bull_call_spread" etc.
    regime_type:          str
    regime_name:          str
    price_type:           str          # "debit" | "credit"
    entry_net:            float        # always positive
    max_profit:           float
    max_loss:             float
    legs:                 list[dict]   # serialised TradeLeg dicts
    daily_prices:         list[dict]   # serialised DailyPrice dicts
    status:               str = "open" # "open" | "closed" | "expired"
    confidence:           float = 0.0  # HMM regime confidence at entry
    recommended_at:       str = ""     # ISO datetime of recommendation
    underlying_at_close:  float = 0.0  # underlying price when position closed/expired


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_trades() -> list[TrackedTrade]:
    if not TRACKED_PATH.exists():
        return []
    try:
        raw = json.loads(TRACKED_PATH.read_text(encoding="utf-8"))
        return [TrackedTrade(**t) for t in raw]
    except Exception:
        return []


def _save_all(trades: list[TrackedTrade]) -> None:
    TRACKED_PATH.parent.mkdir(exist_ok=True, parents=True)
    TRACKED_PATH.write_text(
        json.dumps([asdict(t) for t in trades], indent=2),
        encoding="utf-8",
    )


def save_trade(trade: TrackedTrade) -> None:
    """Append or overwrite a trade (matched by id)."""
    trades = load_trades()
    ids = [t.id for t in trades]
    if trade.id in ids:
        trades[ids.index(trade.id)] = trade
    else:
        trades.append(trade)
    _save_all(trades)


def mark_expired() -> None:
    """Auto-expire trades whose expiry date has passed."""
    today_str = date.today().isoformat()
    trades = load_trades()
    changed = False
    for t in trades:
        if t.status == "open" and t.expiry < today_str:
            t.status = "expired"
            if t.daily_prices:
                t.underlying_at_close = t.daily_prices[-1]["underlying"]
            changed = True
    if changed:
        _save_all(trades)


# ---------------------------------------------------------------------------
# Build a TrackedTrade from recommend.py output
# ---------------------------------------------------------------------------

def build_trade(
    ticker: str,
    expiry: str,
    underlying: float,
    strategy: str,
    regime_type: str,
    regime_name: str,
    price_type: str,
    entry_net: float,
    max_profit: float,
    max_loss: float,
    legs: list[dict],  # each: {action, right, strike, entry_mid, entry_ask, entry_bid}
    confidence: float = 0.0,
) -> TrackedTrade:
    today = date.today()
    dte   = (date.fromisoformat(expiry) - today).days
    tid   = f"{ticker}-{expiry}-{strategy}-{today.isoformat()}-{str(uuid.uuid4())[:4]}"
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    initial = DailyPrice(
        date=today.isoformat(),
        underlying=underlying,
        net_mid=entry_net,
        pnl_dollars=0.0,
    )

    return TrackedTrade(
        id=tid,
        date_recommended=today.isoformat(),
        ticker=ticker,
        expiry=expiry,
        dte_at_entry=dte,
        underlying_at_entry=underlying,
        strategy=strategy,
        regime_type=regime_type,
        regime_name=regime_name,
        price_type=price_type,
        entry_net=entry_net,
        max_profit=max_profit,
        max_loss=max_loss,
        legs=legs,
        daily_prices=[asdict(initial)],
        status="open",
        confidence=confidence,
        recommended_at=now,
    )


# ---------------------------------------------------------------------------
# Price update helpers
# ---------------------------------------------------------------------------

def _fetch_leg_mid(ticker: str, expiry: str, right: str, strike: float) -> Optional[float]:
    """Fetch current mid for a single leg via ThetaData."""
    try:
        from src import thetadata as td
        df = td.get_quotes(ticker, expiry, right=right.lower(), strike_range=40)
        if df.empty:
            return None
        row = df[df["strike"] == strike]
        if row.empty:
            # nearest strike fallback
            row = df.loc[(df["strike"] - strike).abs().idxmin()]
            return float(row["mid"]) if hasattr(row, "mid") else None
        return float(row.iloc[0]["mid"])
    except Exception:
        return None


def _compute_net_mid(legs: list[dict], mids: dict[tuple, float]) -> Optional[float]:
    """
    Net mid = sum of (mid * +1 for BUY, -1 for SELL).
    Returns None if any leg mid is unavailable.
    """
    total = 0.0
    for leg in legs:
        key  = (leg["right"], leg["strike"])
        mid  = mids.get(key)
        if mid is None:
            return None
        sign = 1 if leg["action"] == "BUY" else -1
        total += sign * mid
    return total


def update_trade_prices(trade: TrackedTrade) -> TrackedTrade:
    """
    Fetch current mid for every leg and append a DailyPrice entry.
    Idempotent for the same calendar date (updates existing entry if called twice today).
    """
    if trade.status != "open":
        return trade

    today_str = date.today().isoformat()

    # Fetch current underlying price
    try:
        import yfinance as yf
        S = float(yf.Ticker(trade.ticker).fast_info.last_price)
    except Exception:
        S = trade.underlying_at_entry

    # Fetch mid for each unique (right, strike) pair
    seen: dict[tuple, float] = {}
    for leg in trade.legs:
        key = (leg["right"], leg["strike"])
        if key not in seen:
            mid = _fetch_leg_mid(trade.ticker, trade.expiry, leg["right"], leg["strike"])
            if mid is not None:
                seen[key] = mid

    net = _compute_net_mid(trade.legs, seen)
    if net is None:
        # Can't update — keep last entry
        return trade

    net = abs(net)   # keep positive for comparison

    if trade.price_type == "debit":
        pnl = (net - trade.entry_net) * 100
    else:
        pnl = (trade.entry_net - net) * 100

    dp = DailyPrice(
        date=today_str,
        underlying=round(S, 2),
        net_mid=round(net, 2),
        pnl_dollars=round(pnl, 2),
    )

    # Replace today's entry if it already exists, otherwise append
    daily = trade.daily_prices
    if daily and daily[-1]["date"] == today_str:
        daily[-1] = asdict(dp)
    else:
        daily.append(asdict(dp))

    trade.daily_prices = daily
    return trade


def update_all_open_trades() -> list[TrackedTrade]:
    """Update prices for all open trades and persist. Returns updated list."""
    mark_expired()
    trades = load_trades()
    updated = [update_trade_prices(t) for t in trades]
    _save_all(updated)
    return updated


# ---------------------------------------------------------------------------
# Summary helpers for dashboard
# ---------------------------------------------------------------------------

def latest_pnl(trade: TrackedTrade) -> tuple[float, float]:
    """Return (current_net_mid, pnl_dollars) from most recent daily entry."""
    if not trade.daily_prices:
        return trade.entry_net, 0.0
    last = trade.daily_prices[-1]
    return last["net_mid"], last["pnl_dollars"]


def pnl_pct(trade: TrackedTrade) -> float:
    """P&L as a percentage of max possible loss."""
    _, pnl = latest_pnl(trade)
    ref = abs(trade.max_loss) * 100 if trade.max_loss else 1
    return pnl / ref if ref else 0.0


def days_held(trade: TrackedTrade) -> int:
    try:
        return (date.today() - date.fromisoformat(trade.date_recommended)).days
    except Exception:
        return 0


def dte_remaining(trade: TrackedTrade) -> int:
    try:
        return max((date.fromisoformat(trade.expiry) - date.today()).days, 0)
    except Exception:
        return 0

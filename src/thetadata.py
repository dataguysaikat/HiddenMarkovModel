"""
thetadata.py — ThetaData Terminal v3 REST client (Value subscription).

Available on Value plan:
  - /v3/option/list/expirations   — all expirations for a symbol
  - /v3/option/list/strikes       — all strikes for symbol+expiry
  - /v3/option/snapshot/quote     — real-time bid/ask (strike_range filter)

Greeks (delta, IV) are computed locally via Black-Scholes since the
greeks endpoints require Standard/Professional plan.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

BASE_URL = "http://127.0.0.1:25503/v3"
TIMEOUT  = 10          # seconds per request
RISK_FREE = 0.053      # ~current SOFR / fed funds rate


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict) -> dict | list | None:
    params.setdefault("format", "json")
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        # ThetaData wraps results in {"response": [...]}
        return data.get("response", data) if isinstance(data, dict) else data
    except Exception:
        return None


def is_available() -> bool:
    """Return True if ThetaData terminal is reachable."""
    try:
        r = requests.get(f"{BASE_URL}/option/list/expirations",
                         params={"symbol": "AAPL", "format": "json"}, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Expirations & strikes
# ---------------------------------------------------------------------------

def get_expirations(symbol: str) -> list[str]:
    """Return sorted list of expiry strings (YYYY-MM-DD) from today onward."""
    data = _get("/option/list/expirations", {"symbol": symbol})
    if not data:
        return []
    today_str = date.today().isoformat()
    return sorted(
        r["expiration"] for r in data
        if isinstance(r, dict) and r.get("expiration", "") >= today_str
    )


def find_expiry(symbol: str, target_dte: int = 21,
                dte_min: int = 14, dte_max: int = 45) -> Optional[str]:
    """Return the expiry closest to target_dte within [dte_min, dte_max]."""
    today = date.today()
    best, best_diff = None, 9999
    for exp in get_expirations(symbol):
        dte = (date.fromisoformat(exp) - today).days
        if dte_min <= dte <= dte_max and abs(dte - target_dte) < best_diff:
            best, best_diff = exp, abs(dte - target_dte)
    return best


# ---------------------------------------------------------------------------
# Live quotes
# ---------------------------------------------------------------------------

def get_quotes(symbol: str, expiration: str,
               right: str = "both", strike_range: int = 20) -> pd.DataFrame:
    """
    Fetch real-time bid/ask for all strikes near ATM.

    Returns DataFrame with columns:
        strike, right, bid, ask, mid, bid_size, ask_size, timestamp
    """
    data = _get("/option/snapshot/quote", {
        "symbol":       symbol,
        "expiration":   expiration,
        "right":        right,
        "strike_range": strike_range,
    })
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        contract = item.get("contract", {})
        quotes   = item.get("data", [{}])
        q = quotes[0] if quotes else {}
        bid = float(q.get("bid", 0))
        ask = float(q.get("ask", 0))
        rows.append({
            "strike":    float(contract.get("strike", 0)),
            "right":     contract.get("right", "").upper(),
            "bid":       bid,
            "ask":       ask,
            "mid":       (bid + ask) / 2,
            "bid_size":  int(q.get("bid_size", 0)),
            "ask_size":  int(q.get("ask_size", 0)),
            "timestamp": q.get("timestamp", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["bid"] > 0].sort_values("strike").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _bs_price(S: float, K: float, T: float, r: float,
               sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if right == "CALL" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if right == "CALL":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S: float, K: float, T: float, r: float,
               sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if right == "CALL" else norm.cdf(d1) - 1


def _implied_vol(S: float, K: float, T: float, r: float,
                  market_mid: float, right: str,
                  lo: float = 1e-4, hi: float = 20.0) -> float:
    """Compute implied volatility via Brent root-finding."""
    if T <= 0 or market_mid <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if right == "CALL" else max(K - S, 0)
    if market_mid <= intrinsic:
        return np.nan
    try:
        iv = brentq(
            lambda s: _bs_price(S, K, T, r, s, right) - market_mid,
            lo, hi, xtol=1e-6, maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# Enriched chain
# ---------------------------------------------------------------------------

def get_chain(symbol: str, expiration: str,
              underlying_price: float,
              strike_range: int = 20,
              risk_free: float = RISK_FREE) -> pd.DataFrame:
    """
    Fetch quotes from ThetaData and enrich with BS implied vol and delta.

    Returns DataFrame with columns:
        strike, right, bid, ask, mid, bid_size, ask_size,
        iv, delta, timestamp
    """
    today = date.today()
    T = (date.fromisoformat(expiration) - today).days / 365.0

    df = get_quotes(symbol, expiration, strike_range=strike_range)
    if df.empty:
        return df

    S = underlying_price
    ivs, deltas = [], []
    for _, row in df.iterrows():
        iv = _implied_vol(S, row["strike"], T, risk_free, row["mid"], row["right"])
        delta = _bs_delta(S, row["strike"], T, risk_free, iv, row["right"]) if not np.isnan(iv) else np.nan
        ivs.append(iv)
        deltas.append(delta)

    df["iv"]    = ivs
    df["delta"] = deltas
    return df


def get_calls(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["right"] == "CALL"].sort_values("strike").reset_index(drop=True)


def get_puts(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["right"] == "PUT"].sort_values("strike").reset_index(drop=True)

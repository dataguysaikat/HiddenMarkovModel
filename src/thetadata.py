"""
thetadata.py — ThetaData Terminal REST client.

Endpoints used:
  v3 (port 25503):
    - /v3/option/list/expirations   — all expirations for a symbol
    - /v3/option/snapshot/quote     — real-time bid/ask (strike_range filter)
  v2 (port 25510):
    - /v2/bulk_snapshot/option/greeks — bid/ask + delta, IV, theta, vega, rho

Greeks are fetched directly from ThetaData (dividend-adjusted model).
Black-Scholes fallback is used only when the greeks endpoint is unavailable.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

V3_BASE = "http://127.0.0.1:25503/v3"
V2_BASE = "http://127.0.0.1:25510/v2"
TIMEOUT  = 30          # seconds per request
RISK_FREE = 0.053      # ~current SOFR / fed funds rate


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict, base: str = V3_BASE) -> dict | list | None:
    params.setdefault("format", "json")
    try:
        r = requests.get(f"{base}{path}", params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        # ThetaData wraps results in {"response": [...]}
        return data.get("response", data) if isinstance(data, dict) else data
    except Exception:
        return None


def is_available() -> bool:
    """Return True if ThetaData terminal is reachable (TCP connect check)."""
    import socket
    try:
        s = socket.create_connection(("127.0.0.1", 25503), timeout=3)
        s.close()
        return True
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
# Live quotes (v3)
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
# Bulk greeks (v2) — returns quotes + greeks in one call
# ---------------------------------------------------------------------------

def _fetch_bulk_greeks(symbol: str, expiration: str) -> pd.DataFrame | None:
    """
    Fetch greeks from /v2/bulk_snapshot/option/greeks.

    Returns DataFrame with columns:
        strike, right, bid, ask, mid, delta, iv, theta, vega, rho
    or None if the endpoint is unavailable.
    """
    # v2 expects expiration as YYYYMMDD integer
    exp_v2 = expiration.replace("-", "")
    data = _get("/bulk_snapshot/option/greeks",
                {"root": symbol, "exp": int(exp_v2)},
                base=V2_BASE)
    if not data:
        return None

    rows = []
    for item in data:
        contract = item.get("contract", {})
        ticks = item.get("data", item.get("ticks", [{}]))
        t = ticks[0] if ticks else {}

        bid = float(t.get("bid", 0))
        ask = float(t.get("ask", 0))
        iv  = float(t.get("implied_vol", 0))

        rows.append({
            "strike": float(contract.get("strike", 0)),
            "right":  contract.get("right", "").upper(),
            "bid":    bid,
            "ask":    ask,
            "mid":    (bid + ask) / 2,
            "delta":  float(t.get("delta", np.nan)),
            "iv":     iv if iv > 0 else np.nan,
            "theta":  float(t.get("theta", np.nan)),
            "vega":   float(t.get("vega", np.nan)),
            "rho":    float(t.get("rho", np.nan)),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df[df["bid"] > 0].sort_values("strike").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Black-Scholes helpers (fallback when greeks endpoint unavailable)
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


def _enrich_with_bs(df: pd.DataFrame, underlying_price: float,
                    expiration: str, risk_free: float) -> pd.DataFrame:
    """Add iv and delta columns using Black-Scholes (fallback path)."""
    T = (date.fromisoformat(expiration) - date.today()).days / 365.0
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


# ---------------------------------------------------------------------------
# Enriched chain
# ---------------------------------------------------------------------------

def get_chain(symbol: str, expiration: str,
              underlying_price: float,
              strike_range: int = 20,
              risk_free: float = RISK_FREE) -> pd.DataFrame:
    """
    Fetch option chain with greeks from ThetaData.

    Primary: bulk_snapshot greeks endpoint (v2) — returns quotes + greeks.
    Fallback: snapshot quotes (v3) + local Black-Scholes IV/delta.

    Returns DataFrame with columns:
        strike, right, bid, ask, mid, iv, delta[, theta, vega, rho]
    """
    # --- Try bulk greeks first (single call for quotes + greeks) ---
    greeks_df = _fetch_bulk_greeks(symbol, expiration)
    if greeks_df is not None and not greeks_df.empty:
        # Filter to strikes near ATM (bulk returns ALL strikes)
        atm = underlying_price
        strikes = greeks_df["strike"].unique()
        strikes_sorted = np.sort(strikes)
        atm_idx = np.searchsorted(strikes_sorted, atm)
        lo = max(0, atm_idx - strike_range)
        hi = min(len(strikes_sorted), atm_idx + strike_range + 1)
        keep = set(strikes_sorted[lo:hi])
        df = greeks_df[greeks_df["strike"].isin(keep)].reset_index(drop=True)

        # Fill any missing greeks with BS fallback (illiquid strikes)
        missing = df["delta"].isna() | df["iv"].isna()
        if missing.any():
            T = (date.fromisoformat(expiration) - date.today()).days / 365.0
            S = underlying_price
            for idx in df.index[missing]:
                row = df.loc[idx]
                if np.isnan(row["iv"]):
                    df.at[idx, "iv"] = _implied_vol(S, row["strike"], T, risk_free,
                                                     row["mid"], row["right"])
                iv_val = df.at[idx, "iv"]
                if np.isnan(row["delta"]) and not np.isnan(iv_val):
                    df.at[idx, "delta"] = _bs_delta(S, row["strike"], T, risk_free,
                                                     iv_val, row["right"])
        return df

    # --- Fallback: v3 quotes + Black-Scholes ---
    df = get_quotes(symbol, expiration, strike_range=strike_range)
    if df.empty:
        return df
    return _enrich_with_bs(df, underlying_price, expiration, risk_free)


def get_calls(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["right"] == "CALL"].sort_values("strike").reset_index(drop=True)


def get_puts(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["right"] == "PUT"].sort_values("strike").reset_index(drop=True)

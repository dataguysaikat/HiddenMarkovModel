"""
options.py — Strategy selection and order building.

Active dashboard/scheduler order proposals use ThetaData quote DataFrames for
option prices. Schwab helpers are retained for broker execution compatibility
and legacy chain-shaped inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Graceful guard for schwab-py
try:
    import schwab
    from schwab.orders.options import (
        OptionSymbol,
        OptionOrder,
    )
    from schwab.orders.common import OrderType, Duration, Session
    SCHWAB_AVAILABLE = True
except Exception:  # noqa: BLE001
    SCHWAB_AVAILABLE = False
    schwab = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_MAP: dict[str, str] = {
    "directional_bull": "bull_call_vertical",
    "directional_bear": "bear_put_vertical",
    "vol_expansion": "long_strangle",
    "mean_reverting": "iron_condor",
}


def option_chain_unavailable_meta(
    ticker: str,
    regime_type: str,
    underlying_price: float,
    error: str,
    expiry: str | None = None,
) -> dict:
    """Return proposal metadata when a ThetaData option chain is unavailable."""
    meta = {
        "ticker": ticker,
        "regime_type": regime_type,
        "strategy": STRATEGY_MAP.get(regime_type, "iron_condor"),
        "underlying_price": underlying_price,
        "error": error,
        "legs": [],
        "est_net_price": 0.0,
    }
    if expiry is not None:
        meta["expiry"] = expiry
    return meta


@dataclass
class OptionSelectionParams:
    target_dte: int = 21
    dte_min: int = 14
    dte_max: int = 45
    target_delta_directional: float = 0.40   # for vertical spreads
    target_delta_wing: float = 0.16          # IC short legs
    wing_width_strikes: int = 2              # IC wing width in strikes
    otm_offset_pct: float = 0.03             # 3 % OTM for strangle legs
    quantity: int = 1


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def fetch_option_chain(client: Any, ticker: str, dte_min: int, dte_max: int) -> dict:
    """
    Fetch the option chain from Schwab.  Returns {} if client is None or on error.
    """
    if client is None:
        return {}
    try:
        resp = client.get_option_chain(
            ticker,
            contract_type=client.Options.ContractType.ALL,
            strike_count=40,
            include_quotes=True,
            option_type=client.Options.Type.ALL,
            days_to_expiration=dte_max,
        )
        if resp.status_code != 200:
            return {}
        return resp.json()
    except Exception:  # noqa: BLE001
        return {}


def find_nearest_expiry(chain: dict, target_dte: int) -> Optional[str]:
    """Return the expiry string closest to target_dte, or None if chain empty."""
    if not chain:
        return None
    exps: list[str] = []
    for side in ("callExpDateMap", "putExpDateMap"):
        exps.extend(chain.get(side, {}).keys())
    if not exps:
        return None

    def _dte(exp_str: str) -> int:
        # Format: "YYYY-MM-DD:N" or "YYYY-MM-DD"
        return int(exp_str.split(":")[1]) if ":" in exp_str else 9999

    return min(exps, key=lambda e: abs(_dte(e) - target_dte))


def _get_strikes_for_expiry(chain: dict, expiry: str, contract_type: str) -> list[dict]:
    """Return list of strike dicts [{strike, bid, ask, delta, symbol}, ...]."""
    side_key = "callExpDateMap" if contract_type.upper() == "CALL" else "putExpDateMap"
    exp_map = chain.get(side_key, {}).get(expiry, {})
    strikes = []
    for strike_str, contracts in exp_map.items():
        if not contracts:
            continue
        c = contracts[0]
        strikes.append({
            "strike": float(strike_str),
            "bid": float(c.get("bid", 0)),
            "ask": float(c.get("ask", 0)),
            "delta": float(c.get("delta", 0)),
            "symbol": c.get("symbol", ""),
            "mid": (float(c.get("bid", 0)) + float(c.get("ask", 0))) / 2,
        })
    return sorted(strikes, key=lambda x: x["strike"])


def find_strike_by_delta(
    chain: dict, expiry: str, contract_type: str, target_delta: float
) -> Optional[dict]:
    """Find the strike whose |delta| is closest to target_delta."""
    strikes = _get_strikes_for_expiry(chain, expiry, contract_type)
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(abs(s["delta"]) - target_delta))


def find_strike_by_offset(
    underlying_price: float,
    chain: dict,
    expiry: str,
    contract_type: str,
    offset_pct: float,
    direction: str,  # "above" | "below"
) -> Optional[dict]:
    """Find the strike closest to underlying ± offset_pct."""
    target = underlying_price * (1 + offset_pct) if direction == "above" else underlying_price * (1 - offset_pct)
    strikes = _get_strikes_for_expiry(chain, expiry, contract_type)
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s["strike"] - target))


# ---------------------------------------------------------------------------
# ThetaData DataFrame selectors
# ---------------------------------------------------------------------------

def _nearest_delta_df(df: Any, target: float):
    valid = df.dropna(subset=["delta"])
    if valid.empty:
        return None
    return valid.loc[(valid["delta"].abs() - abs(target)).abs().idxmin()]


def _nearest_strike_df(df: Any, price: float):
    if df.empty:
        return None
    return df.loc[(df["strike"] - price).abs().idxmin()]


def _next_strike_above_df(df: Any, strike: float, n: int = 1):
    above = df[df["strike"] > strike]
    return above.iloc[n - 1] if len(above) >= n else None


def _next_strike_below_df(df: Any, strike: float, n: int = 1):
    below = df[df["strike"] < strike]
    return below.iloc[-n] if len(below) >= n else None


def _tracking_leg_from_quote(row: Any, action: str) -> dict:
    return {
        "action": action,
        "right": str(row["right"]).upper(),
        "strike": float(row["strike"]),
        "entry_mid": float(row.get("mid", 0.0)),
        "entry_ask": float(row.get("ask", 0.0)),
        "entry_bid": float(row.get("bid", 0.0)),
    }


def _order_leg_from_tracking_leg(leg: dict, action: str, quantity: int) -> dict:
    return {
        "symbol": leg.get("symbol", ""),
        "action": action,
        "quantity": quantity,
        "right": leg["right"],
        "strike": float(leg["strike"]),
    }


def build_order_from_thetadata_chain(
    ticker: str,
    regime_type: str,
    underlying_price: float,
    expiry: str,
    calls: Any,
    puts: Any,
    delta_vert: float = 0.40,
    delta_wing: float = 0.16,
    otm_pct: float = 0.03,
    quantity: int = 1,
) -> tuple[Optional[dict], dict]:
    """
    Build an executable order from ThetaData quote DataFrames.

    `calls` and `puts` are the DataFrames returned by thetadata.get_calls()
    and thetadata.get_puts(). Prices in the returned metadata come from
    ThetaData bid/ask mids; the underlying price should be supplied by the
    caller, normally from yfinance.
    """
    strategy = STRATEGY_MAP.get(regime_type, "iron_condor")
    meta: dict = {
        "ticker": ticker,
        "regime_type": regime_type,
        "strategy": strategy,
        "underlying_price": underlying_price,
        "expiry": expiry,
        "error": None,
        "legs": [],
        "est_net_price": 0.0,
    }

    try:
        if strategy == "bull_call_vertical":
            long_call = _nearest_delta_df(calls, delta_vert)
            short_call = _next_strike_above_df(calls, long_call["strike"]) if long_call is not None else None
            if long_call is None or short_call is None:
                meta["error"] = "Insufficient strikes for bull call spread"
                return None, meta
            net_price = round(float(long_call["mid"] - short_call["mid"]), 2)
            if net_price <= 0:
                meta["error"] = f"Bad pricing (net={net_price:.2f})"
                return None, meta
            meta["legs"] = [
                _tracking_leg_from_quote(long_call, "BUY"),
                _tracking_leg_from_quote(short_call, "SELL"),
            ]
            order_legs = [
                _order_leg_from_tracking_leg(meta["legs"][0], "BUY_TO_OPEN", quantity),
                _order_leg_from_tracking_leg(meta["legs"][1], "SELL_TO_OPEN", quantity),
            ]
            price_type = "debit"

        elif strategy == "bear_put_vertical":
            long_put = _nearest_delta_df(puts, -delta_vert)
            short_put = _next_strike_below_df(puts, long_put["strike"]) if long_put is not None else None
            if long_put is None or short_put is None:
                meta["error"] = "Insufficient strikes for bear put spread"
                return None, meta
            net_price = round(float(long_put["mid"] - short_put["mid"]), 2)
            if net_price <= 0:
                meta["error"] = f"Bad pricing (net={net_price:.2f})"
                return None, meta
            meta["legs"] = [
                _tracking_leg_from_quote(long_put, "BUY"),
                _tracking_leg_from_quote(short_put, "SELL"),
            ]
            order_legs = [
                _order_leg_from_tracking_leg(meta["legs"][0], "BUY_TO_OPEN", quantity),
                _order_leg_from_tracking_leg(meta["legs"][1], "SELL_TO_OPEN", quantity),
            ]
            price_type = "debit"

        elif strategy == "long_strangle":
            call_leg = _nearest_strike_df(calls, underlying_price * (1 + otm_pct))
            put_leg = _nearest_strike_df(puts, underlying_price * (1 - otm_pct))
            if call_leg is None or put_leg is None:
                meta["error"] = "Insufficient strikes for strangle"
                return None, meta
            net_price = round(float(call_leg["mid"] + put_leg["mid"]), 2)
            if net_price <= 0:
                meta["error"] = f"Bad pricing (net={net_price:.2f})"
                return None, meta
            meta["legs"] = [
                _tracking_leg_from_quote(call_leg, "BUY"),
                _tracking_leg_from_quote(put_leg, "BUY"),
            ]
            order_legs = [
                _order_leg_from_tracking_leg(meta["legs"][0], "BUY_TO_OPEN", quantity),
                _order_leg_from_tracking_leg(meta["legs"][1], "BUY_TO_OPEN", quantity),
            ]
            price_type = "debit"

        elif strategy == "iron_condor":
            short_call = _nearest_delta_df(calls, delta_wing)
            short_put = _nearest_delta_df(puts, -delta_wing)
            if short_call is None or short_put is None:
                meta["error"] = "No short legs found for iron condor"
                return None, meta
            long_call = _next_strike_above_df(calls, short_call["strike"], n=2)
            long_put = _next_strike_below_df(puts, short_put["strike"], n=2)
            if long_call is None or long_put is None:
                meta["error"] = "No wing strikes found for iron condor"
                return None, meta
            net_price = round(float((short_call["mid"] + short_put["mid"]) - (long_call["mid"] + long_put["mid"])), 2)
            if net_price <= 0:
                meta["error"] = f"Bad pricing (credit={net_price:.2f})"
                return None, meta
            meta["legs"] = [
                _tracking_leg_from_quote(short_call, "SELL"),
                _tracking_leg_from_quote(long_call, "BUY"),
                _tracking_leg_from_quote(short_put, "SELL"),
                _tracking_leg_from_quote(long_put, "BUY"),
            ]
            order_legs = [
                _order_leg_from_tracking_leg(meta["legs"][0], "SELL_TO_OPEN", quantity),
                _order_leg_from_tracking_leg(meta["legs"][1], "BUY_TO_OPEN", quantity),
                _order_leg_from_tracking_leg(meta["legs"][2], "SELL_TO_OPEN", quantity),
                _order_leg_from_tracking_leg(meta["legs"][3], "BUY_TO_OPEN", quantity),
            ]
            price_type = "credit"

        else:
            meta["error"] = f"Unknown strategy: {strategy}"
            return None, meta

    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        return None, meta

    meta["est_net_price"] = net_price
    order = {
        "strategy": strategy,
        "legs": order_legs,
        "net_price": net_price,
        "price_type": price_type,
        "quantity": quantity,
    }
    return order, meta


# ---------------------------------------------------------------------------
# Order builders  (return dicts when schwab-py unavailable)
# ---------------------------------------------------------------------------

def _round_price(price: float) -> float:
    """Round option price to nearest $0.05."""
    return round(round(price / 0.05) * 0.05, 2)


def _build_leg(symbol: str, action: str, qty: int) -> dict:
    return {"symbol": symbol, "action": action, "quantity": qty}


def build_bull_call_vertical(
    long_sym: str, short_sym: str, qty: int, net_debit: float
) -> dict:
    return {
        "strategy": "bull_call_vertical",
        "legs": [
            _build_leg(long_sym, "BUY_TO_OPEN", qty),
            _build_leg(short_sym, "SELL_TO_OPEN", qty),
        ],
        "net_price": _round_price(net_debit),
        "price_type": "debit",
        "quantity": qty,
    }


def build_bear_put_vertical(
    long_sym: str, short_sym: str, qty: int, net_debit: float
) -> dict:
    return {
        "strategy": "bear_put_vertical",
        "legs": [
            _build_leg(long_sym, "BUY_TO_OPEN", qty),
            _build_leg(short_sym, "SELL_TO_OPEN", qty),
        ],
        "net_price": _round_price(net_debit),
        "price_type": "debit",
        "quantity": qty,
    }


def build_iron_condor(
    short_call: str, long_call: str,
    short_put: str, long_put: str,
    qty: int, net_credit: float,
) -> dict:
    """
    Iron condor: sell short call/put, buy long call/put for wing protection.
    Layout: long_put < short_put < underlying < short_call < long_call
    """
    return {
        "strategy": "iron_condor",
        "legs": [
            _build_leg(long_call, "BUY_TO_OPEN", qty),
            _build_leg(short_call, "SELL_TO_OPEN", qty),
            _build_leg(short_put, "SELL_TO_OPEN", qty),
            _build_leg(long_put, "BUY_TO_OPEN", qty),
        ],
        "net_price": _round_price(net_credit),
        "price_type": "credit",
        "quantity": qty,
    }


def build_long_strangle(
    call_sym: str, put_sym: str, qty: int, net_debit: float
) -> dict:
    return {
        "strategy": "long_strangle",
        "legs": [
            _build_leg(call_sym, "BUY_TO_OPEN", qty),
            _build_leg(put_sym, "BUY_TO_OPEN", qty),
        ],
        "net_price": _round_price(net_debit),
        "price_type": "debit",
        "quantity": qty,
    }


# ---------------------------------------------------------------------------
# High-level selector
# ---------------------------------------------------------------------------

def select_and_build_order(
    ticker: str,
    regime_type: str,
    chain: dict,
    underlying_price: float,
    params: Optional[OptionSelectionParams] = None,
) -> tuple[Optional[dict], dict]:
    """
    Select the appropriate strategy for regime_type and build an order dict.

    Returns (order_dict, metadata).
    If chain is empty or strike selection fails, order_dict is None and
    metadata["error"] explains why (suitable for paper simulation without a chain).
    """
    if params is None:
        params = OptionSelectionParams()

    strategy = STRATEGY_MAP.get(regime_type, "iron_condor")
    metadata: dict = {
        "ticker": ticker,
        "regime_type": regime_type,
        "strategy": strategy,
        "underlying_price": underlying_price,
        "error": None,
    }

    if not chain:
        metadata["error"] = "No chain available (paper simulation)"
        metadata["legs"] = []
        metadata["est_net_price"] = 0.0
        return None, metadata

    expiry = find_nearest_expiry(chain, params.target_dte)
    if expiry is None:
        metadata["error"] = "No expiry found in chain"
        return None, metadata

    metadata["expiry"] = expiry

    try:
        if strategy == "bull_call_vertical":
            long_leg = find_strike_by_delta(chain, expiry, "CALL", params.target_delta_directional)
            if long_leg is None:
                raise ValueError("No long call strike found")
            # Short leg: one strike higher
            calls = _get_strikes_for_expiry(chain, expiry, "CALL")
            long_idx = next((i for i, s in enumerate(calls) if s["strike"] == long_leg["strike"]), None)
            short_leg = calls[long_idx + 1] if long_idx is not None and long_idx + 1 < len(calls) else None
            if short_leg is None:
                raise ValueError("No short call strike found")
            net_debit = long_leg["mid"] - short_leg["mid"]
            order = build_bull_call_vertical(long_leg["symbol"], short_leg["symbol"], params.quantity, net_debit)
            metadata["legs"] = order["legs"]
            metadata["est_net_price"] = order["net_price"]

        elif strategy == "bear_put_vertical":
            long_leg = find_strike_by_delta(chain, expiry, "PUT", params.target_delta_directional)
            if long_leg is None:
                raise ValueError("No long put strike found")
            puts = _get_strikes_for_expiry(chain, expiry, "PUT")
            long_idx = next((i for i, s in enumerate(puts) if s["strike"] == long_leg["strike"]), None)
            short_leg = puts[long_idx - 1] if long_idx is not None and long_idx > 0 else None
            if short_leg is None:
                raise ValueError("No short put strike found")
            net_debit = long_leg["mid"] - short_leg["mid"]
            order = build_bear_put_vertical(long_leg["symbol"], short_leg["symbol"], params.quantity, net_debit)
            metadata["legs"] = order["legs"]
            metadata["est_net_price"] = order["net_price"]

        elif strategy == "long_strangle":
            call_leg = find_strike_by_offset(underlying_price, chain, expiry, "CALL", params.otm_offset_pct, "above")
            put_leg = find_strike_by_offset(underlying_price, chain, expiry, "PUT", params.otm_offset_pct, "below")
            if call_leg is None or put_leg is None:
                raise ValueError("Could not find strangle strikes")
            net_debit = call_leg["mid"] + put_leg["mid"]
            order = build_long_strangle(call_leg["symbol"], put_leg["symbol"], params.quantity, net_debit)
            metadata["legs"] = order["legs"]
            metadata["est_net_price"] = order["net_price"]

        elif strategy == "iron_condor":
            short_call = find_strike_by_delta(chain, expiry, "CALL", params.target_delta_wing)
            short_put = find_strike_by_delta(chain, expiry, "PUT", params.target_delta_wing)
            if short_call is None or short_put is None:
                raise ValueError("Could not find IC short legs")

            calls = _get_strikes_for_expiry(chain, expiry, "CALL")
            puts = _get_strikes_for_expiry(chain, expiry, "PUT")

            sc_idx = next((i for i, s in enumerate(calls) if s["strike"] == short_call["strike"]), None)
            long_call = calls[sc_idx + params.wing_width_strikes] if sc_idx is not None and sc_idx + params.wing_width_strikes < len(calls) else None

            sp_idx = next((i for i, s in enumerate(puts) if s["strike"] == short_put["strike"]), None)
            long_put = puts[sp_idx - params.wing_width_strikes] if sp_idx is not None and sp_idx - params.wing_width_strikes >= 0 else None

            if long_call is None or long_put is None:
                raise ValueError("Could not find IC wing strikes")

            net_credit = (short_call["mid"] + short_put["mid"]) - (long_call["mid"] + long_put["mid"])
            order = build_iron_condor(
                short_call["symbol"], long_call["symbol"],
                short_put["symbol"], long_put["symbol"],
                params.quantity, net_credit,
            )
            metadata["legs"] = order["legs"]
            metadata["est_net_price"] = order["net_price"]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    except Exception as exc:  # noqa: BLE001
        metadata["error"] = str(exc)
        metadata.setdefault("legs", [])
        metadata.setdefault("est_net_price", 0.0)
        return None, metadata

    return order, metadata

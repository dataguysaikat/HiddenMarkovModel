"""
recommend.py — Live option recommendations using ThetaData + HMM regimes.
Usage: python -m src.recommend
"""
import json
import pickle
import datetime as dt
import sys
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf

from src.data_loader import load_all_tickers
from src.hmm_model import (
    run_all_tickers, make_features, characterize_regimes,
    regime_forecast, TickerResult,
)
from src import thetadata as td
from src.trade_tracker import build_trade, save_trade

_CACHE_PATH = Path(__file__).parent.parent / "data" / "hmm_cache.pkl"
_CACHE_MAX_AGE_HOURS = 168  # reuse cached HMM model for up to 7 days


def _load_hmm_cache() -> dict | None:
    """Return cached TickerResult dict if it exists and is recent enough."""
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH, "rb") as f:
            d = pickle.load(f)
        updated = d.get("updated_at")
        if updated is None:
            return None
        now = dt.datetime.now(tz=updated.tzinfo)
        age_hours = (now - updated).total_seconds() / 3600
        if age_hours > _CACHE_MAX_AGE_HOURS:
            return None
        return d.get("results") or None
    except Exception:
        return None


def _refresh_regime(cached: TickerResult, fresh_df) -> TickerResult:
    """Re-predict current regime on fresh bars using the cached fitted model."""
    feats = make_features(fresh_df)
    if len(feats) < 100 or cached.model is None:
        return cached
    X = cached.scaler.transform(feats.values)
    post = cached.model.predict_proba(X)          # fast forward-pass, no re-fit
    n = cached.n_states
    df_reg = feats.copy()
    df_reg["regime"] = post.argmax(axis=1)
    df_reg["regime_conf"] = post.max(axis=1)
    for i in range(n):
        df_reg[f"p_regime_{i}"] = post[:, i]
    adaptive_vol = float(feats["vol_20"].quantile(0.60))
    chars = characterize_regimes(cached.model, cached.scaler, df_reg, n,
                                 vol_threshold=adaptive_vol)
    last_post = np.array([float(post[-1, i]) for i in range(n)])
    fc = regime_forecast(cached.model, last_post, horizon_bars=24)
    return TickerResult(
        ticker=cached.ticker,
        model=cached.model,
        scaler=cached.scaler,
        df_reg=df_reg,
        df_prices=fresh_df,
        characteristics=chars,
        forecast=fc,
        current_regime=int(df_reg["regime"].iloc[-1]),
        n_states=n,
        error=None,
    )

# ---------------------------------------------------------------------------
# Config — loaded from config.json at project root
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
_cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8")) if _CONFIG_PATH.exists() else {}
_opt = _cfg.get("option_strategy", {})

TARGET_DTE   = _opt.get("target_dte",   21)
DTE_MIN      = _opt.get("dte_min",      14)
DTE_MAX      = _opt.get("dte_max",      45)
DELTA_VERT   = _opt.get("delta_vert",   0.40)   # long leg delta for vertical spreads
DELTA_WING   = _opt.get("delta_wing",   0.16)   # short leg delta for iron condor
OTM_PCT      = _opt.get("otm_pct",      0.03)   # 3% OTM for strangle legs
STRIKE_RANGE = _opt.get("strike_range", 20)     # strikes above/below ATM to fetch

# Learned policy (written by src/retrain_policy.py after enough closed trades accumulate)
_pol             = _cfg.get("learned_policy", {})
_SKIP_REGIMES    = set(_pol.get("skip_regimes",    []))
_CAUTION_REGIMES = set(_pol.get("caution_regimes", []))
_MIN_CONFIDENCE  = float(_pol.get("min_confidence", 0.0))


# ---------------------------------------------------------------------------
# Strike selectors
# ---------------------------------------------------------------------------

def nearest_delta(df, target: float):
    valid = df.dropna(subset=["delta"])
    if valid.empty:
        return None
    return valid.loc[(valid["delta"].abs() - abs(target)).abs().idxmin()]


def nearest_strike(df, price: float):
    if df.empty:
        return None
    return df.loc[(df["strike"] - price).abs().idxmin()]


def next_strike_above(df, strike: float, n: int = 1):
    above = df[df["strike"] > strike]
    return above.iloc[n - 1] if len(above) >= n else None


def next_strike_below(df, strike: float, n: int = 1):
    below = df[df["strike"] < strike]
    return below.iloc[-(n)] if len(below) >= n else None


# ---------------------------------------------------------------------------
# Underlying price via yfinance
# ---------------------------------------------------------------------------

def get_price(ticker: str, fallback: float) -> float:
    try:
        info = yf.Ticker(ticker).fast_info
        p = float(info.last_price)
        return p if p > 0 else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _row_line(action: str, exp: str, strike: float, right: str,
              price_label: str, price: float,
              delta: float, iv: float) -> str:
    iv_str = f"  IV={iv:.1%}" if not np.isnan(iv) else ""
    d_str  = f"  delta={delta:+.2f}" if not np.isnan(delta) else ""
    return (f"  {action:<5} {exp} ${strike:.2f}{right[0]}  "
            f"{price_label}=${price:.2f}{d_str}{iv_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading market data...")
    bars  = load_all_tickers()
    today = date.today()

    cached = _load_hmm_cache()
    if cached is not None:
        print("Using cached HMM models — re-predicting on fresh bars...")
        results: dict = {}
        for ticker, df in bars.items():
            cr = cached.get(ticker)
            if cr is not None and cr.error is None and cr.model is not None:
                results[ticker] = _refresh_regime(cr, df)
            else:
                results[ticker] = run_all_tickers({ticker: df}, n_states=4)[ticker]
    else:
        print("Fitting HMMs on latest data (this may take several minutes)...")
        results = run_all_tickers(bars, n_states=4)

    td_up = td.is_available()
    print(f"ThetaData terminal: {'connected' if td_up else 'UNAVAILABLE — falling back to yfinance'}")
    if not td_up:
        print("  Start ThetaData Terminal and ensure it's running on http://127.0.0.1:25503")
        return

    print()
    print("=" * 92)
    print(f"  LIVE OPTION RECOMMENDATIONS  --  {today}  (ThetaData real-time quotes)")
    print("=" * 92)

    for ticker, res in results.items():
        if res.error:
            print(f"\n{ticker}: ERROR -- {res.error}")
            continue

        rc   = res.characteristics[res.current_regime]
        S_hm = float(res.df_prices["close"].iloc[-1])
        S    = get_price(ticker, S_hm)
        ann  = rc.mean_log_ret * 252 * 6.5
        conf = res.forecast.get("current_confidence", 0)

        exp = td.find_expiry(ticker, TARGET_DTE, DTE_MIN, DTE_MAX)
        if exp is None:
            print(f"\n{ticker}: no expiry found in {DTE_MIN}-{DTE_MAX} DTE range")
            continue

        dte = (date.fromisoformat(exp) - today).days

        chain = td.get_chain(ticker, exp, S, strike_range=STRIKE_RANGE)
        if chain.empty:
            print(f"\n{ticker}: empty chain from ThetaData")
            continue

        calls = td.get_calls(chain)
        puts  = td.get_puts(chain)

        print()
        print(f"{'='*92}")
        print(f"  {ticker}   price=${S:.2f}   expiry={exp} ({dte} DTE)   "
              f"regime={rc.name}   ann={ann:+.1%}   conf={conf:.0%}")
        print(f"{'='*92}")

        strat = rc.regime_type
        saved = False

        # --- Apply learned policy (populated by src/retrain_policy.py) ---
        if strat in _SKIP_REGIMES:
            print(f"  [POLICY] Skipping {strat} — win rate too low from closed-trade history.")
            continue
        if _MIN_CONFIDENCE > 0 and conf < _MIN_CONFIDENCE:
            print(f"  [POLICY] Skipping — confidence {conf:.0%} below required {_MIN_CONFIDENCE:.0%}.")
            continue
        if strat in _CAUTION_REGIMES:
            print(f"  [CAUTION] {strat} has below-average win rate — proceeding cautiously.")

        try:
            if strat == "directional_bull":
                lc = nearest_delta(calls, DELTA_VERT)
                sc = next_strike_above(calls, lc["strike"]) if lc is not None else None
                if lc is None or sc is None:
                    print("  Strategy: BULL CALL SPREAD -- insufficient strikes"); continue
                net = max(round(lc["mid"] - sc["mid"], 2), 0.05)
                wid = sc["strike"] - lc["strike"]
                print(f"  Strategy: BULL CALL SPREAD  |  Confidence: {conf:.0%}")
                print(_row_line("BUY",  exp, lc["strike"], "C", "ask", lc["ask"], lc["delta"], lc["iv"]))
                print(_row_line("SELL", exp, sc["strike"], "C", "bid", sc["bid"], sc["delta"], sc["iv"]))
                print(f"  Net debit ${net:.2f}  |  Max profit ${wid-net:.2f}  |  Max loss ${net:.2f}  |  B/E ${lc['strike']+net:.2f}")
                legs = [
                    {"action": "BUY",  "right": "CALL", "strike": lc["strike"], "entry_mid": lc["mid"], "entry_ask": lc["ask"], "entry_bid": lc["bid"]},
                    {"action": "SELL", "right": "CALL", "strike": sc["strike"], "entry_mid": sc["mid"], "entry_ask": sc["ask"], "entry_bid": sc["bid"]},
                ]
                trade = build_trade(ticker, exp, S, "bull_call_spread", strat, rc.name, "debit",  net, wid-net, net, legs, conf)
                save_trade(trade); saved = True

            elif strat == "directional_bear":
                lp = nearest_delta(puts, -DELTA_VERT)
                sp = next_strike_below(puts, lp["strike"]) if lp is not None else None
                if lp is None or sp is None:
                    print("  Strategy: BEAR PUT SPREAD -- insufficient strikes"); continue
                net = max(round(lp["mid"] - sp["mid"], 2), 0.05)
                wid = lp["strike"] - sp["strike"]
                print(f"  Strategy: BEAR PUT SPREAD  |  Confidence: {conf:.0%}")
                print(_row_line("BUY",  exp, lp["strike"], "P", "ask", lp["ask"], lp["delta"], lp["iv"]))
                print(_row_line("SELL", exp, sp["strike"], "P", "bid", sp["bid"], sp["delta"], sp["iv"]))
                print(f"  Net debit ${net:.2f}  |  Max profit ${wid-net:.2f}  |  Max loss ${net:.2f}  |  B/E ${lp['strike']-net:.2f}")
                legs = [
                    {"action": "BUY",  "right": "PUT", "strike": lp["strike"], "entry_mid": lp["mid"], "entry_ask": lp["ask"], "entry_bid": lp["bid"]},
                    {"action": "SELL", "right": "PUT", "strike": sp["strike"], "entry_mid": sp["mid"], "entry_ask": sp["ask"], "entry_bid": sp["bid"]},
                ]
                trade = build_trade(ticker, exp, S, "bear_put_spread", strat, rc.name, "debit",  net, wid-net, net, legs, conf)
                save_trade(trade); saved = True

            elif strat == "vol_expansion":
                cl = nearest_strike(calls, S * (1 + OTM_PCT))
                pl = nearest_strike(puts,  S * (1 - OTM_PCT))
                if cl is None or pl is None:
                    print("  Strategy: LONG STRANGLE -- insufficient strikes"); continue
                net = round(cl["mid"] + pl["mid"], 2)
                print(f"  Strategy: LONG STRANGLE  |  Confidence: {conf:.0%}")
                print(_row_line("BUY", exp, cl["strike"], "C", "ask", cl["ask"], cl["delta"], cl["iv"]))
                print(_row_line("BUY", exp, pl["strike"], "P", "ask", pl["ask"], pl["delta"], pl["iv"]))
                print(f"  Net debit ${net:.2f}  |  B/E up ${cl['strike']+net:.2f}  |  B/E dn ${pl['strike']-net:.2f}")
                legs = [
                    {"action": "BUY", "right": "CALL", "strike": cl["strike"], "entry_mid": cl["mid"], "entry_ask": cl["ask"], "entry_bid": cl["bid"]},
                    {"action": "BUY", "right": "PUT",  "strike": pl["strike"], "entry_mid": pl["mid"], "entry_ask": pl["ask"], "entry_bid": pl["bid"]},
                ]
                trade = build_trade(ticker, exp, S, "long_strangle", strat, rc.name, "debit",  net, float("inf"), net, legs, conf)
                save_trade(trade); saved = True

            elif strat == "mean_reverting":
                sc = nearest_delta(calls,  DELTA_WING)
                sp = nearest_delta(puts,  -DELTA_WING)
                if sc is None or sp is None:
                    print("  Strategy: IRON CONDOR -- no short legs found"); continue
                lc = next_strike_above(calls, sc["strike"], n=2)
                lp = next_strike_below(puts,  sp["strike"], n=2)
                if lc is None or lp is None:
                    print("  Strategy: IRON CONDOR -- no wing strikes found"); continue
                cr = max(round((sc["mid"]+sp["mid"]) - (lc["mid"]+lp["mid"]), 2), 0.05)
                ml = round(sc["strike"] - lc["strike"] - cr, 2)
                print(f"  Strategy: IRON CONDOR  |  Confidence: {conf:.0%}")
                print(_row_line("SELL", exp, sc["strike"], "C", "bid", sc["bid"], sc["delta"], sc["iv"]))
                print(_row_line("BUY",  exp, lc["strike"], "C", "ask", lc["ask"], lc["delta"], lc["iv"]))
                print(_row_line("SELL", exp, sp["strike"], "P", "bid", sp["bid"], sp["delta"], sp["iv"]))
                print(_row_line("BUY",  exp, lp["strike"], "P", "ask", lp["ask"], lp["delta"], lp["iv"]))
                print(f"  Net credit ${cr:.2f}  |  Max profit ${cr:.2f}  |  Max loss ${abs(ml):.2f}")
                legs = [
                    {"action": "SELL", "right": "CALL", "strike": sc["strike"], "entry_mid": sc["mid"], "entry_ask": sc["ask"], "entry_bid": sc["bid"]},
                    {"action": "BUY",  "right": "CALL", "strike": lc["strike"], "entry_mid": lc["mid"], "entry_ask": lc["ask"], "entry_bid": lc["bid"]},
                    {"action": "SELL", "right": "PUT",  "strike": sp["strike"], "entry_mid": sp["mid"], "entry_ask": sp["ask"], "entry_bid": sp["bid"]},
                    {"action": "BUY",  "right": "PUT",  "strike": lp["strike"], "entry_mid": lp["mid"], "entry_ask": lp["ask"], "entry_bid": lp["bid"]},
                ]
                trade = build_trade(ticker, exp, S, "iron_condor", strat, rc.name, "credit", cr, cr, abs(ml), legs, conf)
                save_trade(trade); saved = True

        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

        if saved:
            print(f"  [saved to tracked_trades.json]")

    print()
    print("=" * 92)
    print("Quotes from ThetaData terminal (real-time). "
          "Deltas/IV computed via Black-Scholes on mid price.")
    print("Use limit orders at mid or better.")
    if _SKIP_REGIMES or _CAUTION_REGIMES or _MIN_CONFIDENCE > 0:
        print(f"Active learned policy: skip={sorted(_SKIP_REGIMES) or 'none'}  "
              f"caution={sorted(_CAUTION_REGIMES) or 'none'}  "
              f"min_conf={_MIN_CONFIDENCE:.0%}")
    else:
        print("No learned policy yet. Run `python -m src.retrain_policy` after trades close.")
    print("=" * 92)


if __name__ == "__main__":
    main()

"""
hmm_model.py — HMM fitting, regime labeling, forecasting.

Graceful hmmlearn guard: the module loads even if hmmlearn is not installed.
Check HMM_AVAILABLE before calling fit_hmm / run_all_tickers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    from src._hmm_pure import GaussianHMM  # type: ignore[assignment]
    HMM_AVAILABLE = True  # pure-NumPy fallback is available

# ---------------------------------------------------------------------------
# Regime characterisation thresholds
# ---------------------------------------------------------------------------
TREND_THRESHOLD = 0.0003   # hourly mean log-ret (~7 % annualised)
VOL_THRESHOLD = 0.008      # 20-bar rolling std of log-ret


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeCharacteristics:
    label: int
    name: str
    mean_log_ret: float
    mean_vol_20: float
    is_trending: bool
    is_high_vol: bool
    regime_type: str  # "directional_bull" | "directional_bear" | "vol_expansion" | "mean_reverting"


@dataclass
class TickerResult:
    ticker: str
    model: object
    scaler: object
    df_reg: pd.DataFrame
    df_prices: pd.DataFrame
    characteristics: dict[int, RegimeCharacteristics]
    forecast: dict
    current_regime: int
    n_states: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with log_ret, vol_20, log_vol (rows with NaN dropped)."""
    d = df.copy()
    d["log_close"] = np.log(d["close"])
    d["log_ret"] = d["log_close"].diff()
    d["vol_20"] = d["log_ret"].rolling(20).std()
    d["log_vol"] = np.log(d["volume"].replace(0, np.nan))
    feats = d[["log_ret", "vol_20", "log_vol"]].dropna()
    return feats


# ---------------------------------------------------------------------------
# State relabelling (bear → bull by mean log-ret)
# ---------------------------------------------------------------------------

def _relabel_states(model: GaussianHMM, scaler: StandardScaler, n_states: int) -> dict[int, int]:
    """
    Return a mapping {original_label → new_label} where new_label=0 is the
    state with the lowest mean log-ret and new_label=n_states-1 is the highest.
    This makes regime labels deterministic across re-fits.
    """
    # model.means_ is in scaled space → invert first column (log_ret)
    means_scaled = model.means_[:, 0]
    # Reconstruct original log_ret means (only first feature needed for ordering)
    dummy = np.zeros((n_states, scaler.mean_.shape[0]))
    dummy[:, 0] = means_scaled
    means_orig = scaler.inverse_transform(dummy)[:, 0]

    order = np.argsort(means_orig)          # ascending by mean log-ret
    return {int(original): int(new) for new, original in enumerate(order)}


# ---------------------------------------------------------------------------
# HMM fitting
# ---------------------------------------------------------------------------

def fit_hmm(
    feats: pd.DataFrame,
    n_states: int = 4,
    random_state: int = 42,
) -> tuple[GaussianHMM, StandardScaler, pd.DataFrame]:
    """
    Fit GaussianHMM, relabel states bear→bull, return (model, scaler, df_reg).

    df_reg columns: regime, regime_conf, p_regime_0 … p_regime_{n_states-1}
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn is not installed.")

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=random_state,
    )
    model.fit(X)

    post = model.predict_proba(X)           # (T, n_states)
    raw_regimes = post.argmax(axis=1)

    label_map = _relabel_states(model, scaler, n_states)

    # Remap posteriors columns
    new_post = np.zeros_like(post)
    for orig, new in label_map.items():
        new_post[:, new] = post[:, orig]

    out = feats.copy()
    out["regime"] = [label_map[r] for r in raw_regimes]
    out["regime_conf"] = new_post.max(axis=1)
    for i in range(n_states):
        out[f"p_regime_{i}"] = new_post[:, i]

    # Also reorder transition matrix rows/cols so they match new labels
    perm = [None] * n_states
    for orig, new in label_map.items():
        perm[new] = orig
    model.transmat_ = model.transmat_[np.ix_(perm, perm)]
    model.means_ = model.means_[perm]
    model.covars_ = model.covars_[perm]
    model.startprob_ = model.startprob_[perm]

    return model, scaler, out


# ---------------------------------------------------------------------------
# Regime characterisation
# ---------------------------------------------------------------------------

def characterize_regimes(
    model: GaussianHMM,
    scaler: StandardScaler,
    df_reg: pd.DataFrame,
    n_states: int,
    vol_threshold: float | None = None,
) -> dict[int, RegimeCharacteristics]:
    """
    Derive human-readable characteristics for each regime.
    After relabelling, model.means_[i] corresponds to regime i.

    vol_threshold: per-ticker adaptive threshold for high-vol classification.
    If None, falls back to the global VOL_THRESHOLD constant.
    Pass the 60th-percentile of the ticker's vol_20 series so that high-vol
    stocks like TSLA or NVDA are judged against their own distribution, not a
    global constant calibrated for lower-vol names.
    """
    effective_vol_thresh = vol_threshold if vol_threshold is not None else VOL_THRESHOLD

    # Invert scaled means to original feature space
    means_orig = scaler.inverse_transform(model.means_)  # (n_states, 3)

    chars: dict[int, RegimeCharacteristics] = {}
    for i in range(n_states):
        mean_ret = float(means_orig[i, 0])
        mean_vol = float(means_orig[i, 1])  # vol_20 feature mean

        is_trending = abs(mean_ret) > TREND_THRESHOLD
        is_high_vol = mean_vol > effective_vol_thresh

        if is_trending and mean_ret > 0:
            regime_type = "directional_bull"
            name = f"Bull (R{i})"
        elif is_trending and mean_ret < 0:
            regime_type = "directional_bear"
            name = f"Bear (R{i})"
        elif is_high_vol and not is_trending:
            regime_type = "vol_expansion"
            name = f"Vol Exp (R{i})"
        else:
            regime_type = "mean_reverting"
            name = f"Mean Rev (R{i})"

        chars[i] = RegimeCharacteristics(
            label=i,
            name=name,
            mean_log_ret=mean_ret,
            mean_vol_20=mean_vol,
            is_trending=is_trending,
            is_high_vol=is_high_vol,
            regime_type=regime_type,
        )
    return chars


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def regime_forecast(
    model: GaussianHMM,
    current_post: np.ndarray,
    horizon_bars: int,
    label_map: dict | None = None,
) -> dict:
    """
    Propagate current posterior through transition matrix for horizon_bars steps.
    Returns a summary dict compatible with the old hmm.py interface.
    """
    P = model.transmat_
    Pk = np.linalg.matrix_power(P, int(max(1, horizon_bars)))
    future = current_post @ Pk
    curr = int(np.argmax(current_post))
    fut = int(np.argmax(future))
    p_stay = float(future[curr])
    return {
        "current_regime": curr,
        "current_confidence": float(np.max(current_post)),
        "horizon_bars": int(horizon_bars),
        "most_likely_future_regime": fut,
        "future_regime_confidence": float(np.max(future)),
        "prob_stay_same": p_stay,
        "prob_change_by_horizon": float(1.0 - p_stay),
        "future_distribution": future.tolist(),
    }


# ---------------------------------------------------------------------------
# Bulk runner
# ---------------------------------------------------------------------------

def run_all_tickers(
    bars: dict[str, pd.DataFrame],
    n_states: int = 4,
    min_bars: int = 500,
) -> dict[str, TickerResult]:
    """
    Fit HMM for every ticker in `bars`.
    Returns dict[ticker → TickerResult].
    Tickers with fewer than min_bars or fitting errors get result.error set.
    """
    results: dict[str, TickerResult] = {}

    for ticker, df in bars.items():
        try:
            feats = make_features(df)
            if len(feats) < min_bars:
                results[ticker] = TickerResult(
                    ticker=ticker, model=None, scaler=None,
                    df_reg=pd.DataFrame(), df_prices=df,
                    characteristics={}, forecast={},
                    current_regime=-1, n_states=n_states,
                    error=f"Only {len(feats)} feature rows (need {min_bars})",
                )
                continue

            model, scaler, df_reg = fit_hmm(feats, n_states=n_states)
            # Adaptive vol threshold: 60th percentile of this ticker's vol_20
            # series. Prevents high-vol stocks (TSLA, NVDA) from having all
            # their states classified as vol_expansion due to a global constant.
            adaptive_vol_thresh = float(feats["vol_20"].quantile(0.60))
            chars = characterize_regimes(
                model, scaler, df_reg, n_states,
                vol_threshold=adaptive_vol_thresh,
            )

            last_post = np.array([df_reg[f"p_regime_{i}"].iloc[-1] for i in range(n_states)], dtype=float)
            fc = regime_forecast(model, last_post, horizon_bars=24)

            current_regime = int(df_reg["regime"].iloc[-1])

            results[ticker] = TickerResult(
                ticker=ticker,
                model=model,
                scaler=scaler,
                df_reg=df_reg,
                df_prices=df,
                characteristics=chars,
                forecast=fc,
                current_regime=current_regime,
                n_states=n_states,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            results[ticker] = TickerResult(
                ticker=ticker, model=None, scaler=None,
                df_reg=pd.DataFrame(), df_prices=df,
                characteristics={}, forecast={},
                current_regime=-1, n_states=n_states,
                error=str(exc),
            )

    return results

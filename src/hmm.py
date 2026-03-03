import os
import time
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.graph_objects as go

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# -----------------------------
# Config
# -----------------------------
NY_TZ = pytz.timezone("America/New_York")
UTC = pytz.UTC

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

INTERVAL = "60min"  # Alpha Vantage hourly

SYMBOL_PRESETS = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow (DIA)": "DIA",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
}

# -----------------------------
# Local storage
# -----------------------------
def _bars_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}_1h.parquet"


def load_local_bars(symbol: str) -> pd.DataFrame | None:
    p = _bars_path(symbol)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def save_local_bars(symbol: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = df.reset_index(names="timestamp")
    out.to_parquet(_bars_path(symbol), index=False)


def filter_regular_trading_hours(df_utc: pd.DataFrame) -> pd.DataFrame:
    """
    Keep 9:30–16:00 America/New_York bars only.
    For hourly bars, AV typically timestamps on the hour; this filter is conservative.
    """
    if df_utc.empty:
        return df_utc
    idx_ny = df_utc.index.tz_convert(NY_TZ)
    t = idx_ny.time
    keep = (t >= datetime(2000, 1, 1, 9, 30).time()) & (t < datetime(2000, 1, 1, 16, 0).time())
    return df_utc.loc[keep].copy()

# -----------------------------
# Alpha Vantage fetch
# -----------------------------
AV_URL = "https://www.alphavantage.co/query"


def month_range(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    start = datetime.strptime(start_yyyymm, "%Y-%m").replace(tzinfo=NY_TZ)
    end = datetime.strptime(end_yyyymm, "%Y-%m").replace(tzinfo=NY_TZ)
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur = cur + relativedelta(months=1)
    return months


def av_fetch_intraday_month_csv(
    symbol: str,
    yyyymm: str,
    apikey: str,
    interval: str = "60min",
    adjusted: bool = True,
    extended_hours: bool = True,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Premium intraday endpoint with month=YYYY-MM.
    Returns DataFrame indexed by UTC with columns: open, high, low, close, volume.
    """
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": yyyymm,
        "outputsize": "full",
        "datatype": "csv",
        "adjusted": "true" if adjusted else "false",
        "extended_hours": "true" if extended_hours else "false",
        "apikey": apikey,
    }

    r = requests.get(AV_URL, params=params, timeout=timeout)
    text = r.text.strip()

    # Alpha Vantage sometimes returns JSON error/note in plain text
    if text.startswith("{") and ("Error Message" in text or "Note" in text or "Information" in text):
        raise RuntimeError(f"Alpha Vantage response: {text[:300]}")

    # Expected: CSV content with header: timestamp,open,high,low,close,volume
    df = pd.read_csv(StringIO(text))
    if df.empty or "timestamp" not in df.columns:
        raise RuntimeError(f"Unexpected CSV format for {symbol} {yyyymm}. First 200 chars: {text[:200]}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # AV timestamps are typically US/Eastern; localize then convert to UTC
    # If you find your timestamps are already tz-aware, this still works safely.
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    df["timestamp"] = df["timestamp"].dt.tz_convert(UTC)

    df = df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })

    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df


def update_local_data_av(
    symbol: str,
    years: int,
    apikey: str,
    include_extended: bool,
    adjusted: bool,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, str]:
    """
    If no local file: pull last `years` years month-by-month.
    If local exists: pull only from last ~2 months (overlap) through current month.
    """
    df_local = load_local_bars(symbol)

    now_ny = datetime.now(NY_TZ)
    current_month = now_ny.strftime("%Y-%m")

    note = ""
    if df_local is None or df_local.empty:
        start_ny = (now_ny - relativedelta(years=years)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_month = start_ny.strftime("%Y-%m")
        months = month_range(start_month, current_month)
        note = f"Full backfill: {len(months)} monthly requests for ~{years} years."
        df_all = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    else:
        last_utc = df_local.index.max()
        last_ny = last_utc.tz_convert(NY_TZ)

        # overlap ~2 months for safety
        start_ny = (last_ny - relativedelta(months=2)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_month = start_ny.strftime("%Y-%m")
        months = month_range(start_month, current_month)
        note = f"Incremental update: refetching {len(months)} month(s) with overlap."
        df_all = df_local.copy()

    progress = st.progress(0)
    for i, m in enumerate(months):
        progress.progress((i + 1) / max(1, len(months)))
        chunk = av_fetch_intraday_month_csv(
            symbol=symbol,
            yyyymm=m,
            apikey=apikey,
            interval=INTERVAL,
            adjusted=adjusted,
            extended_hours=include_extended,
        )
        df_all = pd.concat([df_all, chunk], axis=0)
        df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
        time.sleep(max(0.0, sleep_seconds))

    if not include_extended:
        df_all = filter_regular_trading_hours(df_all)

    save_local_bars(symbol, df_all)
    return df_all, note

# -----------------------------
# HMM
# -----------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_close"] = np.log(d["close"])
    d["log_ret"] = d["log_close"].diff()
    d["vol_20"] = d["log_ret"].rolling(20).std()
    d["log_vol"] = np.log(d["volume"].replace(0, np.nan))
    feats = d[["log_ret", "vol_20", "log_vol"]].dropna()
    return feats


def fit_hmm(feats: pd.DataFrame, n_states: int, random_state: int = 42):
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=random_state
    )
    model.fit(X)

    post = model.predict_proba(X)
    regimes = post.argmax(axis=1)

    out = feats.copy()
    out["regime"] = regimes
    out["regime_conf"] = post.max(axis=1)
    for i in range(n_states):
        out[f"p_regime_{i}"] = post[:, i]
    return model, scaler, out


def regime_change_forecast(model: GaussianHMM, current_post: np.ndarray, horizon_bars: int) -> dict:
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
        "future_distribution": future,
    }

# -----------------------------
# Plotting
# -----------------------------
def plot_price_with_regimes(df_prices: pd.DataFrame, df_reg: pd.DataFrame) -> go.Figure:
    d = df_prices.join(df_reg[["regime"]], how="inner").dropna(subset=["close", "regime"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["close"], mode="lines", name="Close", line=dict(width=2)))

    palette = [
        "rgba(31,119,180,0.12)",
        "rgba(255,127,14,0.12)",
        "rgba(44,160,44,0.12)",
        "rgba(214,39,40,0.12)",
        "rgba(148,103,189,0.12)",
        "rgba(140,86,75,0.12)",
        "rgba(227,119,194,0.12)",
        "rgba(127,127,127,0.12)",
    ]

    regimes = d["regime"].astype(int).values
    times = d.index
    start_i = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start_i]:
            r = regimes[start_i]
            fig.add_vrect(
                x0=times[start_i], x1=times[i - 1],
                fillcolor=palette[r % len(palette)],
                opacity=1.0, line_width=0,
                annotation_text=f"R{r}",
                annotation_position="top left"
            )
            start_i = i

    fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title="Time (UTC)", yaxis_title="Price")
    return fig


def plot_posteriors(df_reg: pd.DataFrame, n_states: int) -> go.Figure:
    fig = go.Figure()
    for i in range(n_states):
        fig.add_trace(go.Scatter(
            x=df_reg.index, y=df_reg[f"p_regime_{i}"],
            mode="lines", name=f"Regime {i} prob"
        ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20),
                      yaxis_title="Posterior probability", xaxis_title="Time (UTC)")
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="HMM Regime Dashboard (1h, Alpha Vantage)", layout="wide")
st.title("HMM Regime Dashboard (1-hour OHLCV from Alpha Vantage)")

with st.sidebar:
    st.header("Universe")
    pick = st.selectbox("Pick an index / stock", list(SYMBOL_PRESETS.keys()))
    symbol = SYMBOL_PRESETS[pick]
    custom = st.text_input("Or type a custom symbol (optional)", value="")
    if custom.strip():
        symbol = custom.strip().upper()

    st.divider()
    st.header("Alpha Vantage")
    apikey = st.text_input("API key (or set env ALPHAVANTAGE_API_KEY)", type="password")
    if not apikey:
        apikey = os.getenv("ALPHAVANTAGE_API_KEY", "")

    years = st.slider("Years of history (hourly)", 1, 20, 5)
    include_extended = st.checkbox("Include extended hours", value=True)
    adjusted = st.checkbox("Adjusted (splits/dividends)", value=True)

    # Rate limiting: tune to your plan (free is 25/day; premium varies)
    sleep_seconds = st.number_input("Sleep between API calls (seconds)", min_value=0.0, value=1.0, step=0.5)

    st.divider()
    st.header("Model")
    n_states = st.slider("Number of regimes (states)", 2, 6, 4)
    horizon_hours = st.slider("Regime change horizon (hours)", 1, 240, 24)
    run_update = st.button("Update local data")
    run_fit = st.button("Fit / refresh HMM")

status = st.empty()

if not apikey and run_update:
    st.error("Missing API key. Add it in the sidebar or set ALPHAVANTAGE_API_KEY.")
    st.stop()

# Load / update data
try:
    if run_update:
        status.info(f"Updating {symbol} hourly data from Alpha Vantage...")
        df, note = update_local_data_av(
            symbol=symbol,
            years=years,
            apikey=apikey,
            include_extended=include_extended,
            adjusted=adjusted,
            sleep_seconds=sleep_seconds,
        )
        status.success(f"Saved {len(df):,} bars ? {str(_bars_path(symbol))}")
        if note:
            st.info(note)
    else:
        df = load_local_bars(symbol)
        if df is None or df.empty:
            status.warning("No local data yet. Click 'Update local data'.")
            st.stop()
        status.success(f"Loaded {len(df):,} local bars for {symbol}")

except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# Fit model
if not run_fit:
    st.info("Click **Fit / refresh HMM** to run regime detection and forecasts.")
    st.stop()

feats = make_features(df)
if len(feats) < 500:
    st.error("Not enough hourly bars for a stable HMM. Try including extended hours or increasing years.")
    st.stop()

status.info("Fitting HMM...")
model, scaler, df_reg = fit_hmm(feats, n_states=n_states)

last_post = df_reg[[f"p_regime_{i}" for i in range(n_states)]].iloc[-1].values.astype(float)
forecast = regime_change_forecast(model, current_post=last_post, horizon_bars=horizon_hours)
status.success("Model fit complete.")

# Dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Symbol", symbol)
c2.metric("Current regime", f"R{forecast['current_regime']}")
c3.metric("Current confidence", f"{forecast['current_confidence']:.2%}")
c4.metric(f"Prob(change within {horizon_hours}h)", f"{forecast['prob_change_by_horizon']:.2%}")

st.caption(
    "Confidence = posterior probability of the chosen regime on the most recent bar. "
    "Prob(change) uses the HMM transition matrix over your selected horizon."
)

st.subheader("Price chart with regime shading")
st.plotly_chart(plot_price_with_regimes(df, df_reg), use_container_width=True)

st.subheader("Regime posterior probabilities")
st.plotly_chart(plot_posteriors(df_reg, n_states=n_states), use_container_width=True)

st.subheader("Forecast distribution (by horizon)")
fd = pd.DataFrame({"regime": list(range(n_states)), "probability": forecast["future_distribution"]}) \
        .sort_values("probability", ascending=False)
st.dataframe(fd, use_container_width=True)
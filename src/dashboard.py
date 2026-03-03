"""
dashboard.py — Multi-ticker HMM Regime + Options Auto-Trading Dashboard.
Entry point: streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

# Internal modules
from src.data_loader import (
    TICKERS,
    import_csv_to_parquet,
    load_all_tickers,
    load_local_bars,
    update_with_yfinance,
)
from src.hmm_model import (
    HMM_AVAILABLE,
    TickerResult,
    make_features,
    run_all_tickers,
)
from src.options import (
    STRATEGY_MAP,
    OptionSelectionParams,
    fetch_option_chain,
    select_and_build_order,
)
from src.broker import (
    execute_order,
    get_schwab_client,
    load_paper_trades,
    SCHWAB_AVAILABLE,
)
from src.scheduler import get_scheduler, load_cache, cache_mtime

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}

def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
from src.trade_tracker import (
    load_trades,
    update_all_open_trades,
    latest_pnl,
    pnl_pct,
    days_held,
    dte_remaining,
)

# ---------------------------------------------------------------------------
# Cached disk-read wrappers — avoids hitting filesystem on every widget interaction
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _cached_load_trades():
    return load_trades()

@st.cache_data(ttl=60)
def _cached_load_cache():
    return load_cache()

@st.cache_data(ttl=60)
def _thetadata_available() -> bool:
    from src import thetadata as td
    return td.is_available()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
_REGIME_PALETTE = [
    "rgba(31,119,180,0.15)",
    "rgba(255,127,14,0.15)",
    "rgba(44,160,44,0.15)",
    "rgba(214,39,40,0.15)",
    "rgba(148,103,189,0.15)",
    "rgba(140,86,75,0.15)",
]


def _plot_price_regimes(
    df_prices: pd.DataFrame,
    df_reg: pd.DataFrame,
    ticker: str,
    n_states: int,
) -> go.Figure:
    d = df_prices.join(df_reg[["regime"]], how="inner").dropna(subset=["close", "regime"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["close"],
        mode="lines", name="Close",
        line=dict(width=1.5, color="#555"),
    ))

    regimes = d["regime"].astype(int).values
    times = d.index
    start_i = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start_i]:
            r = regimes[start_i]
            fig.add_vrect(
                x0=times[start_i], x1=times[i - 1],
                fillcolor=_REGIME_PALETTE[r % len(_REGIME_PALETTE)],
                opacity=1.0, line_width=0,
                annotation_text=f"R{r}",
                annotation_position="top left",
            )
            start_i = i

    fig.update_layout(
        title=f"{ticker} — price with regime shading",
        height=420, margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Time (UTC)", yaxis_title="Price",
    )
    return fig


def _plot_posteriors(df_reg: pd.DataFrame, n_states: int, ticker: str) -> go.Figure:
    fig = go.Figure()
    for i in range(n_states):
        col = f"p_regime_{i}"
        if col not in df_reg.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_reg.index, y=df_reg[col],
            mode="lines", name=f"R{i} prob",
            stackgroup="one",
        ))
    fig.update_layout(
        title=f"{ticker} — regime posterior probabilities",
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Posterior prob.", xaxis_title="Time (UTC)",
        yaxis=dict(range=[0, 1]),
    )
    return fig


# Bear → Bull palettes: red/orange for bearish states, green shades for bullish states.
# States are sorted ascending by mean_log_ret (index 0 = most bearish, n-1 = most bullish).
# Hardcoded per count so the split is always exactly half-and-half (or nearest integer).
_REGIME_COLOR_PALETTES: dict[int, list[str]] = {
    2: ["#e74c3c", "#2ecc71"],
    3: ["#e74c3c", "#e67e22", "#2ecc71"],
    4: ["#e74c3c", "#e67e22", "#a8d5a2", "#2ecc71"],
    5: ["#c0392b", "#e74c3c", "#e67e22", "#a8d5a2", "#2ecc71"],
    6: ["#c0392b", "#e74c3c", "#e67e22", "#a8d5a2", "#2ecc71", "#27ae60"],
}

def _regime_colors(n_states: int) -> list[str]:
    """Return n colors: lower half red/orange (bearish), upper half green (bullish)."""
    return _REGIME_COLOR_PALETTES.get(n_states, _REGIME_COLOR_PALETTES[4])


def _plot_combined(
    df_prices: pd.DataFrame,
    df_reg: pd.DataFrame,
    ticker: str,
    n_states: int,
    characteristics: dict | None = None,
    default_months: int = 3,
) -> go.Figure:
    """Price + posterior in one figure: shared x-axis, 3-month default, rangeslider.

    Performance: uses go.Scattergl (WebGL) for price traces and eliminates vrects
    entirely — replacing 500+ shape objects with n_states colored line segments.
    """
    d = df_prices.join(df_reg[["regime"]], how="inner").dropna(subset=["close", "regime"]).copy()

    colors = _regime_colors(n_states)

    # row_heights: price chart taller, posterior ≈ 75 % of price height
    price_h, post_h = 0.62, 0.38

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[price_h, post_h],
        vertical_spacing=0.02,
        # Titles dropped — ticker already shown in the selectbox above the chart;
        # removing them saves ~30 px and removes the need for top margin.
    )

    # ── Row 1: one WebGL trace per regime state ────────────────────────────
    for state in range(n_states):
        mask = d["regime"] == state
        y = d["close"].where(mask)          # NaN outside this regime
        rc = characteristics.get(state) if characteristics else None
        label = rc.name if rc else f"R{state}"
        fig.add_trace(go.Scattergl(
            x=d.index, y=y,
            mode="lines", name=label,
            line=dict(color=colors[state], width=1.5),
            legendgroup=f"R{state}",
        ), row=1, col=1)

    # ── Row 2: stacked posteriors (go.Scatter — stackgroup not in Scattergl) ─
    for i in range(n_states):
        col_name = f"p_regime_{i}"
        if col_name not in df_reg.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_reg.index, y=df_reg[col_name],
            mode="lines", name=f"R{i} prob",
            line=dict(color=colors[i]),
            stackgroup="one",
            legendgroup=f"R{i}",
            showlegend=False,
        ), row=2, col=1)

    # ── Default view: last N months ────────────────────────────────────────
    # In Plotly 6 with shared_xaxes=True: row1 xaxis has matches='x2',
    # so set range only on xaxis2 (primary/bottom axis).
    end_dt = d.index[-1]
    start_3m = end_dt - pd.Timedelta(days=default_months * 30)

    fig.update_layout(
        # Target ~500 px so the full chart (price + posterior + rangeslider)
        # fits on screen without scrolling alongside the selectbox above it.
        height=500,
        margin=dict(l=12, r=12, t=10, b=8),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=0.99,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, weight="bold"),
        ),
        yaxis=dict(title="Price ($)", title_standoff=4),
        yaxis2=dict(title="Prob.", range=[0, 1], title_standoff=4),
        xaxis2=dict(
            range=[start_3m, end_dt],
            type="date",
            rangeslider=dict(visible=True, thickness=0.03),
            tickfont=dict(size=10),
        ),
        xaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HMM Regime Dashboard", layout="wide")
st.title("HMM Regime Dashboard — Multi-Ticker + Options")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"]: dict[str, TickerResult] = {}
if "proposed_orders" not in st.session_state:
    st.session_state["proposed_orders"]: list[dict] = {}
if "cache_mtime_seen" not in st.session_state:
    st.session_state["cache_mtime_seen"] = 0.0

# ---------------------------------------------------------------------------
# Start background scheduler (once per process)
# ---------------------------------------------------------------------------
get_scheduler(n_states=4, trade_mode="paper")

# Load from cache if session has no results yet (e.g. page refresh)
if not st.session_state["results"]:
    cache = load_cache()
    if cache:
        st.session_state["results"] = cache["results"]
        st.session_state["proposed_orders"] = cache.get("proposed", [])
        st.session_state["cache_mtime_seen"] = cache_mtime()

# ---------------------------------------------------------------------------
# Auto-refresh fragment — checks for new cache every 30 minutes (matches scheduler)
# ---------------------------------------------------------------------------
@st.fragment(run_every="30m")
def _auto_refresh_check():
    mtime = cache_mtime()
    if mtime > st.session_state.get("cache_mtime_seen", 0.0):
        cache = load_cache()
        if cache:
            st.session_state["results"] = cache["results"]
            st.session_state["proposed_orders"] = cache.get("proposed", [])
            st.session_state["cache_mtime_seen"] = mtime
            updated_at = cache.get("updated_at")
            st.toast(f"Auto-refreshed at {updated_at.strftime('%H:%M ET') if updated_at else 'unknown'}", icon="🔄")

_auto_refresh_check()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_TYPE_COLOURS = {
    "directional_bull": "#2ecc71",
    "directional_bear": "#e74c3c",
    "vol_expansion": "#e67e22",
    "mean_reverting": "#3498db",
}


def _colour_for_type(rtype: str) -> str:
    return _TYPE_COLOURS.get(rtype, "#95a5a6")


def _annualised_ret(mean_log_ret: float) -> float:
    return mean_log_ret * 252 * 6.5   # ~1638 RTH hourly bars/year


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Universe")
    selected_tickers = st.multiselect(
        "Tickers", TICKERS, default=TICKERS, key="ticker_select"
    )

    st.divider()
    st.header("Data")
    # Scheduler status
    from src.scheduler import _is_market_hours
    cache = _cached_load_cache()
    if cache:
        updated_at = cache.get("updated_at")
        ts = updated_at.strftime("%Y-%m-%d %H:%M ET") if updated_at else "unknown"
        st.success(f"Last auto-refresh: {ts}")
    if _is_market_hours():
        st.info("Market open — auto-refreshing hourly at :30")
    else:
        st.caption("Market closed — scheduler paused until next session")

    col_a, col_b = st.columns(2)
    with col_a:
        btn_import = st.button("Import CSVs", use_container_width=True)
    with col_b:
        btn_yf = st.button("Fetch yfinance", use_container_width=True)

    st.divider()
    st.header("Model")
    n_states = st.slider("Regime states", 2, 6, 4)
    horizon_hours = st.slider("Forecast horizon (hours)", 1, 240, 24)
    rth_only = st.checkbox("RTH bars only (already applied)", value=True, disabled=True)

    if HMM_AVAILABLE:
        btn_fit = st.button("Fit HMM — all tickers", use_container_width=True, type="primary")
    else:
        st.error("hmmlearn not installed.\nInstall C++ Build Tools then:\n`pip install hmmlearn`")
        btn_fit = st.button("Fit HMM — all tickers", disabled=True, use_container_width=True)

    st.divider()
    st.header("Strategy Config")
    with st.expander("Option strategy parameters", expanded=False):
        _cfg = _load_config()
        _opt = _cfg.get("option_strategy", {})

        cfg_target_dte   = st.number_input("Target DTE",    min_value=1,    max_value=90,  value=int(_opt.get("target_dte",   21)), step=1)
        cfg_dte_min      = st.number_input("Min DTE",       min_value=1,    max_value=90,  value=int(_opt.get("dte_min",      14)), step=1)
        cfg_dte_max      = st.number_input("Max DTE",       min_value=1,    max_value=180, value=int(_opt.get("dte_max",      45)), step=1)
        cfg_delta_vert   = st.slider("Delta — vertical long leg",  0.10, 0.70, float(_opt.get("delta_vert",  0.40)), step=0.01)
        cfg_delta_wing   = st.slider("Delta — condor short leg",   0.05, 0.40, float(_opt.get("delta_wing",  0.16)), step=0.01)
        cfg_otm_pct      = st.slider("OTM % — strangle legs",      0.01, 0.15, float(_opt.get("otm_pct",    0.03)), step=0.01)
        cfg_strike_range = st.number_input("Strike range (each side)", min_value=5, max_value=100, value=int(_opt.get("strike_range", 20)), step=5)

        if st.button("Save to config.json", use_container_width=True, type="primary"):
            _cfg["option_strategy"] = {
                "target_dte":   cfg_target_dte,
                "dte_min":      cfg_dte_min,
                "dte_max":      cfg_dte_max,
                "delta_vert":   round(cfg_delta_vert, 2),
                "delta_wing":   round(cfg_delta_wing, 2),
                "otm_pct":      round(cfg_otm_pct, 2),
                "strike_range": cfg_strike_range,
            }
            _save_config(_cfg)
            st.success("Saved. Takes effect on next recommend run.")

    st.divider()
    st.header("Trading")
    trade_mode = st.radio("Mode", ["paper", "live"], horizontal=True)
    btn_schwab_info = st.button("Authenticate Schwab (instructions)")
    btn_execute = st.button("Execute trades", use_container_width=True)

# ---------------------------------------------------------------------------
# Sidebar actions
# ---------------------------------------------------------------------------
status_bar = st.empty()

if btn_schwab_info:
    st.info(
        "**Schwab OAuth (one-time setup)**\n\n"
        "1. Fill `SCHWAB_APP_KEY`, `SCHWAB_APP_SECRET`, `SCHWAB_CALLBACK_URL` in your `.env` file.\n"
        "2. Run from a terminal (not Streamlit):\n"
        "   ```\n"
        "   .venv\\Scripts\\activate\n"
        "   python -m src.broker auth\n"
        "   ```\n"
        "3. Log in via the browser, authorise, then close the terminal.\n"
        "4. The token is saved to `data/schwab_token.json`.\n"
        "5. Set `TRADE_MODE=live` in `.env` and restart the dashboard."
    )

if btn_import:
    msgs = []
    for t in (selected_tickers or TICKERS):
        _, msg = import_csv_to_parquet(t)
        msgs.append(msg)
    status_bar.success("\n\n".join(msgs))

if btn_yf:
    msgs = []
    prog = st.progress(0)
    tickers_to_update = selected_tickers or TICKERS
    for idx, t in enumerate(tickers_to_update):
        prog.progress((idx + 1) / len(tickers_to_update))
        _, msg = update_with_yfinance(t)
        msgs.append(msg)
    prog.empty()
    status_bar.success("\n\n".join(msgs))

if btn_fit and HMM_AVAILABLE:
    bars = load_all_tickers(selected_tickers or TICKERS)
    if not bars:
        status_bar.error("No parquet data found. Import CSVs or Fetch yfinance first.")
    else:
        with st.spinner("Fitting HMMs…"):
            results = run_all_tickers(bars, n_states=n_states)
        st.session_state["results"] = results
        # Build proposed orders (no live chain — paper simulation)
        proposed = []
        for t, res in results.items():
            if res.error or not res.characteristics:
                continue
            rc = res.characteristics.get(res.current_regime)
            if rc is None:
                continue
            last_close = float(res.df_prices["close"].iloc[-1])
            order, meta = select_and_build_order(t, rc.regime_type, {}, last_close)
            proposed.append({"ticker": t, "order": order, "meta": meta, "rc": rc})
        st.session_state["proposed_orders"] = proposed
        n_ok = sum(1 for r in results.values() if r.error is None)
        status_bar.success(f"HMM fitted for {n_ok}/{len(results)} tickers.")

if btn_execute:
    if not st.session_state["proposed_orders"]:
        status_bar.warning("No proposed orders.  Fit HMM first.")
    else:
        msgs = []
        for item in st.session_state["proposed_orders"]:
            rec = execute_order(item["order"], item["meta"], mode=trade_mode)
            msgs.append(f"{rec.ticker}: {rec.status} (id={rec.id})")
        status_bar.success("\n\n".join(msgs))
        st.session_state["proposed_orders"] = []

# ---------------------------------------------------------------------------
# Main panel — tabs
# Each tab is wrapped in @st.fragment so interactions within one tab
# do not trigger a full-page rerun of all other tabs.
# ---------------------------------------------------------------------------
if not _thetadata_available():
    st.warning(
        "**ThetaData Terminal is not running.** "
        "Option recommendations and live P&L price updates are unavailable. "
        "Start ThetaData Terminal at http://127.0.0.1:25503, then the warning will clear automatically.",
        icon="⚠️",
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Regime Overview", "Price Charts", "Options & Trading", "Model Diagnostics", "Trade Tracker"]
)


@st.fragment
def _render_tab1():
    results: dict[str, TickerResult] = st.session_state.get("results", {})
    if not results:
        st.info("Fit the HMM (sidebar) to see regime overview.")
        return

    rows = []
    for t, res in results.items():
        if res.error:
            rows.append({"Ticker": t, "Regime": "ERROR", "Name": res.error,
                         "Mean Ret (ann)": "-", "Vol": "-",
                         "Type": "-", "Confidence": "-", "P(change 24h)": "-", "_type": ""})
            continue
        rc = res.characteristics.get(res.current_regime)
        if rc is None:
            continue
        fc = res.forecast
        rows.append({
            "Ticker": t, "Regime": f"R{res.current_regime}", "Name": rc.name,
            "Mean Ret (ann)": f"{_annualised_ret(rc.mean_log_ret):.1%}",
            "Vol": f"{rc.mean_vol_20:.4f}", "Type": rc.regime_type,
            "Confidence": f"{fc.get('current_confidence', 0):.1%}",
            "P(change 24h)": f"{fc.get('prob_change_by_horizon', 0):.1%}",
            "_type": rc.regime_type,
        })

    df_summary = pd.DataFrame(rows)
    display_cols = [c for c in df_summary.columns if c != "_type"]

    def _row_style(row: pd.Series):
        c = _colour_for_type(row.get("_type", ""))
        return [f"background-color: {c}22"] * len(row)

    styled = df_summary[display_cols + ["_type"]].style.apply(_row_style, axis=1)
    st.dataframe(styled.data[display_cols], use_container_width=True)
    st.divider()

    for t, res in results.items():
        with st.expander(f"{t} — detail", expanded=False):
            if res.error:
                st.error(res.error); continue
            rc = res.characteristics.get(res.current_regime)
            if rc is None:
                st.warning("No regime characteristics."); continue
            fc = res.forecast
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Regime", rc.name)
            m2.metric("Type", rc.regime_type)
            m3.metric("Confidence", f"{fc.get('current_confidence', 0):.1%}")
            m4.metric("P(change 24h)", f"{fc.get('prob_change_by_horizon', 0):.1%}")
            df_p = res.df_prices.tail(500)
            df_r = res.df_reg.tail(500)
            st.plotly_chart(_plot_price_regimes(df_p, df_r, t, res.n_states), use_container_width=True)
            st.plotly_chart(_plot_posteriors(df_r, res.n_states, t), use_container_width=True)


@st.fragment
def _render_tab2():
    results: dict[str, TickerResult] = st.session_state.get("results", {})
    if not results:
        st.info("Fit the HMM first.")
        return
    pick_t = st.selectbox("Ticker", list(results.keys()), key="tab2_ticker")
    res = results[pick_t]
    if res.error:
        st.error(res.error); return
    # Combined chart: shared x-axis, 3-month default view, horizontal rangeslider
    try:
        fig = _plot_combined(
            res.df_prices, res.df_reg, pick_t, res.n_states,
            characteristics=res.characteristics,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as _e:
        import traceback as _tb
        st.error(f"Chart error: {_e}")
        st.code(_tb.format_exc())
    st.subheader("Regime statistics")
    colors_tbl = _regime_colors(res.n_states)
    stat_rows = []
    for i, rc in res.characteristics.items():
        mask = res.df_reg["regime"] == i
        stat_rows.append({
            "_state": i,
            "State": f"R{i}", "Name": rc.name, "Type": rc.regime_type,
            "Ann. ret": f"{_annualised_ret(rc.mean_log_ret):.1%}",
            "Mean vol-20": f"{rc.mean_vol_20:.4f}",
            "Bar count": int(mask.sum()), "% of time": f"{mask.mean():.1%}",
        })
    df_stat = pd.DataFrame(stat_rows)
    display_cols = [c for c in df_stat.columns if c != "_state"]

    def _stat_row_style(row: pd.Series):
        state = int(row["_state"])
        hex_color = colors_tbl[state % len(colors_tbl)]
        return [f"background-color: {hex_color}30"] * len(row)  # ~19 % opacity

    styled_stat = df_stat.style.apply(_stat_row_style, axis=1)
    st.dataframe(styled_stat.data[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("""
**How to read these charts**

**Top chart — Price by regime**
Each coloured line segment shows the closing price during a distinct HMM regime.
- 🟢 **Green / light-green** — bullish regimes (positive mean hourly return): *directional bull* or *bullish mean-reverting*.
- 🔴 **Red / orange** — bearish regimes (negative mean hourly return): *directional bear* or *high-volatility*.
Gaps between segments indicate a regime transition. Hover to see exact price and timestamp.

**Bottom chart — Posterior probabilities**
The stacked bands show the model's confidence in each regime at every bar.
A tall band means the model is certain about which regime is active.
Narrow, mixed bands near transitions indicate uncertainty.

**Rangeslider (bottom strip)**
Drag the handles to zoom in/out; click-and-drag inside the strip to pan.
Both charts scroll in sync. Default view spans the last **3 months**.
""")



@st.fragment
def _render_tab3():
    proposed = st.session_state.get("proposed_orders", [])
    st.subheader("Proposed orders")
    if not proposed:
        st.info("Fit HMM to generate proposed orders.")
    else:
        order_rows = []
        for item in proposed:
            meta = item["meta"]
            order_rows.append({
                "Ticker": meta["ticker"], "Regime type": meta["regime_type"],
                "Strategy": meta["strategy"], "Est. net price": meta.get("est_net_price", 0.0),
                "Legs": len(meta.get("legs", [])), "Error": meta.get("error") or "",
            })
        st.dataframe(pd.DataFrame(order_rows), use_container_width=True)
        st.caption("Click **Execute trades** in the sidebar to paper/live execute all proposals.")
    st.divider()
    st.subheader("Trade log")
    trades = load_paper_trades()
    if not trades:
        st.info("No trades recorded yet.")
    else:
        log_rows = []
        for tr in reversed(trades):
            log_rows.append({
                "ID": tr.id, "Time (UTC)": tr.timestamp_utc, "Ticker": tr.ticker,
                "Strategy": tr.strategy, "Regime type": tr.regime_type,
                "Net price": tr.est_net_price, "Qty": tr.quantity,
                "Mode": tr.mode, "Status": tr.status,
                "Schwab ID": tr.schwab_order_id or "", "Error": tr.error or "",
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)


@st.fragment
def _render_tab4():
    results: dict[str, TickerResult] = st.session_state.get("results", {})
    if not results:
        st.info("Fit HMM first."); return
    pick_d = st.selectbox("Ticker", list(results.keys()), key="tab4_ticker")
    res = results[pick_d]
    if res.error or res.model is None:
        st.error(res.error or "No model available."); return
    st.subheader("Transition matrix")
    P = res.model.transmat_
    labels = [f"R{i}" for i in range(res.n_states)]
    fig_tm = go.Figure(go.Heatmap(
        z=P, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in P], texttemplate="%{text}",
    ))
    fig_tm.update_layout(title="Transition probabilities", height=350,
                         margin=dict(l=20, r=20, t=40, b=20),
                         xaxis_title="To regime", yaxis_title="From regime")
    st.plotly_chart(fig_tm, use_container_width=True)
    st.subheader("State means (original scale)")
    means_df_rows = []
    for i, rc in res.characteristics.items():
        means_df_rows.append({
            "State": f"R{i} ({rc.name})", "Mean log-ret": f"{rc.mean_log_ret:.5f}",
            "Mean vol-20": f"{rc.mean_vol_20:.4f}", "Type": rc.regime_type,
        })
    st.dataframe(pd.DataFrame(means_df_rows), use_container_width=True)
    st.subheader("Feature distributions per regime")
    feat_df = make_features(res.df_prices).join(res.df_reg[["regime"]], how="inner")
    fig_dist = go.Figure()
    for i in range(res.n_states):
        subset = feat_df.loc[feat_df["regime"] == i, "log_ret"].dropna()
        if subset.empty: continue
        fig_dist.add_trace(go.Histogram(x=subset, name=f"R{i}", opacity=0.6, nbinsx=80))
    fig_dist.update_layout(barmode="overlay", title="log-ret distribution by regime",
                           height=400, margin=dict(l=20, r=20, t=40, b=20),
                           xaxis_title="log-ret", yaxis_title="count")
    st.plotly_chart(fig_dist, use_container_width=True)


@st.fragment
def _render_tab5():
    st.subheader("Tracked Option Trades")
    col_r1, col_r2 = st.columns([1, 5])
    with col_r1:
        if st.button("Refresh Prices", use_container_width=True, type="primary"):
            with st.spinner("Fetching current option prices…"):
                update_all_open_trades()
            _cached_load_trades.clear()
            st.toast("Prices updated.", icon="✅")

    tracked = _cached_load_trades()

    if not tracked:
        st.info("No tracked trades yet.  Run `python -m src.recommend` to generate "
                "recommendations — they will be saved automatically.")
        return

    st.markdown("#### Recommendations")
    rec_rows = []
    for tr in tracked:
        entry_label = f"{'credit' if tr.price_type == 'credit' else 'debit'} ${tr.entry_net:.2f}"
        rec_rows.append({
            "Rec. Date/Time":  tr.recommended_at if tr.recommended_at else tr.date_recommended,
            "Ticker":          tr.ticker,
            "Regime":          tr.regime_name,
            "Price at Entry":  f"${tr.underlying_at_entry:.2f}",
            "Strategy":        tr.strategy.replace("_", " ").title(),
            "Entry":           entry_label,
            "Confidence":      f"{tr.confidence:.0%}" if tr.confidence else "—",
            "Expiry":          tr.expiry,
            "Status":          tr.status,
        })
    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    st.divider()

    st.markdown("#### Live P&L")
    summary_rows = []
    for tr in tracked:
        net_mid, pnl_d = latest_pnl(tr)
        pct = pnl_pct(tr)
        summary_rows.append({
            "Ticker": tr.ticker, "Strategy": tr.strategy, "Regime": tr.regime_name,
            "Expiry": tr.expiry, "DTE left": dte_remaining(tr), "Days held": days_held(tr),
            "Entry $": tr.entry_net, "Current $": round(net_mid, 2),
            "P&L $": round(pnl_d, 2), "P&L %": f"{pct:.1%}",
            "Max profit $": tr.max_profit if tr.max_profit != float("inf") else "unlimited",
            "Max loss $": tr.max_loss, "Status": tr.status,
        })
    df_tt = pd.DataFrame(summary_rows)

    def _pnl_style(val):
        if isinstance(val, (int, float)):
            color = "#2ecc71" if val >= 0 else "#e74c3c"
            return f"color: {color}; font-weight: bold"
        return ""

    styled_tt = df_tt.style.applymap(_pnl_style, subset=["P&L $"])
    st.dataframe(styled_tt, use_container_width=True)
    st.divider()

    st.subheader("Trade detail")
    for tr in tracked:
        net_mid, pnl_d = latest_pnl(tr)
        pct    = pnl_pct(tr)
        dte    = dte_remaining(tr)
        _opened = tr.recommended_at if tr.recommended_at else tr.date_recommended
        # Date-only for header to avoid truncation; full datetime inside card
        _opened_date = _opened[:10] if _opened else ""
        _opened_full = _opened[:16] if _opened else ""
        label  = (f"{tr.ticker}  |  {tr.strategy}  |  exp {tr.expiry}  "
                  f"|  P&L ${pnl_d:+.2f}  ({pct:+.1%})  |  {tr.status}"
                  f"  |  Open {_opened_date}")
        with st.expander(label, expanded=False):
            # Row 1: P&L metrics
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("Entry net", f"${tr.entry_net:.2f}")
            mc2.metric("Current mid", f"${net_mid:.2f}")
            mc3.metric("P&L $", f"${pnl_d:+.2f}", delta=f"{pct:+.1%}")
            mc4.metric("DTE left", dte)
            mc5.metric("Days held", days_held(tr))
            _conf = f"{tr.confidence:.0%}" if getattr(tr, "confidence", None) else "—"
            mc6.metric("Confidence", _conf)
            # Row 2: underlying prices + trade structure
            mn1, mn2, mn3, mn4, mn5 = st.columns(5)
            mn1.metric("Entry price", f"${tr.underlying_at_entry:.2f}")
            if tr.status == "open":
                _cur_ul = tr.daily_prices[-1]["underlying"] if tr.daily_prices else tr.underlying_at_entry
                mn2.metric("Current price", f"${_cur_ul:.2f}",
                           delta=f"{_cur_ul - tr.underlying_at_entry:+.2f}")
            else:
                _close_ul = getattr(tr, "underlying_at_close", 0.0) or tr.underlying_at_entry
                mn2.metric("Close price", f"${_close_ul:.2f}",
                           delta=f"{_close_ul - tr.underlying_at_entry:+.2f}")
            mn3.metric("Max profit", f"${tr.max_profit:.2f}")
            mn4.metric("Max loss", f"${tr.max_loss:.2f}")
            mn5.metric("Opened", _opened_full)
            st.markdown("**Legs**")
            leg_rows = []
            for leg in tr.legs:
                leg_rows.append({
                    "Action": leg["action"], "Right": leg["right"], "Strike": leg["strike"],
                    "Entry mid": leg.get("entry_mid", "-"),
                    "Entry ask": leg.get("entry_ask", "-"),
                    "Entry bid": leg.get("entry_bid", "-"),
                })
            st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)
            if tr.daily_prices and len(tr.daily_prices) > 1:
                dp_df = pd.DataFrame(tr.daily_prices)
                dp_df["date"] = pd.to_datetime(dp_df["date"])
                dp_df = dp_df.sort_values("date")
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=dp_df["date"], y=dp_df["pnl_dollars"],
                    mode="lines+markers", name="P&L $",
                    line=dict(color="#2ecc71" if pnl_d >= 0 else "#e74c3c", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(46,204,113,0.12)" if pnl_d >= 0 else "rgba(231,76,60,0.12)",
                ))
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                fig_pnl.update_layout(
                    title=f"{tr.ticker} — daily P&L (per contract × 100)",
                    height=280, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Date", yaxis_title="P&L ($)", yaxis=dict(tickprefix="$"))
                st.plotly_chart(fig_pnl, use_container_width=True)
                fig_ul = go.Figure()
                fig_ul.add_trace(go.Scatter(
                    x=dp_df["date"], y=dp_df["underlying"],
                    mode="lines+markers", name="Underlying",
                    line=dict(color="#3498db", width=2),
                ))
                fig_ul.update_layout(
                    title=f"{tr.ticker} — underlying price",
                    height=220, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig_ul, use_container_width=True)
            else:
                st.caption("Not enough daily snapshots yet for a chart (run again tomorrow.")


# ============================================================
# Render tabs via fragments — only the active tab rerenders
# on interaction, not the whole page
# ============================================================
with tab1:
    _render_tab1()

with tab2:
    _render_tab2()

with tab3:
    _render_tab3()

with tab4:
    _render_tab4()

with tab5:
    _render_tab5()

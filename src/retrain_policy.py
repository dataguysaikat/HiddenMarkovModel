"""
retrain_policy.py — Analyse closed trade outcomes and update strategy policy.

Usage:  python -m src.retrain_policy

Effect: reads data/tracked_trades.json, prints a performance report, and
        writes "strategy_performance" + "learned_policy" back to config.json.

recommend.py reads "learned_policy" on startup to:
  - skip regime types whose win rate is too low
  - require a minimum confidence level before entering a trade
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_CONFIG_PATH    = Path(__file__).parent.parent / "config.json"
_MIN_SAMPLE     = 3     # closed trades needed before a regime policy decision is made
_WIN_RATE_SKIP  = 0.25  # win rate below this => skip the regime entirely
_WIN_RATE_FLOOR = 0.40  # win rate below this => flag caution (still trade, but warn)


# ---------------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------------

def _final_pnl(trade) -> float | None:
    """Return the last recorded pnl_dollars for a trade, or None."""
    if not trade.daily_prices:
        return None
    return trade.daily_prices[-1]["pnl_dollars"]


def compute_stats(trades) -> dict[str, dict]:
    """
    Group closed/expired trades by regime_type and compute performance stats.

    Returns dict[regime_type] = {
        count, wins, win_rate, avg_pnl, avg_pnl_pct, status
    }
    where status is one of: "ok" | "caution" | "skip"
    """
    buckets: dict[str, list] = defaultdict(list)
    for t in trades:
        if t.status not in ("closed", "expired"):
            continue
        pnl = _final_pnl(t)
        if pnl is None:
            continue
        max_risk = abs(t.max_loss) * 100 if t.max_loss else None
        pnl_pct  = pnl / max_risk if max_risk else 0.0
        buckets[t.regime_type].append({
            "pnl":     pnl,
            "pnl_pct": pnl_pct,
            "win":     pnl > 0,
            "conf":    t.confidence,
        })

    result: dict[str, dict] = {}
    for regime, rows in buckets.items():
        n        = len(rows)
        wins     = sum(1 for r in rows if r["win"])
        win_rate = wins / n
        avg_pnl     = sum(r["pnl"]     for r in rows) / n
        avg_pnl_pct = sum(r["pnl_pct"] for r in rows) / n

        if win_rate >= _WIN_RATE_FLOOR:
            status = "ok"
        elif win_rate >= _WIN_RATE_SKIP:
            status = "caution"
        else:
            status = "skip"

        result[regime] = {
            "count":        n,
            "wins":         wins,
            "win_rate":     round(win_rate, 4),
            "avg_pnl":      round(avg_pnl, 2),
            "avg_pnl_pct":  round(avg_pnl_pct, 4),
            "status":       status,
        }
    return result


def compute_stats_by_ticker(trades) -> dict[tuple, dict]:
    """
    Group closed/expired trades by (ticker, strategy) and compute performance.

    Returns dict[(ticker, strategy)] = {
        count, wins, win_rate, avg_pnl, total_pnl, avg_pnl_pct
    }
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for t in trades:
        if t.status not in ("closed", "expired"):
            continue
        pnl = _final_pnl(t)
        if pnl is None:
            continue
        max_risk = abs(t.max_loss) * 100 if t.max_loss else None
        pnl_pct  = pnl / max_risk if max_risk else 0.0
        buckets[(t.ticker, t.strategy)].append({
            "pnl":     pnl,
            "pnl_pct": pnl_pct,
            "win":     pnl > 0,
        })

    result: dict[tuple, dict] = {}
    for key, rows in buckets.items():
        n        = len(rows)
        wins     = sum(1 for r in rows if r["win"])
        win_rate = wins / n
        avg_pnl     = sum(r["pnl"]     for r in rows) / n
        total_pnl   = sum(r["pnl"]     for r in rows)
        avg_pnl_pct = sum(r["pnl_pct"] for r in rows) / n
        result[key] = {
            "count":        n,
            "wins":         wins,
            "win_rate":     round(win_rate, 4),
            "avg_pnl":      round(avg_pnl, 2),
            "total_pnl":    round(total_pnl, 2),
            "avg_pnl_pct":  round(avg_pnl_pct, 4),
        }
    return result


def confidence_calibration(trades) -> float:
    """
    Find the lowest confidence threshold where closed-trade win rate >= 50%.

    Evaluates thresholds [0.70, 0.80, 0.90, 0.95] in ascending order.
    Returns 0.0 if there is insufficient data to calibrate.
    """
    THRESHOLDS = [0.70, 0.80, 0.90, 0.95]

    rows = []
    for t in trades:
        if t.status not in ("closed", "expired"):
            continue
        pnl = _final_pnl(t)
        if pnl is None:
            continue
        rows.append({"conf": t.confidence, "win": pnl > 0})

    if len(rows) < _MIN_SAMPLE:
        return 0.0

    for thresh in THRESHOLDS:
        subset = [r for r in rows if r["conf"] >= thresh]
        if len(subset) < _MIN_SAMPLE:
            continue
        win_rate = sum(1 for r in subset if r["win"]) / len(subset)
        if win_rate >= 0.50:
            return thresh

    return 0.0


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"WARNING: config.json parse error: {e}. Using defaults.")
    return {}


def _save_config(cfg: dict) -> None:
    """Atomic write to config.json — prevents corruption on concurrent access."""
    _CONFIG_PATH.parent.mkdir(exist_ok=True, parents=True)
    fd, tmp = tempfile.mkstemp(dir=_CONFIG_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(cfg, indent=2))
        os.replace(tmp, _CONFIG_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_config(stats: dict, min_confidence: float) -> None:
    """
    Merge strategy_performance and learned_policy into config.json.
    Preserves all existing keys (tickers, option_strategy, etc.).
    """
    cfg = _load_config()
    cfg["strategy_performance"] = stats

    skip_regimes    = [r for r, s in stats.items()
                       if s["status"] == "skip"    and s["count"] >= _MIN_SAMPLE]
    caution_regimes = [r for r, s in stats.items()
                       if s["status"] == "caution" and s["count"] >= _MIN_SAMPLE]

    cfg["learned_policy"] = {
        "min_confidence":  min_confidence,
        "skip_regimes":    skip_regimes,
        "caution_regimes": caution_regimes,
    }
    _save_config(cfg)


# ---------------------------------------------------------------------------
# Alert stats — learn which (from_regime, to_regime) transitions actually hurt
# ---------------------------------------------------------------------------

def compute_alert_stats(trades) -> dict:
    """
    For each (entry_regime, alerted_to_regime) pair seen in alert_log entries
    of closed/expired trades, compute:
      - alerted: win rate + avg P&L of trades that were alerted for this pair
      - baseline: win rate + avg P&L of closed trades in the same entry regime
        that had NO alert at all

    Verdict:
      "confirmed_incompatible" — alerted trades lose significantly vs baseline
      "cleared"                — alerted trades perform similarly or better → false positive
      "insufficient_data"      — fewer than _MIN_SAMPLE alerted trades for this pair

    Returns dict keyed by (from_regime, to_regime) tuple.
    """
    # Build baseline: closed trades with no alerts, keyed by entry regime
    baseline: dict[str, list[float]] = {}
    for t in trades:
        if t.status not in ("closed", "expired"):
            continue
        pnl = _final_pnl(t)
        if pnl is None:
            continue
        if not getattr(t, "alert_log", []):
            baseline.setdefault(t.regime_type, []).append(pnl)

    # Build alerted: closed trades with alert_log, keyed by (from, to) pair
    alerted: dict[tuple, list[float]] = {}
    for t in trades:
        if t.status not in ("closed", "expired"):
            continue
        pnl = _final_pnl(t)
        if pnl is None:
            continue
        for entry in getattr(t, "alert_log", []):
            pair = (t.regime_type, entry["to_regime"])
            alerted.setdefault(pair, []).append(pnl)

    result = {}
    for pair, pnls in alerted.items():
        from_regime = pair[0]
        base_pnls   = baseline.get(from_regime, [])

        n_alerted   = len(pnls)
        wr_alerted  = sum(1 for p in pnls if p > 0) / n_alerted
        avg_alerted = sum(pnls) / n_alerted

        n_base   = len(base_pnls)
        wr_base  = sum(1 for p in base_pnls if p > 0) / n_base if n_base else None
        avg_base = sum(base_pnls) / n_base if n_base else None

        if n_alerted < _MIN_SAMPLE:
            verdict = "insufficient_data"
        elif wr_alerted > 0.50:
            verdict = "cleared"          # alert is a false positive — trades survive fine
        elif wr_base is not None and wr_alerted < 0.35 and wr_base > 0.50:
            verdict = "confirmed_incompatible"   # alerts predict real losses
        else:
            verdict = "insufficient_data"        # mixed signal — wait for more data

        result[pair] = {
            "alerted_count":   n_alerted,
            "alerted_win_rate": round(wr_alerted, 4),
            "alerted_avg_pnl":  round(avg_alerted, 2),
            "baseline_count":  n_base,
            "baseline_win_rate": round(wr_base, 4) if wr_base is not None else None,
            "baseline_avg_pnl":  round(avg_base, 2) if avg_base is not None else None,
            "verdict": verdict,
        }

    return result


def update_alert_policy(alert_stats: dict) -> None:
    """Write confirmed/cleared pairs to config.json["regime_alert_policy"]."""
    cfg = _load_config()
    confirmed = [list(pair) for pair, s in alert_stats.items()
                 if s["verdict"] == "confirmed_incompatible"]
    cleared   = [list(pair) for pair, s in alert_stats.items()
                 if s["verdict"] == "cleared"]
    cfg["regime_alert_policy"] = {
        "confirmed_incompatible": confirmed,
        "cleared_incompatible":   cleared,
    }
    _save_config(cfg)


def print_alert_report(alert_stats: dict) -> None:
    print()
    print("=" * 72)
    print("  REGIME ALERT ANALYSIS  (from closed/expired trade alert_log)")
    print("=" * 72)

    if not alert_stats:
        print("  No alert history yet — alert policy uses theory defaults.")
        print("=" * 72)
        return

    print(f"  {'Transition':<42} {'N':>4} {'Win%':>7} {'AvgP&L':>9}  Verdict")
    print("  " + "-" * 68)
    for (frm, to), s in alert_stats.items():
        transition = f"{frm} → {to}"
        verdict_sym = {
            "confirmed_incompatible": "[!!] CONFIRMED",
            "cleared":                "[OK] CLEARED",
            "insufficient_data":      "[??] low-N",
        }.get(s["verdict"], s["verdict"])
        print(f"  {transition:<42} {s['alerted_count']:>4} "
              f"{s['alerted_win_rate']:>7.1%} {s['alerted_avg_pnl']:>+9.2f}  {verdict_sym}")

    actionable = {k: v for k, v in alert_stats.items()
                  if v["verdict"] in ("confirmed_incompatible", "cleared")}
    if not actionable:
        print(f"\n  No actionable verdicts yet (need >= {_MIN_SAMPLE} alerted trades per pair).")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(stats: dict, min_confidence: float) -> None:
    print()
    print("=" * 72)
    print("  STRATEGY PERFORMANCE  (closed + expired trades only)")
    print("=" * 72)

    if not stats:
        print("  No closed/expired trades yet — policy unchanged.")
        print("=" * 72)
        return

    print(f"  {'Regime':<24} {'N':>4} {'Wins':>5} {'Win%':>7} {'AvgP&L':>9}  {'Status'}")
    print("  " + "-" * 62)
    for regime, s in stats.items():
        sym = {"ok": "[OK]", "caution": "[~~]", "skip": "[!!]"}.get(s["status"], "    ")
        low_n = "" if s["count"] >= _MIN_SAMPLE else " (low-N)"
        print(f"  {regime:<24} {s['count']:>4} {s['wins']:>5} "
              f"{s['win_rate']:>7.1%} {s['avg_pnl']:>+9.2f}  {sym}{low_n}")

    print()
    if min_confidence > 0:
        print(f"  Suggested min confidence : {min_confidence:.0%}")
    else:
        print(f"  Min confidence           : not yet calibrated "
              f"(need >= {_MIN_SAMPLE} closed trades per bucket)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from src.trade_tracker import load_trades  # local import to avoid circular deps

    trades = load_trades()
    n_closed = sum(1 for t in trades if t.status in ("closed", "expired"))
    n_open   = sum(1 for t in trades if t.status == "open")
    print(f"Loaded {len(trades)} trades: {n_closed} closed/expired, {n_open} open.")

    # --- Strategy performance ---
    stats    = compute_stats(trades)
    min_conf = confidence_calibration(trades)
    print_report(stats, min_conf)

    sufficient = {r for r, s in stats.items() if s["count"] >= _MIN_SAMPLE}
    if sufficient:
        update_config(stats, min_conf)
        print(f"\nconfig.json updated — strategy policy active for: {', '.join(sorted(sufficient))}")
    else:
        print(f"\nInsufficient data (need >= {_MIN_SAMPLE} closed trades per regime). "
              f"Strategy policy unchanged.")

    skip    = [r for r in stats if stats[r]["status"] == "skip"    and r in sufficient]
    caution = [r for r in stats if stats[r]["status"] == "caution" and r in sufficient]
    if skip:
        print(f"  Will SKIP trades for regimes : {', '.join(skip)}")
    if caution:
        print(f"  Will flag CAUTION for        : {', '.join(caution)}")
    if min_conf > 0:
        print(f"  Will require confidence >=   : {min_conf:.0%}")

    # --- Alert policy ---
    alert_stats = compute_alert_stats(trades)
    print_alert_report(alert_stats)

    actionable_alerts = {k: v for k, v in alert_stats.items()
                         if v["verdict"] in ("confirmed_incompatible", "cleared")}
    if actionable_alerts:
        update_alert_policy(alert_stats)
        confirmed = [f"{f}→{t}" for (f, t), v in alert_stats.items()
                     if v["verdict"] == "confirmed_incompatible"]
        cleared   = [f"{f}→{t}" for (f, t), v in alert_stats.items()
                     if v["verdict"] == "cleared"]
        print(f"\nAlert policy written to config.json.")
        if confirmed:
            print(f"  Confirmed harmful transitions : {', '.join(confirmed)}")
        if cleared:
            print(f"  Cleared (false positive) pairs: {', '.join(cleared)}")
    else:
        print(f"\nAlert policy unchanged — insufficient alerted trade history.")


if __name__ == "__main__":
    main()

"""
broker.py — Paper log + live Schwab execution.

Paper mode stores trades in data/paper_trades.json.
Live mode routes orders through the Schwab API (schwab-py).

OAuth note: The OAuth flow cannot run inside Streamlit (it opens a browser
and blocks on a redirect).  Run it once from the CLI:

    python -m src.broker auth

The token is saved to SCHWAB_TOKEN_PATH (default: data/schwab_token.json).
The Streamlit dashboard then calls get_schwab_client() which reads the
saved token non-interactively.

CLI usage:
    python -m src.broker auth   — run OAuth flow, save token
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

PAPER_TRADES_PATH = DATA_DIR / "paper_trades.json"

# ---------------------------------------------------------------------------
# Schwab guard
# ---------------------------------------------------------------------------
try:
    import schwab
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    schwab = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    id: str
    timestamp_utc: str
    ticker: str
    strategy: str
    regime_type: str
    legs: list[dict]
    est_net_price: float
    quantity: int
    mode: str          # "paper" | "live"
    status: str        # "filled_simulated" | "live_submitted" | "simulated_no_chain"
    schwab_order_id: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Schwab client helpers
# ---------------------------------------------------------------------------

def _token_path() -> Path:
    raw = os.getenv("SCHWAB_TOKEN_PATH", "data/schwab_token.json")
    return Path(raw)


def get_schwab_client(token_path: Optional[Path] = None) -> Any:
    """
    Load a Schwab client from the saved token file.
    Returns None if schwab-py is not installed or the token file doesn't exist.
    """
    if not SCHWAB_AVAILABLE:
        return None
    tp = token_path or _token_path()
    if not tp.exists():
        return None
    try:
        app_key = os.getenv("SCHWAB_APP_KEY", "")
        app_secret = os.getenv("SCHWAB_APP_SECRET", "")
        if not app_key or not app_secret:
            return None
        return schwab.client_from_token_file(str(tp), app_key, app_secret)
    except Exception:  # noqa: BLE001
        return None


def get_account_hash(client: Any) -> Optional[str]:
    """Return the first account hash for the authenticated client."""
    if client is None:
        return None
    try:
        resp = client.get_account_numbers()
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data:
            return data[0].get("hashValue")
        return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Paper trade persistence
# ---------------------------------------------------------------------------

def load_paper_trades() -> list[TradeRecord]:
    if not PAPER_TRADES_PATH.exists():
        return []
    try:
        raw = json.loads(PAPER_TRADES_PATH.read_text(encoding="utf-8"))
        return [TradeRecord(**r) for r in raw]
    except Exception:  # noqa: BLE001
        return []


def save_paper_trade(record: TradeRecord) -> None:
    trades = load_paper_trades()
    trades.append(record)
    PAPER_TRADES_PATH.write_text(
        json.dumps([asdict(t) for t in trades], indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())[:8]


def execute_paper(order: Optional[dict], metadata: dict) -> TradeRecord:
    """
    Simulate a paper trade.  Works whether or not a real order dict was built.
    """
    has_order = order is not None
    status = "filled_simulated" if has_order else "simulated_no_chain"

    record = TradeRecord(
        id=_new_id(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ticker=metadata.get("ticker", ""),
        strategy=metadata.get("strategy", ""),
        regime_type=metadata.get("regime_type", ""),
        legs=order.get("legs", []) if has_order else metadata.get("legs", []),
        est_net_price=order.get("net_price", 0.0) if has_order else metadata.get("est_net_price", 0.0),
        quantity=order.get("quantity", 1) if has_order else 1,
        mode="paper",
        status=status,
        error=metadata.get("error") if not has_order else None,
    )
    save_paper_trade(record)
    return record


def execute_live(client: Any, account_hash: str, order: dict, metadata: dict) -> TradeRecord:
    """
    Submit a live order via Schwab.
    The order dict from options.py uses a plain dict format; we convert it to
    a schwab-py OrderBuilder here.
    """
    schwab_order_id: Optional[str] = None
    error: Optional[str] = None
    status = "live_submitted"

    try:
        if not SCHWAB_AVAILABLE:
            raise RuntimeError("schwab-py not installed")

        from schwab.orders.options import option_buy_to_open_limit, option_sell_to_open_limit
        from schwab.orders.common import one_cancels_other

        # Build a multi-leg order using schwab-py's builder
        builder = schwab.orders.generic.OrderBuilder()
        builder.set_order_type(schwab.orders.common.OrderType.NET_DEBIT if order["price_type"] == "debit"
                               else schwab.orders.common.OrderType.NET_CREDIT)
        builder.set_price(order["net_price"])
        builder.set_duration(schwab.orders.common.Duration.DAY)
        builder.set_session(schwab.orders.common.Session.NORMAL)

        for leg in order["legs"]:
            sym = leg["symbol"]
            qty = leg["quantity"]
            action = leg["action"]
            instr = schwab.orders.options.OptionSymbol(sym).build()
            if action == "BUY_TO_OPEN":
                builder.add_option_leg(schwab.orders.common.OptionInstruction.BUY_TO_OPEN, instr, qty)
            else:
                builder.add_option_leg(schwab.orders.common.OptionInstruction.SELL_TO_OPEN, instr, qty)

        resp = client.place_order(account_hash, builder)
        if resp.status_code in (200, 201):
            loc = resp.headers.get("Location", "")
            schwab_order_id = loc.split("/")[-1] if loc else None
        else:
            status = "live_error"
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:  # noqa: BLE001
        status = "live_error"
        error = str(exc)

    record = TradeRecord(
        id=_new_id(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ticker=metadata.get("ticker", ""),
        strategy=metadata.get("strategy", ""),
        regime_type=metadata.get("regime_type", ""),
        legs=order.get("legs", []),
        est_net_price=order.get("net_price", 0.0),
        quantity=order.get("quantity", 1),
        mode="live",
        status=status,
        schwab_order_id=schwab_order_id,
        error=error,
    )
    save_paper_trade(record)
    return record


def execute_order(
    order: Optional[dict],
    metadata: dict,
    mode: Optional[str] = None,
) -> TradeRecord:
    """
    Route to paper or live execution.

    mode: "paper" | "live" | None (reads TRADE_MODE env var; defaults to "paper")
    Falls back to paper if live client is unavailable.
    """
    if mode is None:
        mode = os.getenv("TRADE_MODE", "paper").lower()

    if mode == "live":
        client = get_schwab_client()
        if client is not None:
            account_hash = get_account_hash(client)
            if account_hash and order is not None:
                return execute_live(client, account_hash, order, metadata)
        # Fall through to paper if live setup is incomplete
        mode = "paper"

    return execute_paper(order, metadata)


# ---------------------------------------------------------------------------
# OAuth flow (CLI only)
# ---------------------------------------------------------------------------

def run_auth_flow() -> None:
    """
    Run the Schwab OAuth flow from the command line and save the token file.
    Builds the authorization URL with only client_id and redirect_uri,
    then exchanges the returned code for a token manually.
    Usage: python -m src.broker auth
    """
    import base64
    import urllib.parse
    import requests as _requests
    from datetime import datetime, timezone

    app_key = os.getenv("SCHWAB_APP_KEY", "")
    app_secret = os.getenv("SCHWAB_APP_SECRET", "")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
    token_path = _token_path()

    if not app_key or not app_secret:
        print("ERROR: Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET in your .env file.")
        sys.exit(1)

    # Build auth URL with only client_id and redirect_uri
    auth_url = (
        "https://api.schwabapi.com/v1/oauth/authorize"
        f"?client_id={app_key}"
        f"&redirect_uri={urllib.parse.quote(callback_url, safe='')}"
    )

    print("=" * 60)
    print("Open this URL in your browser:")
    print()
    print(auth_url)
    print()
    print("1. Log in to Schwab and click Allow")
    print("2. Browser redirects to 127.0.0.1:8182 (page won't load — that's fine)")
    print("3. Copy the FULL URL from the address bar and paste below")
    print("=" * 60)
    print()

    redirect_url = input("Redirect URL> ").strip()
    if not redirect_url:
        print("ERROR: No URL entered.")
        sys.exit(1)

    # Extract the authorization code from the redirect URL
    parsed = urllib.parse.urlparse(redirect_url)
    params = urllib.parse.parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    if not code:
        print(f"ERROR: No 'code' parameter found in redirect URL.\nGot: {redirect_url}")
        sys.exit(1)

    # Exchange code for token
    print("\nExchanging code for token...")
    credentials = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
    resp = _requests.post(
        "https://api.schwabapi.com/v1/oauth/token",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": callback_url,
        },
    )

    if resp.status_code != 200:
        print(f"ERROR: Token exchange failed (HTTP {resp.status_code}):\n{resp.text}")
        sys.exit(1)

    token_data = resp.json()
    token_data["creation_timestamp"] = datetime.now(timezone.utc).timestamp()

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
    print(f"Authentication successful.  Token saved to {token_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.broker auth")
        sys.exit(1)
    if sys.argv[1] == "auth":
        run_auth_flow()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)

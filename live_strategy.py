"""
Adaptive Vol-Managed Portfolio — Live Trading Runner
=====================================================
Strategy: P83 (Adaptive Target Volatility + Pseudo Breadth)

Compatible with the Alpaca Markets API (paper or live trading).
To use a different broker, swap out the Alpaca section at the bottom
with your broker's order-placement API.

SETUP
-----
1. Install dependencies:
       pip install alpaca-trade-api pandas numpy requests

2. Create a free Alpaca account at https://alpaca.markets/
   - Get your API Key and Secret from the dashboard
   - Use the paper trading base URL for testing

3. Set environment variables (or fill in directly below):
       ALPACA_API_KEY     = your key
       ALPACA_SECRET_KEY  = your secret
       ALPACA_BASE_URL    = https://paper-api.alpaca.markets   (paper)

Production-ready implementation of the P83 High-CAGR Champion Strategy (Sharpe 1.098, MaxDD -13.1%).
This script connects to the Alpaca Paper Trading API.

Strategy: P83 (Adaptive Target Volatility + Pseudo Breadth Regime)
Components:
    - Base Component 1: Adaptive Vol Scaling (252d MA of Vol) + RSI-4 Overlay
    - Base Component 2: ATR Turtle Breakout
    - Regime Logic: Weight shifts between Comp1/Comp2 based on cumulative Intraday EMA Breadth
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "YOUR_API_KEY_HERE")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ALPACA API CLIENT
# ─────────────────────────────────────────────────────────────
class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.ticker = "SPY"
        self.rebalance_buffer = 0.02
        self.history_days = 300
        self.headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type":        "application/json",
        }
        self.base_url   = base_url.rstrip("/")
        self.data_url   = "https://data.alpaca.markets"

    def _get(self, url: str, params: dict = None):
        r = requests.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, url: str, payload: dict):
        r = requests.post(url, headers=self.headers, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

    def _delete(self, url: str):
        r = requests.delete(url, headers=self.headers, timeout=15)
        if r.status_code != 204:
            r.raise_for_status()

    def get_account(self) -> dict:
        return self._get(f"{self.base_url}/v2/account")

    def get_portfolio_value(self) -> float:
        return float(self.get_account()["portfolio_value"])

    def get_position_fraction(self, symbol: str) -> float:
        try:
            pos = self._get(f"{self.base_url}/v2/positions/{symbol}")
            return float(pos["market_value"]) / self.get_portfolio_value()
        except:
            return 0.0

    def cancel_all_orders(self):
        self._delete(f"{self.base_url}/v2/orders")

    def submit_market_order(self, symbol: str, notional: float, side: str):
        payload = {
            "symbol": symbol, "notional": f"{abs(notional):.2f}",
            "side": side, "type": "market", "time_in_force": "day",
        }
        logger.info(f"Submitting {side.upper()} order: ${abs(notional):.2f} of {symbol}")
        return self._post(f"{self.base_url}/v2/orders", payload)

    def get_daily_bars(self, symbol: str, days: int = 300) -> pd.DataFrame:
        start = (datetime.utcnow() - timedelta(days=days + 60)).strftime("%Y-%m-%d")
        data = self._get(f"{self.data_url}/v2/stocks/{symbol}/bars", 
                         {"start": start, "timeframe": "1Day", "limit": days + 60, "feed": "iex"})
        df = pd.DataFrame(data["bars"])
        df["t"] = pd.to_datetime(df["t"])
        df.set_index("t", inplace=True)
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        return df.sort_index().tail(days)

    def calculate_signals(self, df):
        """Implements the P83 High-CAGR Champion calculation."""
        logger.info(f"Calculating P83 signals for {df.index[-1].date()}...")
        
        # 1. Component 1: Adaptive Target Volatility
        ret = df['Close'].pct_change().fillna(0)
        vol20 = ret.rolling(20).std() * np.sqrt(252)
        target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
        
        base = (target_vol / vol20).clip(upper=1.0).fillna(0)
        
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(com=3, min_periods=4).mean()
        loss = (-delta).clip(lower=0).ewm(com=3, min_periods=4).mean()
        rsi4 = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        ov = (rsi4 < 30).astype(float) * 0.2
        comp1 = (base + ov).clip(upper=1.0)
        
        # 2. Component 2: ATR Turtle Breakout
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        high20, low10 = df['High'].rolling(20).max(), df['Low'].rolling(10).min()
        size = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
        comp2 = pd.Series(0.0, index=df.index)
        
        active = False
        for i in range(20, len(df)):
            if df['High'].iloc[i] >= high20.iloc[i-1]: active = True
            elif df['Low'].iloc[i] <= low10.iloc[i-1]: active = False
            comp2.iloc[i] = size.iloc[i] if active else 0.0
                    
        # 3. P83 Pseudo Breadth Weighting
        daily_breadth = np.sign(df['Close'] - df['Open'])
        cum_breadth = daily_breadth.cumsum()
        breadth_ema = cum_breadth.ewm(span=20).mean()
        bull_breadth_active = cum_breadth.iloc[-2] > breadth_ema.iloc[-2]
        
        w2 = 0.5 if bull_breadth_active else 0.3
        w1 = 1.0 - w2
        
        target_p83 = w1 * comp1.iloc[-1] + w2 * comp2.iloc[-1]
        
        logger.info(f"  P83 Adaptive Target Vol: {target_vol.iloc[-1]:.3f}")
        logger.info(f"  P83 Bullish Breadth: {bull_breadth_active} -> Comp1: {w1:.2f}, Comp2: {w2:.2f}")
        logger.info(f"  P83 Final Target Exposure: {target_p83:.3f}")
        return target_p83

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def rebalance(client: AlpacaClient):
    logger.info("─" * 60)
    logger.info("Starting daily rebalance")

    portfolio_value = client.get_portfolio_value()
    current_fraction = client.get_position_fraction(client.ticker)
    logger.info(f"Portfolio value:  ${portfolio_value:,.2f}")
    logger.info(f"Current {client.ticker} allocation: {current_fraction:.1%}")

    df = client.get_daily_bars(client.ticker, days=client.history_days)
    target_fraction = client.calculate_signals(df)
    logger.info(f"Target {client.ticker} allocation: {target_fraction:.1%}")

    delta = target_fraction - current_fraction
    if abs(delta) < client.rebalance_buffer:
        logger.info(f"Δ={delta:.1%} < buffer ({client.rebalance_buffer:.1%}) — no trade needed")
        return

    client.cancel_all_orders()
    notional_change = delta * portfolio_value
    
    if notional_change > 0:
        client.submit_market_order(client.ticker, notional_change, side="buy")
    else:
        client.submit_market_order(client.ticker, abs(notional_change), side="sell")

    logger.info(f"Rebalance complete: {current_fraction:.1%} → {target_fraction:.1%} (${notional_change:+,.0f})")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    logger.info("High-CAGR Champion P83 — starting up")
    
    if "YOUR_API_KEY_HERE" in (ALPACA_API_KEY, ALPACA_SECRET_KEY):
        logger.error("API credentials not set. Export ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables.")
        sys.exit(1)

    client = AlpacaClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

    acct = client.get_account()
    logger.info(f"Connected to Alpaca. Account status: {acct['status']}")
    if acct["status"] != "ACTIVE":
        logger.error(f"Account is not active (status={acct['status']}). Aborting.")
        sys.exit(1)

    rebalance(client)


if __name__ == "__main__":
    main()

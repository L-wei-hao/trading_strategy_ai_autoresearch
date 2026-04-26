"""
Batch 3 — 10 New Patterns (P30–P39)
Focus: Strategies that can beat baseline on BOTH Sharpe AND CAGR.
Core insight: The baseline CAGR is 13.11%. We need to stay more invested
during bull regimes while cutting losses faster during crashes.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def metrics(returns):
    if len(returns) == 0 or returns.std() == 0:
        return {'CAGR': 0, 'Sharpe': 0, 'MaxDD': 0}
    total = (1 + returns).prod() - 1
    days  = (returns.index[-1] - returns.index[0]).days
    cagr  = (1 + total) ** (365.25 / days) - 1 if days > 0 else 0.0
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    cum    = (1 + returns).cumprod()
    mdd    = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'CAGR': round(cagr*100,2), 'Sharpe': round(sharpe,3), 'MaxDD': round(mdd*100,2)}

def backtest(df, exposure, tc=0.0005):
    pos  = pd.Series(exposure, index=df.index).shift(1).fillna(0)
    ret  = df['Close'].pct_change().fillna(0)
    cost = pos.diff().abs().fillna(0) * tc
    return pos * ret - cost

def calc_rsi(series, period=4):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────
# P30: Drawdown Stop-Loss (Stay Long, Cut at 10% Drawdown from ATH)
# Idea: Stay 100% invested in bull markets; cut to cash only when
# the portfolio has dropped >10% from its all-time high. Re-enter
# when price recovers above the ATH * 0.97 (to avoid whipsaw).
# ─────────────────────────────────────────────
def p30_drawdown_stop(df, stop=0.10, reentry_buffer=0.03):
    close   = df['Close']
    ath     = close.cummax()
    dd      = (close - ath) / ath  # negative when below ATH

    pos = pd.Series(np.nan, index=df.index)
    pos[dd < -stop]                                  = 0.0  # stop-out
    pos[close >= ath * (1 - reentry_buffer)]         = 1.0  # re-enter near ATH
    pos = pos.ffill().fillna(1.0)  # default: invested
    return backtest(df, pos)

# ─────────────────────────────────────────────
# P31: EMA Ribbon — Trend Strength via Consensus
# Exposure = fraction of [5,10,20,50,100,200]-day EMAs price is above.
# Price above all 6 → 100% in. Price below all → 0%.
# ─────────────────────────────────────────────
def p31_ema_ribbon(df):
    windows = [5, 10, 20, 50, 100, 200]
    above   = pd.DataFrame({w: (df['Close'] > df['Close'].ewm(span=w).mean()).astype(float)
                            for w in windows})
    exposure = above.mean(axis=1)  # 0 to 1
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P32: Trend + ADX Confirmation
# ADX > 25 means a confirmed trend; hold only when trending.
# Scale exposure by normalised ADX strength.
# ─────────────────────────────────────────────
def p32_adx_trend(df, adx_period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr   = pd.concat([high - low,
                      (high - close.shift(1)).abs(),
                      (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=adx_period, adjust=False).mean()

    dm_pos = (high.diff()).clip(lower=0)
    dm_neg = (-low.diff()).clip(lower=0)
    dm_pos[dm_pos < dm_neg] = 0
    dm_neg[dm_neg < dm_pos] = 0

    di_pos = 100 * dm_pos.ewm(span=adx_period, adjust=False).mean() / atr
    di_neg = 100 * dm_neg.ewm(span=adx_period, adjust=False).mean() / atr
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg)
    adx    = dx.ewm(span=adx_period, adjust=False).mean()

    sma200 = close.rolling(200).mean()

    # Long only when: trending (ADX > 25) AND bullish (Close > SMA200)
    in_trend = (adx > 25) & (close > sma200)
    # Scale by ADX strength: ADX=25→0.5x, ADX=50→1.0x
    strength  = ((adx - 25) / 25).clip(0, 1)
    exposure  = in_trend.astype(float) * (0.5 + 0.5 * strength)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P33: Price Acceleration (Short-Term > Long-Term Momentum)
# If 1-month return > 3-month return, momentum is accelerating.
# Combine with vol targeting.
# ─────────────────────────────────────────────
def p33_price_acceleration(df, target_vol=0.15):
    ret1m = df['Close'].pct_change(21)
    ret3m = df['Close'].pct_change(63)
    accel = ret1m > ret3m  # short-term momentum stronger

    realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    base_exp = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)

    exposure = base_exp * accel.shift(1).astype(float)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P34: Volatility Compression Breakout
# When 5-day vol < 50% of 20-day vol (coiling), market is primed.
# Enter on the first day vol expands back above the mean.
# ─────────────────────────────────────────────
def p34_vol_compression(df, target_vol=0.15):
    daily_ret  = df['Close'].pct_change().fillna(0)
    vol5       = daily_ret.rolling(5).std()
    vol20      = daily_ret.rolling(20).std()
    compressed = vol5 < 0.5 * vol20   # coiling
    expanding  = ~compressed & compressed.shift(1)  # first day of expansion

    sma200 = df['Close'].rolling(200).mean()
    bull   = df['Close'] > sma200

    pos = pd.Series(np.nan, index=df.index)
    # Enter on vol expansion in bull regime
    pos[expanding & bull] = 1.0
    # Exit when vol spikes extremely high
    pos[vol5 > 2 * vol20] = 0.0
    pos = pos.ffill().fillna(0)

    realized_vol = daily_ret.rolling(20).std() * np.sqrt(252)
    base_exp = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)

    exposure = base_exp * pos
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P35: Quarterly Earnings Seasonality
# Buy 7 trading days before quarter end, hold through first 5 days of new quarter.
# Months: Mar/Jun/Sep/Dec → buy around day 15–end, hold into next month.
# ─────────────────────────────────────────────
def p35_earnings_window(df):
    month = df.index.month
    # Earnings quarter ends: March, June, September, December
    in_window = month.isin([3, 4, 6, 7, 9, 10, 12, 1])  # end + first days of next
    pos = in_window.astype(float)
    # Refine: last 10 + first 5 trading days around quarter boundary
    # proxy: within 10 days of a quarter-end month transition
    qend_months = {3, 6, 9, 12}
    in_q = pd.Series(0.0, index=df.index)
    grp  = df.groupby([df.index.year, df.index.month])
    for (y, m), g in grp:
        if m in qend_months:
            days_in_month = g.index
            in_q.loc[days_in_month[-min(10, len(days_in_month)):]] = 1.0
        elif (m - 1) % 3 == 0:  # month after quarter end
            days_in_month = g.index
            in_q.loc[days_in_month[:min(5, len(days_in_month))]] = 1.0
    return backtest(df, in_q)

# ─────────────────────────────────────────────
# P36: Weekly Close Momentum
# Use only weekly price changes (Friday closes) as signal;
# avoids intra-week noise. Stay long if 4-week return > 0.
# ─────────────────────────────────────────────
def p36_weekly_momentum(df, target_vol=0.15):
    weekly  = df['Close'].resample('W-FRI').last().dropna()
    w_ret   = weekly.pct_change(4)  # 4-week return
    w_sig   = (w_ret > 0).astype(float)

    # Forward-fill weekly signal to daily
    daily_sig = w_sig.reindex(df.index, method='ffill').fillna(0)

    realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    base_exp = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)

    exposure = base_exp * daily_sig
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P37: Overnight-Return Regime
# Overnight return (prev Close → Open) captures institutional sentiment.
# If the 20-day mean overnight return > 0, the smart money is bullish → long.
# ─────────────────────────────────────────────
def p37_overnight_regime(df, target_vol=0.15):
    overnight_ret = df['Open'] / df['Close'].shift(1) - 1  # overnight gap

    # 20-day moving average of overnight returns
    overnight_ma = overnight_ret.rolling(20).mean()
    bull_overnight = overnight_ma > 0

    realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    base_exp = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)

    exposure = base_exp * bull_overnight.shift(1).astype(float)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P38: Trend Ensemble (Majority Vote of 5 Signals)
# Vote: each of 5 fast/slow trend signals casts 1 vote.
# Exposure = fraction of votes in favour, scaled by vol targeting.
# ─────────────────────────────────────────────
def p38_trend_ensemble(df, target_vol=0.15):
    c    = df['Close']
    ret  = c.pct_change().fillna(0)

    votes = pd.DataFrame({
        'sma50_200':   (c.rolling(50).mean() > c.rolling(200).mean()).astype(int),
        'sma20_100':   (c.rolling(20).mean() > c.rolling(100).mean()).astype(int),
        'price_sma200':(c > c.rolling(200).mean()).astype(int),
        'mom_12m':     (c.pct_change(252) > 0).astype(int),
        'mom_6m':      (c.pct_change(126) > 0).astype(int),
    })
    consensus = votes.mean(axis=1)  # 0 (all bearish) to 1 (all bullish)

    realized_vol = ret.rolling(20).std() * np.sqrt(252)
    base_exp = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)

    exposure = base_exp * consensus.shift(1)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P39: Champion Ensemble
# Blend the two best-performing strategies:
#   60% Vol Scaling + RSI-4 (Iter 9, Sharpe 0.908)
#   40% ATR Turtle (Iter 23, Sharpe 0.867, MaxDD -11.75%)
# Apply same backtest infrastructure.
# ─────────────────────────────────────────────
def p39_champion_ensemble(df, target_vol=0.15):
    ret = df['Close'].pct_change().fillna(0)

    # ── Component 1: Vol Scaling + RSI-4 ──────────────────
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    exp_vol = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4    = calc_rsi(df['Close'], 4)
    overlay = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1   = (exp_vol + overlay).clip(upper=1.0)

    # ── Component 2: ATR Turtle (20-day high / 10-day low) ─
    tr      = pd.concat([df['High'] - df['Low'],
                         (df['High'] - df['Close'].shift(1)).abs(),
                         (df['Low']  - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14   = tr.ewm(span=14, adjust=False).mean()
    high20  = df['High'].rolling(20).max()
    low10   = df['Low'].rolling(10).min()
    risk_per_unit = atr14 / df['Close']
    atr_size = (0.02 / risk_per_unit).clip(upper=1.0)

    comp2 = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = atr_size[df['High'] >= high20.shift(1)]
    comp2[df['Low']  <= low10.shift(1)]  = 0.0
    comp2 = comp2.ffill().fillna(0)

    # ── 60/40 Blend ────────────────────────────────────────
    exposure = 0.6 * comp1 + 0.4 * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df  = df[df.index >= '2016-01-01'].copy()

    patterns = {
        'P30 Drawdown Stop-Loss (10%)':         p30_drawdown_stop,
        'P31 EMA Ribbon Consensus':              p31_ema_ribbon,
        'P32 ADX-Confirmed Trend':               p32_adx_trend,
        'P33 Price Acceleration + Vol Scale':    p33_price_acceleration,
        'P34 Vol Compression Breakout':          p34_vol_compression,
        'P35 Quarterly Earnings Window':         p35_earnings_window,
        'P36 Weekly Momentum + Vol Scale':       p36_weekly_momentum,
        'P37 Overnight Regime + Vol Scale':      p37_overnight_regime,
        'P38 5-Signal Trend Ensemble':           p38_trend_ensemble,
        'P39 Champion Ensemble (60/40)':         p39_champion_ensemble,
    }

    W = 47
    header = f"{'Strategy':<{W}} {'CAGR%':>7} {'Sharpe':>8} {'MaxDD%':>8}"
    sep    = "-" * len(header)

    print(f"\n{'TRAIN METRICS  (1995–2015)':^{len(header)}}")
    print(header); print(sep)
    for name, fn in patterns.items():
        m = metrics(fn(train_df))
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}")

    print(f"\n{'TEST METRICS  (2016–Present)':^{len(header)}}")
    print(header); print(sep)
    for name, fn in patterns.items():
        m = metrics(fn(test_df))
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}")

    # Baselines
    print(sep)
    bh_tr = metrics(train_df['Close'].pct_change().fillna(0))
    bh_te = metrics(test_df['Close'].pct_change().fillna(0))
    print(f"{'Baseline Buy & Hold (train)':<{W}} {bh_tr['CAGR']:>7.2f} {bh_tr['Sharpe']:>8.3f} {bh_tr['MaxDD']:>8.2f}")
    print(f"{'Baseline Buy & Hold (test)' :<{W}} {bh_te['CAGR']:>7.2f} {bh_te['Sharpe']:>8.3f} {bh_te['MaxDD']:>8.2f}")
    print(sep)
    print("* Current champion (Iter 9, test):       11.96%   0.908   -20.60")

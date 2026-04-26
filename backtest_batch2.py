"""
Batch backtest for 10 new patterns.
Each strategy function returns a daily returns series (after transaction costs).
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

def metrics(returns, label=""):
    if len(returns) == 0 or returns.std() == 0:
        return {}
    total = (1 + returns).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    cagr = (1 + total) ** (365.25 / days) - 1 if days > 0 else 0.0
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    cum = (1 + returns).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'CAGR': round(cagr*100, 2), 'Sharpe': round(sharpe, 3), 'MaxDD': round(mdd*100, 2)}

def backtest(df, positions, tc=0.0005):
    """positions: daily fraction allocated (0-1). Executed next day."""
    pos = positions.shift(1).fillna(0)
    ret = df['Close'].pct_change().fillna(0)
    strat = pos * ret
    cost = pos.diff().abs().fillna(0) * tc
    return strat - cost

# ─────────────────────────────────────────────
# PATTERN 20: Z-Score Mean Reversion
# Buy when price is 2 stddev BELOW 252d mean, sell when it reverts to mean
# ─────────────────────────────────────────────
def p20_zscore_mean_reversion(df):
    mu = df['Close'].rolling(252).mean()
    sigma = df['Close'].rolling(252).std()
    zscore = (df['Close'] - mu) / sigma
    # Long when z < -2 (cheap), exit when z > 0 (reverted)
    pos = pd.Series(np.nan, index=df.index)
    pos[zscore < -2.0] = 1.0
    pos[zscore > 0.0] = 0.0
    pos = pos.ffill().fillna(0)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 21: MACD Signal Line Crossover
# ─────────────────────────────────────────────
def p21_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    pos = (macd_line > signal_line).astype(float)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 22: 52-Week High Breakout
# Buy when price closes at a new 1-year high; stay until new low (below 200d SMA)
# ─────────────────────────────────────────────
def p22_52wk_high(df):
    high252 = df['Close'].rolling(252).max()
    sma200 = df['Close'].rolling(200).mean()
    at_high = df['Close'] >= high252.shift(1)  # today's close ≥ prior 252d high
    below_sma = df['Close'] < sma200
    pos = pd.Series(np.nan, index=df.index)
    pos[at_high] = 1.0
    pos[below_sma] = 0.0
    pos = pos.ffill().fillna(0)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 23: ATR-Based Turtle Breakout
# Enter on 20-day high, exit on 10-day low; size by 1 ATR
# ─────────────────────────────────────────────
def p23_atr_turtle(df):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()

    high20 = df['High'].rolling(20).max()
    low10  = df['Low'].rolling(10).min()

    # Position size: target 2% risk-per-unit; max 1.0
    risk_per_unit = atr14 / df['Close']
    size = (0.02 / risk_per_unit).clip(upper=1.0)

    pos = pd.Series(np.nan, index=df.index)
    pos[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    pos[df['Low']  <= low10.shift(1)]  = 0.0
    pos = pos.ffill().fillna(0)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 24: Sell in May / Halloween Effect
# Out May-October, In November-April
# ─────────────────────────────────────────────
def p24_sell_in_may(df):
    month = df.index.month
    pos = pd.Series(1.0, index=df.index)
    pos[month.isin([5, 6, 7, 8, 9, 10])] = 0.0
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 25: Consecutive Down Days Reversal
# Buy after N consecutive down days; exit after M days or 1 up close
# ─────────────────────────────────────────────
def p25_consecutive_down(df, n=4, hold=5):
    ret = df['Close'].pct_change()
    down = (ret < 0).astype(int)
    # Count consecutive down days
    consec = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
    entry = consec >= n
    # Hold for 'hold' days
    pos = pd.Series(0, index=df.index)
    idx = df.index
    for i in range(len(idx) - hold):
        if entry.iloc[i]:
            pos.iloc[i+1:i+1+hold] = 1
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 26: Dual Momentum (Absolute + Relative)
# Long if both: 12m return > 0 AND price > 10m SMA
# ─────────────────────────────────────────────
def p26_dual_momentum(df, target_vol=0.15):
    ret12m = df['Close'].pct_change(252)
    sma10m = df['Close'].rolling(210).mean()
    trend  = (ret12m > 0) & (df['Close'] > sma10m)
    
    realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    exposure = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)
    
    pos = exposure * trend.shift(1).astype(float)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 27: Inside Bar (Low Volatility Compression) Breakout
# Inside bar = Today's High < Prev High AND Today's Low > Prev Low
# Wait for the next-day break above prev high → buy
# ─────────────────────────────────────────────
def p27_inside_bar(df, hold=5):
    inside = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    breakout = (df['Close'] > df['High'].shift(1)) & inside.shift(1)
    pos = pd.Series(0, index=df.index)
    idx = df.index
    for i in range(len(idx) - hold):
        if breakout.iloc[i]:
            pos.iloc[i+1:i+1+hold] = 1
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 28: Normalized Drawdown Reversion
# If current drawdown from ATH is in top-20% historically, go overweight
# ─────────────────────────────────────────────
def p28_drawdown_reversion(df, target_vol=0.15):
    daily_ret = df['Close'].pct_change().fillna(0)
    cum = (1 + daily_ret).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()  # negative
    
    # Historical 20th percentile drawdown (rolling 252d)
    threshold = drawdown.rolling(252).quantile(0.2)  # deeply negative value
    
    # More aggressive exposure when drawdown is severe (below threshold)
    realized_vol = daily_ret.rolling(20).std() * np.sqrt(252)
    exposure = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)
    
    # Boost by 30% when in deep drawdown
    boost = (drawdown.shift(1) < threshold.shift(1)).astype(float) * 0.3
    pos = (exposure + boost).clip(upper=1.0)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# PATTERN 29: Vol-Adjusted Price Momentum (VAPM)
# Rank past N-day returns by their vol-adjusted value
# Scale exposure proportional to the signal strength
# ─────────────────────────────────────────────
def p29_vapm(df, lookback=21):
    daily_ret = df['Close'].pct_change().fillna(0)
    rolling_ret = daily_ret.rolling(lookback).mean()
    rolling_vol = daily_ret.rolling(lookback).std()
    signal = (rolling_ret / rolling_vol).shift(1)  # t-stat
    
    # Normalise to [0,1] using 252d rolling rank
    signal_rank = signal.rolling(252).rank(pct=True)
    
    # Exposure: proportional to rank, scaled to max 1.0
    pos = signal_rank.fillna(0)
    return backtest(df, pos)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df  = df[df.index >= '2016-01-01'].copy()

    patterns = {
        'P20 Z-Score Mean Reversion':            p20_zscore_mean_reversion,
        'P21 MACD Crossover':                    p21_macd,
        'P22 52-Week High Breakout':             p22_52wk_high,
        'P23 ATR Turtle Breakout':               p23_atr_turtle,
        'P24 Sell in May':                       p24_sell_in_may,
        'P25 Consecutive Down Days Reversal':    p25_consecutive_down,
        'P26 Dual Momentum + Vol Scaling':       p26_dual_momentum,
        'P27 Inside Bar Breakout':               p27_inside_bar,
        'P28 Drawdown Reversion Boost':          p28_drawdown_reversion,
        'P29 Vol-Adj Price Momentum (VAPM)':     p29_vapm,
    }

    header = f"{'Strategy':<42} {'CAGR%':>7} {'Sharpe':>8} {'MaxDD%':>9}"
    sep = "-" * len(header)
    print(f"\n{'TRAIN METRICS (1995–2015)':^{len(header)}}")
    print(header); print(sep)
    for name, fn in patterns.items():
        r = fn(train_df)
        m = metrics(r)
        print(f"{name:<42} {m.get('CAGR', 0):>7.2f} {m.get('Sharpe', 0):>8.3f} {m.get('MaxDD', 0):>9.2f}")

    print(f"\n{'TEST METRICS (2016–Present)':^{len(header)}}")
    print(header); print(sep)
    for name, fn in patterns.items():
        r = fn(test_df)
        m = metrics(r)
        print(f"{name:<42} {m.get('CAGR', 0):>7.2f} {m.get('Sharpe', 0):>8.3f} {m.get('MaxDD', 0):>9.2f}")

    # Baselines
    print(sep)
    bh_train = train_df['Close'].pct_change().fillna(0)
    bh_test  = test_df['Close'].pct_change().fillna(0)
    m_tr = metrics(bh_train); m_te = metrics(bh_test)
    print(f"{'Baseline Buy & Hold (train)':<42} {m_tr['CAGR']:>7.2f} {m_tr['Sharpe']:>8.3f} {m_tr['MaxDD']:>9.2f}")
    print(f"{'Baseline Buy & Hold (test)' :<42} {m_te['CAGR']:>7.2f} {m_te['Sharpe']:>8.3f} {m_te['MaxDD']:>9.2f}")

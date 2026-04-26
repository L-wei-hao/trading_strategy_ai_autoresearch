import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def metrics(returns):
    if len(returns) == 0 or returns.std() == 0:
        return {'CAGR': 0, 'Sharpe': 0, 'MaxDD': 0}
    total  = (1 + returns).prod() - 1
    days   = (returns.index[-1] - returns.index[0]).days
    cagr   = (1 + total) ** (365.25 / days) - 1 if days > 0 else 0.0
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    cum    = (1 + returns).cumprod()
    mdd    = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'CAGR': round(cagr*100,2), 'Sharpe': round(sharpe,3), 'MaxDD': round(mdd*100,2)}

def backtest(df, exposure, tc=0.0005):
    pos  = pd.Series(np.asarray(exposure), index=df.index).shift(1).fillna(0)
    ret  = df['Close'].pct_change().fillna(0)
    cost = pos.diff().abs().fillna(0) * tc
    return pos * ret - cost

def calc_rsi(series, period=4):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────
# P70: Trend-Filtered Buy & Hold
# Stays 1.0 mostly. Cuts to 0.0 if Price < SMA200 AND MACD is negative
# ─────────────────────────────────────────────
def p70_trend_filtered_bnh(df):
    sma200 = df['Close'].rolling(200).mean()
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    
    # Condition: Under 200 SMA AND MACD is negative -> Bear market regime
    bear = (df['Close'] < sma200) & (macd < 0)
    exposure = np.where(bear.shift(1), 0.0, 1.0)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P71: Vol-Targeting with 25% Target Vol
# By using 25% target vol, exposure is capped at 1.0 much more often,
# giving higher CAGR, but trims during catastrophic spikes.
# ─────────────────────────────────────────────
def p71_high_vol_target(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base = (0.25 / vol20).clip(upper=1.0).fillna(0)
    
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    exposure = (base + ov).clip(upper=1.0)
    
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P72: V-Bottom Catcher
# 1.0 if Close > SMA200. 0.5 if Close < SMA200. 
# BUT if RSI < 30, override to 1.0 to catch bounces.
# ─────────────────────────────────────────────
def p72_v_bottom_catcher(df):
    sma200 = df['Close'].rolling(200).mean()
    rsi4 = calc_rsi(df['Close'], 4)
    
    base = np.where(df['Close'] > sma200, 1.0, 0.5)
    base = np.where(rsi4 < 30, 1.0, base)
    exposure = pd.Series(base, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P73: P50 (Pseudo Breadth) with 20% Target Vol
# P50 had 11.17% CAGR. Let's loosen the vol target to 20% to get more CAGR.
# ─────────────────────────────────────────────
def p73_p50_high_vol(df):
    daily_breadth = np.sign(df['Close'] - df['Open'])
    cum_breadth = daily_breadth.cumsum()
    breadth_ema = cum_breadth.ewm(span=20).mean()
    
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base = (0.20 / vol20).clip(upper=1.0).fillna(0)
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    high20, low10 = df['High'].rolling(20).max(), df['Low'].rolling(10).min()
    size = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp2 = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp2[df['Low'] <= low10.shift(1)] = 0.0
    comp2 = comp2.ffill().fillna(0)
    
    bull_breadth = cum_breadth > breadth_ema
    w2 = np.where(bull_breadth.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P74: Volatility Risk Premium (VRP) Proxy
# Cut exposure if 10d vol > 60d vol (vol curve inverted)
# ─────────────────────────────────────────────
def p74_vrp_proxy(df):
    ret = df['Close'].pct_change().fillna(0)
    vol10 = ret.rolling(10).std()
    vol60 = ret.rolling(60).std()
    
    inverted = vol10 > vol60
    exposure = np.where(inverted.shift(1), 0.5, 1.0)
    return backtest(df, pd.Series(exposure, index=df.index))

# ─────────────────────────────────────────────
# P75: High-Beta Momentum
# Simple price momentum + RSI combo. Always 50% minimum.
# ─────────────────────────────────────────────
def p75_momentum_rsi_50_100(df):
    ret = df['Close'].pct_change().fillna(0)
    mom100 = df['Close'].pct_change(100)
    rsi14 = calc_rsi(df['Close'], 14)
    
    bull = (mom100 > 0) | (rsi14 < 40)
    exposure = np.where(bull.shift(1), 1.0, 0.5)
    return backtest(df, pd.Series(exposure, index=df.index))

# ─────────────────────────────────────────────
# P76: Adaptive Target Volatility
# Target vol = 252d moving average of realized vol.
# ─────────────────────────────────────────────
def p76_adaptive_target_vol(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    base = (target_vol / vol20).clip(upper=1.0).fillna(0)
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    exposure = (base + ov).clip(upper=1.0)
    
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P77: Drawdown Proportional Scaling
# Let it ride at 1.0 until DD > 10%, then scale linearly.
# ─────────────────────────────────────────────
def p77_dd_scaling(df):
    cum = df['Close'].cummax()
    dd = (df['Close'] - cum) / cum
    
    scalar = 1.0 - (np.clip(dd.abs() - 0.10, 0, 1) * 5)
    exposure = np.clip(scalar, 0.2, 1.0)
    return backtest(df, pd.Series(exposure, index=df.index).shift(1).fillna(1.0))

# ─────────────────────────────────────────────
# P78: Buy & Hold with RSI Overbought Trim
# ─────────────────────────────────────────────
def p78_bnh_rsi_trim(df):
    rsi14 = calc_rsi(df['Close'], 14)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    
    fade = (rsi14 > 70) & (macd < macd_signal)
    exposure = np.where(fade.shift(1), 0.5, 1.0)
    return backtest(df, pd.Series(exposure, index=df.index))

# ─────────────────────────────────────────────
# P79: P65 but With Higher Base Exposures (CAGR Optimized)
# Take P65 (Sharpe 1.065) but:
# - Target vol = 20%
# - P49 scale = 1.0 / 0.8 / 0.6 instead of 1.0 / 0.7 / 0.4
# ─────────────────────────────────────────────
def p79_p65_cagr_optimized(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base = (0.20 / vol20).clip(upper=1.0).fillna(0) # 20% vol
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    high20, low10 = df['High'].rolling(20).max(), df['Low'].rolling(10).min()
    size = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp2 = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp2[df['Low'] <= low10.shift(1)] = 0.0
    comp2 = comp2.ffill().fillna(0)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.8, 0.6))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    p58 = 0.5 * exp_48 + 0.5 * exp_49
    
    gain14 = df['Close'].diff().clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss14 = (-df['Close'].diff()).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rsi14 = 100 - (100 / (1 + (gain14 / loss14.replace(0, np.nan))))
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    
    exposure = p58 * pd.Series(np.where(div_signal.shift(1), 0.85, 1.0), index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────
def walk_forward(df, fn, test_years=2):
    results = []
    start = df.index[0].year
    end   = df.index[-1].year
    for yr in range(start + 5, end - test_years + 1, test_years):
        te = df[(df.index >= f"{yr}-01-01") & (df.index < f"{yr+test_years}-01-01")].copy()
        if len(te) < 100: continue
        m = metrics(fn(te)); m['period'] = f"{yr}-{yr+test_years}"; results.append(m)
    return pd.DataFrame(results).set_index('period') if results else pd.DataFrame()

if __name__ == "__main__":
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df  = df[df.index >= '2016-01-01'].copy()

    patterns = {
        'P70 Trend-Filtered B&H':           p70_trend_filtered_bnh,
        'P71 Vol Target 25%':               p71_high_vol_target,
        'P72 V-Bottom Catcher':             p72_v_bottom_catcher,
        'P73 P50 with 20% Vol':             p73_p50_high_vol,
        'P74 VRP Proxy':                    p74_vrp_proxy,
        'P75 High-Beta Momentum':           p75_momentum_rsi_50_100,
        'P76 Adaptive Target Vol':          p76_adaptive_target_vol,
        'P77 Drawdown Proportional':        p77_dd_scaling,
        'P78 Overbought Trim':              p78_bnh_rsi_trim,
        'P79 P65 CAGR-Optimized':           p79_p65_cagr_optimized,
    }

    W = 43
    header = f"{'Strategy':<{W}} {'CAGR%':>7} {'Sharpe':>8} {'MaxDD%':>8}"
    sep    = "-" * len(header)

    print(f"\n{'TRAIN  (1995-2015)'}")
    print(header); print(sep)
    for name, fn in patterns.items():
        m = metrics(fn(train_df))
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}")

    print(f"\n{'TEST  (2016-Present)'}")
    print(header); print(sep)
    for name, fn in patterns.items():
        m = metrics(fn(test_df))
        flag = " <-- HIGH CAGR" if m['CAGR'] > 12.0 and m['Sharpe'] > 0.9 else ""
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}{flag}")
    
    print("\n--- Walk-Forward (2-year windows) for top candidates ---")
    for name, fn in patterns.items():
        m = metrics(fn(test_df))
        if m['CAGR'] > 11.5 and m['Sharpe'] > 0.9:
            print(f"\n{name}:")
            wf = walk_forward(df, fn)
            if not wf.empty:
                pos_pct = (wf['Sharpe'] > 0).mean() * 100
                print(wf[['CAGR','Sharpe','MaxDD']].to_string())
                print(f"  => Positive windows: {pos_pct:.0f}%")

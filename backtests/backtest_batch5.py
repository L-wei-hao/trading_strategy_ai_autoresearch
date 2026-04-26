"""
Batch 5 — 10 New Patterns (P50–P59)
Goal: Further refine the P48/P49 concepts or find entirely new orthogonal edges.
Target: Sharpe > 1.000, MaxDD < 15.1% (P48), or MaxDD < 9.2% (P49).
"""
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

def atr_turtle_exposure(df):
    tr    = pd.concat([df['High'] - df['Low'],
                       (df['High'] - df['Close'].shift(1)).abs(),
                       (df['Low']  - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    high20 = df['High'].rolling(20).max()
    low10  = df['Low'].rolling(10).min()
    size   = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp   = pd.Series(np.nan, index=df.index)
    comp[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp[df['Low']  <= low10.shift(1)]  = 0.0
    return comp.ffill().fillna(0)

def vol_rsi_exposure(df, target_vol=0.15):
    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4 < 30).astype(float) * 0.2
    return (base + ov).clip(upper=1.0)

# ─────────────────────────────────────────────
# P50: Cross-Sectional Breadth Proxy (Adv/Dec Line Substitute)
# Using S&P 500 OHLCV only, create a pseudo-AD line by summing
# daily sign of Close-Open. If moving average of this is rising,
# trend is broad-based.
# ─────────────────────────────────────────────
def p50_pseudo_breadth(df):
    # Proxy for AD line: +1 if close > open, -1 if close < open
    daily_breadth = np.sign(df['Close'] - df['Open'])
    cum_breadth = daily_breadth.cumsum()
    breadth_ema = cum_breadth.ewm(span=20).mean()
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    
    # Weight P48 ensemble based on breadth
    bull_breadth = cum_breadth > breadth_ema
    w2 = np.where(bull_breadth.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P51: Consecutive Up/Down Days Regime
# Simple streak counting. If there are more down streaks
# than up streaks in the last 20 days, market is struggling.
# ─────────────────────────────────────────────
def p51_streak_regime(df):
    ret = df['Close'].pct_change().fillna(0)
    up_day = (ret > 0).astype(int)
    down_day = (ret < 0).astype(int)
    
    up_streaks = up_day.rolling(20).sum()
    down_streaks = down_day.rolling(20).sum()
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    
    bull_streak = up_streaks > down_streaks
    w2 = np.where(bull_streak.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P52: VIX-Proxy (Bollinger Band Width)
# BB Width = (Upper - Lower) / Middle
# Expanding width means fear/uncertainty (like VIX rising).
# ─────────────────────────────────────────────
def p52_bb_width_vix(df):
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / sma20
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    
    # If BB width is expanding (current > 20d moving avg), reduce exposure
    bbw_sma = bb_width.rolling(20).mean()
    expanding = bb_width > bbw_sma
    
    base_ensemble = 0.6 * comp1 + 0.4 * comp2
    exposure = base_ensemble * np.where(expanding.shift(1), 0.8, 1.0)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P53: Overnight vs Intraday Divergence
# If overnight is consistently positive but intraday is negative,
# retail is selling into institutional gaps (distribution).
# ─────────────────────────────────────────────
def p53_overnight_intraday_divergence(df):
    on_ret = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    in_ret = (df['Close'] - df['Open']) / df['Open']
    
    on_ema = on_ret.ewm(span=20).mean()
    in_ema = in_ret.ewm(span=20).mean()
    
    distribution = (on_ema > 0) & (in_ema < 0)
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    ensemble = 0.6 * comp1 + 0.4 * comp2
    
    exposure = ensemble * np.where(distribution.shift(1), 0.5, 1.0)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P54: Moving Average Distance Sizing
# The further price is above the 200 SMA, the more vulnerable it is
# to a reversion. Scale down slightly as distance increases.
# ─────────────────────────────────────────────
def p54_ma_distance_sizing(df):
    sma200 = df['Close'].rolling(200).mean()
    dist = (df['Close'] - sma200) / sma200
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    ensemble = 0.6 * comp1 + 0.4 * comp2
    
    # Base 1.0. Reduce linearly if dist > 10%, down to 0.7 at 25% dist
    scalar = 1.0 - np.clip((dist - 0.10) / 0.15, 0, 1) * 0.3
    
    exposure = ensemble * scalar.shift(1).fillna(1.0)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P55: Volume Confirmation Trend
# Only increase ATR Turtle weight if the 20-day breakout is accompanied
# by volume greater than the 20-day moving average.
# ─────────────────────────────────────────────
def p55_volume_confirmation(df):
    vol_sma20 = df['Volume'].rolling(20).mean()
    high_vol = df['Volume'] > vol_sma20
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    
    # We only have volume from yesterday to make the decision today
    w2 = np.where(high_vol.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P56: The "Anti-Whipsaw" Choppiness Index Filter
# Choppiness Index (CHOP) measures if market is trending or consolidating.
# High CHOP -> ranging -> reduce breakout weight
# ─────────────────────────────────────────────
def p56_choppiness_filter(df, period=14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    sum_tr = tr.rolling(period).sum()
    high_n = df['High'].rolling(period).max()
    low_n = df['Low'].rolling(period).min()
    
    # Choppiness Index 100 * log10( Sum(TR, n) / (High(n) - Low(n)) ) / log10(n)
    chop = 100 * np.log10(sum_tr / (high_n - low_n)) / np.log10(period)
    
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    
    # High chop (>61.8) -> consolidating -> rely on vol+RSI
    w2 = np.where(chop.shift(1) > 61.8, 0.2, 0.4)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P57: The P48 Intraday Regime + ATR Exit
# Take the P48 Champion but use a faster exit for the ATR Turtle
# component (e.g., 5-day low instead of 10-day low) to cut losses quicker.
# ─────────────────────────────────────────────
def p57_p48_fast_exit(df, target_vol=0.15):
    # Comp1
    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4 < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    
    # Fast Comp2
    tr    = pd.concat([df['High'] - df['Low'],
                       (df['High'] - df['Close'].shift(1)).abs(),
                       (df['Low']  - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    high20 = df['High'].rolling(20).max()
    low5   = df['Low'].rolling(5).min() # Fast exit
    size   = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp2  = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp2[df['Low']  <= low5.shift(1)]  = 0.0
    comp2 = comp2.ffill().fillna(0)
    
    # P48 Intraday regime
    intraday = (df['Close'] - df['Open']) / df['Open']
    intra_ema = intraday.ewm(span=20).mean()
    bull_intra = intra_ema > 0
    
    w2 = np.where(bull_intra.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1.shift(1) + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure) # comp1 already shifted inside this function? Actually need to be careful with shifts here.
    
def p57_p48_fast_exit_fixed(df, target_vol=0.15):
    # Base exposures calculated using current day's data, shifted by backtest()
    comp1 = vol_rsi_exposure(df)
    
    tr    = pd.concat([df['High'] - df['Low'],
                       (df['High'] - df['Close'].shift(1)).abs(),
                       (df['Low']  - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    high20 = df['High'].rolling(20).max()
    low5   = df['Low'].rolling(5).min() # Fast exit
    size   = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp2  = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp2[df['Low']  <= low5.shift(1)]  = 0.0
    comp2 = comp2.ffill().fillna(0)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    intra_ema = intraday.ewm(span=20).mean()
    bull_intra = intra_ema > 0
    
    w2 = np.where(bull_intra.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P58: P48 + Triple Confirmation Blend
# Blend the Sharpe Champion (P48) with the Drawdown Champion (P49)
# ─────────────────────────────────────────────
def p58_champion_blend(df):
    # P48 logic
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    # P49 logic
    ensemble = 0.6 * comp1 + 0.4 * comp2
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.7, 0.4))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    exposure = 0.5 * exp_48 + 0.5 * exp_49
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P59: "The Ultimate Edge"
# Take P48 Intraday Regime, but replace the simple 20-day Vol Scaling
# with the Max(20, 60, 120) multi-scale vol targeting from Iter 11
# This should drastically reduce drawdowns while keeping P48's Sharpe
# ─────────────────────────────────────────────
def p59_multi_scale_p48(df, target_vol=0.15):
    ret = df['Close'].pct_change().fillna(0)
    v20 = ret.rolling(20).std() * np.sqrt(252)
    v60 = ret.rolling(60).std() * np.sqrt(252)
    v120 = ret.rolling(120).std() * np.sqrt(252)
    max_vol = pd.concat([v20, v60, v120], axis=1).max(axis=1)
    
    base = (target_vol / max_vol).clip(upper=1.0).fillna(0)
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    
    comp2 = atr_turtle_exposure(df)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2 = np.where(bull_intra.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
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

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df       = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df  = df[df.index >= '2016-01-01'].copy()

    patterns = {
        'P50 Pseudo Breadth':               p50_pseudo_breadth,
        'P51 Streak Regime':                p51_streak_regime,
        'P52 BB Width VIX Proxy':           p52_bb_width_vix,
        'P53 Overnight/Intraday Div':       p53_overnight_intraday_divergence,
        'P54 MA Distance Sizing':           p54_ma_distance_sizing,
        'P55 Volume Confirmation':          p55_volume_confirmation,
        'P56 Choppiness Filter':            p56_choppiness_filter,
        'P57 P48 Fast ATR Exit':            p57_p48_fast_exit_fixed,
        'P58 Champion Blend (P48+P49)':     p58_champion_blend,
        'P59 Multi-Scale P48':              p59_multi_scale_p48,
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
        flag = " <-- BEATS P48 SHARPE (1.000) OR P49 DD (-9.20)" if m['Sharpe'] > 1.000 or m['MaxDD'] > -9.20 else ""
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}{flag}")
    print(sep)
    print(f"  Current Sharpe Champ P48 (test):      10.82%   1.000   -15.07")
    print(f"  Current DD Champ P49 (test):           7.65%   0.990    -9.20")

    # Walk-forward top candidates
    print("\n--- Walk-Forward (2-year windows) for top candidates ---")
    for name, fn in patterns.items():
        m = metrics(fn(test_df))
        if m['Sharpe'] >= 0.98 or m['MaxDD'] > -11.0:
            print(f"\n{name}:")
            wf = walk_forward(df, fn)
            if not wf.empty:
                pos_pct = (wf['Sharpe'] > 0).mean() * 100
                print(wf[['CAGR','Sharpe','MaxDD']].to_string())
                print(f"  => Positive windows: {pos_pct:.0f}%")

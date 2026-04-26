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

def get_comp1(df, target_vol=0.20):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    
    if isinstance(target_vol, pd.Series):
        base = (target_vol / vol20).clip(upper=1.0).fillna(0)
    else:
        base = (target_vol / vol20).clip(upper=1.0).fillna(0)
        
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    return (base + ov).clip(upper=1.0)

def get_comp2(df):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    high20, low10 = df['High'].rolling(20).max(), df['Low'].rolling(10).min()
    size = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp2 = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp2[df['Low'] <= low10.shift(1)] = 0.0
    return comp2.ffill().fillna(0)

# ─────────────────────────────────────────────
# P80: P79 with 25% target vol (Max CAGR attempt on P65 structure)
# ─────────────────────────────────────────────
def p80_p65_max_vol(df):
    comp1 = get_comp1(df, target_vol=0.25)
    comp2 = get_comp2(df)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    vol20 = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.8, 0.6))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    p58 = 0.5 * exp_48 + 0.5 * exp_49
    
    rsi14 = calc_rsi(df['Close'], 14)
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    
    exposure = p58 * pd.Series(np.where(div_signal.shift(1), 0.85, 1.0), index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P81: Adaptive Target Vol (P76) + Triple Confirm (P49)
# ─────────────────────────────────────────────
def p81_adaptive_triple_confirm(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    comp1 = get_comp1(df, target_vol=target_vol)
    comp2 = get_comp2(df)
    
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.7, 0.4))
    exposure = ensemble * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P82: P65 but Target Vol = Adaptive
# ─────────────────────────────────────────────
def p82_p65_adaptive_vol(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    comp1 = get_comp1(df, target_vol=target_vol)
    comp2 = get_comp2(df)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.8, 0.5))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    p58 = 0.5 * exp_48 + 0.5 * exp_49
    
    rsi14 = calc_rsi(df['Close'], 14)
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    exposure = p58 * pd.Series(np.where(div_signal.shift(1), 0.80, 1.0), index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P83: P50 (Pseudo Breadth) + Adaptive Target Vol
# ─────────────────────────────────────────────
def p83_p50_adaptive(df):
    daily_breadth = np.sign(df['Close'] - df['Open'])
    cum_breadth = daily_breadth.cumsum()
    breadth_ema = cum_breadth.ewm(span=20).mean()
    
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    comp1 = get_comp1(df, target_vol=target_vol)
    comp2 = get_comp2(df)
    
    bull_breadth = cum_breadth > breadth_ema
    w2 = np.where(bull_breadth.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P84: "The 90/10 Baseline"
# Always hold 90%. Use remaining 10% for RSI mean reversion.
# ─────────────────────────────────────────────
def p84_90_10_baseline(df):
    rsi4 = calc_rsi(df['Close'], 4)
    exposure = np.where(rsi4.shift(1) < 30, 1.0, 0.90)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P85: P83 (P50 Adaptive) + RSI Divergence
# ─────────────────────────────────────────────
def p85_p50_adaptive_rsi_div(df):
    daily_breadth = np.sign(df['Close'] - df['Open'])
    cum_breadth = daily_breadth.cumsum()
    breadth_ema = cum_breadth.ewm(span=20).mean()
    
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    comp1 = get_comp1(df, target_vol=target_vol)
    comp2 = get_comp2(df)
    
    bull_breadth = cum_breadth > breadth_ema
    w2 = np.where(bull_breadth.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    p50_exp = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    
    rsi14 = calc_rsi(df['Close'], 14)
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    exposure = p50_exp * pd.Series(np.where(div_signal.shift(1), 0.80, 1.0), index=df.index)
    
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P86: Pure Breadth (No Vol Targeting)
# ─────────────────────────────────────────────
def p86_pure_breadth(df):
    daily_breadth = np.sign(df['Close'] - df['Open'])
    cum_breadth = daily_breadth.cumsum()
    breadth_ema = cum_breadth.ewm(span=20).mean()
    
    bull = cum_breadth > breadth_ema
    exposure = np.where(bull.shift(1), 1.0, 0.5)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P87: Comp1 only (Vol Target = 20%)
# ─────────────────────────────────────────────
def p87_comp1_20(df):
    exposure = get_comp1(df, target_vol=0.20)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P88: P82 but trim is 10% instead of 20%
# ─────────────────────────────────────────────
def p88_p82_light_trim(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    comp1 = get_comp1(df, target_vol=target_vol)
    comp2 = get_comp2(df)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.8, 0.5))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    p58 = 0.5 * exp_48 + 0.5 * exp_49
    
    rsi14 = calc_rsi(df['Close'], 14)
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    exposure = p58 * pd.Series(np.where(div_signal.shift(1), 0.90, 1.0), index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P89: P82 but Floor Base Vol Scaling at 0.5
# ─────────────────────────────────────────────
def p89_p82_floored_vol(df):
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    target_vol = vol20.rolling(252).mean().clip(lower=0.10, upper=0.25).fillna(0.15)
    
    base = (target_vol / vol20).clip(lower=0.5, upper=1.0).fillna(0.5)
    rsi4 = calc_rsi(df['Close'], 4)
    ov = (rsi4 < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    
    comp2 = get_comp2(df)
    
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.8, 0.5))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    p58 = 0.5 * exp_48 + 0.5 * exp_49
    
    rsi14 = calc_rsi(df['Close'], 14)
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    exposure = p58 * pd.Series(np.where(div_signal.shift(1), 0.80, 1.0), index=df.index)
    return backtest(df, exposure)


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
        'P80 P65 with 25% Vol':             p80_p65_max_vol,
        'P81 Adaptive Target + P49':        p81_adaptive_triple_confirm,
        'P82 P65 with Adaptive Vol':        p82_p65_adaptive_vol,
        'P83 P50 with Adaptive Vol':        p83_p50_adaptive,
        'P84 90/10 Baseline':               p84_90_10_baseline,
        'P85 P83 + RSI Div':                p85_p50_adaptive_rsi_div,
        'P86 Pure Breadth Filter':          p86_pure_breadth,
        'P87 Comp1 Only (20% Vol)':         p87_comp1_20,
        'P88 P82 Light Trim (10%)':         p88_p82_light_trim,
        'P89 P82 Floored Vol Scaling':      p89_p82_floored_vol,
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

"""
Batch 6 — 10 Final Refinements (P60–P69)
Goal: The final test. Can we improve P58 (Sharpe 1.047, MaxDD 11.0%)?
Focus: Macro correlations, yield curve, cross-asset momentum, 
and advanced ensemble weighting schemes.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Fetch external macro data if not present
def fetch_macro_data():
    if not os.path.exists("TLT_history.csv"):
        print("Fetching TLT (Treasuries) data...")
        tlt = yf.download("TLT", start="2000-01-01", progress=False)
        tlt.to_csv("TLT_history.csv")
    if not os.path.exists("UUP_history.csv"):
        print("Fetching UUP (Dollar) data...")
        uup = yf.download("UUP", start="2000-01-01", progress=False)
        uup.to_csv("UUP_history.csv")
    
    # Load them
    tlt_df = pd.read_csv("TLT_history.csv", header=[0, 1]) if os.path.exists("TLT_history.csv") else None
    if tlt_df is not None:
        if isinstance(tlt_df.columns, pd.MultiIndex):
            tlt_df = pd.read_csv("TLT_history.csv", skiprows=2, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
            tlt_df = tlt_df.iloc[1:] # drop extra header row
        tlt_df['Date'] = pd.to_datetime(tlt_df['Date'])
        tlt_df.set_index('Date', inplace=True)
        tlt_df['Close'] = pd.to_numeric(tlt_df['Close'], errors='coerce')
        
    uup_df = pd.read_csv("UUP_history.csv", header=[0, 1]) if os.path.exists("UUP_history.csv") else None
    if uup_df is not None:
         if isinstance(uup_df.columns, pd.MultiIndex):
            uup_df = pd.read_csv("UUP_history.csv", skiprows=2, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
            uup_df = uup_df.iloc[1:]
         uup_df['Date'] = pd.to_datetime(uup_df['Date'])
         uup_df.set_index('Date', inplace=True)
         uup_df['Close'] = pd.to_numeric(uup_df['Close'], errors='coerce')
         
    return tlt_df, uup_df

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

# --- Base Components ---
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

def get_p58_exposure(df):
    comp1 = vol_rsi_exposure(df)
    comp2 = atr_turtle_exposure(df)
    
    # P48 intraday
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    # P49 triple
    ensemble = 0.6 * comp1 + 0.4 * comp2
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.7, 0.4))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    return 0.5 * exp_48 + 0.5 * exp_49

# ─────────────────────────────────────────────
# P60: Bond-Equity Correlation Regime
# If SPY and TLT correlation is positive, bonds aren't protecting
# against equity crashes (inflationary regime). Reduce exposure.
# ─────────────────────────────────────────────
def p60_bond_equity_corr(df, tlt_df):
    p58_exp = get_p58_exposure(df)
    if tlt_df is None or len(tlt_df) == 0:
        return backtest(df, p58_exp)
        
    ret_spy = df['Close'].pct_change()
    
    # Align TLT
    tlt_aligned = tlt_df['Close'].reindex(df.index).ffill()
    ret_tlt = tlt_aligned.pct_change()
    
    # 60-day rolling correlation
    corr = ret_spy.rolling(60).corr(ret_tlt)
    
    # If corr > 0.2 (bonds and stocks falling together), trim 20%
    scalar = np.where(corr.shift(1) > 0.2, 0.8, 1.0)
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P61: US Dollar Trend Filter
# A strongly rising dollar (UUP) often precedes equity weakness.
# ─────────────────────────────────────────────
def p61_dollar_trend_filter(df, uup_df):
    p58_exp = get_p58_exposure(df)
    if uup_df is None or len(uup_df) == 0:
        return backtest(df, p58_exp)
        
    uup_aligned = uup_df['Close'].reindex(df.index).ffill()
    uup_sma50 = uup_aligned.rolling(50).mean()
    
    # Dollar in uptrend -> trim equity exposure 10%
    strong_dollar = uup_aligned > uup_sma50
    scalar = np.where(strong_dollar.shift(1), 0.9, 1.0)
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P62: Volatility of Volatility (VVIX Proxy)
# Use std dev of rolling 20d vol to detect structural regime breaks.
# ─────────────────────────────────────────────
def p62_vol_of_vol(df):
    p58_exp = get_p58_exposure(df)
    
    ret = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    vvol20 = vol20.rolling(20).std()
    
    # If Vvol spikes > 90th percentile, structural break imminent
    vvol_spike = vvol20 > vvol20.rolling(252).quantile(0.9)
    
    scalar = np.where(vvol_spike.shift(1), 0.7, 1.0)
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P63: The Skewness Filter
# Negative daily return skewness implies "stairs up, elevator down".
# ─────────────────────────────────────────────
def p63_skewness_filter(df):
    p58_exp = get_p58_exposure(df)
    
    ret = df['Close'].pct_change().fillna(0)
    skew60 = ret.rolling(60).skew()
    
    # Highly negative skew -> limit upside, high downside risk -> trim
    neg_skew = skew60 < -1.0
    
    scalar = np.where(neg_skew.shift(1), 0.8, 1.0)
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P64: Multi-Scale P58 (Combines P58 with P59)
# Use the Max(20,60,120) vol targeting from P59 *inside* P58.
# ─────────────────────────────────────────────
def p64_p58_multiscale_vol(df, target_vol=0.15):
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
    
    # P48
    intraday = (df['Close'] - df['Open']) / df['Open']
    bull_intra = intraday.ewm(span=20).mean() > 0
    w2_48 = np.where(bull_intra.shift(1), 0.5, 0.3)
    exp_48 = pd.Series(1 - w2_48, index=df.index) * comp1 + pd.Series(w2_48, index=df.index) * comp2
    
    # P49
    ensemble = 0.6 * comp1 + 0.4 * comp2
    s1 = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    s2 = (comp2 > 0).astype(int)
    s3 = (v20 < v20.rolling(252).quantile(0.7)).astype(int)
    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.7, 0.4))
    exp_49 = ensemble * pd.Series(scalar, index=df.index)
    
    exposure = 0.5 * exp_48 + 0.5 * exp_49
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P65: RSI Divergence
# Price makes higher high, but RSI makes lower high.
# ─────────────────────────────────────────────
def p65_rsi_divergence(df):
    p58_exp = get_p58_exposure(df)
    
    rsi14 = calc_rsi(df['Close'], 14)
    
    price_high = df['Close'].rolling(20).max() == df['Close']
    rsi_high = rsi14.rolling(20).max() == rsi14
    
    # Price high but RSI not high
    div = price_high & (~rsi_high)
    div_signal = div.rolling(5).max() > 0
    
    scalar = np.where(div_signal.shift(1), 0.8, 1.0)
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P66: The Gamma Squeeze Filter
# When 5-day return > 5% but Vol is expanding rapidly.
# ─────────────────────────────────────────────
def p66_gamma_squeeze(df):
    p58_exp = get_p58_exposure(df)
    
    ret5 = df['Close'].pct_change(5)
    ret = df['Close'].pct_change().fillna(0)
    vol5 = ret.rolling(5).std()
    vol20 = ret.rolling(20).std()
    
    # Sharp up move + expanding short-term vol (often a blow-off top)
    blow_off = (ret5 > 0.05) & (vol5 > vol20 * 1.5)
    
    scalar = np.where(blow_off.shift(1), 0.5, 1.0) # Cut exposure in half
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P67: P58 + Dynamic Rebalancing Threshold
# Introduce a minimum weight change threshold to P58 to act as a 
# low-pass filter on trades.
# ─────────────────────────────────────────────
def p67_p58_rebalance_buffer(df, buffer=0.05):
    raw_exp = get_p58_exposure(df).fillna(0)
    clean_exp = np.zeros(len(raw_exp))
    curr = 0.0
    
    for i in range(len(raw_exp)):
        target = raw_exp.iloc[i]
        if abs(target - curr) > buffer:
            curr = target
        clean_exp[i] = curr
        
    exposure = pd.Series(clean_exp, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P68: Downside Beta Hedge
# If 20-day returns have higher downside beta than upside beta,
# market microstructure is weak.
# ─────────────────────────────────────────────
def p68_downside_beta(df):
    p58_exp = get_p58_exposure(df)
    
    ret = df['Close'].pct_change().fillna(0)
    up_days = ret.clip(lower=0)
    down_days = ret.clip(upper=0).abs()
    
    up_vol = up_days.rolling(20).std()
    down_vol = down_days.rolling(20).std()
    
    # Market falling faster than it rises
    weak = down_vol > up_vol * 1.5
    
    scalar = np.where(weak.shift(1), 0.7, 1.0)
    exposure = p58_exp * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P69: Kelly Criterion Sizing on P58
# Scale P58 base exposure by rolling Kelly fraction.
# ─────────────────────────────────────────────
def p69_kelly_sizing(df):
    raw_exp = get_p58_exposure(df)
    ret = df['Close'].pct_change().fillna(0)
    strat_ret = raw_exp.shift(1) * ret
    
    # 252d rolling win rate & win/loss ratio
    wins = (strat_ret > 0).astype(int).rolling(252).mean()
    avg_win = strat_ret[strat_ret > 0].rolling(252).mean().reindex(df.index).ffill()
    avg_loss = strat_ret[strat_ret < 0].abs().rolling(252).mean().reindex(df.index).ffill()
    
    w_l_ratio = (avg_win / avg_loss).fillna(1.0)
    
    # Kelly = W - [ (1 - W) / R ]
    kelly = wins - ((1 - wins) / w_l_ratio.replace(0, np.nan))
    kelly = kelly.fillna(0.5).clip(0, 1.0) # Half-Kelly effectively max 1.0
    
    # Use kelly to scale down if edge drops
    scalar = kelly.shift(1).fillna(1.0)
    exposure = raw_exp * scalar
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────
def walk_forward(df, fn, test_years=2, *args):
    results = []
    start = df.index[0].year
    end   = df.index[-1].year
    for yr in range(start + 5, end - test_years + 1, test_years):
        te = df[(df.index >= f"{yr}-01-01") & (df.index < f"{yr+test_years}-01-01")].copy()
        if len(te) < 100: continue
        m = metrics(fn(te, *args)); m['period'] = f"{yr}-{yr+test_years}"; results.append(m)
    return pd.DataFrame(results).set_index('period') if results else pd.DataFrame()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df       = load_data("SP500_history_2000_to_current.csv")
    tlt_df, uup_df = fetch_macro_data()
    
    train_df = df[df.index < '2016-01-01'].copy()
    test_df  = df[df.index >= '2016-01-01'].copy()

    patterns = {
        'P60 Bond-Equity Corr':             (p60_bond_equity_corr, [tlt_df]),
        'P61 US Dollar Trend':              (p61_dollar_trend_filter, [uup_df]),
        'P62 Vol-of-Vol Break':             (p62_vol_of_vol, []),
        'P63 Skewness Filter':              (p63_skewness_filter, []),
        'P64 Multi-Scale P58':              (p64_p58_multiscale_vol, []),
        'P65 RSI Divergence':               (p65_rsi_divergence, []),
        'P66 Gamma Squeeze':                (p66_gamma_squeeze, []),
        'P67 Rebalance Buffer 5%':          (p67_p58_rebalance_buffer, []),
        'P68 Downside Beta':                (p68_downside_beta, []),
        'P69 Rolling Kelly Sizing':         (p69_kelly_sizing, []),
    }

    W = 43
    header = f"{'Strategy':<{W}} {'CAGR%':>7} {'Sharpe':>8} {'MaxDD%':>8}"
    sep    = "-" * len(header)

    print(f"\n{'TRAIN  (1995-2015)'}")
    print(header); print(sep)
    for name, (fn, args) in patterns.items():
        m = metrics(fn(train_df, *args))
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}")

    print(f"\n{'TEST  (2016-Present)'}")
    print(header); print(sep)
    for name, (fn, args) in patterns.items():
        m = metrics(fn(test_df, *args))
        flag = " <-- BEATS P58 (1.047) OR P49 DD (-9.20)" if m['Sharpe'] > 1.047 or m['MaxDD'] > -9.20 else ""
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}{flag}")
    print(sep)
    print(f"  Ultimate Champ P58 (test):             9.55%   1.047   -11.03")

    # Walk-forward top candidates
    print("\n--- Walk-Forward (2-year windows) for top candidates ---")
    for name, (fn, args) in patterns.items():
        m = metrics(fn(test_df, *args))
        if m['Sharpe'] >= 1.03 or m['MaxDD'] > -11.0:
            print(f"\n{name}:")
            wf = walk_forward(df, fn, 2, *args)
            if not wf.empty:
                pos_pct = (wf['Sharpe'] > 0).mean() * 100
                print(wf[['CAGR','Sharpe','MaxDD']].to_string())
                print(f"  => Positive windows: {pos_pct:.0f}%")

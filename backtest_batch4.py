"""
Batch 4 — 10 New Patterns (P40–P49)
Goal: Beat current champion P39 (Sharpe 0.964, MaxDD -14.1%)
Focus: Regime-switching, adaptive sizing, and structural market microstructure signals.
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

# ─────────────────────────────────────────────
# P40: Volatility Regime Switch
# Low vol  → favour mean-reversion (RSI overlay heavy)
# High vol → favour trend-following (ATR turtle heavy)
# ─────────────────────────────────────────────
def p40_vol_regime_switch(df, target_vol=0.15):
    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    exp_v = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (exp_v + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)

    # vol percentile rank (rolling 252d)
    vol_rank = vol20.rolling(252).rank(pct=True).fillna(0.5)
    # high vol -> more turtle, low vol -> more vol+RSI
    w2 = vol_rank.shift(1)       # turtle weight grows with vol
    w1 = 1 - w2                  # vol+RSI weight shrinks with vol
    exposure = w1 * comp1 + w2 * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P41: Rolling Sharpe Position Sizing
# Use the strategy's own rolling 63-day Sharpe to scale position.
# High recent Sharpe → more aggressive; low/negative → scale down.
# ─────────────────────────────────────────────
def p41_rolling_sharpe_sizing(df, target_vol=0.15):
    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    base_returns = (base + ov).clip(upper=1.0).shift(1) * ret  # approx strategy returns

    roll_sharpe = base_returns.rolling(63).mean() / base_returns.rolling(63).std() * np.sqrt(252)
    # Normalise: clamp at [-2, 2], map to [0, 1]
    size_scalar = ((roll_sharpe.shift(1).clip(-2, 2) + 2) / 4).fillna(0.5)

    exposure = (base + ov).clip(upper=1.0) * size_scalar
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P42: Autocorrelation Regime Detector
# +ve 1-lag autocorrelation → trending → ride trend (Vol Scaling)
# -ve 1-lag autocorrelation → mean-reverting → lean on RSI
# ─────────────────────────────────────────────
def p42_autocorr_regime(df, target_vol=0.15, window=20):
    ret   = df['Close'].pct_change().fillna(0)
    acorr = ret.rolling(window).corr(ret.shift(1))  # 1-lag autocorrelation

    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)

    # Trending regime: acorr > 0 → pure vol scaling
    # MR regime: acorr < 0 → vol scaling + RSI dip overlay
    ov = ((acorr.shift(1) < 0) & (rsi4.shift(1) < 30)).astype(float) * 0.2
    exposure = (base + ov).clip(upper=1.0)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P43: Drawdown-Stop Wrapper over Champion Ensemble
# Run P39 logic but add a hard 12% drawdown stop from rolling 252d ATH
# Re-enter when price recovers above ATH * 0.97
# ─────────────────────────────────────────────
def p43_ensemble_with_stop(df, stop=0.12, reentry=0.03, target_vol=0.15):
    ret   = df['Close'].pct_change().fillna(0)
    # P39 base exposure
    vol20  = ret.rolling(20).std() * np.sqrt(252)
    exp_v  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4   = calc_rsi(df['Close'], 4)
    ov     = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1  = (exp_v + ov).clip(upper=1.0)
    comp2  = atr_turtle_exposure(df)
    base_exposure = 0.6 * comp1 + 0.4 * comp2

    # Rolling 252-day ATH stop
    ath    = df['Close'].rolling(252, min_periods=1).max()
    dd     = (df['Close'] - ath) / ath    # negative
    stopped = pd.Series(np.nan, index=df.index)
    stopped[dd < -stop]                          = 0.0
    stopped[df['Close'] >= ath * (1 - reentry)] = 1.0
    stopped = stopped.ffill().fillna(1.0)   # default: active

    exposure = base_exposure * stopped
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P44: Candle Body Strength Trend Filter
# Body = |Close - Open| / (High - Low) → 1 = strong candle
# Use 10-day EMA of body strength as a trend quality filter
# ─────────────────────────────────────────────
def p44_candle_strength(df, target_vol=0.15):
    body       = (df['Close'] - df['Open']).abs()
    range_     = (df['High']  - df['Low']).replace(0, np.nan)
    body_ratio = (body / range_).fillna(0)  # 0 to 1
    body_ema   = body_ratio.ewm(span=10).mean()

    # Directional strength: positive if bullish candles dominate
    bullish = (df['Close'] > df['Open']).astype(float)
    bear    = (df['Close'] < df['Open']).astype(float)
    dir_score = (bullish - bear).ewm(span=10).mean()  # -1 to +1
    trend_ok = (dir_score.shift(1) > 0) & (body_ema.shift(1) > 0.4)

    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)

    # Use ensemble but only when candle quality confirms
    strength_scalar = trend_ok.shift(1).astype(float)
    exposure = (0.6 * comp1 + 0.4 * comp2) * (0.5 + 0.5 * strength_scalar)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P45: EWMA Volatility Forecast (GARCH-lite)
# Use exponentially weighted vol (λ=0.94, like RiskMetrics) instead of
# simple rolling std — more responsive to recent shocks
# ─────────────────────────────────────────────
def p45_ewma_vol(df, lam=0.94, target_vol=0.15):
    ret      = df['Close'].pct_change().fillna(0)
    # EWMA variance (RiskMetrics)
    ewma_var = ret.pow(2).ewm(alpha=1-lam, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var * 252)

    base  = (target_vol / ewma_vol.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)
    exposure = 0.6 * comp1 + 0.4 * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P46: High-Low Spread Momentum (Breadth Proxy)
# (High - Low) / Close = intraday range normalised
# Shrinking range over 5 days → compression → buy
# Expanding range → uncertainty → reduce
# ─────────────────────────────────────────────
def p46_hl_spread_momentum(df, target_vol=0.15):
    hl_pct   = (df['High'] - df['Low']) / df['Close']
    hl_ma5   = hl_pct.rolling(5).mean()
    hl_ma20  = hl_pct.rolling(20).mean()
    compress = (hl_ma5 < hl_ma20)  # intraday range contracting

    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)

    # Boost ATR turtle component when range is compressing (breakout setup)
    w2 = np.where(compress.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P47: Momentum Crash Protection
# When trailing 1-month return is extreme (>+8%), reduce exposure 20%
# to guard against momentum crashes/reversals
# ─────────────────────────────────────────────
def p47_momentum_crash_protect(df, target_vol=0.15):
    ret1m = df['Close'].pct_change(21)
    crash_risk = ret1m.shift(1) > 0.08   # too far, too fast

    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)
    ensemble = 0.6 * comp1 + 0.4 * comp2

    # Scale back 20% when overextended
    protection = np.where(crash_risk, 0.80, 1.0)
    exposure = ensemble * pd.Series(protection, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P48: Intraday Return Regime
# Open-to-Close return (intraday) trend — if the market is
# consistently closing near the high (vs open), trend is healthy.
# Use 20-day EMA of (Close-Open)/Open as regime filter.
# ─────────────────────────────────────────────
def p48_intraday_regime(df, target_vol=0.15):
    intraday  = (df['Close'] - df['Open']) / df['Open']
    intra_ema = intraday.ewm(span=20).mean()
    bull_intra = intra_ema > 0

    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)

    # When intraday regime is bullish, lean more on turtle breakout
    w2 = np.where(bull_intra.shift(1), 0.5, 0.3)
    w1 = 1 - w2
    exposure = pd.Series(w1, index=df.index)*comp1 + pd.Series(w2, index=df.index)*comp2
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# P49: Triple Confirmation Champion
# All three must agree: Vol regime OK + ATR in breakout + Price above SMA200
# When all three agree → full ensemble weight
# When 2/3 agree → 70% weight
# When ≤1 agrees → 40% weight
# ─────────────────────────────────────────────
def p49_triple_confirm(df, target_vol=0.15):
    ret   = df['Close'].pct_change().fillna(0)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    base  = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4  = calc_rsi(df['Close'], 4)
    ov    = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1 = (base + ov).clip(upper=1.0)
    comp2 = atr_turtle_exposure(df)
    ensemble = 0.6 * comp1 + 0.4 * comp2

    # Three signals
    sma200     = df['Close'].rolling(200).mean()
    s1 = (df['Close'] > sma200).astype(int)            # trend OK
    s2 = (comp2 > 0).astype(int)                       # turtle in position
    s3 = (vol20 < vol20.rolling(252).quantile(0.7)).astype(int)  # vol not extreme

    score = (s1.shift(1) + s2.shift(1) + s3.shift(1)).fillna(0)
    scalar = np.where(score >= 3, 1.0, np.where(score == 2, 0.7, 0.4))
    exposure = ensemble * pd.Series(scalar, index=df.index)
    return backtest(df, exposure)

# ─────────────────────────────────────────────
# WALK-FORWARD for candidates
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
        'P40 Vol Regime Switch':            p40_vol_regime_switch,
        'P41 Rolling Sharpe Sizing':        p41_rolling_sharpe_sizing,
        'P42 Autocorr Regime Detector':     p42_autocorr_regime,
        'P43 Ensemble + Drawdown Stop':     p43_ensemble_with_stop,
        'P44 Candle Strength Filter':       p44_candle_strength,
        'P45 EWMA Vol (RiskMetrics)':       p45_ewma_vol,
        'P46 H-L Spread Momentum':          p46_hl_spread_momentum,
        'P47 Momentum Crash Protection':    p47_momentum_crash_protect,
        'P48 Intraday Regime':              p48_intraday_regime,
        'P49 Triple Confirmation':          p49_triple_confirm,
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
        flag = " <-- CHAMPION?" if m['Sharpe'] > 0.964 else ""
        print(f"{name:<{W}} {m['CAGR']:>7.2f} {m['Sharpe']:>8.3f} {m['MaxDD']:>8.2f}{flag}")
    print(sep)
    print(f"  Current champion P39 (test):          9.91%   0.964   -14.08")
    print(f"  Baseline B&H (test):                 13.11%   0.775   -33.92")

    # Walk-forward any candidates above Sharpe 0.90
    print("\n--- Walk-Forward (2-year windows) for top candidates ---")
    for name, fn in patterns.items():
        m = metrics(fn(test_df))
        if m['Sharpe'] >= 0.90:
            print(f"\n{name}:")
            wf = walk_forward(df, fn)
            if not wf.empty:
                pos_pct = (wf['Sharpe'] > 0).mean() * 100
                print(wf[['CAGR','Sharpe','MaxDD']].to_string())
                print(f"  => Positive windows: {pos_pct:.0f}%")

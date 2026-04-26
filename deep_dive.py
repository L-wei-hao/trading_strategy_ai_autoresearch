"""
Deep-dive into the two breakout winners from Batch 3:
  P37: Overnight Regime + Vol Scale  (Test Sharpe 1.103, MaxDD -11.6%)
  P39: Champion Ensemble 60/40       (Test Sharpe 0.964, MaxDD -14.1%)

Robustness checks:
  1. Walk-forward validation (5-year windows)
  2. Sensitivity to key parameters
  3. Full-sample equity curve statistics
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
        return {'CAGR': 0, 'Sharpe': 0, 'MaxDD': 0, 'WinRate': 0}
    total  = (1 + returns).prod() - 1
    days   = (returns.index[-1] - returns.index[0]).days
    cagr   = (1 + total) ** (365.25 / days) - 1 if days > 0 else 0.0
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    cum    = (1 + returns).cumprod()
    mdd    = ((cum - cum.cummax()) / cum.cummax()).min()
    wr     = (returns > 0).sum() / (returns != 0).sum()
    return {'CAGR': round(cagr*100,2), 'Sharpe': round(sharpe,3),
            'MaxDD': round(mdd*100,2),  'WinRate': round(wr*100,1)}

def calc_rsi(series, period=4):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────
# P37: Overnight Regime + Vol Scale
# ─────────────────────────────────────────────
def p37_overnight_regime(df, target_vol=0.15, window=20):
    overnight_ret = df['Open'] / df['Close'].shift(1) - 1
    overnight_ma  = overnight_ret.rolling(window).mean()
    bull_overnight = overnight_ma > 0
    realized_vol   = df['Close'].pct_change().rolling(window).std() * np.sqrt(252)
    base_exp       = (target_vol / realized_vol.shift(1)).clip(upper=1.0).fillna(0)
    exposure       = base_exp * bull_overnight.shift(1).astype(float)
    pos  = exposure.shift(1).fillna(0)
    ret  = df['Close'].pct_change().fillna(0)
    cost = pos.diff().abs().fillna(0) * 0.0005
    return pos * ret - cost

# ─────────────────────────────────────────────
# P39: Champion Ensemble (60/40)
# ─────────────────────────────────────────────
def p39_champion_ensemble(df, target_vol=0.15):
    ret = df['Close'].pct_change().fillna(0)
    # Vol Scaling + RSI
    vol20   = ret.rolling(20).std() * np.sqrt(252)
    exp_vol = (target_vol / vol20.shift(1)).clip(upper=1.0).fillna(0)
    rsi4    = calc_rsi(df['Close'], 4)
    overlay = (rsi4.shift(1) < 30).astype(float) * 0.2
    comp1   = (exp_vol + overlay).clip(upper=1.0)
    # ATR Turtle
    tr      = pd.concat([df['High'] - df['Low'],
                         (df['High'] - df['Close'].shift(1)).abs(),
                         (df['Low']  - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr14   = tr.ewm(span=14, adjust=False).mean()
    high20  = df['High'].rolling(20).max()
    low10   = df['Low'].rolling(10).min()
    size    = (0.02 / (atr14 / df['Close'])).clip(upper=1.0)
    comp2   = pd.Series(np.nan, index=df.index)
    comp2[df['High'] >= high20.shift(1)] = size[df['High'] >= high20.shift(1)]
    comp2[df['Low']  <= low10.shift(1)]  = 0.0
    comp2   = comp2.ffill().fillna(0)
    exposure = 0.6 * comp1 + 0.4 * comp2
    pos  = exposure.shift(1).fillna(0)
    cost = pos.diff().abs().fillna(0) * 0.0005
    return pos * ret - cost

# ─────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────
def walk_forward(df, fn, train_years=5, test_years=2):
    results = []
    start = df.index[0].year
    end   = df.index[-1].year
    for yr in range(start + train_years, end - test_years + 1, test_years):
        te_start = f"{yr}-01-01"
        te_end   = f"{yr + test_years}-01-01"
        train_end = f"{yr}-01-01"
        te_df = df[(df.index >= te_start) & (df.index < te_end)].copy()
        if len(te_df) < 100:
            continue
        r = fn(te_df)
        m = metrics(r)
        m['period'] = f"{yr}–{yr+test_years}"
        results.append(m)
    return pd.DataFrame(results).set_index('period')

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df       = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df  = df[df.index >= '2016-01-01'].copy()

    print("=" * 65)
    print("DEEP DIVE: P37 Overnight Regime + Vol Scale")
    print("=" * 65)
    r_tr = p37_overnight_regime(train_df)
    r_te = p37_overnight_regime(test_df)
    print(f"  Train (1995-2015): {metrics(r_tr)}")
    print(f"  Test  (2016-now):  {metrics(r_te)}")

    print("\n  Walk-Forward (2-year windows):")
    wf = walk_forward(df, p37_overnight_regime)
    print(wf.to_string())

    # Sensitivity: overnight window
    print("\n  Sensitivity — overnight MA window:")
    for w in [5, 10, 20, 40, 60]:
        r = p37_overnight_regime(test_df, window=w)
        print(f"    window={w:>3d}: {metrics(r)}")

    print("\n" + "=" * 65)
    print("DEEP DIVE: P39 Champion Ensemble (60/40)")
    print("=" * 65)
    r_tr = p39_champion_ensemble(train_df)
    r_te = p39_champion_ensemble(test_df)
    print(f"  Train (1995-2015): {metrics(r_tr)}")
    print(f"  Test  (2016-now):  {metrics(r_te)}")

    print("\n  Walk-Forward (2-year windows):")
    wf = walk_forward(df, p39_champion_ensemble)
    print(wf.to_string())

    # Sensitivity: blend ratio
    print("\n  Sensitivity — ensemble blend (Vol+RSI weight):")
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        def blend(df, w=w1):
            # inline re-blend
            ret = df['Close'].pct_change().fillna(0)
            vol20   = ret.rolling(20).std() * np.sqrt(252)
            exp_vol = (0.15 / vol20.shift(1)).clip(upper=1.0).fillna(0)
            rsi4    = calc_rsi(df['Close'], 4)
            overlay = (rsi4.shift(1) < 30).astype(float) * 0.2
            c1      = (exp_vol + overlay).clip(upper=1.0)
            tr      = pd.concat([df['High']-df['Low'],
                                 (df['High']-df['Close'].shift(1)).abs(),
                                 (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
            atr14   = tr.ewm(span=14, adjust=False).mean()
            high20  = df['High'].rolling(20).max()
            low10   = df['Low'].rolling(10).min()
            size    = (0.02/(atr14/df['Close'])).clip(upper=1.0)
            c2      = pd.Series(np.nan, index=df.index)
            c2[df['High']>=high20.shift(1)] = size[df['High']>=high20.shift(1)]
            c2[df['Low']<=low10.shift(1)]   = 0.0
            c2      = c2.ffill().fillna(0)
            exposure = w*c1 + (1-w)*c2
            pos  = exposure.shift(1).fillna(0)
            cost = pos.diff().abs().fillna(0) * 0.0005
            return pos*ret - cost
        r = blend(test_df)
        print(f"    Vol+RSI weight={w1:.1f}: {metrics(r)}")

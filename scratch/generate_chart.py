import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

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

def p83_exp(df):
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
    return pd.Series(w1, index=df.index) * comp1 + pd.Series(w2, index=df.index) * comp2

def p85_exp(df):
    p50_exp = p83_exp(df)
    rsi14 = calc_rsi(df['Close'], 14)
    div_signal = ((df['Close'] == df['Close'].rolling(20).max()) & ~(rsi14 == rsi14.rolling(20).max())).rolling(5).max() > 0
    return p50_exp * pd.Series(np.where(div_signal.shift(1), 0.80, 1.0), index=df.index)

def run_backtest(df, exposure, tc=0.0005):
    pos  = pd.Series(np.asarray(exposure), index=df.index).shift(1).fillna(0)
    ret  = df['Close'].pct_change().fillna(0)
    cost = pos.diff().abs().fillna(0) * tc
    return (pos * ret - cost)

if __name__ == "__main__":
    df = load_data("SP500_history_2000_to_current.csv")
    test_df = df[df.index >= '2016-01-01'].copy()
    
    # 1. Buy & Hold
    bh_ret = test_df['Close'].pct_change().fillna(0)
    bh_cum = (1 + bh_ret).cumprod() * 1000000
    
    # 2. P83
    p83_returns = run_backtest(test_df, p83_exp(test_df))
    p83_cum = (1 + p83_returns).cumprod() * 1000000
    
    # 3. P85
    p85_returns = run_backtest(test_df, p85_exp(test_df))
    p85_cum = (1 + p85_returns).cumprod() * 1000000
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(bh_cum, label=f'S&P 500 Buy & Hold (Final: ${bh_cum.iloc[-1]:,.0f})', color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(p83_cum, label=f'P83 High-CAGR Champion (Final: ${p83_cum.iloc[-1]:,.0f})', color='#1f77b4', linewidth=2)
    plt.plot(p85_cum, label=f'P85 Max-Sharpe Champion (Final: ${p85_cum.iloc[-1]:,.0f})', color='#d62728', linewidth=2)
    
    plt.title('Growth of $1,000,000: S&P 500 vs. Quantitative Champions (2016-Present)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Shade drawdowns for Buy & Hold
    bh_roll_max = bh_cum.cummax()
    plt.fill_between(bh_cum.index, bh_cum, bh_roll_max, color='gray', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('performance_graph.png', dpi=300)
    print("Graph saved to performance_graph.png")

import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_metrics(returns, daily_rf=0.0):
    if len(returns) == 0:
        return {}
    
    total_return = (1 + returns).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    cagr = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0.0
        
    excess_returns = returns - daily_rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0
        
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Max Drawdown': max_drawdown
    }

def calc_rsi(series, period=4):
    delta = series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.ewm(com=period-1, min_periods=period).mean()
    roll_down = down.abs().ewm(com=period-1, min_periods=period).mean()
    RS = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + RS))

def test_final_strategy(df, target_vol=0.15):
    df = df.copy()
    daily_returns = df['Close'].pct_change().fillna(0)
    
    # 1. Multi-Scale Volatility
    vol20 = daily_returns.rolling(window=20).std() * np.sqrt(252)
    vol60 = daily_returns.rolling(window=60).std() * np.sqrt(252)
    vol120 = daily_returns.rolling(window=120).std() * np.sqrt(252)
    max_vol = pd.concat([vol20, vol60, vol120], axis=1).max(axis=1)
    
    exposure_vol = target_vol / max_vol.shift(1)
    exposure_vol = exposure_vol.clip(upper=1.0).fillna(0)
    
    # 2. RSI Mean Reversion Overlay
    df['RSI'] = calc_rsi(df['Close'], 4)
    rsi_overlay = (df['RSI'].shift(1) < 30).astype(float) * 0.2
    
    final_exposure = (exposure_vol + rsi_overlay).clip(upper=1.0)
    
    # 3. Strategy returns
    strategy_returns = final_exposure * daily_returns
    
    # 4. Costs
    transaction_cost = 0.0005
    trades = final_exposure.diff().abs().fillna(0)
    costs = trades * transaction_cost
    
    return strategy_returns - costs

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df = df[df.index >= '2016-01-01'].copy()
    
    print("Multi-Scale Volatility Scaling + RSI Overlay")
    returns_train = test_final_strategy(train_df)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    returns_test = test_final_strategy(test_df)
    print("Test Metrics:", calculate_metrics(returns_test))

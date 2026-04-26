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
    return {'CAGR': cagr, 'Sharpe': sharpe, 'Max Drawdown': max_drawdown}

def test_risk_adjusted_momentum(df, lookback=252):
    df = df.copy()
    daily_returns = df['Close'].pct_change().fillna(0)
    
    # 1-year Return / 1-year Volatility
    ann_ret = df['Close'].pct_change(lookback)
    ann_vol = daily_returns.rolling(lookback).std() * np.sqrt(252)
    
    signal = (ann_ret / ann_vol).shift(1)
    
    # Strategy: Long if Signal > 0
    positions = (signal > 0).astype(float)
    
    # Vol Scaling overlay (Target 15%)
    realized_vol = daily_returns.rolling(20).std() * np.sqrt(252)
    exposure = 0.15 / realized_vol.shift(1)
    exposure = exposure.clip(upper=1.0).fillna(0)
    
    final_exposure = positions * exposure
    
    strategy_returns = final_exposure * daily_returns
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
    
    print("Risk-Adjusted Momentum (12-Month) + Vol Scaling")
    returns_train = test_risk_adjusted_momentum(train_df)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    returns_test = test_risk_adjusted_momentum(test_df)
    print("Test Metrics:", calculate_metrics(returns_test))

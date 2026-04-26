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

def test_vol_of_vol(df, target_vol=0.15):
    df = df.copy()
    daily_returns = df['Close'].pct_change().fillna(0)
    realized_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
    
    # Vol-of-Vol (standard deviation of the realized vol over 20 days)
    vol_of_vol = realized_vol.rolling(window=20).std()
    
    # Base Exposure
    exposure = target_vol / realized_vol.shift(1)
    exposure = exposure.clip(upper=1.0).fillna(0)
    
    # Penalty if Vol-of-Vol is in the top 20th percentile
    threshold = vol_of_vol.rolling(252).quantile(0.8)
    penalty = (vol_of_vol.shift(1) > threshold).astype(float) * 0.5
    
    final_exposure = (exposure - penalty).clip(lower=0)
    
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
    
    print("Volatility-of-Volatility Scaling (Top 20% Penalty)")
    returns_train = test_vol_of_vol(train_df)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    returns_test = test_vol_of_vol(test_df)
    print("Test Metrics:", calculate_metrics(returns_test))

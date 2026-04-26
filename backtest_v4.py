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
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
    
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor
    }

def run_backtest(df, signals, transaction_cost=0.0005):
    positions = signals.shift(1).fillna(0)
    daily_returns = df['Close'].pct_change().fillna(0)
    strategy_returns = positions * daily_returns
    trades = positions.diff().abs().fillna(0)
    costs = trades * transaction_cost
    return strategy_returns - costs

def test_fast_trend(df, fast=20, slow=50, regime=200):
    df = df.copy()
    df['SMA_fast'] = df['Close'].rolling(window=fast).mean()
    df['SMA_slow'] = df['Close'].rolling(window=slow).mean()
    df['SMA_regime'] = df['Close'].rolling(window=regime).mean()
    
    # Condition: Fast > Slow AND Close > Regime
    buy_signal = (df['SMA_fast'] > df['SMA_slow']) & (df['Close'] > df['SMA_regime'])
    
    positions = pd.Series(0, index=df.index)
    positions[buy_signal] = 1
    
    return positions

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df = df[df.index >= '2016-01-01'].copy()
    
    print("Fast Trend Strategy (20/50 SMA) + 200 SMA Regime")
    signals_train = test_fast_trend(train_df)
    returns_train = run_backtest(train_df, signals_train)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    signals_test = test_fast_trend(test_df)
    returns_test = run_backtest(test_df, signals_test)
    print("Test Metrics:", calculate_metrics(returns_test))

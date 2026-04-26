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
    
    # CAGR
    total_return = (1 + returns).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    if days > 0:
        cagr = (1 + total_return) ** (365.25 / days) - 1
    else:
        cagr = 0.0
        
    # Sharpe Ratio
    excess_returns = returns - daily_rf
    if excess_returns.std() > 0:
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    else:
        sharpe = 0.0
        
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Win rate and Profit factor
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
    """
    signals: pd.Series of 1 (long), -1 (short), or 0 (cash) aligned with df.index
    We assume signal is generated at the end of day t, and we trade at the close of day t+1
    Wait, to avoid lookahead bias, if signal is based on day t close, we trade at day t+1 open, or day t+1 close.
    Let's assume we hold position from close to close.
    So position at day t is signal from day t-1.
    """
    positions = signals.shift(1).fillna(0)
    
    # Daily returns of the underlying
    daily_returns = df['Close'].pct_change().fillna(0)
    
    # Strategy returns
    strategy_returns = positions * daily_returns
    
    # Transaction costs
    # Cost incurred when position changes
    trades = positions.diff().abs().fillna(0)
    costs = trades * transaction_cost
    
    strategy_returns = strategy_returns - costs
    
    return strategy_returns

def test_sma_crossover(df, short_window=50, long_window=200):
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    
    # Signal: 1 if short > long, else 0
    signals = pd.Series(0, index=df.index)
    signals[short_window:] = np.where(df['SMA_short'][short_window:] > df['SMA_long'][short_window:], 1, 0)
    
    return signals

if __name__ == "__main__":
    df = load_data("SP500_history_2000_to_current.csv")
    
    # Train/Test split
    train_df = df[df.index < '2016-01-01'].copy()
    test_df = df[df.index >= '2016-01-01'].copy()
    
    print("Baseline Buy & Hold (Train):")
    bh_train = train_df['Close'].pct_change().fillna(0)
    print(calculate_metrics(bh_train))
    
    print("\nBaseline Buy & Hold (Test):")
    bh_test = test_df['Close'].pct_change().fillna(0)
    print(calculate_metrics(bh_test))
    
    print("\nSMA Crossover Strategy (50, 200)")
    
    # Train
    signals_train = test_sma_crossover(train_df, 50, 200)
    returns_train = run_backtest(train_df, signals_train)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    # Test
    signals_test = test_sma_crossover(test_df, 50, 200)
    returns_test = run_backtest(test_df, signals_test)
    print("Test Metrics:", calculate_metrics(returns_test))

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

def test_bb_mean_reversion(df, bb_window=20, bb_std=2.0, sma_regime=200):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=bb_window).mean()
    df['STD'] = df['Close'].rolling(window=bb_window).std()
    df['Lower_BB'] = df['SMA'] - bb_std * df['STD']
    df['SMA_long'] = df['Close'].rolling(window=sma_regime).mean()
    
    # Entry: Close < Lower_BB AND Close > SMA_long
    buy_signal = (df['Close'] < df['Lower_BB']) & (df['Close'] > df['SMA_long'])
    
    # Exit: Close > SMA (mean reversion target) OR Close < SMA_long (stop loss / regime change)
    sell_signal = (df['Close'] > df['SMA']) | (df['Close'] < df['SMA_long'])
    
    pos_logic = pd.Series(np.nan, index=df.index)
    pos_logic[buy_signal] = 1
    pos_logic[sell_signal] = -1
    pos_logic = pos_logic.ffill().fillna(-1)
    positions = pos_logic.replace(-1, 0)
    
    return positions

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df = df[df.index >= '2016-01-01'].copy()
    
    print("Bollinger Band Mean Reversion with Regime & Stop Loss")
    signals_train = test_bb_mean_reversion(train_df)
    returns_train = run_backtest(train_df, signals_train)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    signals_test = test_bb_mean_reversion(test_df)
    returns_test = run_backtest(test_df, signals_test)
    print("Test Metrics:", calculate_metrics(returns_test))

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

def test_tom_pattern(df):
    df = df.copy()
    
    # Identify days relative to month end
    # We want last day of month and first 3 days of next month
    df['DayOfMonth'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    # Find last trading day of month
    df['IsLastDay'] = df.groupby(['Year', 'Month']).cumcount(ascending=False) == 0
    # Find first 3 trading days of month
    df['IsFirst3Days'] = df.groupby(['Year', 'Month']).cumcount() < 3
    
    # Strategy: Hold if IsLastDay or IsFirst3Days
    # Wait, if we buy at the CLOSE of the second-to-last day?
    # Let's say: Signal on Last Day, hold for 4 days.
    # Simpler: Signal is 1 if it's the last day of month OR one of the first 3 days of month.
    signals = (df['IsLastDay'] | df['IsFirst3Days']).astype(int)
    
    daily_returns = df['Close'].pct_change().fillna(0)
    # Positions: shift signal by 1? 
    # If it's the last day, we want to HAVE the position for that day's return?
    # Usually ToM starts at the close of the penultimate day.
    
    positions = signals # If signal is 1, we hold for that day's return
    
    strategy_returns = positions * daily_returns
    
    # Costs
    transaction_cost = 0.0005
    trades = positions.diff().abs().fillna(0)
    costs = trades * transaction_cost
    
    return strategy_returns - costs

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    df = load_data("SP500_history_2000_to_current.csv")
    train_df = df[df.index < '2016-01-01'].copy()
    test_df = df[df.index >= '2016-01-01'].copy()
    
    print("Turn-of-the-Month Pattern (Last day + First 3 days)")
    returns_train = test_tom_pattern(train_df)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    returns_test = test_tom_pattern(test_df)
    print("Test Metrics:", calculate_metrics(returns_test))
    
    print("\nBaseline Buy & Hold (Test) for comparison:")
    bh_test = test_df['Close'].pct_change().fillna(0)
    print(calculate_metrics(bh_test))

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

def test_volume_climax(df):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    
    # Volume Z-score (20-day window)
    df['Vol_Mean'] = df['Volume'].rolling(20).mean()
    df['Vol_Std'] = df['Volume'].rolling(20).std()
    df['Vol_Z'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
    
    # Panic Bottom: Price drops > 2% on Vol Z > 2.0
    panic_bottom = (df['Returns'] < -0.02) & (df['Vol_Z'] > 2.0)
    
    # Euphoria Top: Price rises > 2% on Vol Z > 2.0
    euphoria_top = (df['Returns'] > 0.02) & (df['Vol_Z'] > 2.0)
    
    # Strategy: Buy panic, Sell euphoria? 
    # Or just use as signals.
    # Let's say: Hold for 5 days after panic.
    positions = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if panic_bottom.iloc[i]:
            positions.iloc[i:i+6] = 1 # Hold for 5 days
        if euphoria_top.iloc[i]:
            positions.iloc[i:i+6] = 0 # Exit
            
    strategy_returns = positions.shift(1).fillna(0) * df['Returns']
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
    
    print("Volume Climax Pattern (Panic Buy / Euphoria Exit)")
    returns_train = test_volume_climax(train_df)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    returns_test = test_volume_climax(test_df)
    print("Test Metrics:", calculate_metrics(returns_test))

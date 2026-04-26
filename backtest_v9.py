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

def calc_rsi(series, period=4):
    delta = series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.ewm(com=period-1, min_periods=period).mean()
    roll_down = down.abs().ewm(com=period-1, min_periods=period).mean()
    RS = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + RS))

def test_vol_scaling_rsi(df, target_vol=0.15, vol_window=20, rsi_buy=30):
    df = df.copy()
    daily_returns = df['Close'].pct_change().fillna(0)
    
    # Calculate realized volatility
    realized_vol = daily_returns.rolling(window=vol_window).std() * np.sqrt(252)
    
    # Exposure Calculation
    exposure = target_vol / realized_vol.shift(1)
    exposure = exposure.clip(upper=1.0).fillna(0)
    
    # RSI Mean Reversion Overlay
    df['RSI'] = calc_rsi(df['Close'], 4)
    # If RSI is oversold, add 20% exposure (buy the dip)
    rsi_overlay = (df['RSI'].shift(1) < rsi_buy).astype(float) * 0.2
    
    final_exposure = (exposure + rsi_overlay).clip(upper=1.0)
    
    # Strategy returns
    strategy_returns = final_exposure * daily_returns
    
    # Transaction costs
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
    
    print("Volatility Scaling + RSI Dip Overlay")
    returns_train = test_vol_scaling_rsi(train_df)
    print("Train Metrics:", calculate_metrics(returns_train))
    
    returns_test = test_vol_scaling_rsi(test_df)
    print("Test Metrics:", calculate_metrics(returns_test))

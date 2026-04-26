import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    df = load_data("SP500_history_2000_to_current.csv")
    df['DayOfWeek'] = df.index.dayofweek # 0=Monday, 4=Friday
    df['Returns'] = df['Close'].pct_change()
    
    print("Mean Returns by Day of Week:")
    print(df.groupby('DayOfWeek')['Returns'].mean() * 252)
    
    print("\nSharpe Ratio by Day of Week:")
    stats = df.groupby('DayOfWeek')['Returns'].agg(['mean', 'std'])
    print(np.sqrt(252) * stats['mean'] / stats['std'])

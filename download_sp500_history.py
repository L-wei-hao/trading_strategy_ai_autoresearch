import yfinance as yf
import pandas as pd
from datetime import datetime

def download_sp500_history():
    # ^GSPC is the ticker symbol for the S&P 500
    ticker = "^GSPC"
    
    # Define the date range based on your URL (period1=946684800 is Jan 1, 2000)
    start_date = "1995-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # Fetch the data
    # Note: Yahoo Finance API only provides Daily (1d) granularity for this length of history.
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    if data.empty:
        print("No data found or download failed.")
        return

    # Clean up the format for CSV
    # yfinance sometimes returns multi-index columns; we flatten them for a standard CSV
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Save to CSV
    filename = f"SP500_history_2000_to_current.csv"
    data.to_csv(filename)
    
    print(f"Success! Data saved to {filename}")
    print(data.head()) # Show first few rows

if __name__ == "__main__":
    download_sp500_history()
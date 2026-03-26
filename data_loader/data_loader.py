import yfinance as yf
import pandas as pd
import os
from nsepython import index_history
from datetime import datetime, timedelta

def fetch_data(symbol, start_date, end_date, filename):
    """
    Fetches historical data for a given symbol and dates.
    Uses yfinance as primary and specialized scrapers as fallback.
    """
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    # Primary: yfinance
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # yfinance 0.2.x often returns multi-index columns (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Drop the ticker level to keep only OHLCV
        data.columns = data.columns.droplevel(1)
    
    # Specialized Fallback for India VIX if yfinance is empty or unreliable
    vix_failure = data.empty
    if not vix_failure and (symbol == "^INDIAVIX" or "VIX" in symbol):
        # If 'Close' is all zeros (common issue on Yahoo for India VIX)
        # Use axis=None to handle it if it's accidentally still a DataFrame
        if (data['Close'] == 0).all(axis=None):
            vix_failure = True
            
    if (symbol == "^INDIAVIX" or "VIX" in symbol) and vix_failure:
        print(f"yfinance failed for {symbol}. Attempting nsepython fallback...")
        try:
            # NSE expects DD-MMM-YYYY or DD-MM-YYYY
            # Converting to NSE format if needed
            nse_start = pd.to_datetime(start_date).strftime("%d-%m-%Y")
            nse_end = pd.to_datetime(end_date).strftime("%d-%m-%Y")
            
            data_nse = index_history("INDIA VIX", nse_start, nse_end)
            if not data_nse.empty:
                # Rename columns to match standard format
                data_nse = data_nse.rename(columns={
                    'Index Name': 'Ticker',
                    'INDEX_NAME': 'Ticker',
                    'HistoricalDate': 'Date',
                    'OPEN_INDEX_VAL': 'Open',
                    'CLOSE_INDEX_VAL': 'Close',
                    'HIGH_INDEX_VAL': 'High',
                    'LOW_INDEX_VAL': 'Low'
                })
                data_nse['Date'] = pd.to_datetime(data_nse['Date'])
                data_nse.set_index('Date', inplace=True)
                data = data_nse[['Open', 'High', 'Low', 'Close']]
                print("Successfully fetched ^INDIAVIX from NSE via nsepython.")
        except Exception as e:
            print(f"nsepython fallback failed for {symbol}: {e}")

    if not data.empty:
        # Save to CSV while preserving the format expected by the simulation
        # The simulation expects yfinance-style multi-index or simple index
        data.to_csv(filename)
        print(f"Saved {symbol} data to {filename}")
    else:
        print(f"No data found for {symbol} after all attempts.")
    
    return data

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    start_date = "2020-01-01"
    end_date = "2020-06-01"
    
    nifty_spot = fetch_data("^NSEI", start_date, end_date, "data/nifty_spot.csv")
    vix_data = fetch_data("^INDIAVIX", start_date, end_date, "data/vix.csv")

    if vix_data.empty:
        print("Warning: ^INDIAVIX not found on Yahoo Finance. Attempting alternative symbols or methods...")

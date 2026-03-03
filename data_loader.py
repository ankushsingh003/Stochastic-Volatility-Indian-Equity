import yfinance as yf
import pandas as pd
import os

def fetch_data(symbol, start_date, end_date, filename):
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date)
    if not data.empty:
        data.to_csv(filename)
        print(f"Saved {symbol} data to {filename}")
    else:
        print(f"No data found for {symbol}")
    return data

if __name__ == "__main__":
    # Create data directory
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # COVID-19 Period: Feb 2020 to May 2020
    start_date = "2020-01-01"
    end_date = "2020-06-01"
    
    # Nifty 50 Spot (^NSEI)
    nifty_spot = fetch_data("^NSEI", start_date, end_date, "data/nifty_spot.csv")
    
    # India VIX (^INDIAVIX) - Note: yfinance might not have ^INDIAVIX or it might be unreliable.
    # Alternative: fetched from NSE if needed, but let's try yfinance first.
    vix_data = fetch_data("^INDIAVIX", start_date, end_date, "data/vix.csv")

    if vix_data.empty:
        print("Warning: ^INDIAVIX not found on Yahoo Finance. Attempting alternative symbols or methods...")
        # Sometimes it's listed as INDIAVIX.NS or similar in some sources, but yfinance usually doesn't have it.
        # If it fails, we will need a fallback strategy for volatility.

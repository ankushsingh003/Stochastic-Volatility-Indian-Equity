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
    if not os.path.exists("data"):
        os.makedirs("data")
    
    start_date = "2020-01-01"
    end_date = "2020-06-01"
    
    nifty_spot = fetch_data("^NSEI", start_date, end_date, "data/nifty_spot.csv")
    vix_data = fetch_data("^INDIAVIX", start_date, end_date, "data/vix.csv")

    if vix_data.empty:
        print("Warning: ^INDIAVIX not found on Yahoo Finance. Attempting alternative symbols or methods...")

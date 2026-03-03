import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from heston_model.heston_model import HestonModel
from bates_model.bates_model import BatesModel
from rough_volatility.rough_heston_model import RoughHestonModel
from engines.monte_carlo_engine import MonteCarloEngine
from engines.backtest_engine import run_delta_hedge_backtest

def run_comparative_backtest():
    nifty_df = pd.read_csv("data/nifty_spot.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    vix_df = pd.read_csv("data/vix.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    
    start_date = "2020-03-09"
    end_date = "2020-03-27"
    test_data = nifty_df[start_date:end_date]["Close"]
    spot_prices = test_data.values
    
    s0 = spot_prices[0]
    vix0 = vix_df.loc[start_date, "Close"]
    if hasattr(vix0, "iloc"): vix0 = vix0.iloc[0]
    vix0 = vix0 / 100.0
    v0 = vix0**2
    r, T, dt = 0.05, 20/365, 1/252
    K = s0
    num_paths = 5000
    
    h_model = HestonModel(s0=s0, v0=v0, kappa=2.0, theta=v0, sigma=0.5, rho=-0.7)
    b_model = BatesModel(s0=s0, v0=v0, kappa=2.0, theta=v0, sigma=0.5, rho=-0.7, lamb=1.0, mu_j=-0.05, sigma_j=0.1)
    rh_model = RoughHestonModel(s0=s0, v0=v0, kappa=2.0, theta=v0, sigma=0.5, rho=-0.7, H=0.1)
    
    h_engine = MonteCarloEngine(h_model)
    b_engine = MonteCarloEngine(b_model)
    rh_engine = MonteCarloEngine(rh_model)
    
    print("Running Heston Backtest...")
    h_results = run_delta_hedge_backtest(h_model, h_engine, spot_prices, K, T, r, dt, num_paths)
    
    print("Running Bates Backtest...")
    b_results = run_delta_hedge_backtest(b_model, b_engine, spot_prices, K, T, r, dt, num_paths)

    print("Running Rough Heston Backtest...")
    rh_results = run_delta_hedge_backtest(rh_model, rh_engine, spot_prices, K, T, r, dt, num_paths)
    
    if not os.path.exists("backtest_results"):
        os.makedirs("backtest_results")
        
    plt.figure(figsize=(12, 8))
    plt.plot(test_data.index, h_results['portfolio_value'], label='Heston Portfolio')
    plt.plot(test_data.index, b_results['portfolio_value'], label='Bates Portfolio')
    plt.plot(test_data.index, rh_results['portfolio_value'], label='Rough Heston Portfolio')
    plt.axhline(y=h_results['payoff'], color='r', linestyle='--', label='Actual Option Payoff')
    plt.title("Delta Hedging Comparison: Standard vs Bates vs Rough Heston")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("backtest_results/hedging_comparison_all.png")
    
    print(f"Heston Hedging Error: {h_results['hedging_error']:.2f}")
    print(f"Bates Hedging Error: {b_results['hedging_error']:.2f}")
    print(f"Rough Heston Hedging Error: {rh_results['hedging_error']:.2f}")

if __name__ == "__main__":
    run_comparative_backtest()

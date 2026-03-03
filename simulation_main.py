import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from heston_model import HestonModel, estimate_initial_params
from monte_carlo_engine import MonteCarloEngine

def run_simulation():
    nifty_df = pd.read_csv("data/nifty_spot.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    vix_df = pd.read_csv("data/vix.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    
    target_date = "2020-03-23"
    
    s0 = nifty_df.loc[target_date, "Close"]
    vix0 = vix_df.loc[target_date, "Close"] / 100.0
    
    if isinstance(s0, pd.Series): s0 = s0.iloc[0]
    if isinstance(vix0, pd.Series): vix0 = vix0.iloc[0]
    v0 = vix0**2
    
    hist_vix = vix_df[:target_date]["Close"] / 100.0
    theta, kappa, sigma = estimate_initial_params(hist_vix)
    rho = -0.7
    
    model = HestonModel(s0=s0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    engine = MonteCarloEngine(model)
    
    T = 1/12
    dt = 1/252
    num_paths = 20000
    
    strikes = np.linspace(s0 * 0.8, s0 * 1.2, 10)
    prices = []
    
    for K in strikes:
        p, _ = engine.price_european_option(K=K, T=T, dt=dt, num_paths=num_paths, option_type='call')
        prices.append(p)
        
    if not os.path.exists("output"):
        os.makedirs("output")
        
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, prices, marker='o', linestyle='-', color='b')
    plt.title(f"Nifty 50 Call Option Prices (Heston MC) - {target_date}")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Premium")
    plt.grid(True)
    plt.savefig("output/option_prices.png")
    
    results = pd.DataFrame({'Strike': strikes, 'CallPrice': prices})
    results.to_csv("output/simulation_results.csv", index=False)

if __name__ == "__main__":
    run_simulation()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from heston_model import HestonModel, estimate_initial_params
from monte_carlo_engine import MonteCarloEngine

def run_simulation():
    # Load data
    # Load data - skip the ticker and empty date rows
    nifty_df = pd.read_csv("data/nifty_spot.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    vix_df = pd.read_csv("data/vix.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    
    # Target date: March 23, 2020 (near bottom of crash)
    target_date = "2020-03-23"
    
    # Get Nifty spot and VIX on that date
    s0 = nifty_df.loc[target_date, "Close"]
    vix0 = vix_df.loc[target_date, "Close"] / 100.0 # Convert to decimal
    
    # If the result is a Series, take the first value
    if isinstance(s0, pd.Series): s0 = s0.iloc[0]
    if isinstance(vix0, pd.Series): vix0 = vix0.iloc[0]
    v0 = vix0**2 # Variance
    
    print(f"Simulation for {target_date}")
    print(f"Nifty Spot: {s0:.2f}, India VIX: {vix0*100:.2f}%")
    
    # Estimate Heston params from historical VIX
    # We use a window before the target date for calibration
    hist_vix = vix_df[:target_date]["Close"] / 100.0
    theta, kappa, sigma = estimate_initial_params(hist_vix)
    rho = -0.7 # Typical negative correlation for equities
    
    print(f"Heston Params: kappa={kappa}, theta={theta:.4f}, sigma={sigma:.4f}, rho={rho}")
    
    model = HestonModel(s0=s0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    engine = MonteCarloEngine(model)
    
    # Simulation settings
    T = 1/12 # 1 month to expiry
    dt = 1/252 # daily steps
    num_paths = 20000
    
    strikes = np.linspace(s0 * 0.8, s0 * 1.2, 10)
    prices = []
    
    print("Pricing options for different strikes...")
    for K in strikes:
        p, _ = engine.price_european_option(K=K, T=T, dt=dt, num_paths=num_paths, option_type='call')
        prices.append(p)
        
    # Visualization
    if not os.path.exists("output"):
        os.makedirs("output")
        
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, prices, marker='o', linestyle='-', color='b')
    plt.title(f"Nifty 50 Call Option Prices (Heston MC) - {target_date}")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Premium")
    plt.grid(True)
    plt.savefig("output/option_prices.png")
    print("Saved plot to output/option_prices.png")
    
    # Export results
    results = pd.DataFrame({'Strike': strikes, 'CallPrice': prices})
    results.to_csv("output/simulation_results.csv", index=False)
    print("Saved results to output/simulation_results.csv")

if __name__ == "__main__":
    run_simulation()

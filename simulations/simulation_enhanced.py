import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from heston_model.heston_model import HestonModel, estimate_initial_params, calibrate_heston
from engines.monte_carlo_engine import MonteCarloEngine
from black_scholes.black_scholes import black_scholes_call, implied_volatility

def run_simulation_enhanced():
    nifty_df = pd.read_csv("data/nifty_spot.csv", index_col=0, parse_dates=True)
    vix_df = pd.read_csv("data/vix.csv", index_col=0, parse_dates=True)
    
    target_date = "2020-03-23"
    r = 0.05
    
    s0 = nifty_df.loc[target_date, "Close"]
    vix0 = vix_df.loc[target_date, "Close"] / 100.0
    if isinstance(s0, pd.Series): s0 = s0.iloc[0]
    if isinstance(vix0, pd.Series): vix0 = vix0.iloc[0]
    v0 = vix0**2

    print(f"Enhanced Simulation for {target_date}")
    print(f"Nifty Spot: {s0:.2f}, India VIX: {vix0*100:.2f}%")

    hist_vix = vix_df[:target_date]["Close"] / 100.0
    theta_est, kappa_est, sigma_est = estimate_initial_params(hist_vix)
    rho_est = -0.7
    
    T = 30/365
    dt = 1/252
    strikes = np.linspace(s0 * 0.9, s0 * 1.1, 8)
    
    heston_prices = []
    bs_prices = []
    ivs = []
    
    model = HestonModel(s0=s0, v0=v0, kappa=kappa_est, theta=theta_est, sigma=sigma_est, rho=rho_est)
    engine = MonteCarloEngine(model)
    
    print("Pricing Heston vs Black-Scholes...")
    for K in strikes:
        hp, _ = engine.price_european_option(K=K, T=T, dt=dt, num_paths=10000, r=r)
        heston_prices.append(hp)
        
        bp = black_scholes_call(s0, K, T, r, vix0)
        bs_prices.append(bp)
        
        iv = implied_volatility(hp, s0, K, T, r)
        ivs.append(iv)
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(strikes, heston_prices, 'b-o', label='Heston (MC)')
    plt.plot(strikes, bs_prices, 'r--x', label='Black-Scholes (Flat VIX)')
    plt.title("Heston vs Black-Scholes Prices")
    plt.xlabel("Strike")
    plt.ylabel("Premium")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(strikes, [i*100 for i in ivs], 'g-s')
    plt.title("Volatility Smile (IV from Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("output/comparison_and_smile.png")
    
    results_df = pd.DataFrame({
        'Strike': strikes,
        'HestonPrice': heston_prices,
        'BSPrice': bs_prices,
        'ImpliedVol': ivs
    })
    results_df.to_csv("output/enhanced_results.csv", index=False)

if __name__ == "__main__":
    run_simulation_enhanced()

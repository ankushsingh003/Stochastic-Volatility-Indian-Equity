import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from heston_model import HestonModel, estimate_initial_params
from monte_carlo_engine import MonteCarloEngine
from black_scholes import black_scholes_call, implied_volatility

def run_full_analysis():
    # 1. Setup and Data Loading
    print("--- Heston Model Full Analysis ---")
    
    # Load data
    try:
        nifty_df = pd.read_csv("data/nifty_spot.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
        vix_df = pd.read_csv("data/vix.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data files not found. Please run data_loader.py first.")
        return

    # Analysis Date: March 23, 2020 (Volatile period)
    target_date = "2020-03-23"
    r = 0.05 # Risk-free rate
    
    s0 = nifty_df.loc[target_date, "Close"]
    vix0 = vix_df.loc[target_date, "Close"] / 100.0
    if isinstance(s0, pd.Series): s0 = s0.iloc[0]
    if isinstance(vix0, pd.Series): vix0 = vix0.iloc[0]
    v0 = vix0**2

    print(f"Target Date: {target_date}")
    print(f"Spot: {s0:.2f}, VIX: {vix0*100:.2f}%")

    # 2. Parameter Estimation
    hist_vix = vix_df[:target_date]["Close"] / 100.0
    theta_est, kappa_est, sigma_est = estimate_initial_params(hist_vix)
    rho_est = -0.7 # Standard negative correlation for equity
    
    print(f"Estimated Params: kappa={kappa_est:.2f}, theta={theta_est:.4f}, sigma={sigma_est:.4f}, rho={rho_est}")

    # 3. Model Initialization
    model = HestonModel(s0=s0, v0=v0, kappa=kappa_est, theta=theta_est, sigma=sigma_est, rho=rho_est)
    
    # Check Feller Condition
    if model.is_feller_satisfied():
        print("Feller Condition: Satisfied (Variance stays positive)")
    else:
        print("Feller Condition: NOT Satisfied (Variance may hit zero)")

    engine = MonteCarloEngine(model)

    # 4. Pricing and Greeks Calculation
    T = 30/365 # 1 month to expiry
    dt = 1/252
    num_paths = 10000
    
    strikes = np.linspace(s0 * 0.85, s0 * 1.15, 10)
    heston_results = []
    bs_prices = []
    
    print("Calculating prices and Greeks...")
    for K in strikes:
        # Heston Greeks and Price
        res = engine.calculate_greeks(K, T, dt, num_paths, r=r)
        
        # BS Price for comparison
        bp = black_scholes_call(s0, K, T, r, vix0)
        
        # Implied Volatility
        iv = implied_volatility(res['price'], s0, K, T, r)
        
        res['strike'] = K
        res['bs_price'] = bp
        res['iv'] = iv
        heston_results.append(res)

    results_df = pd.DataFrame(heston_results)

    # 5. Visualization
    if not os.path.exists("output"):
        os.makedirs("output")

    plt.figure(figsize=(15, 10))

    # Subplot 1: Price Comparison
    plt.subplot(2, 2, 1)
    plt.plot(results_df['strike'], results_df['price'], 'b-o', label='Heston Price')
    plt.plot(results_df['strike'], results_df['bs_price'], 'r--x', label='BS Price')
    plt.title("Heston vs Black-Scholes Prices")
    plt.xlabel("Strike")
    plt.ylabel("Premium")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Volatility Smile
    plt.subplot(2, 2, 2)
    plt.plot(results_df['strike'], results_df['iv'] * 100, 'g-s')
    plt.title("Heston Implied Volatility Smile")
    plt.xlabel("Strike")
    plt.ylabel("IV (%)")
    plt.grid(True)

    # Subplot 3: Delta
    plt.subplot(2, 2, 3)
    plt.plot(results_df['strike'], results_df['delta'], 'm-d')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Option Delta (Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Delta")
    plt.grid(True)

    # Subplot 4: Gamma
    plt.subplot(2, 2, 4)
    plt.plot(results_df['strike'], results_df['gamma'], 'c-^')
    plt.title("Option Gamma (Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Gamma")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output/full_analysis_greeks.png")
    print("Saved plots to output/full_analysis_greeks.png")

    # Save results to CSV
    results_df.to_csv("output/full_analysis_results.csv", index=False)
    print("Saved results to output/full_analysis_results.csv")

if __name__ == "__main__":
    run_full_analysis()

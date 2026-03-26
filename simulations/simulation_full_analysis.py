import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from heston_model.heston_model import HestonModel, estimate_initial_params
from engines.monte_carlo_engine import MonteCarloEngine
from black_scholes.black_scholes import black_scholes_call, implied_volatility

def run_full_analysis():
    try:
        nifty_df = pd.read_csv("data/nifty_spot.csv", index_col=0, parse_dates=True)
        vix_df = pd.read_csv("data/vix.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return

    target_date = "2020-03-23"
    r = 0.05
    
    s0 = nifty_df.loc[target_date, "Close"]
    vix0 = vix_df.loc[target_date, "Close"] / 100.0
    if isinstance(s0, pd.Series): s0 = s0.iloc[0]
    if isinstance(vix0, pd.Series): vix0 = vix0.iloc[0]
    v0 = vix0**2

    hist_vix = vix_df[:target_date]["Close"] / 100.0
    theta_est, kappa_est, sigma_est = estimate_initial_params(hist_vix)
    rho_est = -0.7
    
    model = HestonModel(s0=s0, v0=v0, kappa=kappa_est, theta=theta_est, sigma=sigma_est, rho=rho_est)
    engine = MonteCarloEngine(model)

    T = 30/365
    dt = 1/252
    num_paths = 10000
    
    strikes = np.linspace(s0 * 0.85, s0 * 1.15, 10)
    heston_results = []
    
    for K in strikes:
        res = engine.calculate_greeks(K, T, dt, num_paths, r=r)
        bp = black_scholes_call(s0, K, T, r, vix0)
        iv = implied_volatility(res['price'], s0, K, T, r)
        
        res['strike'] = K
        res['bs_price'] = bp
        res['iv'] = iv
        heston_results.append(res)

    results_df = pd.DataFrame(heston_results)

    if not os.path.exists("output"):
        os.makedirs("output")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(results_df['strike'], results_df['price'], 'b-o', label='Heston Price')
    plt.plot(results_df['strike'], results_df['bs_price'], 'r--x', label='BS Price')
    plt.title("Heston vs Black-Scholes Prices")
    plt.xlabel("Strike")
    plt.ylabel("Premium")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(results_df['strike'], results_df['iv'] * 100, 'g-s')
    plt.title("Heston Implied Volatility Smile")
    plt.xlabel("Strike")
    plt.ylabel("IV (%)")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(results_df['strike'], results_df['delta'], 'm-d')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Option Delta (Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Delta")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(results_df['strike'], results_df['gamma'], 'c-^')
    plt.title("Option Gamma (Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Gamma")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output/full_analysis_greeks.png")
    results_df.to_csv("output/full_analysis_results.csv", index=False)

if __name__ == "__main__":
    run_full_analysis()

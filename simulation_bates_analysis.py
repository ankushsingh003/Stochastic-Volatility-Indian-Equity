import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bates_model import BatesModel
from heston_model import HestonModel
from monte_carlo_engine import MonteCarloEngine
from black_scholes import implied_volatility

def compare_heston_bates():
    s0, v0, r = 10000, 0.04, 0.05
    kappa, theta, sigma, rho = 2.0, 0.04, 0.3, -0.7
    lamb, mu_j, sigma_j = 0.5, -0.05, 0.1
    
    T = 1/52 
    dt = 1/252
    num_paths = 20000
    
    strikes = np.linspace(s0 * 0.9, s0 * 1.1, 10)
    
    h_model = HestonModel(s0, v0, kappa, theta, sigma, rho)
    b_model = BatesModel(s0, v0, kappa, theta, sigma, rho, lamb, mu_j, sigma_j)
    
    h_engine = MonteCarloEngine(h_model)
    b_engine = MonteCarloEngine(b_model)
    
    h_ivs = []
    b_ivs = []
    
    for K in strikes:
        hp, _ = h_engine.price_european_option(K, T, dt, num_paths, r=r)
        bp, _ = b_engine.price_european_option(K, T, dt, num_paths, r=r)
        
        h_ivs.append(implied_volatility(hp, s0, K, T, r))
        b_ivs.append(implied_volatility(bp, s0, K, T, r))
        
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, [i*100 for i in h_ivs], 'b-o', label='Heston IV')
    plt.plot(strikes, [i*100 for i in b_ivs], 'r-s', label='Bates IV')
    plt.title(f"Volatility Smile Comparison (T = {T*365:.1f} days)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/bates_comparison.png")

if __name__ == "__main__":
    compare_heston_bates()
